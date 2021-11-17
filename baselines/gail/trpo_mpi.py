"""
Disclaimer: The trpo part highly rely on trpo_mpi at @openai/baselines
"""

import time
from contextlib import contextmanager
from mpi4py import MPI
from collections import deque

import tensorflow as tf
import numpy as np

import baselines.common.tf_util as U
from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
from baselines.common import colorize
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg


def traj_segment_generator(pi, env, reward_giver, horizon, stochastic):
    # Initialize state variables
    t = 0
    new = True
    rew = 0.0
    true_rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.zeros([horizon, env.action_space.n], 'float32')
    names = ['' for _ in range(horizon)]

    while True:
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "nextvpred": vpred * (1 - new), 'names': names,
                   "ep_rets": ep_rets, "ep_true_rets": ep_true_rets}
            _, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        names[i] = env.scene.info.id

        rew = reward_giver.get_reward(ob, ac)
        print('{:>6.4f}'.format(round(rew, 4)), end=', ')
        ob, true_rew, new, _ = env.step(np.argmax(ac))
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_true_ret += true_rew
        if new:
            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            ob = env.reset()
            print('step', t)
        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def learn(env, policy_func, reward_giver, expert_dataset, rank,
          pretrained_weight, *,
          g_step, d_step, entcoeff, save_per_iter,
          ckpt_dir, timesteps_per_batch,
          gamma, lam,
          max_kl, cg_iters, cg_damping=1e-2,
          vf_stepsize=3e-4, d_stepsize=3e-4, vf_iters=3,
          max_timesteps=0, max_iters=0,
          callback=None):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=pretrained_weight)
    oldpi = policy_func("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    ob_ph = U.get_placeholder_cached(name="ob")
    ac_ph = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = entcoeff * meanent

    vferr = tf.reduce_mean(tf.square(pi.vpred - ret))

    ratio = tf.exp(pi.pd.logp(ac_ph) - oldpi.pd.logp(ac_ph))  # advantage * pnew / pold
    surrgain = tf.reduce_mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "g_entropy"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.startswith("pi/pol") or v.name.startswith("pi/logstd")]
    vf_var_list = [v for v in all_var_list if v.name.startswith("pi/vff")]
    # assert len(var_list) == len(vf_var_list) + 1
    d_adam = MpiAdam(reward_giver.get_trainable_variables())
    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob_ph, ac_ph, atarg], losses)
    compute_lossandgrad = U.function([ob_ph, ac_ph, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob_ph, ac_ph, atarg], fvp)
    compute_vflossandgrad = U.function([ob_ph, ret], U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    # writer = tf.summary.FileWriter('logs', tf.get_default_session().graph)
    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    d_adam.sync()
    vfadam.sync()
    if rank == 0:
        print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, reward_giver, timesteps_per_batch, stochastic=True)

    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    rewbuffer = deque(maxlen=g_step*timesteps_per_batch)  # rolling buffer for episode rewards
    true_rewbuffer = deque(maxlen=g_step*timesteps_per_batch)

    assert sum([max_iters > 0, max_timesteps > 0]) == 1

    # if provide pretrained weight
    if pretrained_weight:
        # U.load_state(pretrained_weight, var_list=pi.get_variables())
        U.load_variables(ckpt_dir, variables=pi.get_variables())
        print('Load previous trained variables successfully!')

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_iters and iters_so_far >= max_iters:
            break

        # Save model
        if rank == 0 and iters_so_far % save_per_iter == 0 and ckpt_dir is not None:
            U.save_variables(ckpt_dir, variables=pi.get_variables())

        logger.log("********** Iteration %i ************" % iters_so_far)

        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p
        # ------------------ Update G ------------------
        logger.log("Optimizing Policy...")
        ob_array, ac_array, num_array = [], [], []
        for _ in range(g_step):
            with timed("sampling"):
                seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, gamma, lam)
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            ob_array.append(ob)
            ac_array.append(ac)
            num_array += seg['names']

            vpredbefore = seg["vpred"]  # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
            if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

            args = seg["ob"], seg["ac"], atarg
            fvpargs = [arr[::5] for arr in args]

            assign_old_eq_new()  # set old parameter values to new parameter values
            with timed("computegrad"):
                *lossbefore, g = compute_lossandgrad(*args)
            lossbefore = allmean(np.array(lossbefore))
            g = allmean(g)
            if np.allclose(g, 0):
                logger.log("Got zero gradient. not updating")
            else:
                with timed("cg"):
                    stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank == 0)
                assert np.isfinite(stepdir).all()
                shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / max_kl)
                fullstep = stepdir / lm
                expectedimprove = g.dot(fullstep)
                surrbefore = lossbefore[0]
                stepsize = 1.0
                thbefore = get_flat()
                for _ in range(10):
                    thnew = thbefore + fullstep * stepsize
                    set_from_flat(thnew)
                    meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                    improve = surr - surrbefore
                    logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                    if not np.isfinite(meanlosses).all():
                        logger.log("Got non-finite value of losses -- bad!")
                    elif kl > max_kl * 1.5:
                        logger.log("violated KL constraint. shrinking step.", kl)
                    elif improve < 0:
                        logger.log("surrogate didn't improve. shrinking step.")
                    else:
                        logger.log("Step size OK!")
                        break
                    stepsize *= .5
                else:
                    logger.log("couldn't compute a good step")
                    set_from_flat(thbefore)

                if nworkers > 1 and iters_so_far % 20 == 0:
                    paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum()))  # list of tuples
                    assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])
            with timed("vf"):
                for _ in range(vf_iters):
                    for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                                                             include_final_partial_batch=False, batch_size=128):
                        if hasattr(pi, "ob_rms"):
                            pi.ob_rms.update(mbob)  # update running mean/std for policy
                        g = allmean(compute_vflossandgrad(mbob, mbret))
                        vfadam.update(g, vf_stepsize)

        for (name, loss) in zip(loss_names, meanlosses):
            logger.record_tabular(name, loss)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        ob_array = np.concatenate(ob_array)
        ac_array = np.concatenate(ac_array)
        # print(ob_array.shape, ac_array.shape)

        batch_size = len(ob_array) // d_step
        d_losses = []  # list of tuples, each of which gives the loss for a mini-batch
        for ob_batch, ac_batch in dataset.iterbatches((ob_array, ac_array),
                                                      include_final_partial_batch=False,
                                                      batch_size=batch_size):
            real_batch_size = len(ob_batch)
            ob_expert, ac_expert = expert_dataset.get_next_batch(real_batch_size, names=num_array[:real_batch_size])
            # print(ob_batch.shape, ac_batch.shape, ob_expert.shape, ac_expert.shape)
            # update running mean/std for reward_giver
            if hasattr(reward_giver, "obs_rms"): reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))

            ac_expert = np.eye(env.action_space.n)[ac_expert]
            *newlosses, g = reward_giver.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
            d_adam.update(allmean(g), d_stepsize)
            d_losses.append(newlosses)

        for (name, loss) in zip(reward_giver.loss_name, np.mean(d_losses, axis=0)):
            logger.record_tabular(name, loss)

        lrlocal = (seg["ep_rets"], seg["ep_true_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        rews, true_rets = map(flatten_lists, zip(*listoflrpairs))
        true_rewbuffer.extend(true_rets)
        rewbuffer.extend(rews)

        timesteps_so_far += len(rews)
        iters_so_far += 1
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpTrueRewMean", np.mean(true_rewbuffer))
        logger.record_tabular("StepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank == 0:
            logger.dump_tabular()


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
