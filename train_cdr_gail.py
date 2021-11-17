"""
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
"""

import argparse
import os.path as osp
import logging
from mpi4py import MPI

import gym

from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines import logger
from baselines.gail.dataset.mujoco_dset import Mujoco_Dset
from baselines.gail.adversary import TransitionClassifier
from baselines.gail import trpo_mpi, mlp_policy, behavior_clone

from fltenv.env import ConflictEnv


def args_parser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='.\\dataset\\random_policy_125_all.npz')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'test'], default='train')
    # for evaluation
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    # Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=3)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=128)
    parser.add_argument('--adversary_hidden_size', type=int, default=128)
    # Algorithms Configuration
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=1e-3)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Training Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=2)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=int(1e5))
    # Behavior Cloning
    boolean_flag(parser, 'pretrained', default=True, help='Use BC to pretrain')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=int(1e3))
    return parser.parse_args()


def main():
    args = args_parser()

    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)

    env = ConflictEnv(limit=0, reverse=args.task == 'evaluate')
    # env = ConflictEnv(limit=0, act='Box', reverse=args.task == 'evaluate')
    env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)

    task_name = "gail.seed_{}.iters_{}".format(args.seed, args.num_timesteps)
    if args.pretrained:
        task_name += ".BC_{}".format(args.BC_max_iter)
    checkpoint_dir = osp.abspath(args.checkpoint_dir)
    save_dir = osp.join(checkpoint_dir, task_name)
    print(save_dir)

    if args.task == 'train':
        # expert demonstrations
        dataset = Mujoco_Dset(expert_path=args.expert_path)

        reward_giver = TransitionClassifier(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff)

        if args.pretrained:
            # Pretrain with behavior cloning
            behavior_clone.learn(env, policy_fn, dataset, ckpt_dir=save_dir, max_iters=args.BC_max_iter)

        # Set up for MPI seed
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        worker_seed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(worker_seed)
        env.seed(worker_seed)

        trpo_mpi.learn(env, policy_fn, reward_giver, dataset, rank,
                       pretrained_weight=args.pretrained,
                       g_step=args.g_step, d_step=args.d_step,
                       entcoeff=args.policy_entcoeff,
                       max_timesteps=args.num_timesteps,
                       ckpt_dir=save_dir,
                       save_per_iter=args.save_per_iter,
                       timesteps_per_batch=32,
                       max_kl=0.01, cg_iters=10, cg_damping=0.1,
                       gamma=0.995, lam=0.97,
                       vf_iters=5, vf_stepsize=1e-3)
    else:
        output_gail_policy(env,
                           policy_fn,
                           save_dir,
                           task_name+'_'+args.task,
                           stochastic_policy=False)
    env.close()


def output_gail_policy(env, policy_func, load_model_path, task_name, stochastic_policy=False, reuse=False):
    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=reuse)
    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    print(load_model_path)
    U.load_variables(load_model_path, variables=pi.get_variables())
    print(task_name)
    env.evaluate(pi.act, save_path=task_name, stochastic=stochastic_policy)


if __name__ == '__main__':
    main()
