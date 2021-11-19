"""
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
"""
import tensorflow as tf
import numpy as np

from baselines import logger
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common import tf_util as U


def logsigmoid(a):
    """Equivalent to tf.log(tf.sigmoid(a))"""
    return -tf.nn.softplus(-a)


"""
Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51
"""


def logit_bernoulli_entropy(logits):
    ent = (1. - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent


class TransitionClassifier(object):
    def __init__(self, env, hidden_size, entcoeff=0.001, scope="adversary"):
        self.scope = scope
        self.observation_shape = env.observation_space.shape
        self.actions_shape = (env.action_space.n, )
        # self.actions_shape = env.action_space.shape

        self.generator_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape, name="observations_ph")
        self.generator_acs_ph = tf.placeholder(tf.float32, (None,) + self.actions_shape, name="actions_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape, name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(tf.float32, (None,) + self.actions_shape, name="expert_actions_ph")

        # Build graph
        generator_logits = self.__build_graph(self.generator_obs_ph, self.generator_acs_ph, hidden_size=hidden_size)
        expert_logits = self.__build_graph(self.expert_obs_ph, self.expert_acs_ph, hidden_size=hidden_size, reuse=True)

        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) <= 0.001))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) >= 0.999))

        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits, labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)

        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff * entropy

        # Loss + Accuracy terms
        self.total_loss = generator_loss + expert_loss + entropy_loss
        self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc, self.total_loss]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc", "total_loss"]

        # Build Reward for policy
        # self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)
        self.reward_op = -(1 - tf.nn.sigmoid(generator_logits))

        var_list = self.get_trainable_variables()
        self.adam = MpiAdam(var_list, epsilon=1e-5)
        self.lossandgrad = U.function(
            [self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph],
            self.losses + [U.flatgrad(self.total_loss, var_list)])

    def __build_graph(self, obs_ph, acs_ph, hidden_size=128, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=self.observation_shape)
            obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std
            _input = tf.concat([obs, acs_ph], axis=1)  # concatenate the two input -> form a transition
            p_h1 = tf.contrib.layers.fully_connected(_input, hidden_size, activation_fn=tf.nn.tanh)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, hidden_size, activation_fn=tf.nn.tanh)
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_reward(self, obs, acs):
        sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(acs.shape) == 1:
            acs = np.expand_dims(acs, 0)
        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: acs}
        reward = sess.run(self.reward_op, feed_dict)
        return reward


if __name__ == '__main__':
    from tqdm import tqdm

    from fltenv.env import ConflictEnv
    from baselines.gail.dataset.mujoco_dset import Mujoco_Dset

    env = ConflictEnv(limit=0)

    num_samples = 125
    dataset = Mujoco_Dset(expert_path='dataset\\random_policy_{}_all.npz'.format(num_samples))

    reward_giver = TransitionClassifier(env, 128, entcoeff=0.1)
    adam = reward_giver.adam

    U.initialize()
    adam.sync()

    num_steps = int(2e5)
    num_actions = env.action_space.n
    actions = list(range(num_actions))

    U.load_variables('discriminator_200000_125', variables=reward_giver.get_variables())
    ob_expert, ac_expert = dataset.get_next_batch(-1)
    ac_expert = np.eye(num_actions)[ac_expert]
    rew = reward_giver.get_reward(ob_expert, ac_expert)
    print(max(rew), min(rew))

    col = []
    for iter_so_far in tqdm(range(1, num_steps+1)):
        ob_expert, ac_expert = dataset.get_next_batch(32)
        ac_expert = ac_expert.squeeze()

        ob_batch = ob_expert[:, :]
        ac_batch = []
        for ac_e in ac_expert:
            np.random.shuffle(actions)
            for act in actions:
                if act != ac_e:
                    ac_batch.append(act)
                    break
            else:
                raise NotImplementedError
        ac_batch = np.eye(num_actions)[ac_batch]
        ac_expert = np.eye(num_actions)[ac_expert]
        *train_loss, g = reward_giver.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
        adam.update(g, 1e-4)

        # print(train_loss)
        col.append(train_loss)

        if iter_so_far % 100 == 0:
            train_loss = np.mean(col, axis=0)
            print(train_loss.shape)
            logger.record_tabular("step", iter_so_far)
            logger.record_tabular("g_loss", train_loss[0])
            logger.record_tabular("e_loss", train_loss[1])
            logger.record_tabular("entropy", train_loss[2])
            logger.record_tabular("entropy_loss", train_loss[3])
            logger.record_tabular("g_acc", train_loss[4])
            logger.record_tabular("e_acc", train_loss[5])
            logger.record_tabular("t_loss", train_loss[6])

            ob_expert, ac_expert = dataset.get_next_batch(-1)
            ac_expert = np.squeeze(ac_expert)

            ob_batch = ob_expert[:, :]
            ac_batch = []
            for ac_e in ac_expert:
                np.random.shuffle(actions)
                for act in actions:
                    if act != ac_e:
                        ac_batch.append(act)
                        break
                else:
                    raise NotImplementedError
            ac_batch = np.eye(num_actions)[ac_batch]
            ac_expert = np.eye(num_actions)[ac_expert]

            rewards_expert = reward_giver.get_reward(ob_expert, ac_expert)
            logger.record_tabular("rewards_expert", np.mean(rewards_expert))

            rewards_batch = reward_giver.get_reward(ob_batch, ac_batch)
            logger.record_tabular("rewards_batch", np.mean(rewards_batch))

            print(ac_expert.shape, ac_batch.shape, rewards_expert.shape, rewards_batch.shape)

            logger.dump_tabular()

            if iter_so_far % 10000 == 0:
                U.save_variables('discriminator_{}_{}'.format(num_steps, num_samples),
                                 variables=reward_giver.get_variables())



