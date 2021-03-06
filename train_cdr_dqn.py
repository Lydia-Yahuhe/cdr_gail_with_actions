import numpy as np

from baselines import deepq
from baselines.common import models

from fltenv.env import ConflictEnv

root = ".\\dataset\\my_model"


def train(test=False, path='dqn_policy'):
    env = ConflictEnv(limit=0)
    network = models.mlp(num_hidden=256, num_layers=2)
    if not test:
        act = deepq.learn(
            env,
            network=network,  # 隐藏节点，隐藏层数
            lr=5e-4,
            batch_size=32,
            total_timesteps=200000,
            buffer_size=100000,

            param_noise=True,
            prioritized_replay=True,
        )
        print('Save model to my_model.pkl')
        act.save(root+'.pkl')
    else:
        act = deepq.learn(env,
                          network=network,
                          total_timesteps=0,
                          load_path=root+".pkl")
        env.evaluate(act, save_path=path)
    env.close()


if __name__ == '__main__':
    train()
    # train(test=True, path='random_policy')
    # for key, value in np.load('.\\dataset\\dqn_policy.npz').items():
    #     print(key, value.shape, value)
