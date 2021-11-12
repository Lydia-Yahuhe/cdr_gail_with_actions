from fltenv import ConflictScene
from fltenv.cmd import CmdCount
from fltenv.env import ConflictEnv

from fltsim.visual import *


def generate_random_policy(save_path='policy'):
    num_array = []
    obs_array = []
    act_array = []
    rew_array = []
    n_obs_array = []

    env = ConflictEnv(limit=0)
    size = len(env.train)
    action_list = list(range(CmdCount))

    count = 0
    for info in env.train:
        print('>>>', info.id)
        np.random.shuffle(action_list)
        for i, action in enumerate(action_list):
            scene = ConflictScene(info)

            obs_collected = {'num': [], 'obs': [], 'act': [], 'rew': [], 'n_obs': []}

            obs, done, result = scene.get_states(), False, {}
            while not done:
                # action = np.random.randint(0, CmdCount)
                next_obs, rew, done, result = env.step(action, scene=scene)

                obs_collected['num'].append(scene.info.id)
                obs_collected['obs'].append(obs)
                obs_collected['act'].append(action)
                obs_collected['rew'].append(rew)
                obs_collected['n_obs'].append(next_obs)
                obs = next_obs

            if result['result']:
                num_array += obs_collected['num']
                obs_array += obs_collected['obs']
                act_array += obs_collected['act']
                rew_array += obs_collected['rew']
                n_obs_array += obs_collected['n_obs']
                count += 1

                print('\t', i, count/size*100)
                break
            else:
                print('\t', i)

    num_array = np.array(num_array)
    obs_array = np.array(obs_array, dtype=np.float64)
    act_array = np.array(act_array, dtype=np.float64)
    rew_array = np.array(rew_array, dtype=np.float64)
    n_obs_array = np.array(n_obs_array, dtype=np.float64)

    print('Success Rate is {}%'.format(count * 100.0 / size))
    print(obs_array.shape, act_array.shape, rew_array.shape, n_obs_array.shape)
    np.savez(save_path + '.npz', num=num_array, obs=obs_array, acs=act_array, rews=rew_array, n_obs=n_obs_array)


generate_random_policy('random_policy_1250_all')
