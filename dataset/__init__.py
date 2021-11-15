import math

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns

from fltenv.cmd import CmdCount
from fltsim.utils import convert_with_align


def kl_divergence(p, q):
    return scipy.stats.entropy(p, q)


def get_kde(x, data_array, bandwidth=0.1):
    def gauss(x):
        import math
        return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * (x ** 2))

    N = len(data_array)
    res = 0
    if len(data_array) == 0:
        return 0
    for i in range(len(data_array)):
        res += gauss((x - data_array[i]) / bandwidth)
    res /= (N * bandwidth)
    return res


def get_pdf_or_kde(input_array, bins):
    bandwidth = 1.05 * np.std(input_array) * (len(input_array) ** (-1 / 5))
    x_array = np.linspace(0, bins - 1, num=100)
    y_array = [get_kde(x_array[i], input_array, bandwidth) for i in range(x_array.shape[0])]
    return y_array


def visual(x, y, bins, show=True):
    # 图表设置
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    # plt.rcParams['figure.figsize'] = (50, 50)  # 设定图片大小
    f = plt.figure()  # 确定画布

    sns.set()  # 设置seaborn默认格式
    np.random.seed(0)  # 设置随机种子数

    # KL散度
    dqn_kde = get_pdf_or_kde(x, bins=bins)
    gail_kde = get_pdf_or_kde(y, bins=bins)
    kl = round(kl_divergence(dqn_kde, gail_kde), 3)

    f.add_subplot(2, 1, 1)
    plt.hist([x, y], bins=bins, range=(0, bins), density=True, align='left', label=['DQN', 'GAIL'])  # 绘制x的密度直方图

    plt.xlabel('Action index')
    plt.ylabel("Frequency")
    # plt.xticks(np.arange(-1, bins, 1))  # 设置x轴刻度值的字体大小
    # plt.yticks(np.arange(0.0, 1.05, 0.1))  # 设置y轴刻度值的字体大小
    plt.title("The histogram of DQN and GAIL policy", fontsize=12)  # 设置子图标题
    plt.legend()

    f.add_subplot(2, 1, 2)
    sns.distplot(x, bins=bins, hist=False, label='DQN')  # 绘制x的密度直方图
    sns.distplot(y, bins=bins, hist=False, label='GAIL')  # 绘制y的密度直方图
    plt.xlabel('Action index')
    plt.ylabel("KDE")
    # plt.xticks(np.arange(0, bins, 10))  # 设置x轴刻度值的字体大小
    # plt.yticks(np.arange(0.0, 1.0, 0.1))  # 设置y轴刻度值的字体大小
    plt.title("The similarity of DQN and GAIL policy, KL divergence={}".format(kl), fontsize=12)  # 设置子图标题
    plt.legend()

    plt.subplots_adjust(wspace=0.2, hspace=0.5)  # 调整两幅子图的间距
    plt.savefig('distribution_action.svg')

    if show:
        plt.show()

    return kl


def parse_idx_2_action(idx):
    time_idx, cmd_idx = idx // 9, idx % 9
    time_cmd = 240-int(time_idx) * 15  # time cmd
    [alt_idx, hdg_idx] = convert_with_align(cmd_idx, x=3, align=2)  # 将idx转化成三进制数
    alt_cmd = (int(alt_idx) - 1) * 600.0  # alt cmd
    hdg_cmd = (int(hdg_idx) - 1) * 45  # hdg cmd
    return time_cmd, alt_cmd, hdg_cmd


def analysis_two_actions(a1, a2):
    cmd_list_a1 = parse_idx_2_action(a1)
    cmd_list_a2 = parse_idx_2_action(a2)

    print(a1, a2, cmd_list_a1, cmd_list_a2)
    time_diff = abs(cmd_list_a1[0] - cmd_list_a2[0]) / 240
    alt_diff = abs(cmd_list_a1[1] - cmd_list_a2[1]) / 600
    hdg_diff = abs(cmd_list_a1[2] - cmd_list_a2[2]) / 45

    return time_diff**2+alt_diff**2+hdg_diff**2


def visual_action_distribution():
    bins = CmdCount

    expert_path = 'random_policy_125_all.npz'
    size = int(expert_path.split('_')[2]) * 0.8

    # Learned Policy
    dqn_policy = np.load(expert_path)
    dqn_rew = dqn_policy['rews']
    print('LP SR:', len(dqn_rew) / size * 100, np.mean(dqn_rew))

    # Imitation Policy
    gail_policy = np.load('gail.seed_0.iters_5000000.BC_100000_evaluate.npz')
    gail_rew = gail_policy['rews']
    print('IP SR:', len(gail_rew) / size * 100, np.mean(gail_rew))

    dqn_acs, gail_acs = dqn_policy['acs'], gail_policy['acs']
    print('LP acs:', dqn_acs)
    print('IP acs:', gail_acs)

    # Action Distribution (AD)
    kl = visual(dqn_acs, gail_acs, bins=bins, show=True)
    print('KL divergence:', kl)

    dqn_num = list(dqn_policy['num'])
    gail_num = list(gail_policy['num'])
    print('LP num:', dqn_num)
    print('IP num:', gail_num)

    # Matching Degree (MD)
    num_ip_in_lp = [int(num in dqn_num) for num in gail_num]
    num_ip_lp = sum(num_ip_in_lp)
    num_lp = len(dqn_num)
    print('MD: ', num_ip_lp / num_lp * 100, np.mean(num_ip_in_lp) == 1.0)

    # Mean Square Error (MSE)
    square_error_list = []
    similarity_list = []
    for i, num in enumerate(gail_num):
        if num not in dqn_num:
            continue

        idx = dqn_num.index(num)
        dqn_action = int(dqn_acs[idx])
        gail_action = int(gail_acs[i])

        # Action Similarity(AS)
        similarity = analysis_two_actions(dqn_action, gail_action)
        similarity_list.append(similarity)
        square_error_list.append(math.pow(dqn_action - gail_action, 2))

    print('MSE:', np.mean(square_error_list))
    print('AS:', np.mean(similarity_list))


# 随机数生成
def random_policy_generator():
    data_set = np.load('.\\dqn_policy.npz')

    tmp = {}
    for key, value in data_set.items():
        print(key, value.shape)
        if key == 'acs':
            value = np.random.randint(1, 10, value.shape)
        tmp[key] = value

    np.savez('dqn_policy_1.npz', **tmp)


if __name__ == '__main__':
    # random_policy_generator()
    visual_action_distribution()
