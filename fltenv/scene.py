import numpy as np
import time
from contextlib import contextmanager

from baselines.common import colorize

from fltenv.agent_Set import AircraftAgentSet
from fltenv.cmd import CmdCount, int_2_atc_cmd, check_cmd


@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'), end=' ')
    tstart = time.time()
    yield
    print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))


class ConflictScene:
    def __init__(self, info, limit=0):
        self.info = info

        self.conflict_ac, self.clock = info.conflict_ac, info.time

        self.agentSet = AircraftAgentSet(fpl_list=info.fpl_list, start=info.start)
        self.agentSet.do_step(self.clock - 300 + limit, basic=True)
        self.conflict_pos = info.other[0]

        # print('\nNew scenario--------------------------------')
        # print(' Conflict Info: ', self.conflict_ac, self.clock, self.agentSet.time, len(info.fpl_list))

        self.cmd_check_dict = {ac: {'HDG': [], 'ALT': [], 'SPD': []} for ac in self.conflict_ac}
        self.cmd_info = {}

    def now(self):
        return self.agentSet.time

    # def get_states(self):
    #     state = [[0.0 for _ in range(7)] for _ in range(50)]
    #
    #     j = 0
    #     for agent in self.agentSet.agent_en:
    #         pos = agent.position
    #         v_spd, h_spd, hdg = agent.status.vSpd, agent.status.hSpd, agent.status.heading
    #         ele = [int(agent.id in self.conflict_ac),
    #                pos[0] - self.conflict_pos[0],
    #                pos[1] - self.conflict_pos[1],
    #                (pos[2] - self.conflict_pos[2]) / 3000,
    #                (h_spd - 150) / 100,
    #                 v_spd / 20,
    #                 hdg / 180]
    #
    #         state[min(50 - 1, j)] = ele
    #         j += 1
    #
    #     return np.concatenate(state)

    def get_state(self, ac_en, limit=50):
        state = [[0.0 for _ in range(7)] for _ in range(limit)]

        j = 0
        for agent in ac_en:
            pos = agent.position
            v_spd, h_spd, hdg = agent.status.vSpd, agent.status.hSpd, agent.status.heading
            ele = [int(agent.id in self.conflict_ac),
                   pos[0] - self.conflict_pos[0],
                   pos[1] - self.conflict_pos[1],
                   (pos[2] - self.conflict_pos[2]) / 3000,
                   (h_spd - 150) / 100,
                   v_spd / 20,
                   hdg / 180]

            state[min(limit - 1, j)] = ele
            j += 1
        return state

    def get_states(self):
        states = [self.get_state(self.agentSet.agent_en)]

        ghost = AircraftAgentSet(other=self.agentSet)
        ghost.do_step(duration=60)

        states.append(self.get_state(ghost.agent_en))

        ghost.do_step(duration=60)
        states.append(self.get_state(ghost.agent_en))

        states = np.vstack(states)
        return np.concatenate(states)

    def do_step(self, action):
        agent_id, idx = self.conflict_ac[0], action

        # 指令解析
        now = self.now()
        agent = self.agentSet.agents[agent_id]
        [hold, *cmd_list] = int_2_atc_cmd(now + 1, idx, agent)
        print('{:>4d}, {:>4d}'.format(idx, hold), end=', ')

        # 执行hold，并探测冲突
        while self.now() < now + hold:
            self.agentSet.do_step(duration=15)
            conflicts = self.agentSet.detect_conflict_list(search=self.conflict_ac)
            if len(conflicts) > 0:
                return False, None  # solved, done, cmd

        # 分配动作
        for cmd in cmd_list:
            cmd.ok, reason = check_cmd(cmd, agent, self.cmd_check_dict[agent_id])
            # print(now, hold, cmd.assignTime, self.now())
            print('{:>+5d}, {}'.format(int(cmd.delta), int(cmd.ok)), end=', ')
            agent.assign_cmd(cmd)
        cmd_info = {'agent': agent_id, 'cmd': cmd_list, 'hold': hold}
        self.cmd_info[now] = cmd_info

        # 执行动作并探测冲突
        has_conflict = self.__do_step(self.clock + 300, duration=15)

        return not has_conflict, cmd_info  # solved, done, cmd

    def __do_step(self, end_time, duration):
        while self.now() < end_time:
            self.agentSet.do_step(duration=duration)
            conflicts = self.agentSet.detect_conflict_list(search=self.conflict_ac)
            if len(conflicts) > 0:
                return True
        return False
