import csv

from fltenv import ConflictScene
from fltsim.load import load_and_split_data
from fltsim.utils import equal

scenarios, _ = load_and_split_data('scenarios_gail_final', split_ratio=1.0)


def generate_surrounding_aircraft_tracks():
    for i, scenario in enumerate(scenarios):
        if i != 98:
            continue

        scene = ConflictScene(scenario, limit=0, read=False)
        c_clock = scenario.time
        print(scenario.id, scenario.start, scenario.end, len(scenario.fpl_list))
        agent_set = scene.agentSet
        print(len(agent_set.agents))

        print(agent_set.time, end=', ')
        conflicts = []
        while agent_set.time < c_clock+300:
            agent_set.do_step()
            conflicts += agent_set.detect_conflict_list(search=scenario.conflict_ac)
        print(agent_set.time)
        assert len(conflicts) > 0

        c_acs = []
        for c in conflicts:
            print(c.id)
            c_acs += c.id.split('-')
        c_acs = list(set(c_acs))

        print(len(conflicts), c_acs, scene.conflict_ac, conflicts[0].time, c_clock)
        assert equal(c_acs, scene.conflict_ac) and conflicts[0].time == c_clock

        with open('No.'+scenario.id+'.csv', 'w', newline='') as f:
            f = csv.writer(f)
            time_range = list(range(c_clock-300, c_clock+301))
            print(time_range)
            for key, agent in agent_set.agents.items():
                if key in scenario.conflict_ac:
                    continue

                fpl = agent.fpl
                timestamps, max_timestamp = [], 0
                for timestamp, track in agent.tracks.items():
                    max_timestamp = timestamp
                    if timestamp not in time_range:
                        continue

                    timestamps.append(timestamp)
                    f.writerow([key, timestamp]+track)
                if len(timestamps) <= 0:
                    print(fpl.startTime, max_timestamp, fpl.routing.id, c_clock-300, c_clock+300, 'False')
                else:
                    print(min(timestamps), max(timestamps), agent.fpl.startTime, 'True')


# generate_surrounding_aircraft_tracks()


import numpy as np
import math


def tmp_test():
    for i, scenario in enumerate(scenarios):
        if i != 98:
            continue

        scene_read = ConflictScene(scenario, limit=0)
        scene_no_read = ConflictScene(scenario, limit=0, read=False)
        scenario.to_string()

        obs_read, check_1 = scene_read.get_states()
        print(len(scene_read.agentSet.agents))
        obs_no_read, check_2 = scene_no_read.get_states()
        a = scene_no_read.agentSet.agents['CBJ5765']
        print(scene_no_read.agentSet.time, a.fpl.startTime, a.is_enroute())
        print(len(scene_no_read.agentSet.agents))

        print(check_1)
        print(check_2)
        print(len(check_2) == len(check_1))
        for k, c1 in enumerate(check_1):
            if c1 != check_2[k]:
                print(k, 'Not Equal', c1, check_2[k])
                break

        sum_square = 0.0
        for j, ob1 in enumerate(list(obs_read)):
            sum_square = math.pow(list(obs_no_read)[j] - ob1, 2)
            if sum_square > 0.5:
                print(j, ob1, list(obs_no_read)[j])

        print(sum_square)


tmp_test()
