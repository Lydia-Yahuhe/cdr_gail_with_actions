import csv

from fltenv import ConflictScene
from fltsim.load import load_and_split_data
from fltsim.utils import equal

scenarios, _ = load_and_split_data('scenarios_gail_final', split_ratio=1.0)


def generate_surrounding_aircraft_tracks():
    for i, scenario in enumerate(scenarios):
        scene = ConflictScene(scenario, limit=0)
        c_clock = scenario.time
        print(scenario.id, scenario.start, scenario.end, len(scenario.fpl_list))
        agent_set = scene.agentSet
        print(len(agent_set.agents))

        print(agent_set.time, end=', ')
        conflicts = []
        while agent_set.time < c_clock+300:
            agent_set.do_step()
            conflicts += agent_set.detect_conflict_list(search=scene.conflict_ac)
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


generate_surrounding_aircraft_tracks()
