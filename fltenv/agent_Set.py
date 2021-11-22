from __future__ import annotations
from dataclasses import dataclass

from fltsim.aircraft import AircraftAgent
from fltsim.utils import make_bbox, distance, build_rt_index
from fltsim.visual import save_to_kml


@dataclass
class Conflict:
    id: str
    time: int
    hDist: float
    vDist: float
    pos0: tuple
    pos1: tuple

    def to_dict(self):
        return dict(id=self.id, time=self.time, hDist=self.hDist, vDist=self.vDist,
                    pos0=self.pos0, pos1=self.pos1)


class AircraftAgentSet:
    def __init__(self, fpl_list=None, start=None, supply=None, other=None):
        if fpl_list:
            self.time = start or -1
            self.agents = {}
            for fpl in fpl_list:
                if supply is not None and fpl.id not in supply[1]:
                    continue
                self.agents[fpl.id] = AircraftAgent(fpl)

            self.tracks = supply[0] if supply is not None else {}
            self.check_list = []
        else:
            self.time = other.time
            self.agents = {a_id: agent.copy() for a_id, agent in other.agents.items()}
            self.tracks = other.tracks
            self.check_list = other.check_list[:]

        self.agent_en, self.agent_en_ = [], []

    def do_step(self, duration=1, basic=False):
        now = self.time
        duration -= now * int(basic)
        self.agent_en = []

        for key, agent in self.agents.items():
            if agent.is_finished():
                continue

            agent.do_step(now, duration)
            if agent.is_enroute():
                self.agent_en.append([key]+agent.get_x_data())

        now += duration

        ac_en = []
        for key, track in self.tracks.items():
            if now in track.keys():
                ac_en.append(track[now])

        self.agent_en_ = self.agent_en + ac_en
        self.time = now

    def detect_conflict_list(self, search=None):
        conflicts, agent_en = [], self.agent_en_

        if len(agent_en) <= 1:
            return []

        r_tree = build_rt_index(agent_en)
        check_list = []
        for [a0, *state_0] in self.agent_en:
            if search is not None and a0 not in search:
                continue

            bbox = make_bbox(state_0[:3], (0.1, 0.1, 299))

            for i in r_tree.intersection(bbox):
                [a1, *state_1] = agent_en[i]
                if a0 == a1 or a0+'-'+a1 in check_list+self.check_list:
                    continue

                check_list.append(a1+'-'+a0)
                self.detect_conflict(a0, a1, state_0, state_1, conflicts)

        return conflicts

    def detect_conflict(self, a0, a1, pos0, pos1, conflicts):
        h_dist = distance(pos0[:2], pos1[:2])
        v_dist = abs(pos0[2] - pos1[2])
        if h_dist >= 10000 or v_dist >= 300.0:
            return

        self.check_list.append(a0 + '-' + a1)
        self.check_list.append(a1 + '-' + a0)

        conflicts.append(Conflict(id=a0+"-"+a1, time=self.time, hDist=h_dist, vDist=v_dist, pos0=pos0, pos1=pos1))

    def visual(self, save_path='agentSet', limit=None):
        tracks_real = {}
        tracks_plan = {}
        for a_id, agent in self.agents.items():
            if limit is not None and a_id not in limit:
                continue

            tracks_real[a_id] = [tuple(track[:3]) for track in agent.tracks.values()]
            tracks_plan[a_id] = [(point.location.lng, point.location.lat, 8100.0)
                                 for point in agent.fpl.routing.waypointList]
        save_to_kml(tracks_real, tracks_plan, save_path=save_path)
