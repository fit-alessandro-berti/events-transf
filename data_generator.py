
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Event:
    activity: str
    timestamp: float  # minutes since case start
    resource: str
    cost: float
    pos: int


@dataclass
class Trace:
    case_id: str
    model_id: str
    events: List[Event]
    end_time: float
    next_activity_ids: List[int]  # filled later
    remaining_times: List[float]


class ProcessModel:
    def __init__(
        self,
        name: str,
        start_activity: str,
        transitions: Dict[str, List[Tuple[str, float]]],
        duration_params: Dict[str, Tuple[float, float]],
        resource_probs: Dict[str, List[Tuple[str, float]]],
        cost_params: Dict[str, Tuple[float, float]],
    ):
        self.name = name
        self.start_activity = start_activity
        self.transitions = transitions
        self.duration_params = duration_params
        self.resource_probs = resource_probs
        self.cost_params = cost_params

    def _sample_duration(self, act: str) -> float:
        mu, sigma = self.duration_params[act]
        return float(np.random.lognormal(mean=mu, sigma=sigma))

    def _sample_resource(self, act: str) -> str:
        choices, probs = zip(*self.resource_probs[act])
        return random.choices(list(choices), weights=list(probs), k=1)[0]

    def _sample_cost(self, act: str) -> float:
        m, s = self.cost_params[act]
        return float(np.random.normal(loc=m, scale=s))

    def _sample_next(self, act: str) -> str:
        choices, probs = zip(*self.transitions.get(act, []))
        return random.choices(list(choices), weights=list(probs), k=1)[0]

    def simulate_case(self, case_id: str) -> Trace:
        t = 0.0
        pos = 0
        events: List[Event] = []
        act = self.start_activity
        while act != "END":
            dur = self._sample_duration(act)
            res = self._sample_resource(act)
            cost = self._sample_cost(act)
            events.append(Event(activity=act, timestamp=t, resource=res, cost=cost, pos=pos))
            pos += 1
            t += dur
            act = self._sample_next(act)
        end_time = t
        remaining_times = [end_time - e.timestamp for e in events]
        return Trace(case_id=case_id, model_id=self.name, events=events, end_time=end_time,
                     next_activity_ids=[], remaining_times=remaining_times)


def build_process_models() -> List[ProcessModel]:
    R = ["r1", "r2", "r3", "r4"]

    m1 = ProcessModel(
        name="Model_1",
        start_activity="A",
        transitions={
            "A": [("B", 1.0)],
            "B": [("C", 0.7), ("X", 0.3)],
            "C": [("D", 0.85), ("B", 0.15)],
            "X": [("D", 1.0)],
            "D": [("END", 1.0)],
        },
        duration_params={
            "A": (math.log(15), 0.30),
            "B": (math.log(25), 0.35),
            "C": (math.log(20), 0.40),
            "X": (math.log(10), 0.25),
            "D": (math.log(30), 0.30),
        },
        resource_probs={
            "A": [(R[0], 0.7), (R[1], 0.3)],
            "B": [(R[1], 0.6), (R[2], 0.4)],
            "C": [(R[2], 0.7), (R[3], 0.3)],
            "X": [(R[0], 0.5), (R[3], 0.5)],
            "D": [(R[1], 0.5), (R[2], 0.5)],
        },
        cost_params={
            "A": (50, 5),
            "B": (80, 8),
            "C": (60, 6),
            "X": (30, 3),
            "D": (120, 12),
        },
    )

    m2 = ProcessModel(
        name="Model_2",
        start_activity="A",
        transitions={
            "A": [("B", 0.5), ("E", 0.5)],
            "B": [("C", 1.0)],
            "C": [("G", 1.0)],
            "E": [("F", 1.0)],
            "F": [("F", 0.2), ("G", 0.8)],
            "G": [("END", 1.0)],
        },
        duration_params={
            "A": (math.log(12), 0.25),
            "B": (math.log(40), 0.35),
            "C": (math.log(18), 0.30),
            "E": (math.log(14), 0.25),
            "F": (math.log(22), 0.40),
            "G": (math.log(28), 0.30),
        },
        resource_probs={
            "A": [(R[0], 0.6), (R[1], 0.4)],
            "B": [(R[2], 0.7), (R[3], 0.3)],
            "C": [(R[2], 0.5), (R[1], 0.5)],
            "E": [(R[3], 0.7), (R[0], 0.3)],
            "F": [(R[3], 0.6), (R[2], 0.4)],
            "G": [(R[1], 0.8), (R[0], 0.2)],
        },
        cost_params={
            "A": (40, 4),
            "B": (90, 10),
            "C": (55, 5),
            "E": (45, 4),
            "F": (65, 6),
            "G": (110, 11),
        },
    )

    m3 = ProcessModel(
        name="Model_3",
        start_activity="R",
        transitions={
            "R": [("S", 1.0)],
            "S": [("T", 0.6), ("U", 0.4)],
            "T": [("V", 1.0)],
            "U": [("S", 0.15), ("V", 0.85)],
            "V": [("END", 1.0)],
        },
        duration_params={
            "R": (math.log(16), 0.30),
            "S": (math.log(24), 0.35),
            "T": (math.log(18), 0.25),
            "U": (math.log(26), 0.35),
            "V": (math.log(32), 0.30),
        },
        resource_probs={
            "R": [("r1", 0.5), ("r3", 0.5)],
            "S": [("r2", 0.6), ("r4", 0.4)],
            "T": [("r3", 0.7), ("r2", 0.3)],
            "U": [("r4", 0.6), ("r1", 0.4)],
            "V": [("r2", 0.7), ("r3", 0.3)],
        },
        cost_params={
            "R": (48, 5),
            "S": (70, 7),
            "T": (58, 6),
            "U": (62, 6),
            "V": (125, 12),
        },
    )

    return [m1, m2, m3]


def generate_traces(n_cases_per_model: int = 500, seed: int = 13):
    random.seed(seed)
    np.random.seed(seed)

    models = build_process_models()
    traces = []
    all_activities = set()
    all_resources = set()

    cid = 0
    for m in models:
        for _ in range(n_cases_per_model):
            tr = m.simulate_case(case_id=f"{m.name}-{cid}")
            cid += 1
            traces.append(tr)
            for e in tr.events:
                all_activities.add(e.activity)
                all_resources.add(e.resource)

    activity2id = {a: i for i, a in enumerate(sorted(all_activities))}
    resource2id = {r: i for i, r in enumerate(sorted(all_resources))}

    for tr in traces:
        tr.next_activity_ids = []
        for i, e in enumerate(tr.events):
            e.activity_id = activity2id[e.activity]
            e.resource_id = resource2id[e.resource]
            if i < len(tr.events) - 1:
                tr.next_activity_ids.append(activity2id[tr.events[i + 1].activity])

    return traces, activity2id, resource2id
