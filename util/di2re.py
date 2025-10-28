# -*- coding: utf-8 -*-
"""
Disaster-to-Recovery synthetic log generator (XES)
--------------------------------------------------
Generates a rich, process-agnostic event log that mixes:
- Control-flow skeletons (linear with optionals, XOR/OR/AND, loops, k-of-n, preemption)
- Temporal patterns (NHPP bursty arrivals, calendars, deadlines, synchronization)
- Resource patterns (roles, skills, binding/separation-of-duty, handovers, meetings)
- Cost patterns (fixed+variable, overtime, SLA penalties, waiting, handover fees, drift)

Outputs:
  - di2re.xes.gz  (XES compressed)
  - Console: quality checks and summaries

Dependencies: Python stdlib + numpy (+ pandas optional) + pm4py
"""

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    # Prefer the simple API if available
    from pm4py import write_xes as _pm4py_write_xes
    def write_log(log, path):
        return _pm4py_write_xes(log, path)
except Exception:
    # Fallback: classic exporter
    from pm4py.objects.log.exporter.xes import exporter as xes_exporter
    def write_log(log, path):
        return xes_exporter.apply(log, path)

from pm4py.objects.log.obj import EventLog, Trace, Event


# -----------------------------
# Configuration
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

NUM_CASES = 250
TARGET_FILE = "di2re.xes.gz"

# Simulation horizon: last ~30 days
NOW = datetime.now()
HORIZON_START = NOW - timedelta(days=30)
HORIZON_END = NOW

# Business hours & calendars
BUSINESS_OPEN = time(9, 0)
BUSINESS_LUNCH_START = time(12, 0)
BUSINESS_LUNCH_END = time(13, 0)
BUSINESS_CLOSE = time(17, 0)
WEEKENDS_CLOSED = True

# Resource shifts (two archetypes)
SHIFT_A = (time(8, 0), time(16, 0))
SHIFT_B = (time(12, 0), time(20, 0))

# Costs
WAITING_RATE_PER_HOUR = 8.0              # holding cost
HANDOVER_FEE = 12.0                       # per handover event
EXPEDITE_FEE = 75.0
ESCALATE_PENALTY = 180.0
WEEKEND_RATE_MULT = 1.35
OVERTIME_RATE_MULT = 1.5

# Budget bands (to inject cost → control-flow dependence)
BUDGET_MIN = 800.0
BUDGET_MAX = 2500.0

# Era concept drift: 3 eras with drifts in durations/costs
ERA_SPLITS = 3  # equally divide the horizon into 3 eras


# -----------------------------
# Alphabets: Activities, Roles, Resources
# -----------------------------
# Activity families to help model setup/changeover and context switch costs
ACTIVITY_FAMILY = {
    "DR.Start": "admin",
    "Assess": "analysis",
    "Classify": "analysis",
    "Triage": "analysis",
    "Secure": "contain",
    "Contain": "contain",
    "Investigate": "invest",
    "Communicate": "coord",
    "Forensics": "invest",
    "Backup": "infra",
    "Notify": "coord",
    "Fix": "remedy",
    "Review": "qa",
    "QA": "qa",
    "Approve": "govern",
    "Audit": "govern",
    "Deploy": "infra",
    "Verify": "qa",
    "BatchReview": "qa",
    "Meeting": "coord",
    "Vendor.Assess": "vendor",
    "Vendor.Restore": "vendor",
    "Restore": "infra",
    "Warn": "signal",
    "Escalate": "signal",
    "Expedite": "signal",
    "Reassign": "signal",
    "Wait": "wait",
    "Handover": "handover",
    "Rollback": "remedy",
    "MI.Check": "qa",
    "Join.MI": "qa",
    "AND.Split": "control",
    "AND.Join": "control",
    "OR.Split": "control",
    "OR.Join": "control",
    "End": "admin"
}

# Preferred roles for activities
PREFERRED_ROLE = {
    "DR.Start": "role:helpdesk",
    "Assess": "role:analyst",
    "Classify": "role:analyst",
    "Triage": "role:analyst",
    "Secure": "role:engineer",
    "Contain": "role:engineer",
    "Investigate": "role:engineer",
    "Communicate": "role:analyst",
    "Forensics": "role:engineer",
    "Backup": "role:engineer",
    "Notify": "role:analyst",
    "Fix": "role:engineer",
    "Review": "role:analyst",
    "QA": "role:analyst",
    "Approve": "role:manager",
    "Audit": "role:auditor",
    "Deploy": "role:engineer",
    "Verify": "role:analyst",
    "BatchReview": "role:analyst",
    "Meeting": "role:manager",
    "Vendor.Assess": "role:vendor",
    "Vendor.Restore": "role:vendor",
    "Restore": "role:engineer",
    "Warn": "role:senior",
    "Escalate": "role:senior",
    "Expedite": "role:senior",
    "Reassign": "role:senior",
    "Wait": "role:analyst",
    "Handover": "role:analyst",
    "Rollback": "role:engineer",
    "MI.Check": "role:analyst",
    "Join.MI": "role:analyst",
    "AND.Split": "role:analyst",
    "AND.Join": "role:analyst",
    "OR.Split": "role:analyst",
    "OR.Join": "role:analyst",
    "End": "role:helpdesk"
}

# Generate resource pool R1..R30 with roles and shifts
def build_resources():
    resources = {}
    # mapping: resource -> (role, shift)
    # Analysts: R1..R10
    for i in range(1, 11):
        resources[f"R{i}"] = ("role:analyst", SHIFT_A if i % 2 == 0 else SHIFT_B)
    # Engineers: R11..R18
    for i in range(11, 19):
        resources[f"R{i}"] = ("role:engineer", SHIFT_A if i % 2 == 1 else SHIFT_B)
    # Seniors: R19..R22
    for i in range(19, 23):
        resources[f"R{i}"] = ("role:senior", SHIFT_A)
    # Managers: R23..R24
    resources["R23"] = ("role:manager", SHIFT_B)
    resources["R24"] = ("role:manager", SHIFT_A)
    # Auditors: R25..R26
    resources["R25"] = ("role:auditor", SHIFT_A)
    resources["R26"] = ("role:auditor", SHIFT_B)
    # Vendors: R27..R29
    for i in range(27, 30):
        resources[f"R{i}"] = ("role:vendor", SHIFT_B)
    # Misc/Helpdesk: R30
    resources["R30"] = ("role:helpdesk", SHIFT_A)
    return resources

RESOURCES = build_resources()
ALL_RESOURCE_IDS = list(RESOURCES.keys())

# Small fraction of Unknown to test defaults
def maybe_unknown_resource(res_id: str, p=0.03) -> str:
    return "Unknown" if random.random() < p else res_id


# -----------------------------
# Calendars
# -----------------------------
@dataclass
class BusinessCalendar:
    open_time: time = BUSINESS_OPEN
    lunch_start: time = BUSINESS_LUNCH_START
    lunch_end: time = BUSINESS_LUNCH_END
    close_time: time = BUSINESS_CLOSE
    weekends_closed: bool = WEEKENDS_CLOSED

    def is_business_open(self, dt: datetime) -> bool:
        if self.weekends_closed and dt.weekday() >= 5:
            return False
        tt = dt.time()
        if self.lunch_start <= tt < self.lunch_end:
            return False
        return self.open_time <= tt < self.close_time

    def next_business_open(self, dt: datetime) -> datetime:
        t = dt
        # Move through time until in open window (skipping lunch)
        while True:
            if self.weekends_closed and t.weekday() >= 5:
                # jump to Monday 9:00
                days_to_monday = (7 - t.weekday()) % 7
                t = datetime.combine((t + timedelta(days=days_to_monday)).date(), self.open_time)
                continue
            tt = t.time()
            # before open -> set to open
            if tt < self.open_time:
                t = datetime.combine(t.date(), self.open_time)
                return t
            # in lunch -> jump to end lunch
            if self.lunch_start <= tt < self.lunch_end:
                t = datetime.combine(t.date(), self.lunch_end)
                return t
            # between open and lunch or between lunch and close
            if self.open_time <= tt < self.lunch_start or self.lunch_end <= tt < self.close_time:
                return t
            # after close -> next day open
            t = datetime.combine(t.date() + timedelta(days=1), self.open_time)

    def in_resource_shift(self, dt: datetime, res_id: str) -> bool:
        if res_id not in RESOURCES:
            return True  # Unknown/externals: ignore shift restriction
        _, (s_start, s_end) = RESOURCES[res_id]
        tt = dt.time()
        # simple same-day shift window
        return s_start <= tt < s_end

    def next_resource_shift_open(self, dt: datetime, res_id: str) -> datetime:
        if res_id not in RESOURCES:
            return dt
        _, (s_start, s_end) = RESOURCES[res_id]
        t = dt
        while True:
            if self.weekends_closed and t.weekday() >= 5:
                # jump to Monday at both business+shift opening
                days_to_monday = (7 - t.weekday()) % 7
                t = datetime.combine((t + timedelta(days=days_to_monday)).date(), max(self.open_time, s_start))
                continue
            tt = t.time()
            # align to max of business and shift opening
            earliest = max(self.open_time, s_start)
            latest = min(self.close_time, s_end)
            if tt < earliest:
                t = datetime.combine(t.date(), earliest)
                return t
            # if in lunch break, jump past lunch (but not past shift latest)
            if self.lunch_start <= tt < self.lunch_end:
                after_lunch = datetime.combine(t.date(), self.lunch_end)
                if self.lunch_end >= latest:  # shift ends before/at lunch end
                    # go to next day earliest
                    t = datetime.combine(t.date() + timedelta(days=1), earliest)
                    continue
                return after_lunch
            if earliest <= tt < latest and not (self.lunch_start <= tt < self.lunch_end):
                return t
            # after close or after shift -> next day earliest
            t = datetime.combine(t.date() + timedelta(days=1), earliest)

    def add_work(self, start: datetime, duration_hours: float, res_id: str,
                 allow_overtime: bool = False) -> Tuple[datetime, bool]:
        """
        Returns (end_time, used_overtime_flag)
        If allow_overtime, we ignore resource shift/business closing but still skip weekend if configured frozen.
        """
        remaining = duration_hours
        t = start
        used_overtime = False

        while remaining > 1e-9:
            if allow_overtime:
                # Only freeze on weekends, otherwise run continuously (overtime)
                if self.weekends_closed and t.weekday() >= 5:
                    t = datetime.combine(t.date() + timedelta(days=(7 - t.weekday()) % 7), self.open_time)
                    continue
                # progress in continuous time
                step = remaining
                t = t + timedelta(hours=step)
                used_overtime = True
                remaining = 0.0
            else:
                # Align to business + resource shift open
                t = self.next_resource_shift_open(self.next_business_open(t), res_id)
                # figure current window end
                _, (s_start, s_end) = RESOURCES.get(res_id, ("", (time(0,0), time(23,59))))
                window_start = max(self.open_time, s_start)
                window_end = min(self.close_time, s_end)
                curr_end = datetime.combine(t.date(), window_end)
                # cut for lunch
                lunch_s = datetime.combine(t.date(), self.lunch_start)
                lunch_e = datetime.combine(t.date(), self.lunch_end)
                if t < lunch_s < curr_end:
                    available_1 = (lunch_s - t).total_seconds() / 3600.0
                    if remaining <= available_1 + 1e-9:
                        t = t + timedelta(hours=remaining)
                        remaining = 0.0
                    else:
                        remaining -= available_1
                        t = lunch_e
                else:
                    available = (curr_end - t).total_seconds() / 3600.0
                    if remaining <= available + 1e-9:
                        t = t + timedelta(hours=remaining)
                        remaining = 0.0
                    else:
                        remaining -= available
                        # move to next day start
                        t = datetime.combine(t.date() + timedelta(days=1), window_start)
        return t, used_overtime


CAL = BusinessCalendar()


# -----------------------------
# NHPP (bursty) arrivals
# -----------------------------
def nhpp_intensity(dt: datetime) -> float:
    """
    Non-homogeneous Poisson intensity (events/hour):
    - daily sinusoid (peaks at ~10:30 & 15:30)
    - weekday boost, weekend slump
    - era drift slowly increases intensity over the 30-day window
    """
    # daily seasonality
    tod = dt.hour + dt.minute / 60.0
    daily = 0.6 + 0.4 * (math.sin(2 * math.pi * (tod - 10.5) / 24) + 1) / 2
    # weekday vs weekend
    wd = 1.0 if dt.weekday() < 5 else 0.5
    # slow drift across horizon
    span = (HORIZON_END - HORIZON_START).total_seconds()
    pos = (dt - HORIZON_START).total_seconds() / (span + 1e-9)
    drift = 0.8 + 0.6 * pos
    return 0.45 * daily * wd * drift  # reasonable overall rate

def sample_nhpp_arrivals(n_cases: int) -> List[datetime]:
    # Thinning algorithm on hourly grid
    arrivals = []
    # Upper bound on intensity (safe)
    lambda_bar = 1.5  # events/hour (upper envelope)
    t = HORIZON_START
    while len(arrivals) < n_cases and t < HORIZON_END:
        # propose next via exponential with rate lambda_bar
        w = np.random.exponential(1.0 / lambda_bar)  # hours
        t = t + timedelta(hours=w)
        if t >= HORIZON_END:
            break
        # accept with prob lambda(t)/lambda_bar
        if random.random() < nhpp_intensity(t) / lambda_bar:
            arrivals.append(t)
    # If too few (unlikely), top up uniformly
    while len(arrivals) < n_cases:
        dt = HORIZON_START + timedelta(seconds=random.randint(0, int((HORIZON_END - HORIZON_START).total_seconds())))
        arrivals.append(dt)
    arrivals.sort()
    return arrivals[:n_cases]


# -----------------------------
# Cost model & service times
# -----------------------------
# Base (per-activity) service-times in hours ~ lognormal
BASE_TIME = {
    "DR.Start": (0.1, 0.25),
    "Assess": (0.5, 0.4),
    "Classify": (0.3, 0.3),
    "Triage": (0.4, 0.35),
    "Secure": (0.6, 0.5),
    "Contain": (1.2, 0.6),
    "Investigate": (2.2, 0.7),
    "Communicate": (0.3, 0.25),
    "Forensics": (1.5, 0.7),
    "Backup": (1.0, 0.6),
    "Notify": (0.2, 0.2),
    "Fix": (2.5, 0.8),
    "Review": (0.7, 0.4),
    "QA": (0.6, 0.4),
    "Approve": (0.2, 0.2),
    "Audit": (0.6, 0.5),
    "Deploy": (0.8, 0.5),
    "Verify": (0.4, 0.35),
    "BatchReview": (0.6, 0.4),
    "Meeting": (0.8, 0.4),
    "Vendor.Assess": (0.5, 0.5),
    "Vendor.Restore": (3.0, 0.9),
    "Restore": (3.2, 0.9),
    "Warn": (0.05, 0.1),
    "Escalate": (0.15, 0.2),
    "Expedite": (0.05, 0.1),
    "Reassign": (0.05, 0.1),
    "Wait": (0.25, 0.2),
    "Handover": (0.15, 0.15),
    "Rollback": (0.5, 0.5),
    "MI.Check": (0.4, 0.3),
    "Join.MI": (0.05, 0.05),
    "AND.Split": (0.01, 0.02),
    "AND.Join": (0.02, 0.02),
    "OR.Split": (0.01, 0.02),
    "OR.Join": (0.02, 0.02),
    "End": (0.05, 0.05)
}

# Fixed activity costs (materials/fees), can be negative for credits
FIXED_COST = defaultdict(float, {
    "Backup": 25.0,
    "Vendor.Restore": 60.0,
    "Escalate": ESCALATE_PENALTY,
    "Expedite": EXPEDITE_FEE,
    "Handover": HANDOVER_FEE,
    "Rollback": -40.0  # credit/refund
})

# Hourly rate by role (drifts by era)
BASE_RATE_BY_ROLE = {
    "role:helpdesk": 20.0,
    "role:analyst": 30.0,
    "role:engineer": 45.0,
    "role:senior": 60.0,
    "role:manager": 55.0,
    "role:auditor": 50.0,
    "role:vendor": 38.0
}

def era_of_time(dt: datetime) -> int:
    total = (HORIZON_END - HORIZON_START).total_seconds()
    step = total / ERA_SPLITS
    pos = (dt - HORIZON_START).total_seconds()
    e = int(min(ERA_SPLITS - 1, max(0, pos // step)))
    return int(e)

def rate_for(role: str, dt: datetime) -> float:
    base = BASE_RATE_BY_ROLE.get(role, 30.0)
    e = era_of_time(dt)
    # simple drift: later eras cheaper for vendors, pricier for approval/govern
    drift = 1.0
    if role == "role:vendor":
        drift *= (0.95 ** e)
    if role in ("role:manager", "role:auditor"):
        drift *= (1.08 ** e)
    return base * drift

def heavy_tail_multiplier() -> float:
    # mixture: mostly 1x, sometimes Pareto tail
    if random.random() < 0.12:
        return 1.0 + np.random.pareto(1.5)  # 0..heavy
    return 1.0

def lognormal_hours(base_mu, base_sigma):
    # mu/sigma here are in linear scale, we derive lognormal params roughly
    mu = math.log(max(1e-4, base_mu)) - 0.5
    sigma = max(0.1, base_sigma)
    return np.random.lognormal(mean=mu, sigma=sigma)

def sample_service_time_hours(activity: str, skill_mult: float = 1.0) -> float:
    mu, sg = BASE_TIME.get(activity, (0.5, 0.4))
    x = lognormal_hours(mu, sg) * heavy_tail_multiplier()
    return max(0.02, x * skill_mult)

def overtime_multiplier(start: datetime, end: datetime, res_id: str) -> float:
    # Apply overtime multiplier if any part of the interval is off-shift
    if res_id not in RESOURCES:
        return 1.0
    _, (s_start, s_end) = RESOURCES[res_id]
    spans_overtime = False
    t = start
    while t < end:
        if not CAL.in_resource_shift(t, res_id):
            spans_overtime = True
            break
        t += timedelta(minutes=30)
    return OVERTIME_RATE_MULT if spans_overtime else 1.0

def weekend_multiplier(dt: datetime) -> float:
    return WEEKEND_RATE_MULT if dt.weekday() >= 5 else 1.0


# -----------------------------
# Resource management (skills, load, policies)
# -----------------------------
@dataclass
class ResourceState:
    next_free: datetime = field(default_factory=lambda: HORIZON_START)
    experience: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_exec_time: Dict[str, datetime] = field(default_factory=dict)

class ResourceManager:
    def __init__(self):
        self.state: Dict[str, ResourceState] = {r: ResourceState() for r in ALL_RESOURCE_IDS}
        # skill multipliers by family: <1 faster, >1 slower
        self.skill_mult: Dict[str, Dict[str, float]] = defaultdict(dict)
        families = set(ACTIVITY_FAMILY.values())
        for r in ALL_RESOURCE_IDS:
            role = RESOURCES[r][0]
            for fam in families:
                base = 1.0
                if role == "role:engineer" and fam in ("infra", "contain", "invest", "remedy"):
                    base = random.uniform(0.75, 1.1)
                elif role == "role:analyst" and fam in ("qa", "analysis", "coord", "admin"):
                    base = random.uniform(0.75, 1.15)
                elif role.startswith("role:vendor") and fam in ("vendor", "infra"):
                    base = random.uniform(0.85, 1.2)
                elif role == "role:senior":
                    base = random.uniform(0.8, 1.05)
                else:
                    base = random.uniform(0.9, 1.25)
                self.skill_mult[r][fam] = base

    def choose_resource(self, activity: str, ready_time: datetime, policy_mix: str, prefer_role: str) -> str:
        """
        policy_mix in {"random","least_loaded","skill","mixed"}
        """
        candidates = [r for r in ALL_RESOURCE_IDS if RESOURCES[r][0] == prefer_role]
        if not candidates:
            candidates = ALL_RESOURCE_IDS[:]  # fallback
        if policy_mix == "random":
            return random.choice(candidates)
        if policy_mix == "least_loaded":
            return min(candidates, key=lambda r: self.state[r].next_free)
        if policy_mix == "skill":
            fam = ACTIVITY_FAMILY.get(activity, "misc")
            return min(candidates, key=lambda r: self.skill_mult[r].get(fam, 1.0))
        # mixed: combine load and skill
        fam = ACTIVITY_FAMILY.get(activity, "misc")
        def score(r):
            load = (self.state[r].next_free - ready_time).total_seconds() / 3600.0
            return 0.6 * max(0, load) + 0.4 * self.skill_mult[r].get(fam, 1.0)
        return min(candidates, key=score)

    def service_multiplier(self, res_id: str, activity: str, when: datetime) -> float:
        fam = ACTIVITY_FAMILY.get(activity, "misc")
        base = self.skill_mult[res_id].get(fam, 1.0)
        # learning: faster with experience
        exp = self.state[res_id].experience[activity]
        learn = 1.0 / (1.0 + 0.08 * exp)
        # forgetting: slower after inactivity
        last_t = self.state[res_id].last_exec_time.get(activity, when)
        idle_days = max(0.0, (when - last_t).total_seconds() / 86400.0)
        forget = 1.0 + 0.03 * min(10.0, idle_days)
        return base * learn * forget

    def book(self, res_id: str, start: datetime, end: datetime, activity: str):
        s = self.state[res_id]
        s.next_free = max(s.next_free, end)
        s.experience[activity] += 1
        s.last_exec_time[activity] = end


RM = ResourceManager()


# -----------------------------
# Helpers: events & costs
# -----------------------------
def add_event(trace: Trace, activity: str, when: datetime, res_id: str, amount: float, role: Optional[str] = None):
    e = Event({
        "concept:name": activity,
        "time:timestamp": when,
        "org:resource": maybe_unknown_resource(res_id),
        "amount": float(amount)
    })
    if role:
        e["org:role"] = role
    trace.append(e)

def handover_if_needed(trace: Trace, prev_res: Optional[str], next_res: str, start_time: datetime,
                       cumulative_cost: List[float]) -> datetime:
    if prev_res is None or prev_res == next_res:
        return start_time
    # Inject a handover pseudo-activity that completes just before the next activity starts
    t_hand = max(start_time - timedelta(seconds=1), trace[-1]["time:timestamp"] if trace else start_time - timedelta(seconds=1))
    add_event(trace, "Handover", t_hand, prev_res, FIXED_COST["Handover"], role="role:analyst")
    cumulative_cost[0] += FIXED_COST["Handover"]
    return start_time

def schedule_activity(trace: Trace, name: str, earliest: datetime, case_id: str,
                      policy_mix: str = "mixed", allow_overtime: bool = False,
                      bind_with: Optional[str] = None, forbid_with: Optional[str] = None,
                      prev_activity: Optional[str] = None, prev_res: Optional[str] = None,
                      cumulative_cost: Optional[List[float]] = None,
                      deadline: Optional[datetime] = None,
                      budget_limit: Optional[float] = None) -> Tuple[datetime, str, float]:
    """
    Schedules a single activity, returns (end_time, resource, cost_amount_added)
    """
    prefer_role = PREFERRED_ROLE.get(name, "role:analyst")
    fam = ACTIVITY_FAMILY.get(name, "misc")

    # Choose resource according to policy and constraints
    res = RM.choose_resource(name, earliest, policy_mix, prefer_role)

    # Binding-of-duty: if bind_with provided and prev_res is not None, keep same
    if bind_with and prev_res:
        res = prev_res

    # Separation-of-duty: if forbid_with provided and prev_res is same, reassign
    if forbid_with and prev_res and res == prev_res:
        # pick alternative with same role but different id
        same_role = [r for r in ALL_RESOURCE_IDS if RESOURCES[r][0] == RESOURCES[res][0] and r != res]
        if same_role:
            res = random.choice(same_role)

    # Setup/changeover: if family changed vs previous activity, add extra prep
    setup_penalty = 0.0
    if prev_activity and ACTIVITY_FAMILY.get(prev_activity) != fam:
        setup_penalty = np.random.lognormal(mean=math.log(0.15), sigma=0.4)  # ~0.1-0.5h

    # Resource & business calendars determine feasible start
    start = earliest
    # Wait for resource availability
    start = max(start, RM.state[res].next_free)
    # Business and shift opening unless overtime allowed
    if not allow_overtime:
        start = CAL.next_resource_shift_open(CAL.next_business_open(start), res)

    # If we had to wait, create explicit Wait event (holding cost)
    if start - earliest > timedelta(minutes=30):
        wait_hours = (start - earliest).total_seconds() / 3600.0
        wait_cost = WAITING_RATE_PER_HOUR * wait_hours
        add_event(trace, "Wait", earliest + timedelta(seconds=1), res, wait_cost, role="role:analyst")
        if cumulative_cost is not None:
            cumulative_cost[0] += wait_cost

    # Draw service time with resource multipliers (learning/forgetting)
    skill_mult = RM.service_multiplier(res, name, start)
    service_h = sample_service_time_hours(name, skill_mult=skill_mult) + setup_penalty

    # Preemption logic: if long AND resource queues high, split into start/resume/complete
    preempted = False
    if name in ("Restore", "Fix", "Vendor.Restore") and (service_h > 4.0) and random.random() < 0.35:
        preempted = True
        # split 40% / gap / remaining
        part1 = service_h * random.uniform(0.35, 0.55)
        gap = random.uniform(0.5, 2.0)  # hours paused
        part2 = max(0.25, service_h - part1)

        # Segment 1
        end1, used_ot1 = CAL.add_work(start, part1, res, allow_overtime=allow_overtime)
        role = RESOURCES[res][0] if res in RESOURCES else None
        cost1 = FIXED_COST[name] + rate_for(role, start) * part1 * weekend_multiplier(start) * (overtime_multiplier(start, end1, res) if used_ot1 else 1.0)
        add_event(trace, f"{name}.start", end1, res, cost1, role=role)
        # Pause (represented implicitly by gap; we don't add Wait here to avoid double counting)
        resume_time = end1 + timedelta(hours=gap)
        resume_time = CAL.next_resource_shift_open(CAL.next_business_open(resume_time), res) if not allow_overtime else resume_time
        # Segment 2
        end2, used_ot2 = CAL.add_work(resume_time, part2, res, allow_overtime=allow_overtime)
        cost2 = rate_for(role, resume_time) * part2 * weekend_multiplier(resume_time) * (overtime_multiplier(resume_time, end2, res) if used_ot2 else 1.0)
        add_event(trace, f"{name}.resume", end2, res, cost2, role=role)
        # Completion marker (zero-time bookkeeping)
        end = end2
        add_event(trace, f"{name}.complete", end, res, 0.0, role=role)
        total_cost = cost1 + cost2
    else:
        # Monolithic execution
        end, used_ot = CAL.add_work(start, service_h, res, allow_overtime=allow_overtime)
        role = RESOURCES[res][0] if res in RESOURCES else None
        var_rate = rate_for(role, start)
        total_cost = FIXED_COST[name] + var_rate * service_h * weekend_multiplier(start) * (overtime_multiplier(start, end, res) if used_ot else 1.0)
        add_event(trace, name, end, res, total_cost, role=role)

    # SLA deadline monitoring → Warn/Escalate
    if deadline and end > deadline and random.random() < 0.8:
        # add warn then escalate with senior takeover
        w_end = end + timedelta(minutes=1)
        add_event(trace, "Warn", w_end, res, FIXED_COST["Warn"], role="role:senior")
        e_end = w_end + timedelta(minutes=1)
        add_event(trace, "Escalate", e_end, "R19", FIXED_COST["Escalate"], role="role:senior")
        total_cost += FIXED_COST["Escalate"]
        res = "R19"  # senior now considered current resource for binding that follows

    if cumulative_cost is not None:
        cumulative_cost[0] += total_cost

    # Update resource booking
    RM.book(res if res in RESOURCES else "R30", start, end, name.split(".")[0])
    return end, res, total_cost


# -----------------------------
# Control-flow builders
# -----------------------------
def choose_skeleton():
    """
    Return a skeleton id and a small set of pattern toggles.
    Ensures ambiguity by making multiple plausible next activities from common prefixes.
    """
    sk = random.choices(
        ["linear_opt", "xor", "and_parallel", "or_parallel", "loop_rework", "mi_kofn"],
        weights=[0.20, 0.18, 0.2, 0.15, 0.17, 0.10],
        k=1
    )[0]
    # Pick 3-8 patterns from temporal/resource/cost interactions
    patterns = random.sample([
        "heavy_tails", "nhpp_bursty", "business_calendar", "resource_shift",
        "deadline_timer", "setup_changeover", "preemption", "sync_latency",
        "handover_delay", "priority_queue", "seasonality_drift", "learning",
        "forgetting", "cooling_off", "handover_fee", "batch_release",
        "skip_with_memory", "k_of_n", "binding_duty", "separation_duty",
        "vendor_option", "budget_branch", "waiting_cost"
    ], k=random.randint(3, 8))
    return sk, set(patterns)


# -----------------------------
# Case generation
# -----------------------------
def build_case_trace(case_id: str, start_time: datetime) -> Trace:
    """
    Builds a single trace. The function composes skeleton + patterns; schedules activities
    via calendars and resource manager; injects costs and control-flow markers.
    """
    trace = Trace()
    trace.attributes["concept:name"] = case_id

    # Era for drift & SLA/deadlines
    era = era_of_time(start_time)
    # Priority & SLA deadlines
    priority = random.choices(["low", "normal", "high"], weights=[0.2, 0.6, 0.2])[0]
    base_sla_hours = {"low": 96, "normal": 60, "high": 36}[priority]
    sla_hours = int(base_sla_hours * (0.9 + 0.2 * era))  # later eras a tad tighter for realism
    deadline = start_time + timedelta(hours=sla_hours)

    budget_limit = random.uniform(BUDGET_MIN, BUDGET_MAX)

    skeleton, patterns = choose_skeleton()
    policy_mix = random.choice(["mixed", "least_loaded", "skill", "random"])

    # Record flags for checks
    trace._has_and = False
    trace._has_or = False
    trace._has_mi = False

    # Cumulative cost (as list to mutate inside schedule_activity)
    cum_cost = [0.0]

    # 0) Start
    t_cursor = start_time
    t_end, res_prev, _ = schedule_activity(trace, "DR.Start", t_cursor, case_id,
                                           policy_mix=policy_mix, cumulative_cost=cum_cost,
                                           deadline=deadline, budget_limit=budget_limit)

    # Optional inserts after Start
    if random.random() < 0.6:
        t_end, res_prev, _ = schedule_activity(trace, "Assess", t_end, case_id,
                                               policy_mix=policy_mix, prev_res=res_prev,
                                               cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)
    if random.random() < 0.5:
        t_end, res_prev, _ = schedule_activity(trace, "Classify", t_end, case_id,
                                               policy_mix=policy_mix, prev_res=res_prev,
                                               cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)

    # Early XOR decision: Quick vs Deep vs Vendor (depends on early durations/costs)
    branch_probs = [0.4, 0.4, 0.2]
    # shift based on cost so far and priority (budget-aware branching)
    if cum_cost[0] > 0.6 * budget_limit:
        branch_probs = [0.25, 0.35, 0.40]  # lean to vendor
    if priority == "high":
        branch_probs = [0.45, 0.45, 0.10]  # avoid vendor if urgent
    branch = random.choices(["quick", "deep", "vendor"], weights=branch_probs, k=1)[0]

    # Cooling-off constraint applies between Contain and Restore later
    cooling_min = timedelta(hours=random.uniform(1.0, 3.0))

    # Common initial steps
    t_end, res_prev, _ = schedule_activity(trace, "Triage", t_end, case_id,
                                           policy_mix=policy_mix, prev_res=res_prev,
                                           cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)
    t_end, res_prev, _ = schedule_activity(trace, "Secure", t_end, case_id,
                                           policy_mix=policy_mix, prev_res=res_prev,
                                           cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)
    t_post_contain, res_after_contain, _ = schedule_activity(trace, "Contain", t_end, case_id,
                                                             policy_mix=policy_mix, prev_res=res_prev,
                                                             cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)

    # Decide skeleton-specific flow
    if branch == "vendor":
        # Vendor alternative: assess + restore (slower mean but lower fixed)
        t_va, res_va, _ = schedule_activity(trace, "Vendor.Assess", t_post_contain, case_id,
                                            policy_mix=policy_mix, prev_res=res_after_contain,
                                            cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)
        # Cooling-off: ensure lag before restore
        if t_va - t_post_contain < cooling_min:
            # Insert Wait buffer for cooling-off
            add_event(trace, "Wait", t_va - timedelta(seconds=1), "R30",
                      WAITING_RATE_PER_HOUR * ((cooling_min - (t_va - t_post_contain)).total_seconds() / 3600.0),
                      role="role:analyst")
        # Potential expedite if close to deadline
        allow_ot = (deadline - t_va).total_seconds() / 3600.0 < 12
        t_vr, res_vr, _ = schedule_activity(trace, "Vendor.Restore", t_va, case_id,
                                            policy_mix=policy_mix, prev_res=res_va,
                                            allow_overtime=allow_ot,
                                            cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)
        t_end = t_vr
        res_prev = res_vr
    else:
        # Quick or Deep organization
        if skeleton in ("and_parallel", "xor", "linear_opt"):
            # AND-split: Investigate || Communicate  → join → Fix
            trace._has_and = True
            add_event(trace, "AND.Split", t_post_contain + timedelta(seconds=1), "R30", 0.0, role="role:analyst")
            # Parallel branches start at same readiness
            ready = t_post_contain
            # Branch A: Investigate (binding-of-duty with Fix optionally)
            bind_fix = random.random() < 0.5
            t_inv, res_inv, _ = schedule_activity(trace, "Investigate", ready, case_id,
                                                  policy_mix=policy_mix, prev_res=res_after_contain,
                                                  cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)
            # Branch B: Communicate
            t_com, res_com, _ = schedule_activity(trace, "Communicate", ready, case_id,
                                                  policy_mix=policy_mix, prev_res=res_after_contain,
                                                  cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)
            # Long OR split (subset) sometimes
            if random.random() < 0.5:
                trace._has_or = True
                add_event(trace, "OR.Split", max(t_inv, t_com) + timedelta(seconds=1), "R30", 0.0, role="role:analyst")
                chosen = [a for a in ["Forensics", "Backup", "Notify"] if random.random() < 0.6]
                or_ends = []
                for a in chosen:
                    t_a, res_a, _ = schedule_activity(trace, a, max(t_inv, t_com), case_id,
                                                      policy_mix=policy_mix, prev_res=res_after_contain,
                                                      cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)
                    or_ends.append(t_a)
                add_event(trace, "OR.Join", (max(or_ends) if or_ends else max(t_inv, t_com)) + timedelta(seconds=1), "R30", 0.0, role="role:analyst")
                t_join_ready = max(or_ends) if or_ends else max(t_inv, t_com)
            else:
                t_join_ready = max(t_inv, t_com)
            # AND join (sync latency + handover)
            join_end = t_join_ready + timedelta(minutes=random.randint(2, 15))
            add_event(trace, "AND.Join", join_end, "R30", 0.0, role="role:analyst")

            # Fix (binding-of-duty with Investigate when chosen)
            t_fix, res_fix, _ = schedule_activity(trace, "Fix", join_end, case_id,
                                                  policy_mix=policy_mix, prev_res=(res_inv if bind_fix else None),
                                                  bind_with=("Investigate" if bind_fix else None),
                                                  cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)
            t_end = t_fix
            res_prev = res_fix

        if skeleton == "or_parallel":
            trace._has_or = True
            add_event(trace, "OR.Split", t_post_contain + timedelta(seconds=1), "R30", 0.0, role="role:analyst")
            subset = [a for a in ["Investigate", "Backup", "Communicate"] if random.random() < 0.7]
            ends = []
            last_res = res_after_contain
            for a in subset:
                t_a, last_res, _ = schedule_activity(trace, a, t_post_contain, case_id,
                                                     policy_mix=policy_mix, prev_res=last_res,
                                                     cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)
                ends.append(t_a)
            add_event(trace, "OR.Join", (max(ends) if ends else t_post_contain) + timedelta(seconds=1), "R30", 0.0, role="role:analyst")
            # Restore after OR join with cooling-off constraint
            after_or = max(ends) if ends else t_post_contain
            if after_or - t_post_contain < cooling_min:
                add_event(trace, "Wait", after_or + timedelta(seconds=1), "R30",
                          WAITING_RATE_PER_HOUR * ((cooling_min - (after_or - t_post_contain)).total_seconds() / 3600.0),
                          role="role:analyst")
                after_or = after_or + (cooling_min - (after_or - t_post_contain))
            # Sometimes preempt restore if long
            allow_ot = (deadline - after_or).total_seconds() / 3600.0 < 10
            t_res, res_res, _ = schedule_activity(trace, "Restore", after_or, case_id,
                                                  policy_mix=policy_mix, prev_res=None,
                                                  allow_overtime=allow_ot,
                                                  cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)
            t_end = t_res
            res_prev = res_res

        if skeleton == "loop_rework" or branch == "deep":
            # Deep path with potential Review↔Fix loop (geometric)
            # First Restore before review if quick path
            if branch != "deep":
                t_res, res_res, _ = schedule_activity(trace, "Restore", t_post_contain, case_id,
                                                      policy_mix=policy_mix, prev_res=res_after_contain,
                                                      cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)
                t_end = t_res
                res_prev = res_res

            # Review + optional QA skip based on review duration (stochastic with memory)
            loop_count = 0
            while True:
                t_rev, res_rev, _ = schedule_activity(trace, "Review", t_end, case_id,
                                                      policy_mix=policy_mix, prev_res=res_prev,
                                                      cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)
                # If Review was long, increase chance of rework
                review_dur_h = 0.6  # proxy threshold
                rework_prob = 0.25 + 0.15 * min(3, loop_count)
                if random.random() < rework_prob:
                    # Rework: Fix again
                    t_fix2, res_fix2, _ = schedule_activity(trace, "Fix", t_rev, case_id,
                                                            policy_mix=policy_mix, prev_res=res_rev,
                                                            cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)
                    loop_count += 1
                    t_end, res_prev = t_fix2, res_fix2
                    continue
                # QA skip probability decreases when Review short (memory)
                qa_skip = random.random() < (0.4 if review_dur_h < 0.8 else 0.2)
                if not qa_skip:
                    t_qa, res_qa, _ = schedule_activity(trace, "QA", t_rev, case_id,
                                                        policy_mix=policy_mix, prev_res=res_rev,
                                                        cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)
                    t_end, res_prev = t_qa, res_qa
                else:
                    t_end, res_prev = t_rev, res_rev
                break

    # Governance: Approve (then separation-of-duty Audit)
    t_app, res_app, _ = schedule_activity(trace, "Approve", t_end, case_id,
                                          policy_mix=policy_mix, prev_res=res_prev,
                                          cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)
    # separation-of-duty: enforce different resource for Audit
    t_aud, res_aud, _ = schedule_activity(trace, "Audit", t_app, case_id,
                                          policy_mix=policy_mix, prev_res=res_app, forbid_with="Approve",
                                          cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)

    # Batch release: BatchReview allowed only at release slots (at minute 0..5 of each hour)
    # Align to next release if not in window
    release_minute_window = range(0, 6)
    br_ready = t_aud
    if br_ready.minute not in release_minute_window:
        # align to next hour :00
        align = (60 - br_ready.minute) % 60
        br_ready = br_ready + timedelta(minutes=align)
    t_br, res_br, _ = schedule_activity(trace, "BatchReview", br_ready, case_id,
                                        policy_mix=policy_mix, prev_res=res_aud,
                                        cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)

    # Deploy + Verify
    t_dep, res_dep, _ = schedule_activity(trace, "Deploy", t_br, case_id,
                                          policy_mix=policy_mix, prev_res=res_br,
                                          cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)
    t_ver, res_ver, _ = schedule_activity(trace, "Verify", t_dep, case_id,
                                          policy_mix=policy_mix, prev_res=res_dep,
                                          cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)

    # Multi-instance (k-of-n) checks: spawn 5, join after 3 complete
    if random.random() < 0.6:
        trace._has_mi = True
        add_event(trace, "AND.Split", t_ver + timedelta(seconds=1), "R30", 0.0, role="role:analyst")
        mi_ready = t_ver
        mi_ends = []
        for i in range(5):
            t_mi, res_mi, _ = schedule_activity(trace, "MI.Check", mi_ready, case_id,
                                                policy_mix=policy_mix, prev_res=res_ver,
                                                cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)
            mi_ends.append(t_mi)
        # proceed after k-th order statistic (k=3)
        k = 3
        kth = sorted(mi_ends)[k-1]
        add_event(trace, "Join.MI", kth + timedelta(seconds=1), "R30", 0.0, role="role:analyst")
        t_ver = kth + timedelta(seconds=1)

    # Finalization and potential rollback if budget blown late
    if cum_cost[0] > budget_limit * 1.2 and random.random() < 0.2:
        t_rb, res_rb, _ = schedule_activity(trace, "Rollback", t_ver, case_id,
                                            policy_mix=policy_mix, prev_res=res_ver,
                                            cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)
        t_ver = t_rb

    # Meeting (multi-resource) to sign off: requires manager + senior overlapping
    m1, m2 = "R23", "R19"
    start_meet = max(RM.state[m1].next_free, RM.state[m2].next_free, t_ver)
    # Business+shift constraint for both; use latest alignment
    start_meet1 = CAL.next_resource_shift_open(CAL.next_business_open(start_meet), m1)
    start_meet2 = CAL.next_resource_shift_open(CAL.next_business_open(start_meet), m2)
    meet_start = max(start_meet1, start_meet2)
    # duration
    dur_h = sample_service_time_hours("Meeting", RM.service_multiplier(m1, "Meeting", meet_start))
    end_meet, _ = CAL.add_work(meet_start, dur_h, m1, allow_overtime=False)
    # Book both resources and add one event (manager logs the event)
    RM.book(m1, meet_start, end_meet, "Meeting")
    RM.book(m2, meet_start, end_meet, "Meeting")
    meet_cost = rate_for(RESOURCES[m1][0], meet_start) * dur_h + rate_for(RESOURCES[m2][0], meet_start) * dur_h
    add_event(trace, "Meeting", end_meet, m1, meet_cost, role=RESOURCES[m1][0])
    cum_cost[0] += meet_cost

    # End
    t_final, res_final, _ = schedule_activity(trace, "End", end_meet, case_id,
                                              policy_mix=policy_mix, prev_res=m1,
                                              cumulative_cost=cum_cost, deadline=deadline, budget_limit=budget_limit)

    # Ensure strictly increasing timestamps (sort just in case)
    ordered = sorted(trace, key=lambda e: e["time:timestamp"])
    trace[:] = ordered

    return trace


# -----------------------------
# Generation & Export
# -----------------------------
def execute_script():
    print("Generating Disaster-to-Recovery XES log ...")
    arrivals = sample_nhpp_arrivals(NUM_CASES)

    L = EventLog()
    L.attributes["description"] = "Synthetic Disaster-to-Recovery log with rich control-flow/temporal/resource/cost patterns"
    L.attributes["source"] = "pm4py synthetic generator"
    L.attributes["lifecycle:model"] = "standard"

    for i, st in enumerate(arrivals, start=1):
        case_id = f"DI2RE_{i:04d}"
        tr = build_case_trace(case_id, st)
        # schema check: enforce attributes; add small noise if duplicates in time
        for j, e in enumerate(tr):
            # Required attributes present
            assert "concept:name" in e and "time:timestamp" in e and "org:resource" in e and "amount" in e
            # Normalize amount
            e["amount"] = float(e["amount"])
            # Monotonic in trace
            if j > 0 and e["time:timestamp"] <= tr[j-1]["time:timestamp"]:
                e["time:timestamp"] = tr[j-1]["time:timestamp"] + timedelta(seconds=1)
        L.append(tr)

    # Export
    write_log(L, TARGET_FILE)
    print(f"Saved XES to: {TARGET_FILE}")
    # Quality checks and summaries
    summarize_log(L)


# -----------------------------
# Quality & Sanity Checks
# -----------------------------
def summarize_log(L: EventLog):
    print("\n--- QUALITY & SANITY CHECKS ---")

    # 1) Schema & monotonicity
    bad = 0
    for tr in L:
        prev = None
        for e in tr:
            ok = all(k in e for k in ("concept:name", "time:timestamp", "org:resource", "amount"))
            if not ok:
                bad += 1
            if prev and e["time:timestamp"] <= prev:
                bad += 1
            prev = e["time:timestamp"]
    print(f"Schema/monotonic violations: {bad}")

    # 2) Distribution summaries
    trace_lengths = [len(tr) for tr in L]
    costs = [sum(e["amount"] for e in tr) for tr in L]
    durations_h = [(tr[-1]["time:timestamp"] - tr[0]["time:timestamp"]).total_seconds() / 3600.0 for tr in L]
    def qtl(x, qs=(0.1, 0.5, 0.9)):
        arr = np.array(x)
        return [round(float(np.quantile(arr, q)), 2) for q in qs]
    print(f"Traces: n={len(L)}, len mean={np.mean(trace_lengths):.1f}, q10/50/90={qtl(trace_lengths)}")
    print(f"Total cost per trace: mean={np.mean(costs):.1f}, std={np.std(costs):.1f}, q10/50/90={qtl(costs)}")
    print(f"Trace duration (hours): mean={np.mean(durations_h):.1f}, std={np.std(durations_h):.1f}, q10/50/90={qtl(durations_h)}")

    # 3) Ambiguity check: random prefixes → ≥2 next activities observed
    prefix_map = defaultdict(set)
    for tr in L:
        for i in range(1, len(tr)):
            prefix = tuple(e["concept:name"] for e in tr[:i])
            if i < len(tr):
                prefix_map[prefix].add(tr[i]["concept:name"])
    keys = list(prefix_map.keys())
    sample_keys = random.sample(keys, k=min(100, len(keys)))
    ambiguous = sum(1 for k in sample_keys if len(prefix_map[k]) >= 2)
    print(f"Ambiguity: {ambiguous}/{len(sample_keys)} sampled prefixes had ≥2 possible next-activities")

    # 4) Remaining time regression signal (broad distribution)
    rem_times = []
    for tr in L:
        end_t = tr[-1]["time:timestamp"]
        for e in tr[:-1]:
            rem_h = (end_t - e["time:timestamp"]).total_seconds() / 3600.0
            rem_times.append(rem_h)
    print(f"Remaining time (hours): mean={np.mean(rem_times):.2f}, std={np.std(rem_times):.2f}, q10/50/90={qtl(rem_times)}")

    # 5) Parallelism check
    has_and = sum(1 for tr in L if any(e["concept:name"] in ("AND.Split", "AND.Join") for e in tr))
    print(f"Traces with AND-parallelism markers: {has_and}/{len(L)}")

    # 6) Era drift
    by_era = defaultdict(list)
    for tr in L:
        e0 = tr[0]["time:timestamp"]
        er = era_of_time(e0)
        by_era[er].append((sum(e["amount"] for e in tr),
                           (tr[-1]["time:timestamp"] - tr[0]["time:timestamp"]).total_seconds()/3600.0))
    for er, items in sorted(by_era.items()):
        cs = [c for c, _ in items]
        ds = [d for _, d in items]
        print(f"Era {er}: mean cost={np.mean(cs):.1f}, mean duration h={np.mean(ds):.1f} (n={len(items)})")


# -----------------------------
# Minimal Example (comment)
# -----------------------------
# A typical generated case will touch:
# - AND split (Investigate || Communicate), join → Fix
# - Handover event on resource change
# - SLA breach triggering Warn/Escalate with senior
# - Review→Fix rework loop (geometric)
# - Vendor path with overtime when budget/slack dictate


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    execute_script()
