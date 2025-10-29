"""
Campaign-to-Lead synthetic XES generator (trace-by-trace, pm4py)

Key features implemented (mapping to the spec):
- Activities/resources: 30+ CTL_* labels; resources R1..R30 with roles (analyst, senior, vendor) + a small fraction 'Unknown'.
- Control-flow templates library: linear+optional inserts; XOR; loops/rework (Review→Fix); AND/OR splits/joins; preemption/resume;
  skip patterns; long-term dependencies (binding-of-duty, separation-of-duty); multi-instance k-of-n.
- Temporal: NHPP bursty arrivals; business & shift calendars (9–17 with lunch, weekends frozen, per-resource shifts); escalation & warn;
  heavy‑tailed service times; synchronization latency; batching for QA; learning/forgetting; cooling‑off; handover/context-switch delay.
- Resource: roles & skill heterogeneity; assignment policy mixture (skill/least-loaded/random); reassignment; multi-resource meeting.
- Cost: fixed + rate*duration (+overtime); SLA penalty; rework premium; expedite fee; handover fee; waiting (holding); budget‑aware branching;
  era-based tariffs & drift.
- Scheduling: global resource next_free; per-case construction & calendars; parallel blocks materialized then scheduled; joins wait for max(child end).

Each event has: concept:name, time:timestamp (datetime), org:resource (string), amount (float).
The trace attribute concept:name is set to the case id so `case:concept:name` appears in DataFrames.

Minimal example (comment near the end) hits: AND-split, handover delay, SLA Escalate, one rework, vendor path with overtime.
"""

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time, date
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from copy import deepcopy

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

from pm4py.objects.log.obj import EventLog, Trace, Event
import pm4py


# ---------------------------
# Utility helpers
# ---------------------------

def td_minutes(m: float) -> timedelta:
    return timedelta(minutes=float(m))

def td_hours(h: float) -> timedelta:
    return timedelta(hours=float(h))

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def add_eps_if_needed(previous: Optional[datetime], t: datetime) -> datetime:
    """Ensure strictly increasing timestamps by adding a small epsilon if needed."""
    if previous is None or t > previous:
        return t
    return previous + timedelta(seconds=1)

def weighted_choice(items_with_w):
    total = sum(w for _, w in items_with_w)
    r = random.random() * total
    acc = 0.0
    for item, w in items_with_w:
        acc += w
        if r <= acc:
            return item
    return items_with_w[-1][0]

def business_week_start(d: date) -> date:
    while d.weekday() >= 5:
        d = d + timedelta(days=1)
    return d


# ---------------------------
# Calendars
# ---------------------------

class BusinessCalendar:
    """Business hours: 09:00–17:00, lunch 12:00–13:00, weekends frozen."""
    def __init__(self, open_h=9, close_h=17, lunch=(12, 13)):
        self.open_h = open_h
        self.close_h = close_h
        self.lunch_start, self.lunch_end = lunch

    def is_weekday(self, dt: datetime) -> bool:
        return dt.weekday() < 5

    def next_open(self, dt: datetime) -> datetime:
        t = dt
        while t.weekday() >= 5:
            t = datetime.combine(t.date() + timedelta(days=1), time(self.open_h, 0))
        if t.hour < self.open_h:
            return datetime.combine(t.date(), time(self.open_h, 0))
        if self.lunch_start <= t.hour < self.lunch_end:
            return datetime.combine(t.date(), time(self.lunch_end, 0))
        if t.hour >= self.close_h:
            nxt = business_week_start(t.date() + timedelta(days=1))
            return datetime.combine(nxt, time(self.open_h, 0))
        return t

    def add_work(self,
                 start: datetime,
                 hours: float,
                 resource_shift: Optional[Tuple[int, int]] = None,
                 allow_overtime: bool = False) -> Tuple[datetime, float]:
        """
        Adds 'hours' of work respecting calendar (and optional resource shift).
        If allow_overtime=True, allows up to +2h beyond shift end (capped at 22:00). Returns (end_time, overtime_hours_approx).
        """
        remaining = float(hours)
        t = self.next_open(start)
        overtime_hours = 0.0

        while remaining > 1e-9:
            if t.weekday() >= 5:
                t = datetime.combine(t.date() + timedelta(days=(7 - t.weekday())), time(self.open_h, 0))
                continue

            day_open = datetime.combine(t.date(), time(self.open_h, 0))
            day_close_nominal = datetime.combine(t.date(), time(self.close_h, 0))
            lunch_start = datetime.combine(t.date(), time(self.lunch_start, 0))
            lunch_end = datetime.combine(t.date(), time(self.lunch_end, 0))

            if resource_shift is not None:
                r_open = datetime.combine(t.date(), time(resource_shift[0], 0))
                r_close = datetime.combine(t.date(), time(resource_shift[1], 0))
                day_open = max(day_open, r_open)
                day_close_nominal = min(day_close_nominal, r_close)

            day_close_ext = day_close_nominal
            if allow_overtime:
                ext_hour = min(22, (resource_shift[1] if resource_shift else self.close_h) + 2)
                day_close_ext = datetime.combine(t.date(), time(ext_hour, 0))

            if t < day_open:
                t = day_open
            if t >= day_close_ext:
                nxt = business_week_start(t.date() + timedelta(days=1))
                t = datetime.combine(nxt, time(self.open_h, 0))
                continue

            if t < lunch_start:
                window_end = min(lunch_start, day_close_ext)
            elif t < lunch_end:
                t = lunch_end
                continue
            else:
                window_end = day_close_ext

            if window_end <= t:
                t = window_end
                continue

            avail_hours = (window_end - t).total_seconds() / 3600.0
            work_now = min(avail_hours, remaining)
            end = t + td_hours(work_now)

            base_close = min(day_close_nominal, window_end)
            if allow_overtime and end > base_close:
                over_part = (end - max(base_close, t)).total_seconds() / 3600.0
                overtime_hours += max(0.0, over_part)

            remaining -= work_now
            t = end

        return t, overtime_hours


class ResourceCalendar:
    """Per-resource shifts: R1..R10 08–16, R11..R20 09–17, R21..R30 10–18."""
    def __init__(self):
        self.shifts: Dict[str, Tuple[int, int]] = {}
        for i in range(1, 31):
            if 1 <= i <= 10:
                self.shifts[f"R{i}"] = (8, 16)
            elif 11 <= i <= 20:
                self.shifts[f"R{i}"] = (9, 17)
            else:
                self.shifts[f"R{i}"] = (10, 18)

    def get_shift(self, rid: str) -> Tuple[int, int]:
        return self.shifts.get(rid, (9, 17))


# ---------------------------
# Activity universe / roles
# ---------------------------

ALL_ACTIVITIES = [
    "CTL_StartCampaign",
    "CTL_ImportAudience",
    "CTL_SegmentAudience",
    "CTL_PrioritizeSegments",
    "CTL_DesignAssets",
    "CTL_ReviewAssets",
    "CTL_FixAssets",
    "CTL_QA",
    "CTL_ApproveCampaign",
    "CTL_Expedite",
    "CTL_LaunchCampaign",
    "CTL_VendorExecute",
    "CTL_MonitorPerformance",
    "CTL_AdjustParameters",
    "CTL_EmailBlast",
    "CTL_SocialPost",
    "CTL_CallOutreach",
    "CTL_ABTest_V1",
    "CTL_ABTest_V2",
    "CTL_ABTest_V3",
    "CTL_Join",
    "CTL_ScoreLeads",
    "CTL_QualifyLeads",
    "CTL_EnrichData",
    "CTL_RouteLead",
    "CTL_Audit",
    "CTL_Warn",
    "CTL_Escalate",
    "CTL_Reassign",
    "CTL_Handover",
    "CTL_Wait",
    "CTL_Cancel",
    "CTL_Rollback",
    "CTL_SLA_Penalty",
    "CTL_Archive",
    "CTL_End"
]

ACT_FAMILY = {
    "CTL_StartCampaign": "Control",
    "CTL_ImportAudience": "Data",
    "CTL_SegmentAudience": "Analytics",
    "CTL_PrioritizeSegments": "Analytics",
    "CTL_DesignAssets": "Design",
    "CTL_ReviewAssets": "Review",
    "CTL_FixAssets": "Rework",
    "CTL_QA": "QA",
    "CTL_ApproveCampaign": "Approval",
    "CTL_Expedite": "Control",
    "CTL_LaunchCampaign": "Execution",
    "CTL_VendorExecute": "Execution",
    "CTL_MonitorPerformance": "Monitoring",
    "CTL_AdjustParameters": "Adjustment",
    "CTL_EmailBlast": "Followup",
    "CTL_SocialPost": "Followup",
    "CTL_CallOutreach": "Followup",
    "CTL_ABTest_V1": "Experiment",
    "CTL_ABTest_V2": "Experiment",
    "CTL_ABTest_V3": "Experiment",
    "CTL_Join": "Sync",
    "CTL_ScoreLeads": "Analytics",
    "CTL_QualifyLeads": "Qualification",
    "CTL_EnrichData": "Data",
    "CTL_RouteLead": "Routing",
    "CTL_Audit": "Audit",
    "CTL_Warn": "Control",
    "CTL_Escalate": "Control",
    "CTL_Reassign": "Control",
    "CTL_Handover": "Handover",
    "CTL_Wait": "Wait",
    "CTL_Cancel": "Control",
    "CTL_Rollback": "Control",
    "CTL_SLA_Penalty": "Control",
    "CTL_Archive": "Control",
    "CTL_End": "Control"
}

PREFERRED_ROLE = {
    "CTL_ImportAudience": "analyst",
    "CTL_SegmentAudience": "analyst",
    "CTL_PrioritizeSegments": "analyst",
    "CTL_DesignAssets": "analyst",
    "CTL_ReviewAssets": "analyst",
    "CTL_FixAssets": "analyst",
    "CTL_QA": "analyst",
    "CTL_ApproveCampaign": "senior",
    "CTL_Expedite": "senior",
    "CTL_LaunchCampaign": "analyst",
    "CTL_VendorExecute": "vendor",
    "CTL_MonitorPerformance": "analyst",
    "CTL_AdjustParameters": "analyst",
    "CTL_EmailBlast": "analyst",
    "CTL_SocialPost": "analyst",
    "CTL_CallOutreach": "analyst",
    "CTL_ABTest_V1": "analyst",
    "CTL_ABTest_V2": "analyst",
    "CTL_ABTest_V3": "analyst",
    "CTL_ScoreLeads": "analyst",
    "CTL_QualifyLeads": "analyst",
    "CTL_EnrichData": "analyst",
    "CTL_RouteLead": "analyst",
    "CTL_Audit": "senior",
    "CTL_Handover": "system",
    "CTL_Warn": "senior",
    "CTL_Escalate": "senior",
    "CTL_Reassign": "senior",
    "CTL_Wait": "system",
    "CTL_Cancel": "senior",
    "CTL_Rollback": "senior",
    "CTL_SLA_Penalty": "senior",
    "CTL_Archive": "system",
    "CTL_StartCampaign": "system",
    "CTL_Join": "system",
    "CTL_End": "system"
}


# ---------------------------
# Cost Model
# ---------------------------

class CostModel:
    def __init__(self, era: int):
        self.fixed_by_family = {
            "Design": 40.0, "Review": 25.0, "Rework": 30.0, "QA": 20.0,
            "Approval": 35.0, "Execution": 50.0, "Monitoring": 10.0, "Adjustment": 15.0,
            "Followup": 8.0, "Experiment": 20.0, "Analytics": 12.0, "Qualification": 15.0,
            "Data": 10.0, "Routing": 6.0, "Audit": 25.0, "Handover": 5.0, "Wait": 0.0,
            "Sync": 5.0, "Control": 10.0
        }
        vendor_rate = [55.0, 50.0, 42.0][era]
        analyst_rate = [60.0, 58.0, 56.0][era]
        senior_rate = [85.0, 88.0, 92.0][era]
        system_rate = 15.0
        self.rate_by_role = {
            "analyst": analyst_rate,
            "senior": senior_rate,
            "vendor": vendor_rate,
            "system": system_rate,
            "unknown": 60.0
        }
        self.overtime_mult = 1.5
        self.handover_fee = 12.0
        self.waiting_cost_per_hour = 4.0
        self.escalation_penalty = 240.0 if era < 2 else 260.0
        self.expedite_fee = 120.0
        self.rework_premium_mult = 1.20 + 0.05 * era
        self.sla_penalty = 300.0 + 40.0 * era

    def event_cost(self, activity: str, role: str, duration_h: float, waiting_h: float = 0.0,
                   overtime_h: float = 0.0, is_rework_iter: int = 0, is_handover: bool = False,
                   extra_fixed: float = 0.0) -> float:
        fam = ACT_FAMILY.get(activity, "Control")
        fixed = self.fixed_by_family.get(fam, 8.0)
        rate = self.rate_by_role.get(role, self.rate_by_role["unknown"])
        if fam == "Rework" and is_rework_iter > 0:
            fixed *= (self.rework_premium_mult ** is_rework_iter)
            rate *= (1.0 + 0.10 * is_rework_iter)
        overtime_fee = rate * (self.overtime_mult - 1.0) * max(0.0, overtime_h)
        handover_fee = self.handover_fee if is_handover else 0.0
        wait_cost = self.waiting_cost_per_hour * max(0.0, waiting_h)
        return float(fixed + rate * max(0.0, duration_h) + overtime_fee + handover_fee + wait_cost + extra_fixed)


# ---------------------------
# Resources & assignment
# ---------------------------

class ResourceManager:
    def __init__(self, rng: random.Random, rc: ResourceCalendar):
        self.rng = rng
        self.rc = rc
        self.resources = [f"R{i}" for i in range(1, 31)]
        self.role: Dict[str, str] = {}
        for i in range(1, 31):
            if i <= 12:
                self.role[f"R{i}"] = "analyst"
            elif i <= 20:
                self.role[f"R{i}"] = "senior"
            else:
                self.role[f"R{i}"] = "vendor"
        self.next_free: Dict[str, datetime] = {}
        fams = set(ACT_FAMILY.values())
        self.skill: Dict[str, Dict[str, float]] = {r: {f: clamp(self.rng.normalvariate(1.0, 0.12), 0.75, 1.35) for f in fams} for r in self.resources}
        self.experience: Dict[str, Dict[str, int]] = {r: defaultdict(int) for r in self.resources}
        self.last_exec: Dict[str, datetime] = {}
        self.last_family: Dict[str, Optional[str]] = {r: None for r in self.resources}

    def get_shift(self, rid: str) -> Tuple[int, int]:
        return self.rc.get_shift(rid)

    def init_time(self, start: datetime):
        for r in self.resources:
            self.next_free[r] = start

    def _candidates_by_role(self, role: str) -> List[str]:
        if role == "system":
            return [r for r in self.resources if self.role[r] in ("analyst", "senior")]
        return [r for r in self.resources if self.role[r] == role]

    def assign(self, activity: str, ready_time: datetime, prefer_role: str, policy_mix=(0.5, 0.3, 0.2)) -> str:
        candidates = self._candidates_by_role(prefer_role) or self.resources
        a, b, c = policy_mix
        p = self.rng.random()
        if p < a:
            fam = ACT_FAMILY.get(activity, "Control")
            sorted_cands = sorted(candidates, key=lambda rid: (self.skill[rid][fam], self.next_free[rid]))
            return sorted_cands[0]
        elif p < a + b:
            sorted_cands = sorted(candidates, key=lambda rid: self.next_free[rid])
            return sorted_cands[0]
        else:
            return self.rng.choice(candidates)

    def learning_forgetting_mult(self, rid: str, family: str, now: datetime) -> float:
        exp = self.experience[rid][family]
        learn = clamp(1.0 - 0.03 * math.sqrt(exp), 0.70, 1.0)
        last = self.last_exec.get(rid, None)
        if last is None:
            forget = 1.0
        else:
            days = max(0.0, (now - last).total_seconds() / 86400.0)
            forget = clamp(1.0 + 0.05 * math.log1p(max(0.0, days / 10.0)), 1.0, 1.25)
        return learn * forget

    def changeover_setup_minutes(self, rid: str, new_family: str) -> float:
        prev = self.last_family.get(rid, None)
        if prev is None or prev == new_family:
            return 0.0
        return clamp(self.rng.gauss(18, 6), 5, 45)

    def book(self, rid: str, family: str, end_time: datetime):
        self.experience[rid][family] += 1
        self.last_exec[rid] = end_time
        self.last_family[rid] = family
        self.next_free[rid] = end_time


# ---------------------------
# Service time sampler (heavy-tailed)
# ---------------------------

class ServiceSampler:
    def __init__(self, rng: random.Random, era: int):
        self.rng = rng
        base_means = {
            "Design": 120, "Review": 55, "Rework": 75, "QA": 40,
            "Approval": 30, "Execution": 60, "Monitoring": 15, "Adjustment": 30,
            "Followup": 12, "Experiment": 80, "Analytics": 25, "Qualification": 35,
            "Data": 20, "Routing": 8, "Audit": 25, "Sync": 8, "Control": 5, "Handover": 12, "Wait": 0
        }
        self.mean_by_family = base_means.copy()
        if era == 1:
            self.mean_by_family["Review"] *= 0.90
            self.mean_by_family["Approval"] *= 1.05
        elif era >= 2:
            self.mean_by_family["Review"] *= 0.82
            self.mean_by_family["Approval"] *= 1.12

    def _mix_lognorm_pareto(self, mean_minutes: float) -> float:
        mu = math.log(max(1e-6, mean_minutes * 0.8))
        sigma = 0.55
        ln = self.rng.lognormvariate(mu, sigma)
        if self.rng.random() < 0.10:
            alpha = 2.0
            xm = max(5.0, mean_minutes * 0.8)
            pareto = xm / (self.rng.random() ** (1.0 / alpha))
            return 0.7 * ln + 0.3 * pareto
        return ln

    def sample_minutes(self, activity: str, rid: str, rm: ResourceManager, now: datetime,
                       queue_backlog_h: float = 0.0) -> float:
        fam = ACT_FAMILY.get(activity, "Control")
        base_mean = self.mean_by_family.get(fam, 10.0)
        draw = self._mix_lognorm_pareto(base_mean)
        skill = rm.skill[rid][fam]
        lf = rm.learning_forgetting_mult(rid, fam, now)
        press = clamp(1.0 - 0.05 * min(5.0, queue_backlog_h), 0.75, 1.0)
        minutes = clamp(draw * skill * lf * press, 1.0, 8 * 60.0)
        minutes += rm.changeover_setup_minutes(rid, fam)
        return minutes


# ---------------------------
# Case & scheduling state
# ---------------------------

@dataclass
class CaseState:
    case_id: str
    era: int
    priority: str
    start_time: datetime
    deadline_hours: float
    budget_limit: float
    calendar: BusinessCalendar
    res_mgr: ResourceManager
    sampler: ServiceSampler
    cost_model: CostModel
    events: List[Dict[str, Any]] = field(default_factory=list)
    curr_time: datetime = None
    cum_cost: float = 0.0
    warned: bool = False
    escalated: bool = False
    expedited: bool = False
    binding: Dict[str, str] = field(default_factory=dict)
    separation_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [("CTL_ApproveCampaign", "CTL_Audit")])
    last_assigned_resource: Optional[str] = None
    has_parallel: bool = False
    branch_imbalance_minutes: List[float] = field(default_factory=list)
    review_total_minutes: float = 0.0

    def __post_init__(self):
        self.curr_time = self.start_time

    def remaining_time_to_deadline(self) -> float:
        elapsed_h = (self.curr_time - self.start_time).total_seconds() / 3600.0
        return self.deadline_hours - elapsed_h


# ---------------------------
# NHPP arrivals (bursty)
# ---------------------------

def nhpp_generate_arrivals(rng: random.Random, start: datetime, n: int,
                           horizon_days: int = 180) -> List[datetime]:
    cal = BusinessCalendar()
    t = start
    arrivals = []
    end = start + timedelta(days=horizon_days)

    def hour_peak_weight(h):
        w1 = math.exp(-((h - 10.5) ** 2) / (2 * 1.2 ** 2))
        w2 = math.exp(-((h - 15.0) ** 2) / (2 * 1.3 ** 2))
        return 0.3 + 0.7 * (0.6 * w1 + 0.4 * w2)

    def day_w(dow):
        return [0.7, 1.0, 1.2, 1.2, 0.9, 0.2, 0.0][dow]

    base = 3.2
    max_lambda = base * 1.25

    while len(arrivals) < n and t < end:
        dt_hours = rng.expovariate(max_lambda)
        t = t + td_hours(dt_hours)
        if t.weekday() >= 5:
            t = datetime.combine(business_week_start(t.date()), time(9, 0))
        hour = t.hour + t.minute / 60.0
        lam = base * day_w(t.weekday()) * hour_peak_weight(hour)
        if rng.random() <= lam / max_lambda:
            t = cal.next_open(t)
            arrivals.append(t)

    while len(arrivals) < n:
        d = start.date() + timedelta(days=len(arrivals) // 8 + 1)
        d = business_week_start(d)
        h = rng.choice([9, 10, 11, 13, 14, 15, 16])
        arrivals.append(datetime.combine(d, time(h, rng.choice([0, 15, 30, 45]))))
    return arrivals[:n]


# ---------------------------
# Flow construction & scheduling
# ---------------------------

def geometric_rework(rng: random.Random, pass_prob: float = 0.65, max_iters: int = 4) -> int:
    iters = 0
    while iters < max_iters and rng.random() > pass_prob:
        iters += 1
    return iters

def choose_or_subset(rng: random.Random, tasks: List[str]) -> List[str]:
    k = rng.randint(1, len(tasks))
    return rng.sample(tasks, k)

def should_preempt(rng: random.Random, service_hours: float, queue_backlog_hours: float) -> bool:
    base = 0.10 if service_hours > 2.5 else 0.03
    p = base + 0.07 * clamp(queue_backlog_hours / 2.0, 0.0, 1.0)
    return rng.random() < p

def assign_resource_for_activity(cs: CaseState, activity: str, prefer_role: str) -> str:
    if activity in ("CTL_ReviewAssets", "CTL_ApproveCampaign") and "CTL_DesignAssets" in cs.binding:
        if activity == "CTL_ReviewAssets":
            return cs.binding.get("CTL_DesignAssets")
    if activity == "CTL_Audit" and cs.binding.get("CTL_ApproveCampaign"):
        avoid = cs.binding["CTL_ApproveCampaign"]
        cand = [r for r in cs.res_mgr._candidates_by_role("senior") if r != avoid]
        if cand:
            return cand[0]
    rid = cs.res_mgr.assign(activity, cs.curr_time, prefer_role)
    return rid

def maybe_insert_wait_event(cs: CaseState, new_start: datetime, reason: str = "queue") -> None:
    if new_start > cs.curr_time:
        waiting_h = (new_start - cs.curr_time).total_seconds() / 3600.0
        amount = cs.cost_model.event_cost("CTL_Wait", "system", 0.0, waiting_h=waiting_h)
        cs.events.append({
            "concept:name": "CTL_Wait",
            "time:timestamp": new_start,
            "org:resource": "Rsys" if random.random() > 0.2 else "Unknown",
            "amount": float(amount),
            "note": f"wait:{reason}"
        })
        cs.cum_cost += amount
        cs.curr_time = new_start

def insert_handover_if_needed(cs: CaseState, new_resource: str):
    prev_r = cs.last_assigned_resource
    if prev_r is not None and prev_r != new_resource:
        delta_min = clamp(random.gauss(15, 5), 5, 45)
        end_time, _ = cs.calendar.add_work(cs.curr_time, delta_min / 60.0)
        amount = cs.cost_model.event_cost("CTL_Handover", "system", delta_min / 60.0, is_handover=True)
        cs.events.append({
            "concept:name": "CTL_Handover",
            "time:timestamp": end_time,
            "org:resource": "Rsys" if random.random() > 0.3 else "Unknown",
            "amount": float(amount)
        })
        cs.cum_cost += amount
        cs.curr_time = end_time

def schedule_task(cs: CaseState, activity: str, allow_overtime: bool = False,
                  meeting_with_analyst: bool = False,
                  rework_iter: int = 0) -> float:
    prefer_role = PREFERRED_ROLE.get(activity, "analyst")

    multi_resources: Optional[List[str]] = None
    if activity == "CTL_ApproveCampaign" and meeting_with_analyst:
        senior = assign_resource_for_activity(cs, activity, "senior")
        analyst = assign_resource_for_activity(cs, activity, "analyst")
        if analyst == senior:
            alt = [r for r in cs.res_mgr._candidates_by_role("analyst") if r != senior]
            if alt:
                analyst = alt[0]
        multi_resources = [senior, analyst]
        cs.binding["CTL_ApproveCampaign"] = senior

    rid = assign_resource_for_activity(cs, activity, prefer_role) if multi_resources is None else None
    if activity == "CTL_DesignAssets":
        cs.binding["CTL_DesignAssets"] = rid

    if rid is not None:
        insert_handover_if_needed(cs, rid)

    rc = cs.res_mgr
    shift = rc.get_shift(rid) if rid is not None else None

    start_candidate = cs.curr_time
    if rid is not None:
        start_candidate = max(start_candidate, rc.next_free[rid])
    if multi_resources:
        for rr in multi_resources:
            start_candidate = max(start_candidate, rc.next_free[rr])
    start_candidate = cs.calendar.next_open(start_candidate)

    if activity == "CTL_QA":
        next_release = (start_candidate.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=60))
        start_candidate = max(start_candidate, next_release)

    maybe_insert_wait_event(cs, start_candidate, reason="cal/queue/shift")

    backlog_h = 0.0
    if rid is not None:
        backlog_h = max(0.0, (rc.next_free[rid] - cs.curr_time).total_seconds() / 3600.0)

    rid_eff = multi_resources[0] if (rid is None and multi_resources) else rid
    minutes = cs.sampler.sample_minutes(activity, rid_eff, rc, cs.curr_time, queue_backlog_h=backlog_h)
    duration_h = minutes / 60.0

    preempt = should_preempt(random, duration_h, backlog_h) if activity not in ("CTL_Wait", "CTL_Handover") else False
    total_overtime_h = 0.0

    if preempt:
        start_label = f"{activity}.start"
        end1, ot1 = cs.calendar.add_work(cs.curr_time, 0.10, resource_shift=shift, allow_overtime=allow_overtime or cs.expedited or cs.escalated)
        role = PREFERRED_ROLE.get(activity, "analyst")
        amount = cs.cost_model.event_cost(activity, role, 0.10, overtime_h=ot1, is_rework_iter=rework_iter)
        cs.events.append({
            "concept:name": start_label,
            "time:timestamp": end1,
            "org:resource": rid if rid else "|".join(multi_resources),
            "amount": float(amount),
            "org:role": role if random.random() < 0.7 else None
        })
        cs.cum_cost += amount
        if rid: rc.book(rid, ACT_FAMILY[activity], end1)
        if multi_resources:
            for rr in multi_resources: rc.book(rr, ACT_FAMILY[activity], end1)

        stop_wait_h = clamp(random.uniform(0.5, 2.0), 0.25, 3.0)
        stop_end, _ = cs.calendar.add_work(end1, stop_wait_h, resource_shift=None, allow_overtime=False)
        wait_amt = cs.cost_model.event_cost("CTL_Wait", "system", 0.0, waiting_h=stop_wait_h)
        cs.events.append({
            "concept:name": "CTL_Wait",
            "time:timestamp": stop_end,
            "org:resource": "Rsys" if random.random() > 0.2 else "Unknown",
            "amount": float(wait_amt),
            "note": "preemption"
        })
        cs.cum_cost += wait_amt
        cs.curr_time = stop_end

        if rid: rc.next_free[rid] = max(rc.next_free[rid], stop_end)
        if multi_resources:
            for rr in multi_resources: rc.next_free[rr] = max(rc.next_free[rr], stop_end)

        resume_label = f"{activity}.resume"
        end2, ot2 = cs.calendar.add_work(cs.curr_time, 0.05, resource_shift=shift, allow_overtime=allow_overtime or cs.expedited or cs.escalated)
        amount2 = cs.cost_model.event_cost(activity, role, 0.05, overtime_h=ot2, is_rework_iter=rework_iter)
        cs.events.append({
            "concept:name": resume_label,
            "time:timestamp": end2,
            "org:resource": rid if rid else "|".join(multi_resources),
            "amount": float(amount2),
            "org:role": role if random.random() < 0.7 else None
        })
        cs.cum_cost += amount2
        if rid: rc.book(rid, ACT_FAMILY[activity], end2)
        if multi_resources:
            for rr in multi_resources: rc.book(rr, ACT_FAMILY[activity], end2)
        cs.curr_time = end2
        total_overtime_h += ot1 + ot2

        remaining_h = max(0.0, duration_h - 0.15)
        end3, ot3 = cs.calendar.add_work(cs.curr_time, remaining_h, resource_shift=shift, allow_overtime=allow_overtime or cs.expedited or cs.escalated)
        amount3 = cs.cost_model.event_cost(activity, role, remaining_h, overtime_h=ot3, is_rework_iter=rework_iter)
        e3 = {
            "concept:name": f"{activity}.complete",
            "time:timestamp": end3,
            "org:resource": rid if rid else "|".join(multi_resources),
            "amount": float(amount3),
            "org:role": role if random.random() < 0.7 else None
        }
        if random.random() < 0.03:
            e3["org:resource"] = "Unknown"
        cs.events.append(e3)
        cs.cum_cost += amount3
        fam = ACT_FAMILY.get(activity, "Control")
        if rid: cs.res_mgr.book(rid, fam, end3)
        if multi_resources:
            for rr in multi_resources: cs.res_mgr.book(rr, fam, end3)
        cs.curr_time = end3
        total_overtime_h += ot3

    else:
        end, ot = cs.calendar.add_work(cs.curr_time, duration_h, resource_shift=shift, allow_overtime=allow_overtime or cs.expedited or cs.escalated)
        role = PREFERRED_ROLE.get(activity, "analyst")
        amount = cs.cost_model.event_cost(activity, role, duration_h, overtime_h=ot, is_rework_iter=rework_iter)
        e = {
            "concept:name": activity,
            "time:timestamp": end,
            "org:resource": rid if rid else ("|".join(multi_resources) if multi_resources else "Rsys"),
            "amount": float(amount),
            "org:role": role if random.random() < 0.7 else None
        }
        if random.random() < 0.03:
            e["org:resource"] = "Unknown"
        cs.events.append(e)
        cs.cum_cost += amount
        fam = ACT_FAMILY.get(activity, "Control")
        if rid: cs.res_mgr.book(rid, fam, end)
        if multi_resources:
            for rr in multi_resources: cs.res_mgr.book(rr, fam, end)
        cs.curr_time = end
        total_overtime_h += ot

    if rid:
        cs.last_assigned_resource = rid
    elif multi_resources:
        cs.last_assigned_resource = multi_resources[0]

    if activity.startswith("CTL_ReviewAssets"):
        cs.review_total_minutes += minutes

    if activity == "CTL_Expedite":
        cs.expedited = True

    return minutes

def schedule_parallel_all(cs: CaseState, branches: List[List[Tuple[str, dict]]], join_label: str = "CTL_Join",
                          join_handover_minutes_range=(5, 20)) -> None:
    cs.has_parallel = True
    start_time = cs.curr_time
    branch_end_times: List[datetime] = []

    for b in branches:
        cs.curr_time = start_time
        for act, kwargs in b:
            schedule_task(cs, act, **kwargs)
        branch_end_times.append(cs.curr_time)

    join_ready = max(branch_end_times)
    join_delay_min = clamp(random.gauss(sum(join_handover_minutes_range)/2, 3), join_handover_minutes_range[0], join_handover_minutes_range[1])
    join_end, _ = cs.calendar.add_work(join_ready, join_delay_min / 60.0)
    amount = cs.cost_model.event_cost("CTL_Join", "system", join_delay_min / 60.0)
    cs.events.append({
        "concept:name": join_label,
        "time:timestamp": join_end,
        "org:resource": "Rsys" if random.random() > 0.2 else "Unknown",
        "amount": float(amount)
    })
    cs.cum_cost += amount
    cs.curr_time = join_end

    mins = [(t - start_time).total_seconds() / 60.0 for t in branch_end_times]
    if len(mins) >= 2:
        cs.branch_imbalance_minutes.append(max(mins) - min(mins))

def build_and_schedule_case(cs: CaseState, rng: random.Random) -> None:
    schedule_task(cs, "CTL_StartCampaign")
    early_next = weighted_choice([
        ("CTL_ImportAudience", 0.45),
        ("CTL_SegmentAudience", 0.35),
        ("CTL_DesignAssets", 0.20)
    ])
    schedule_task(cs, early_next)
    if rng.random() < 0.5:
        schedule_task(cs, "CTL_PrioritizeSegments")
    if early_next != "CTL_DesignAssets":
        schedule_task(cs, "CTL_DesignAssets")

    iters = geometric_rework(rng, pass_prob=0.62, max_iters=3)
    for i in range(iters):
        schedule_task(cs, "CTL_ReviewAssets")
        schedule_task(cs, "CTL_FixAssets", rework_iter=i+1)
    schedule_task(cs, "CTL_ReviewAssets")

    review_minutes = cs.review_total_minutes
    skip_prob = 0.60 if review_minutes <= 75 else 0.30
    if rng.random() > skip_prob:
        schedule_task(cs, "CTL_QA")

    warn_threshold_h = 0.80 * cs.deadline_hours
    elapsed_h = (cs.curr_time - cs.start_time).total_seconds() / 3600.0
    if not cs.warned and elapsed_h >= warn_threshold_h:
        amt = cs.cost_model.event_cost("CTL_Warn", "senior", 0.0) + 25.0
        cs.events.append({
            "concept:name": "CTL_Warn",
            "time:timestamp": cs.curr_time,
            "org:resource": "R13",
            "amount": float(amt)
        })
        cs.cum_cost += amt
        cs.warned = True

    schedule_task(cs, "CTL_ApproveCampaign", meeting_with_analyst=True)

    vendor_bias = 0.25 + (0.35 if cs.cum_cost > cs.budget_limit else 0.0)
    choice = weighted_choice([("vendor", vendor_bias), ("inhouse", 1.0 - vendor_bias)])
    if cs.warned and not cs.expedited and rng.random() < 0.40:
        schedule_task(cs, "CTL_Expedite")
    if choice == "vendor":
        schedule_task(cs, "CTL_VendorExecute", allow_overtime=cs.warned or cs.expedited)
    else:
        schedule_task(cs, "CTL_LaunchCampaign")

    channels = choose_or_subset(rng, ["CTL_EmailBlast", "CTL_SocialPost", "CTL_CallOutreach"])
    branches = [[(c, {})] for c in channels]
    schedule_parallel_all(cs, branches, join_label="CTL_Join")

    if set(channels) >= {"CTL_EmailBlast", "CTL_CallOutreach"} and rng.random() < 0.5:
        buffer_end, _ = cs.calendar.add_work(cs.curr_time, 2.0, allow_overtime=False)
        amt = cs.cost_model.event_cost("CTL_Wait", "system", 0.0, waiting_h=2.0)
        cs.events.append({
            "concept:name": "CTL_Wait",
            "time:timestamp": buffer_end,
            "org:resource": "Rsys",
            "amount": float(amt),
            "note": "cooling-off"
        })
        cs.cum_cost += amt
        cs.curr_time = buffer_end

    branches = [
        [("CTL_MonitorPerformance", {})],
        [("CTL_AdjustParameters", {})]
    ]
    schedule_parallel_all(cs, branches, join_label="CTL_Join")

    variants = ["CTL_ABTest_V1", "CTL_ABTest_V2", "CTL_ABTest_V3"]
    rng.shuffle(variants)
    ab_branches = [[(variants[0], {})], [(variants[1], {})]]
    schedule_parallel_all(cs, ab_branches, join_label="CTL_Join")
    if rng.random() < 0.4:
        schedule_task(cs, variants[2])
        amt = -abs(30.0 + 10.0 * cs.era)
        cs.events.append({
            "concept:name": "CTL_Rollback",
            "time:timestamp": cs.curr_time,
            "org:resource": "R15",
            "amount": float(amt)
        })
        cs.cum_cost += amt

    schedule_task(cs, "CTL_ScoreLeads")
    schedule_task(cs, "CTL_QualifyLeads")
    schedule_task(cs, "CTL_EnrichData")

    if rng.random() < 0.15:
        amt = cs.cost_model.event_cost("CTL_Reassign", "senior", 0.25)
        cs.events.append({
            "concept:name": "CTL_Reassign",
            "time:timestamp": cs.curr_time,
            "org:resource": "R14",
            "amount": float(amt)
        })
        cs.cum_cost += amt

    schedule_task(cs, "CTL_RouteLead")

    if rng.random() < 0.25:
        schedule_task(cs, "CTL_Audit")

    if (cs.curr_time - cs.start_time).total_seconds() / 3600.0 > cs.deadline_hours:
        amt = cs.cost_model.escalation_penalty
        cs.events.append({
            "concept:name": "CTL_Escalate",
            "time:timestamp": cs.curr_time,
            "org:resource": "R18",
            "amount": float(amt)
        })
        cs.cum_cost += amt
        cs.escalated = True
        pen = cs.cost_model.sla_penalty
        cs.events.append({
            "concept:name": "CTL_SLA_Penalty",
            "time:timestamp": cs.curr_time,
            "org:resource": "R18",
            "amount": float(pen)
        })
        cs.cum_cost += pen

    schedule_task(cs, "CTL_Archive")
    schedule_task(cs, "CTL_End")


# ---------------------------
# Distinct trace signature
# ---------------------------

def trace_signature(events: List[Dict[str, Any]]) -> Tuple[str, ...]:
    return tuple(e["concept:name"] for e in events)


# ---------------------------
# Generation driver
# ---------------------------

def generate_log(max_distinct_traces: int = 250,
                 seed: int = 42,
                 out_path: str = "camp2lead.xes.gz") -> EventLog:
    rng = random.Random(seed)
    np.random.seed(seed)

    start_sim = datetime(2025, 1, 6, 9, 0)  # Monday
    arrivals = nhpp_generate_arrivals(rng, start_sim, n=max_distinct_traces * 3, horizon_days=240)

    res_cal = ResourceCalendar()
    res_mgr = ResourceManager(rng, res_cal)
    res_mgr.init_time(arrivals[0])

    seen: set = set()
    traces: List[Trace] = []

    def era_for_time(dt: datetime) -> int:
        m = dt.month
        if m <= 4:
            return 0
        elif m <= 8:
            return 1
        else:
            return 2

    attempt = 0
    for idx, st in enumerate(arrivals):
        attempt += 1
        era = era_for_time(st)
        cal = BusinessCalendar()
        sampler = ServiceSampler(rng, era)
        cost_model = CostModel(era)
        priority = weighted_choice([("normal", 0.8), ("high", 0.2)])
        if priority == "high":
            deadline_h = rng.uniform(24, 40)
            budget = rng.uniform(1400, 2100)
        else:
            deadline_h = rng.uniform(40, 64)
            budget = rng.uniform(1600, 2600)

        cs = CaseState(case_id=f"C{idx+1:04d}",
                       era=era,
                       priority=priority,
                       start_time=st,
                       deadline_hours=deadline_h,
                       budget_limit=budget,
                       calendar=cal,
                       res_mgr=res_mgr,
                       sampler=sampler,
                       cost_model=cost_model)

        build_and_schedule_case(cs, rng)

        last_t = None
        for e in cs.events:
            t = e["time:timestamp"]
            e["time:timestamp"] = add_eps_if_needed(last_t, t)
            last_t = e["time:timestamp"]

        sig = trace_signature(cs.events)
        if sig in seen:
            if attempt > max_distinct_traces * 10:
                pass
            continue

        t = Trace()
        t.attributes["concept:name"] = cs.case_id
        t.attributes["era"] = str(era)
        t.attributes["priority"] = cs.priority
        t.attributes["deadline_hours"] = float(cs.deadline_hours)
        t.attributes["budget_limit"] = float(cs.budget_limit)
        for e in sorted(cs.events, key=lambda x: x["time:timestamp"]):
            ev = Event({
                "concept:name": e["concept:name"],
                "time:timestamp": e["time:timestamp"],
                "org:resource": e.get("org:resource", "Unknown"),
                "amount": float(e.get("amount", 0.0))
            })
            if "org:role" in e and e["org:role"] is not None:
                ev["org:role"] = e["org:role"]
            if "note" in e:
                ev["note"] = e["note"]
            t.append(ev)

        seen.add(sig)
        traces.append(t)

        if len(traces) >= max_distinct_traces:
            break

    L = EventLog()
    for tr in traces:
        L.append(tr)

    pm4py.write_xes(L, out_path)

    # ---------------------------
    # Sanity & quality checks
    # ---------------------------
    print("\n=== Generation complete ===")
    print(f"Saved log to: {out_path}")
    print(f"Traces generated: {len(L)} (distinct sequences)")
    def schema_ok(log: EventLog) -> Tuple[bool, int]:
        missing = 0
        for tr in log:
            last = None
            for e in tr:
                ok = all(k in e for k in ("concept:name", "time:timestamp", "org:resource", "amount"))
                if not ok:
                    missing += 1
                if last and not (e["time:timestamp"] > last):
                    missing += 1
                last = e["time:timestamp"]
        return (missing == 0), missing
    ok, cnt = schema_ok(L)
    print(f"Schema & monotonic timestamps: {'OK' if ok else 'Issues'} (violations={cnt})")

    try:
        df = pm4py.convert_to_dataframe(L)
        df["event_idx"] = df.groupby("case:concept:name").cumcount() + 1
        g = df.groupby("case:concept:name")
        length = g.size()
        amounts = df["amount"]
        inter_event = df.sort_values(["case:concept:name", "time:timestamp"]).groupby("case:concept:name")["time:timestamp"].diff().dropna().dt.total_seconds() / 60.0
        q = lambda s, p: float(np.quantile(s, p))
        print("\n-- Distribution summaries --")
        print(f"Trace length: mean={length.mean():.2f}, q25={q(length, 0.25):.0f}, q50={q(length, 0.50):.0f}, q75={q(length, 0.75):.0f}")
        print(f"Event amount: mean={amounts.mean():.2f}, q25={q(amounts, 0.25):.2f}, q50={q(amounts, 0.50):.2f}, q75={q(amounts, 0.75):.2f}, max={amounts.max():.2f}")
        if not inter_event.empty:
            print(f"Inter-event minutes: mean={inter_event.mean():.1f}, q25={q(inter_event, 0.25):.1f}, q50={q(inter_event, 0.50):.1f}, q75={q(inter_event, 0.75):.1f}, max={inter_event.max():.1f}")

        print("\n-- Ambiguity check (prefix → ≥2 next activities) --")
        seqs = g["concept:name"].apply(list).to_dict()
        traces_list = list(seqs.values())
        prefix_samples = 100
        ambiguous = 0
        rng2 = np.random.default_rng(123)
        for _ in range(prefix_samples):
            seq = rng2.choice(traces_list)
            if len(seq) < 3:
                continue
            k = rng2.integers(1, max(1, len(seq)-1))
            prefix = tuple(seq[:k])
            next_set = set()
            for s2 in traces_list:
                if len(s2) > k and tuple(s2[:k]) == prefix:
                    next_set.add(s2[k])
            if len(next_set) >= 2:
                ambiguous += 1
        print(f"Ambiguous prefixes (out of {prefix_samples}): {ambiguous}")

        print("\n-- Remaining time signal --")
        df_sorted = df.sort_values(["case:concept:name", "time:timestamp"])
        end_time = df_sorted.groupby("case:concept:name")["time:timestamp"].transform("max")
        rem_min = (end_time - df_sorted["time:timestamp"]).dt.total_seconds() / 60.0
        print(f"Remaining minutes: mean={rem_min.mean():.1f}, std={rem_min.std():.1f}, q10={q(rem_min,0.10):.1f}, q50={q(rem_min,0.50):.1f}, q90={q(rem_min,0.90):.1f}")

        print("\n-- Parallelism --")
        has_and = df["concept:name"].eq("CTL_Join").groupby(df["case:concept:name"]).any()
        pct = 100.0 * has_and.mean()
        print(f"Traces with ≥1 AND/OR join: {pct:.1f}%")
        print("(Branch imbalance collected during simulation; not reprinted here.)")

        print("\n-- Era drift --")
        era_means = df_sorted.groupby("case:concept:name").agg(
            start=("time:timestamp", "min"), end=("time:timestamp", "max"), total=("amount", "sum"))
        era_means["duration_h"] = (era_means["end"] - era_means["start"]).dt.total_seconds() / 3600.0
        eras = df_sorted.drop_duplicates("case:concept:name")[["case:concept:name", "case:era"]].set_index("case:concept:name")
        era_means = era_means.join(eras)
        for e_val, sub in era_means.groupby("case:era"):
            print(f"Era {e_val}: mean duration={sub['duration_h'].mean():.2f} h, mean cost={sub['total'].mean():.2f}")

    except Exception:
        print("\n(pandas not available) Basic stats only.")
        lengths = [len(tr) for tr in L]
        print(f"Trace length mean={sum(lengths)/len(lengths):.2f}, min={min(lengths)}, max={max(lengths)}")

    print("\nMinimal example note (not executed):")
    print("Start → Design(start/resume/complete due to preemption) → Review → Fix(rework once) → Review → Approve(meeting) →")
    print("  (Monitor || Adjust) → Join(+handover) → VendorExecute(overtime) → Escalate(SLA) → End")
    return L


# ---------------------------
# Main
# ---------------------------

def execute_script():
    log = generate_log(max_distinct_traces=250, seed=42, out_path="camp2lead.xes.gz")
    print(f"\nFinal log size: {len(log)} traces. Path: camp2lead.xes.gz")

if __name__ == "__main__":
    execute_script()
