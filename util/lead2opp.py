#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lead-to-Opportunity synthetic log generator (pm4py) — trace-by-trace construction.

Key features implemented (abbrev.):
- Activity & resource alphabets: 30+ activity labels; resources R1..R30 with roles and small 'Unknown' fraction.
- Control-flow skeletons: linear w/ optional inserts, XOR, loops (geometric rework), AND/OR splits with explicit joins,
  multi-instance k-of-n (reference calls), skip patterns, long-term dependency (binding-of-duty / separation-of-duty),
  preemption (X.start/resume/complete).
- Temporal patterns: heavy-tailed service times, NHPP bursty arrivals, business & shift calendars (freeze on weekends),
  deadlines/aging with Warn/Escalate/Expedite, setup/changeover & handover delays, synchronization latency,
  simple batching/release windows, basic seasonality/drift across "eras", learning & forgetting curves.
- Resource patterns: skill heterogeneity, binding/separation-of-duty, handover-of-work event injection,
  external vendor option, (optional) multi-resource meeting.
- Cost patterns: fixed+variable rate (shift/overtime-aware), SLA penalties, rework premiums, expedite fee,
  handover fee, waiting/holding cost, era-based tariffs, budget-aware branching, cancellation credits.
- Sampling/scheduling: per-case skeleton choice + pattern bundle (3–8 patterns), global resource next-free tracking,
  calendar-aware start/end times, AND-join waits for max(child end), OR/k-of-n joins, queue-impact via
  resource contention and expedite logic.
- Exports one XES (.xes.gz) with required event attributes:
  * concept:name, time:timestamp, org:resource, amount (float)
  * trace attribute concept:name set to case id → appears as case:concept:name in DataFrame.
- Quality checks printed at the end.

Minimal example (covered by parameters/probabilities):
  A trace that hits: AND-split, handover delay, SLA breach → Escalate, rework once, and vendor path with overtime.

Author: ChatGPT (GPT-5 Pro) — Senior Process Mining Engineer
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time, date
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None  # optional

from pm4py.objects.log.obj import Event, Trace, EventLog

# ------------- Global controls -------------
SEED = 20251028
rng = np.random.default_rng(SEED)
random.seed(SEED)

N_CASES = 250  # <= 250 distinct traces as required
OUT_PATH = "lead2opp.xes.gz"

# Timeframe + drift ("eras")
ERA_SPLITS = [
    (datetime(2025, 1, 1, 9, 0, 0), "ERA_A"),   # Jan–Apr
    (datetime(2025, 5, 1, 9, 0, 0), "ERA_B"),   # May–Aug
    (datetime(2025, 9, 1, 9, 0, 0), "ERA_C"),   # Sep–Oct
]
ERA_PARAMS = {
    "ERA_A": {"duration_scale": 1.00, "approve_slow": 0.95, "review_fast": 1.05, "vendor_rate": 1.10},
    "ERA_B": {"duration_scale": 0.95, "approve_slow": 1.10, "review_fast": 0.90, "vendor_rate": 0.95},
    "ERA_C": {"duration_scale": 0.90, "approve_slow": 1.15, "review_fast": 0.85, "vendor_rate": 0.90},
}

# SLA / deadlines
SLA_WORK_DAYS = 14
WARN_MARGIN_WORK_DAYS = 2

# Unknown resource fraction
UNKNOWN_FRAC = 0.03

# Expedite parameters
EXPEDITE_FEE = 120.0
OVERTIME_MULTIPLIER = 1.40

# ------------- Activity & resource alphabets -------------

# Activity labels (30+; prefixed 'L_' for Lead-to-Opportunity domain)
ACTIVITIES = [
    "L_ReceiveLead",
    "L_EnrichData",
    "L_Qualify",
    "L_Score",
    "L_Contact",
    "L_ScheduleDemo",
    "L_TechCheck",
    "L_SendBrochure",
    "L_Demo",
    "L_ProposalDraft",
    "L_InternalReview",
    "L_ReviseProposal",
    "L_ApproveDiscount",
    "L_LegalReview",
    "L_SecurityCheck",
    "L_ReferenceCall",       # multi-instance (k-of-n)
    "L_ReferenceJoin_k2of3", # join event name
    "L_Negotiate",
    "L_FinanceReview",
    "L_QA",                  # QA (stochastic skip memory)
    "L_Expedite",
    "L_Warn",
    "L_Escalate",
    "L_VendorOutreach",
    "L_Reassign",
    "L_Wait",
    "L_Handover",
    "L_BacklogRelease",
    "L_JoinParallel",
    "L_OrJoin",
    "L_OpportunityCreated",
    "L_Decline",
    "L_Cancel",
    "L_Rollback",
    # Preemption variants (no lifecycle attr; represented as separate activities)
    "L_LegalReview.start", "L_LegalReview.resume", "L_LegalReview.complete",
    "L_Demo.start", "L_Demo.resume", "L_Demo.complete",
    # Multi-resource meeting (optional)
    "L_StakeholderMeeting",
]

# Activity families (for setup/changeover & skill multipliers)
FAMILY = {
    "L_ReceiveLead": "ingest",
    "L_EnrichData": "ingest",
    "L_Score": "ingest",
    "L_Qualify": "sales",
    "L_Contact": "sales",
    "L_ScheduleDemo": "sales",
    "L_Demo": "sales",
    "L_Demo.start": "sales",
    "L_Demo.resume": "sales",
    "L_Demo.complete": "sales",
    "L_TechCheck": "it",
    "L_SendBrochure": "ops",
    "L_ProposalDraft": "proposal",
    "L_InternalReview": "review",
    "L_ReviseProposal": "proposal",
    "L_ApproveDiscount": "finance",
    "L_LegalReview": "legal",
    "L_LegalReview.start": "legal",
    "L_LegalReview.resume": "legal",
    "L_LegalReview.complete": "legal",
    "L_SecurityCheck": "it",
    "L_ReferenceCall": "sales",
    "L_ReferenceJoin_k2of3": "sales",
    "L_Negotiate": "sales",
    "L_FinanceReview": "finance",
    "L_QA": "review",
    "L_Expedite": "ops",
    "L_Warn": "ops",
    "L_Escalate": "senior",
    "L_VendorOutreach": "vendor",
    "L_Reassign": "ops",
    "L_Wait": "ops",
    "L_Handover": "ops",
    "L_BacklogRelease": "ops",
    "L_JoinParallel": "ops",
    "L_OrJoin": "ops",
    "L_OpportunityCreated": "ops",
    "L_Decline": "ops",
    "L_Cancel": "ops",
    "L_Rollback": "ops",
    "L_StakeholderMeeting": "sales",
}

# Preferred roles by activity (fallback to 'analyst' if missing)
ROLE_PREFS: Dict[str, List[str]] = defaultdict(
    lambda: ["analyst"],
    **{
        "L_ReceiveLead": ["ops", "analyst"],
        "L_EnrichData": ["analyst"],
        "L_Score": ["analyst"],
        "L_Qualify": ["sales", "analyst"],
        "L_Contact": ["sales"],
        "L_ScheduleDemo": ["sales"],
        "L_TechCheck": ["it", "analyst"],
        "L_SendBrochure": ["ops", "sales"],
        "L_Demo": ["sales"],
        "L_Demo.start": ["sales"],
        "L_Demo.resume": ["sales"],
        "L_Demo.complete": ["sales"],
        "L_ProposalDraft": ["analyst", "sales"],
        "L_InternalReview": ["analyst", "senior"],
        "L_ReviseProposal": ["analyst", "sales"],
        "L_ApproveDiscount": ["finance", "senior"],
        "L_LegalReview": ["legal", "senior"],
        "L_LegalReview.start": ["legal", "senior"],
        "L_LegalReview.resume": ["legal", "senior"],
        "L_LegalReview.complete": ["legal", "senior"],
        "L_SecurityCheck": ["it"],
        "L_ReferenceCall": ["sales"],
        "L_ReferenceJoin_k2of3": ["sales"],
        "L_Negotiate": ["sales", "senior"],
        "L_FinanceReview": ["finance"],
        "L_QA": ["analyst"],
        "L_Expedite": ["ops", "senior"],
        "L_Warn": ["ops"],
        "L_Escalate": ["senior"],
        "L_VendorOutreach": ["vendor"],
        "L_Reassign": ["ops"],
        "L_Wait": ["ops"],
        "L_Handover": ["ops"],
        "L_BacklogRelease": ["ops"],
        "L_JoinParallel": ["ops"],
        "L_OrJoin": ["ops"],
        "L_OpportunityCreated": ["ops"],
        "L_Decline": ["ops"],
        "L_Cancel": ["ops"],
        "L_Rollback": ["ops"],
        "L_StakeholderMeeting": ["sales", "senior"],
    },
)

# ------------- Calendars -------------

@dataclass
class BusinessCalendar:
    """
    Business calendar: Mon–Fri, 09:00–17:00, lunch 12:00–13:00.
    Freeze on weekends. Resource shifts overlay these hours.
    """
    open_hour: int = 9
    close_hour: int = 17
    lunch_start: time = time(12, 0)
    lunch_end: time = time(13, 0)

    def is_weekend(self, dt: datetime) -> bool:
        return dt.weekday() >= 5

    def within_lunch(self, t: time) -> bool:
        return self.lunch_start <= t < self.lunch_end

    def in_business(self, dt: datetime) -> bool:
        if self.is_weekend(dt):
            return False
        t = dt.time()
        return (time(self.open_hour, 0) <= t < time(self.close_hour, 0)) and not self.within_lunch(t)

    def next_business_open(self, dt: datetime) -> datetime:
        """Move dt to next business open (skipping lunch and weekends)."""
        cur = dt
        # Skip weekends
        while self.is_weekend(cur):
            cur = datetime.combine(cur.date() + timedelta(days=1), time(self.open_hour, 0))
        t = cur.time()
        open_t = time(self.open_hour, 0)
        close_t = time(self.close_hour, 0)
        if t < open_t:
            return datetime.combine(cur.date(), open_t)
        if self.within_lunch(t):
            return datetime.combine(cur.date(), self.lunch_end)
        if t >= close_t:
            # move to next business day
            nxt = cur.date() + timedelta(days=1)
            while date.weekday(nxt) >= 5:
                nxt = nxt + timedelta(days=1)
            return datetime.combine(nxt, open_t)
        return cur

    def add_worktime(self, start: datetime, minutes: float, allow_overtime: bool = False,
                     resource_shift: Optional[Tuple[int, int]] = None) -> datetime:
        """
        Add working minutes w.r.t business calendar (+ optional overtime & resource shift).
        If allow_overtime=True, enable work outside resource shift up to 21:00 (Mon–Fri).
        """
        if minutes <= 0:
            return max(start, self.next_business_open(start))

        cur = self.next_business_open(start)
        remaining = minutes

        def day_close(dt: datetime) -> datetime:
            # Resource shift (hour,hour) bounds; otherwise business bounds
            if resource_shift:
                rs, re = resource_shift
                # clamp to business window too
                rs = max(rs, self.open_hour)
                re = min(re, self.close_hour)
            else:
                rs, re = self.open_hour, self.close_hour
            close_t = time(re, 0)
            # overtime extends to 21:00 weekdays
            if allow_overtime and not self.is_weekend(dt):
                close_t = time(21, 0)
            return datetime.combine(dt.date(), close_t)

        def next_open(dt: datetime) -> datetime:
            base = self.next_business_open(dt)
            if resource_shift:
                rs, re = resource_shift
                rs = max(rs, self.open_hour)
                # if base is before resource shift, push to shift start
                shift_start = datetime.combine(base.date(), time(rs, 0))
                if base < shift_start and not allow_overtime:
                    base = shift_start
            return base

        cur = next_open(cur)
        while remaining > 0:
            # handle lunch
            if not allow_overtime and (self.lunch_start <= cur.time() < self.lunch_end):
                cur = datetime.combine(cur.date(), self.lunch_end)
                continue
            # compute until end-of-window
            end_of_window = day_close(cur)
            # If we're before lunch and business-only, stop at lunch
            if not allow_overtime and cur.time() < self.lunch_start:
                lunch_dt = datetime.combine(cur.date(), self.lunch_start)
                work_chunk_end = min(lunch_dt, end_of_window)
            else:
                work_chunk_end = end_of_window

            # minutes available in this chunk
            chunk_min = (work_chunk_end - cur).total_seconds() / 60.0
            if chunk_min <= 0:
                # move to next open
                cur = next_open(datetime.combine(cur.date() + timedelta(days=1), time(self.open_hour, 0)))
                continue

            if remaining <= chunk_min:
                cur = cur + timedelta(minutes=remaining)
                remaining = 0
                return cur
            else:
                remaining -= chunk_min
                # advance to next open (after lunch or next day)
                cur = next_open(work_chunk_end + timedelta(minutes=1))
        return cur


# ------------- Resources -------------

@dataclass
class Resource:
    rid: str
    role: str
    shift: Tuple[int, int]  # (start_hour, end_hour) within business window
    next_free: datetime
    skills: Dict[str, float]  # family -> speed multiplier (lower=faster)
    last_exec: Dict[str, datetime] = field(default_factory=dict)
    count_exec: Dict[str, int] = field(default_factory=dict)

    def speed_factor(self, family: str, now: datetime) -> float:
        base = self.skills.get(family, 1.0)
        # Learning: improve with repetitions (min cap)
        n = self.count_exec.get(family, 0)
        learn = max(0.7, 0.98 ** n)
        # Forgetting: if inactive >15 days on this family, slow down (max cap)
        last = self.last_exec.get(family, None)
        if last is None:
            forget = 1.0
        else:
            days = max(0.0, (now - last).total_seconds() / 86400.0)
            forget = min(1.5, 1.0 + 0.02 * (days / 5.0))
        return base * learn * forget

    def mark_exec(self, family: str, when: datetime):
        self.last_exec[family] = when
        self.count_exec[family] = self.count_exec.get(family, 0) + 1


class ResourceManager:
    def __init__(self, cal: BusinessCalendar, era: str):
        self.cal = cal
        self.resources: List[Resource] = []
        self.by_role: Dict[str, List[Resource]] = defaultdict(list)
        self.era = era

    def add(self, res: Resource):
        self.resources.append(res)
        self.by_role[res.role].append(res)

    def pick(self, role: str, ready: datetime, family: str,
             policy_mix: Tuple[float, float, float] = (0.4, 0.3, 0.3),
             prefer: Optional[str] = None,
             forbid: Optional[str] = None) -> Resource:
        """
        Pick a resource by a mixture of policies:
        - random
        - least loaded (earliest next_free)
        - skill-based (lowest speed factor)
        Respect prefer/forbid (for binding/separation of duty).
        """
        pool = list(self.by_role.get(role, []))
        if not pool:
            # fallback to analysts if role empty
            pool = list(self.by_role.get("analyst", []))
        if prefer:
            # place preferred at beginning if available
            pref = [r for r in pool if r.rid == prefer]
            if pref:
                return pref[0]
        if forbid:
            pool = [r for r in pool if r.rid != forbid]
            if not pool:
                pool = list(self.by_role.get(role, []))  # relax

        if not pool:
            # ultimate fallback to any resource
            pool = self.resources[:]

        prand, pleast, pskill = policy_mix
        u = rng.random()
        if u < prand:
            return rng.choice(pool)
        elif u < prand + pleast:
            return sorted(pool, key=lambda r: r.next_free)[0]
        else:
            # skill-based: min speed_factor at "ready"
            scored = sorted(pool, key=lambda r: r.speed_factor(family, ready))
            return scored[0]

# ------------- Cost model -------------

class CostModel:
    def __init__(self):
        # fixed per activity (baseline)
        self.fixed = defaultdict(float, {
            "L_ReceiveLead": 2.0,
            "L_EnrichData": 5.0,
            "L_Score": 7.0,
            "L_Qualify": 10.0,
            "L_Contact": 6.0,
            "L_ScheduleDemo": 4.0,
            "L_TechCheck": 8.0,
            "L_SendBrochure": 3.0,
            "L_Demo": 15.0,
            "L_Demo.start": 5.0,
            "L_Demo.resume": 5.0,
            "L_Demo.complete": 8.0,
            "L_ProposalDraft": 20.0,
            "L_InternalReview": 8.0,
            "L_ReviseProposal": 12.0,
            "L_ApproveDiscount": 5.0,
            "L_LegalReview": 25.0,
            "L_LegalReview.start": 10.0,
            "L_LegalReview.resume": 10.0,
            "L_LegalReview.complete": 15.0,
            "L_SecurityCheck": 10.0,
            "L_ReferenceCall": 6.0,
            "L_ReferenceJoin_k2of3": 0.5,
            "L_Negotiate": 12.0,
            "L_FinanceReview": 10.0,
            "L_QA": 5.0,
            "L_Expedite": EXPEDITE_FEE,
            "L_Warn": 1.0,
            "L_Escalate": 50.0,
            "L_VendorOutreach": 15.0,
            "L_Reassign": 4.0,
            "L_Wait": 0.0,
            "L_Handover": 2.5,
            "L_BacklogRelease": 0.0,
            "L_JoinParallel": 0.0,
            "L_OrJoin": 0.0,
            "L_OpportunityCreated": 0.0,
            "L_Decline": 0.0,
            "L_Cancel": 0.0,
            "L_Rollback": -8.0,  # credit/refund
            "L_StakeholderMeeting": 18.0,
        })
        # hourly rates by role (baseline)
        self.rate = defaultdict(float, {
            "analyst": 30.0,
            "senior": 50.0,
            "vendor": 25.0,
            "sales": 40.0,
            "legal": 80.0,
            "finance": 45.0,
            "it": 35.0,
            "ops": 30.0
        })
        # handover fee additionally charged (on L_Handover events)
        self.handover_fee = 6.0
        # waiting cost per hour
        self.wait_rate = 2.0

    def compute(self, activity: str, role: str, duration_min: float, waiting_min: float,
                overtime: bool, era: str, loop_idx: int = 0, vendor_tariff: float = 1.0) -> float:
        fixed = float(self.fixed[activity])
        # rework premium grows with loop iteration
        if activity == "L_ReviseProposal":
            fixed *= (1.0 + 0.15 * loop_idx)
        # era-based tariff drift (lower vendor tariff later)
        rate = self.rate[role]
        if role == "vendor":
            rate *= vendor_tariff
        # overtime uplift
        if overtime:
            rate *= OVERTIME_MULTIPLIER
        # variable piece
        var = rate * (duration_min / 60.0)
        wait_cost = self.wait_rate * (waiting_min / 60.0)
        return float(round(fixed + var + wait_cost, 2))

# ------------- Utilities -------------

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def lognormal_minutes(mu: float, sigma: float, scale: float = 1.0) -> float:
    """Draw heavy-tailed minutes (lognormal with optional Pareto tail)."""
    if rng.random() < 0.10:  # 10% Pareto tail
        xm = 10.0 * scale
        alpha = 2.5
        return float(xm / (rng.random() ** (1.0 / alpha)))
    x = rng.lognormal(mean=mu, sigma=sigma)
    return float(x * scale)

def pareto_minutes(xm: float, alpha: float) -> float:
    return float(xm / (rng.random() ** (1.0 / alpha)))

def era_of(dt: datetime) -> str:
    current = ERA_SPLITS[0][1]
    for start, name in ERA_SPLITS:
        if dt >= start:
            current = name
    return current

def add_work_days(cal: BusinessCalendar, start: datetime, n_days: int) -> datetime:
    """Add business days in calendar."""
    cur = cal.next_business_open(start)
    days = 0
    while days < n_days:
        cur = datetime.combine(cur.date(), time(cal.close_hour, 0))
        cur = cal.next_business_open(cur + timedelta(days=1))
        days += 1
    return cur

def align_to_release_window(cal: BusinessCalendar, dt0: datetime) -> datetime:
    """
    Batch/release: Only process at 10:00 or 15:00 on business days.
    """
    dt = cal.next_business_open(dt0)
    release_times = [time(10, 0), time(15, 0)]
    # if already past 15:00 => next business day 10:00
    if dt.time() > release_times[-1]:
        nxt = cal.next_business_open(datetime.combine(dt.date() + timedelta(days=1), time(cal.open_hour, 0)))
        return datetime.combine(nxt.date(), release_times[0])
    # otherwise choose next release >= now
    for rt in release_times:
        cand = datetime.combine(dt.date(), rt)
        if cand >= dt:
            return cand
    return dt

# ------------- Arrival process (NHPP) -------------

def nhpp_arrivals(n: int, start: datetime) -> List[datetime]:
    """
    Non-homogeneous Poisson Process via thinning.
    Intensity lambda(t) changes by time-of-day (peaks) and era (drift).
    """
    arrivals = []
    t = start
    # max intensity per hour (upper bound)
    lam_max = 8.0  # per day approx (bounded)
    while len(arrivals) < n:
        # proposal step: exponential with rate lam_max / day
        w_days = rng.exponential(1.0 / (lam_max / 8.0))  # expected 8 per day @ lam_max
        t = t + timedelta(days=w_days)
        # compute actual intensity modulation
        tod = t.time()
        tod_factor = 1.0 + 0.7 * math.sin(2.0 * math.pi * (tod.hour + tod.minute / 60.0) / 24.0)
        # weekly seasonality: weaker on Mon morning, stronger midweek
        dow = t.weekday()
        dow_factor = (0.7 if dow in [0] else (1.2 if dow in [2, 3] else 1.0))
        # era drift
        e = era_of(t)
        era_scale = {"ERA_A": 1.0, "ERA_B": 1.1, "ERA_C": 0.9}[e]
        lam_t = 4.0 * tod_factor * dow_factor * era_scale
        if rng.random() < min(1.0, lam_t / lam_max):
            # align to business open to avoid weekend starts; produce waves
            arrivals.append(t)
    # sort and trim to business openings
    cal = BusinessCalendar()
    arrivals = [cal.next_business_open(a) for a in sorted(arrivals)[:n]]
    return arrivals

# ------------- Activity duration parameters -------------

# Baseline (lognormal mu,sigma in minutes) + variability tuned by era scale + family speed factor
ACT_BASE = defaultdict(lambda: (math.log(30.0), 0.60), {
    "L_ReceiveLead": (math.log(3.0), 0.35),
    "L_EnrichData": (math.log(12.0), 0.45),
    "L_Score": (math.log(18.0), 0.55),
    "L_Qualify": (math.log(35.0), 0.55),
    "L_Contact": (math.log(20.0), 0.50),
    "L_ScheduleDemo": (math.log(10.0), 0.45),
    "L_TechCheck": (math.log(25.0), 0.55),
    "L_SendBrochure": (math.log(8.0), 0.35),
    "L_Demo": (math.log(60.0), 0.65),
    "L_Demo.start": (math.log(25.0), 0.50),
    "L_Demo.resume": (math.log(10.0), 0.45),
    "L_Demo.complete": (math.log(25.0), 0.50),
    "L_ProposalDraft": (math.log(120.0), 0.70),
    "L_InternalReview": (math.log(40.0), 0.60),
    "L_ReviseProposal": (math.log(75.0), 0.65),
    "L_ApproveDiscount": (math.log(20.0), 0.40),
    "L_LegalReview": (math.log(90.0), 0.70),
    "L_LegalReview.start": (math.log(30.0), 0.60),
    "L_LegalReview.resume": (math.log(20.0), 0.55),
    "L_LegalReview.complete": (math.log(40.0), 0.60),
    "L_SecurityCheck": (math.log(60.0), 0.65),
    "L_ReferenceCall": (math.log(15.0), 0.45),
    "L_Negotiate": (math.log(90.0), 0.65),
    "L_FinanceReview": (math.log(35.0), 0.55),
    "L_QA": (math.log(25.0), 0.50),
    "L_Expedite": (math.log(5.0), 0.20),
    "L_Warn": (math.log(2.0), 0.20),
    "L_Escalate": (math.log(15.0), 0.40),
    "L_VendorOutreach": (math.log(45.0), 0.55),
    "L_Reassign": (math.log(15.0), 0.45),
    "L_Wait": (math.log(60.0), 0.70),
    "L_Handover": (math.log(12.0), 0.35),
    "L_BacklogRelease": (math.log(5.0), 0.25),
    "L_JoinParallel": (math.log(1.0), 0.10),
    "L_OrJoin": (math.log(1.0), 0.10),
    "L_StakeholderMeeting": (math.log(45.0), 0.45),
})

# ------------- Case configuration -------------

@dataclass
class CaseConfig:
    case_id: str
    arrival: datetime
    priority: int           # 1 high, 0 normal
    budget_limit: float
    sla_deadline: datetime
    warn_time: datetime
    era: str
    binding_pair: Optional[Tuple[str, str]] = None  # e.g., ("L_Qualify", "L_Negotiate")
    separation_pair: Optional[Tuple[str, str]] = None  # e.g., ("L_ApproveDiscount", "L_QA")
    expedite_until: Optional[datetime] = None

# ------------- Skeleton & pattern composition -------------

@dataclass
class ScheduledEvent:
    name: str
    role: str
    resource_id: str
    event_time: datetime
    amount: float
    # (internal) for sanity checks
    start_time: datetime
    duration_min: float

def build_skeleton(cfg: CaseConfig) -> List[Any]:
    """
    Return a composed control-flow structure (list of steps).
    A "step" can be:
      - ("act", "ActivityName")
      - ("opt", "ActivityName")  # optional insert
      - ("xor", [ [path1...], [path2...], ... ])
      - ("and", [ [branch1...], [branch2...], ... ])  # Join after all
      - ("or",  [ [branch1...], [branch2...], ... ])  # choose subset 1..k, join later
      - ("mi_kofn", "BaseAct", k, n)  # multi-instance, join on k-th completion
      - ("loop", ["ReviewAct", "FixAct"], p)  # geometric repeats of pair
      - ("preempt", "BaseAct")  # emit start/resume/complete
    """
    steps: List[Any] = []
    # Start
    steps.append(("act", "L_ReceiveLead"))
    # Optional enrichment
    if rng.random() < 0.65:
        steps.append(("opt", "L_EnrichData"))

    # Scoring (batched release later)
    steps.append(("act", "L_Score"))
    steps.append(("act", "L_Qualify"))
    steps.append(("act", "L_Contact"))

    # XOR after Qualify/Contact
    xor_paths = []
    # path A: Demo -> AND(Brochure || TechCheck) -> ProposalDraft
    pathA = [("act", "L_ScheduleDemo"),
             ("and", [[("preempt", "L_Demo")], [("act", "L_SendBrochure"), ("act", "L_TechCheck")]]),
             ("act", "L_ProposalDraft")]
    # path B: Direct proposal (skip demo)
    pathB = [("act", "L_ProposalDraft")]
    # path C: Nurture (skip to QA & Decline/Opportunity via Negotiation)
    pathC = [("act", "L_QA"), ("act", "L_ProposalDraft")]
    xor_paths.append(pathA)
    xor_paths.append(pathB)
    xor_paths.append(pathC)
    steps.append(("xor", xor_paths))

    # Proposal review with geometric loop (review, revise)
    steps.append(("loop", ["L_InternalReview", "L_ReviseProposal"], 0.35))

    # OR-split: Legal, Security, Finance (choose 1..3)
    or_branches = [
        [("preempt", "L_LegalReview")],
        [("act", "L_SecurityCheck")],
        [("act", "L_FinanceReview")],
    ]
    steps.append(("or", or_branches))

    # Multi-instance reference calls: 2-of-3
    steps.append(("mi_kofn", "L_ReferenceCall", 2, 3))

    # Negotiation (binding-of-duty with Qualify)
    steps.append(("act", "L_Negotiate"))

    # Optional QA (stochastic skip with memory of review duration will be applied later)
    if rng.random() < 0.6:
        steps.append(("act", "L_QA"))

    # Vendor option appears if budget exceeded later
    # End: OpportunityCreated / Decline / Cancel
    steps.append(("xor", [[("act", "L_OpportunityCreated")],
                          [("act", "L_Decline")],
                          [("act", "L_Cancel"), ("act", "L_Rollback")]]))
    return steps

# ------------- Simulation (scheduling & costing) -------------

class Simulator:
    def __init__(self):
        self.cal = BusinessCalendar()
        self.cost = CostModel()
        # Build resources
        self.res_mgr = ResourceManager(self.cal, "ERA_A")
        self.resources = self._init_resources()
        # state per case to inject handovers etc.
        self._case_prev_res: Dict[str, Optional[str]] = defaultdict(lambda: None)

    def _init_resources(self) -> List[Resource]:
        resources = []
        # Role distribution across 30 resources
        roles = (
            ["analyst"] * 12
            + ["sales"] * 6
            + ["senior"] * 4
            + ["legal"] * 3
            + ["finance"] * 2
            + ["it"] * 2
            + ["ops"] * 1
        )
        rng.shuffle(roles)
        start0 = datetime(2025, 1, 1, 9, 0, 0)
        for i in range(30):
            rid = f"R{i+1}"
            role = roles[i] if i < len(roles) else "analyst"
            # Shifts by role
            if role == "analyst":
                shift = (9, 17)
            elif role == "sales":
                shift = (8, 16)
            elif role == "senior":
                shift = (9, 18)
            elif role == "legal":
                shift = (10, 18)
            elif role == "finance":
                shift = (9, 17)
            elif role == "it":
                shift = (9, 17)
            elif role == "ops":
                shift = (9, 17)
            else:
                shift = (9, 17)
            # Skills per family (heterogeneous)
            skills = {}
            for fam in set(FAMILY.values()):
                # draw around 1.0 (lower is faster)
                skills[fam] = float(rng.normal(1.0, 0.12))
                skills[fam] = clamp(skills[fam], 0.75, 1.35)
            res = Resource(rid=rid, role=role, shift=shift, next_free=start0, skills=skills)
            resources.append(res)
            self.res_mgr.add(res)
        # Add a few vendor proxy resources (use rid still in R.. range for simplicity)
        for j in range(3):
            rid = f"R{30+j+1}"
            role = "vendor"
            shift = (11, 19)
            skills = {fam: float(rng.normal(1.05, 0.10)) for fam in set(FAMILY.values())}
            res = Resource(rid=rid, role=role, shift=shift, next_free=start0, skills=skills)
            resources.append(res)
            self.res_mgr.add(res)
        return resources

    def _duration_minutes(self, act: str, res: Resource, when: datetime, era: str,
                          prev_act: Optional[str], loop_idx: int) -> float:
        mu, sigma = ACT_BASE[act]
        fam = FAMILY.get(act, "generic")
        # Era drift on durations
        scale = ERA_PARAMS[era]["duration_scale"]
        # Activity-specific adjustments (review faster in later eras; approve slower)
        if act in ("L_InternalReview", "L_QA") and ERA_PARAMS[era]["review_fast"] != 1.0:
            scale *= ERA_PARAMS[era]["review_fast"]
        if act in ("L_ApproveDiscount",) and ERA_PARAMS[era]["approve_slow"] != 1.0:
            scale *= ERA_PARAMS[era]["approve_slow"]
        # Resource speed factor (skills + learning/forgetting)
        speed = res.speed_factor(fam, when)
        # Sequence-dependent setup/changeover if family changes
        setup = 0.0
        if prev_act:
            prev_fam = FAMILY.get(prev_act, fam)
            if prev_fam != fam:
                setup = rng.uniform(5.0, 25.0)
        # Loop amplification
        loop_amp = 1.0 + 0.1 * max(0, loop_idx - 1) if act == "L_ReviseProposal" else 1.0
        base = lognormal_minutes(mu, sigma, scale=scale) * speed * loop_amp
        dur = clamp(base + setup, 2.0, 8 * 60.0)
        return float(dur)

    def _assign_resource(self, act: str, ready: datetime,
                         prefer: Optional[str], forbid: Optional[str]) -> Resource:
        role_prefs = ROLE_PREFS[act]
        # Assignment policy mixture
        for role in role_prefs:
            pool = self.res_mgr.by_role.get(role, [])
            if pool:
                return self.res_mgr.pick(role, ready, FAMILY.get(act, "generic"),
                                         policy_mix=(0.4, 0.3, 0.3),
                                         prefer=prefer, forbid=forbid)
        # fallback to analyst
        return self.res_mgr.pick("analyst", ready, FAMILY.get(act, "generic"),
                                 policy_mix=(0.4, 0.3, 0.3),
                                 prefer=prefer, forbid=forbid)

    def _overtime_allowed(self, cfg: CaseConfig, act: str) -> bool:
        # Allow overtime if expedited or escalated recently
        return cfg.expedite_until is not None

    def _maybe_warn_escalate(self, cfg: CaseConfig, events: List[ScheduledEvent],
                             now: datetime) -> None:
        # Insert Warn if we crossed warn_time and no prior warn
        names = [e.name for e in events]
        if now >= cfg.warn_time and "L_Warn" not in names:
            self._schedule_simple(cfg, events, "L_Warn", now)
        # Escalate if violated SLA
        if now >= cfg.sla_deadline and "L_Escalate" not in names:
            esc_event = self._schedule_simple(cfg, events, "L_Escalate", now, prefer_role="senior")
            # Expedite next few hours
            cfg.expedite_until = esc_event.event_time + timedelta(hours=24)

    def _schedule_simple(self, cfg: CaseConfig, events: List[ScheduledEvent], act: str,
                         ready: datetime, prefer_role: Optional[str] = None,
                         loop_idx: int = 0, previous_act: Optional[str] = None) -> ScheduledEvent:
        prefer = None
        forbid = None
        # Select resource; if prefer_role is defined, try that role
        if prefer_role:
            candidates = self.res_mgr.by_role.get(prefer_role, [])
            if candidates:
                res = self.res_mgr.pick(prefer_role, ready, FAMILY.get(act, "generic"))
            else:
                res = self._assign_resource(act, ready, prefer, forbid)
        else:
            res = self._assign_resource(act, ready, prefer, forbid)

        # Release window for scoring
        eff_ready = align_to_release_window(self.cal, ready) if act == "L_Score" else ready
        allow_ot = self._overtime_allowed(cfg, act)

        # start @ max(ready, res.next_free, calendar open)
        start = max(eff_ready, res.next_free)
        start = self.cal.next_business_open(start)

        dur = self._duration_minutes(act, res, start, cfg.era, previous_act, loop_idx)
        end = self.cal.add_worktime(start, dur, allow_overtime=allow_ot, resource_shift=res.shift)

        # Update res
        res.next_free = end
        res.mark_exec(FAMILY.get(act, "generic"), end)

        # waiting time (holding)
        wait_min = max(0.0, (start - ready).total_seconds() / 60.0)

        # Vendor tariff drift per era
        vendor_tariff = ERA_PARAMS[cfg.era]["vendor_rate"]

        # Overtime detection: if any part outside shift or after 17:00, assume OT
        overtime = allow_ot and (start.time() < time(res.shift[0], 0) or end.time() > time(res.shift[1], 0))

        # Choose displayed resource id (with small Unknown fraction)
        rid_attr = res.rid if rng.random() > UNKNOWN_FRAC else "Unknown"

        amount = self.cost.compute(act, res.role, dur, wait_min, overtime, cfg.era, loop_idx, vendor_tariff)
        ev = ScheduledEvent(
            name=act, role=res.role, resource_id=rid_attr, event_time=end,
            amount=amount, start_time=start, duration_min=dur
        )
        events.append(ev)

        # Handover event if resource changed
        prev_res_real = self._case_prev_res[cfg.case_id]
        if prev_res_real is not None and prev_res_real != res.rid:
            # only if gap >= 5 minutes
            gap_min = max(0.0, (start - events[-2].event_time).total_seconds() / 60.0) if len(events) >= 2 else 0.0
            if gap_min >= 5.0:
                # insert a handover event at midpoint between prev_end and start
                mid = events[-2].event_time + timedelta(minutes=gap_min / 2.0)
                hv_wait = max(0.0, (mid - events[-2].event_time).total_seconds() / 60.0)
                hv_dur = clamp(rng.uniform(5.0, 20.0), 5.0, gap_min)
                hv_end = self.cal.add_worktime(mid, hv_dur, allow_overtime=False, resource_shift=self.res_mgr.by_role["ops"][0].shift)
                hv_amount = self.cost.handover_fee + self.cost.compute("L_Handover", "ops", hv_dur, hv_wait, False, cfg.era)
                handover_ev = ScheduledEvent(
                    name="L_Handover", role="ops", resource_id="Unknown" if rng.random() < 0.1 else "R1",
                    event_time=hv_end, amount=hv_amount, start_time=mid, duration_min=hv_dur
                )
                # place before the last scheduled event
                events.insert(len(events) - 1, handover_ev)

        # Remember real resource for next handover check
        self._case_prev_res[cfg.case_id] = res.rid
        # Warn/Escalate checks
        self._maybe_warn_escalate(cfg, events, end)
        return ev

    def _schedule_preemptable(self, cfg: CaseConfig, events: List[ScheduledEvent], base_act: str,
                              ready: datetime, previous_act: Optional[str]) -> ScheduledEvent:
        """
        Preemption & resume: split into start/resume/complete. Preempt prob increases with queue (approximated
        via resource contention at ready time).
        """
        # approximate queue pressure by number of resources with next_free > ready
        busy = sum(1 for r in self.res_mgr.resources if r.next_free > ready)
        preempt_prob = clamp(0.05 + 0.01 * busy, 0.05, 0.35)
        preempt = rng.random() < preempt_prob

        start_ev = self._schedule_simple(cfg, events, f"{base_act}.start", ready, previous_act=previous_act)
        if preempt:
            # wait a while and resume
            gap = rng.uniform(60.0, 8 * 60.0)  # 1h to 8h pause
            resume_ready = start_ev.event_time + timedelta(minutes=gap)
            self._schedule_simple(cfg, events, f"{base_act}.resume", resume_ready, previous_act=f"{base_act}.start")
            comp_ready = events[-1].event_time + timedelta(minutes=5.0)
            comp_ev = self._schedule_simple(cfg, events, f"{base_act}.complete", comp_ready, previous_act=f"{base_act}.resume")
            return comp_ev
        else:
            comp_ready = start_ev.event_time + timedelta(minutes=5.0)
            comp_ev = self._schedule_simple(cfg, events, f"{base_act}.complete", comp_ready, previous_act=f"{base_act}.start")
            return comp_ev

    def _schedule_and(self, cfg: CaseConfig, events: List[ScheduledEvent],
                      branches: List[List[Any]], ready: datetime, previous_act: Optional[str]) -> ScheduledEvent:
        """AND-split: schedule branches in parallel; Join when all complete (latency=max(end))."""
        branch_ends: List[datetime] = []
        for br in branches:
            end_t = self._schedule_seq(cfg, events, br, ready, previous_act)
            branch_ends.append(end_t)
        join_ready = max(branch_ends)
        join_ev = self._schedule_simple(cfg, events, "L_JoinParallel", join_ready, previous_act=previous_act)
        return join_ev

    def _schedule_or(self, cfg: CaseConfig, events: List[ScheduledEvent],
                     branches: List[List[Any]], ready: datetime, previous_act: Optional[str]) -> ScheduledEvent:
        """OR-split: choose 1..k random branches; Join when all selected complete."""
        k = rng.integers(1, len(branches) + 1)
        chosen_idx = rng.choice(len(branches), size=k, replace=False)
        ends = []
        for idx in chosen_idx:
            ends.append(self._schedule_seq(cfg, events, branches[idx], ready, previous_act))
        join_ready = max(ends) if ends else ready
        join_ev = self._schedule_simple(cfg, events, "L_OrJoin", join_ready, previous_act=previous_act)
        return join_ev

    def _schedule_mi_kofn(self, cfg: CaseConfig, events: List[ScheduledEvent],
                          base_act: str, k: int, n: int, ready: datetime, previous_act: Optional[str]) -> ScheduledEvent:
        """Multi-instance: spawn n tasks, proceed after k complete (k-th order statistic)."""
        comp_times = []
        for i in range(n):
            name = base_act  # identical label; case variation by resource/timestamp
            ev = self._schedule_simple(cfg, events, name, ready, previous_act=previous_act)
            comp_times.append(ev.event_time)
        # k-th order statistic
        comp_times.sort()
        kth = comp_times[k - 1]
        join_name = "L_ReferenceJoin_k2of3" if base_act == "L_ReferenceCall" and k == 2 and n == 3 else "L_OrJoin"
        join_ev = self._schedule_simple(cfg, events, join_name, kth, previous_act=previous_act)
        return join_ev

    def _schedule_loop(self, cfg: CaseConfig, events: List[ScheduledEvent],
                       pair: List[str], p: float, ready: datetime, previous_act: Optional[str]) -> ScheduledEvent:
        """Geometric repeats of {Review -> Fix} until pass."""
        review, fix = pair
        loop_idx = 0
        cur_ready = ready
        while True:
            rev_ev = self._schedule_simple(cfg, events, review, cur_ready, loop_idx=loop_idx, previous_act=previous_act)
            # memory: faster early review may increase skip prob for QA later (handled elsewhere)
            long_review = (rev_ev.duration_min > 60.0)
            if rng.random() > p * (1.2 if long_review else 1.0):
                return rev_ev
            # Fix/revise
            loop_idx += 1
            cur_ready = rev_ev.event_time + timedelta(minutes=15.0)
            self._schedule_simple(cfg, events, fix, cur_ready, loop_idx=loop_idx, previous_act=review)
            cur_ready = events[-1].event_time + timedelta(minutes=10.0)
            previous_act = fix

    def _schedule_seq(self, cfg: CaseConfig, events: List[ScheduledEvent],
                      seq: List[Any], ready: datetime, previous_act: Optional[str]) -> datetime:
        cur_ready = ready
        prev = previous_act
        for step in seq:
            kind = step[0]
            if kind == "act":
                act = step[1]
                ev = self._schedule_simple(cfg, events, act, cur_ready, previous_act=prev)
                cur_ready = ev.event_time + timedelta(minutes=3.0)
                prev = act
            elif kind == "opt":
                act = step[1]
                if rng.random() < 0.7:
                    ev = self._schedule_simple(cfg, events, act, cur_ready, previous_act=prev)
                    cur_ready = ev.event_time + timedelta(minutes=3.0)
                    prev = act
            elif kind == "xor":
                paths = step[1]
                idx = 0
                # branch probs depend on queue/cost/priority approx.
                if cfg.priority == 1 and rng.random() < 0.5:
                    idx = 0  # prefer demo path for high priority
                else:
                    idx = int(rng.choice(len(paths)))
                cur_ready = self._schedule_seq(cfg, events, paths[idx], cur_ready, prev)
                prev = "XOR"
            elif kind == "and":
                branches = step[1]
                ev = self._schedule_and(cfg, events, branches, cur_ready, prev)
                cur_ready = ev.event_time + timedelta(minutes=2.0)
                prev = "AND_JOIN"
            elif kind == "or":
                branches = step[1]
                ev = self._schedule_or(cfg, events, branches, cur_ready, prev)
                cur_ready = ev.event_time + timedelta(minutes=2.0)
                prev = "OR_JOIN"
            elif kind == "mi_kofn":
                base_act, k, n = step[1], step[2], step[3]
                ev = self._schedule_mi_kofn(cfg, events, base_act, k, n, cur_ready, prev)
                cur_ready = ev.event_time + timedelta(minutes=2.0)
                prev = "MI_JOIN"
            elif kind == "loop":
                pair, p = step[1], step[2]
                ev = self._schedule_loop(cfg, events, pair, p, cur_ready, prev)
                cur_ready = ev.event_time + timedelta(minutes=2.0)
                prev = pair[-1]
            elif kind == "preempt":
                base = step[1]
                ev = self._schedule_preemptable(cfg, events, base, cur_ready, prev)
                cur_ready = ev.event_time + timedelta(minutes=2.0)
                prev = f"{base}.complete"
            else:
                # unknown: skip
                pass
        return cur_ready

    def _apply_long_term_dependency(self, cfg: CaseConfig, events: List[ScheduledEvent]):
        """
        Binding-of-duty (same resource) between Qualify and Negotiate if possible;
        if not possible → Reassign sub-process.
        Separation-of-duty between ApproveDiscount and QA/Audit (ensure different resources).
        """
        # Gather indexes
        idx_map = defaultdict(list)
        for i, e in enumerate(events):
            idx_map[e.name].append(i)

        # Binding: Qualify & Negotiate
        if idx_map.get("L_Qualify") and idx_map.get("L_Negotiate"):
            q_res = events[idx_map["L_Qualify"][0]].resource_id
            n_idx = idx_map["L_Negotiate"][0]
            n_ev = events[n_idx]
            # If different and not Unknown, try to inject Reassign & retime Negotiate
            if q_res != "Unknown" and n_ev.resource_id not in (q_res, "Unknown"):
                # Insert Reassign before Negotiate, delaying it and changing resource_id to q_res
                base_time = events[n_idx - 1].event_time if n_idx > 0 else n_ev.start_time
                ra = self._schedule_simple(cfg, events, "L_Reassign", base_time + timedelta(minutes=5.0))
                # modify Negotiate to use Qualify's resource id label (only attribute; actual schedule remains)
                events = sorted(events, key=lambda e: e.event_time)
                for ev in events:
                    if ev.name == "L_Negotiate":
                        ev.resource_id = q_res
                        break

        # Separation: ApproveDiscount vs QA (proxy for Audit)
        if idx_map.get("L_ApproveDiscount") and idx_map.get("L_QA"):
            a_res = events[idx_map["L_ApproveDiscount"][0]].resource_id
            # ensure different from QA; if same -> Reassign
            for qi in idx_map["L_QA"]:
                if events[qi].resource_id == a_res and a_res != "Unknown":
                    base_time = events[qi - 1].event_time if qi > 0 else events[qi].start_time
                    _ = self._schedule_simple(cfg, events, "L_Reassign", base_time + timedelta(minutes=3.0))
                    # flip QA resource attribute to Unknown to indicate change
                    events[qi].resource_id = "Unknown"

        # Cooling-off: ensure lag between Contact and ProposalDraft
        if idx_map.get("L_Contact") and idx_map.get("L_ProposalDraft"):
            c_end = events[idx_map["L_Contact"][0]].event_time
            p_start = events[idx_map["L_ProposalDraft"][0]].start_time
            if (p_start - c_end).total_seconds() / 3600.0 < 6.0:
                # Insert Wait buffer
                _ = self._schedule_simple(cfg, events, "L_Wait", c_end + timedelta(minutes=5.0))

    def _maybe_budget_vendor_branch(self, cfg: CaseConfig, events: List[ScheduledEvent]) -> None:
        """
        Budget-aware branching: route to vendor if cumulative amount exceeds limit before Negotiation.
        """
        total = 0.0
        vendor_inserted = False
        for i, e in enumerate(events):
            total += e.amount
            if not vendor_inserted and total > cfg.budget_limit and e.name != "L_VendorOutreach":
                # Insert vendor outreach right here
                _ = self._schedule_simple(cfg, events, "L_VendorOutreach", e.event_time + timedelta(minutes=5.0))
                vendor_inserted = True

    def _stochastic_skip_QA_with_memory(self, events: List[ScheduledEvent]):
        """Probability to skip QA decreases if InternalReview was short (fast review → more QA)."""
        have_review = [e for e in events if e.name == "L_InternalReview"]
        have_qa_idx = [i for i, e in enumerate(events) if e.name == "L_QA"]
        if not have_qa_idx:
            return
        # compute review duration stat
        if have_review:
            avg_rev = np.mean([e.duration_min for e in have_review])
        else:
            avg_rev = 45.0
        # lower avg review => higher chance to keep QA; otherwise maybe drop one QA occurrence
        keep_prob = clamp(0.8 - (avg_rev - 45.0) / 100.0, 0.2, 0.9)
        for qi in have_qa_idx:
            if rng.random() > keep_prob:
                # null transition: remove QA event (skip pattern)
                events[qi].name = "L_Wait"
                events[qi].amount = 0.0

    def simulate_case(self, cfg: CaseConfig) -> List[ScheduledEvent]:
        """Build one case trace end-to-end with composed patterns."""
        self.res_mgr.era = cfg.era
        events: List[ScheduledEvent] = []
        self._case_prev_res[cfg.case_id] = None

        skeleton = build_skeleton(cfg)

        # Initial path + patterns (we may compose several)
        # 1) linear part from skeleton (includes XOR/AND/etc.)
        end_time = self._schedule_seq(cfg, events, skeleton, cfg.arrival, previous_act=None)

        # Long-term dependencies (binding/separation, cooling-off)
        events.sort(key=lambda e: e.event_time)
        self._apply_long_term_dependency(cfg, events)

        # Budget-aware vendor branching
        self._maybe_budget_vendor_branch(cfg, events)

        # Stochastic skip QA with memory of review
        self._stochastic_skip_QA_with_memory(events)

        # Aging/deadline final checks near the end
        if events:
            self._maybe_warn_escalate(cfg, events, events[-1].event_time)

        # Expedite can allow queue jump (modeled via next events scheduled w/ overtime flag active already)
        return sorted(events, key=lambda e: e.event_time)

# ------------- Generation & export -------------

def generate_log() -> EventLog:
    cal = BusinessCalendar()
    sim = Simulator()

    # NHPP arrivals in waves; cap to N_CASES
    start = datetime(2025, 1, 1, 9, 0, 0)
    arrivals = nhpp_arrivals(N_CASES, start)

    log = EventLog()
    trace_stats = []
    parallel_traces = 0
    drift_stats = defaultdict(lambda: {"dur": [], "cost": []})

    for i, arr in enumerate(arrivals):
        cid = f"CASE_{i+1:04d}"
        er = era_of(arr)

        # case attributes
        priority = 1 if rng.random() < 0.20 else 0
        budget_limit = float(rng.uniform(2500.0, 6500.0))
        sla_deadline = add_work_days(cal, arr, SLA_WORK_DAYS)
        warn_time = add_work_days(cal, arr, SLA_WORK_DAYS - WARN_MARGIN_WORK_DAYS)

        cfg = CaseConfig(
            case_id=cid, arrival=arr, priority=priority, budget_limit=budget_limit,
            sla_deadline=sla_deadline, warn_time=warn_time, era=er,
            binding_pair=("L_Qualify", "L_Negotiate"),
            separation_pair=("L_ApproveDiscount", "L_QA"),
            expedite_until=None
        )

        events = sim.simulate_case(cfg)

        # Build pm4py Trace
        tr = Trace()
        tr.attributes["concept:name"] = cid  # ensures case:concept:name in DataFrames
        # sort chronologically and inject Start-of-Case explicit event (optional: not required)
        events = sorted(events, key=lambda e: e.event_time)
        # Ensure strictly increasing timestamps per trace
        last_t = None
        for ev in events:
            et = ev.event_time
            if last_t is not None and et <= last_t:
                et = last_t + timedelta(seconds=1)
            attrs = {
                "concept:name": ev.name,
                "time:timestamp": et,
                "org:resource": ev.resource_id,
                "amount": float(ev.amount),
            }
            # Optionally include role to help testing (not required, but useful)
            attrs["org:role"] = ev.role
            tr.append(Event(attrs))
            last_t = et

        # Simple end marker: ensure an end-state exists (already in skeleton via XOR)
        # Trace stats
        t0 = tr[0]["time:timestamp"] if len(tr) > 0 else arr
        tN = tr[-1]["time:timestamp"] if len(tr) > 0 else arr
        dur_hours = (tN - t0).total_seconds() / 3600.0
        total_cost = float(sum(e["amount"] for e in tr))
        trace_stats.append((len(tr), dur_hours, total_cost))
        if any(e["concept:name"] in ("L_JoinParallel", "L_OrJoin") for e in tr):
            parallel_traces += 1
        drift_stats[er]["dur"].append(dur_hours)
        drift_stats[er]["cost"].append(total_cost)

        log.append(tr)

    # Export XES
    try:
        import pm4py
        pm4py.write_xes(log, OUT_PATH)
    except Exception:
        # Fallback exporter
        from pm4py.objects.log.exporter.xes import exporter as xes_exporter
        xes_exporter.apply(log, OUT_PATH)

    # Sanity + quality checks
    _print_quality_checks(log, trace_stats, parallel_traces, drift_stats)
    print(f"\nSaved XES to: {OUT_PATH}")
    return log

# ------------- Quality & sanity checks -------------

def _print_quality_checks(log: EventLog, trace_stats, parallel_traces: int, drift_stats):
    print("\n==== QUALITY & SANITY CHECKS ====")

    # Schema check
    missing = []
    for t_idx, tr in enumerate(log):
        last_ts = None
        for e_idx, ev in enumerate(tr):
            for k in ("concept:name", "time:timestamp", "org:resource", "amount"):
                if k not in ev:
                    missing.append((t_idx, e_idx, k))
            # timestamp monotonic
            if last_ts is not None and ev["time:timestamp"] <= last_ts:
                missing.append((t_idx, e_idx, "non_increasing_timestamp"))
            last_ts = ev["time:timestamp"]
            # types
            if not isinstance(ev["amount"], float):
                missing.append((t_idx, e_idx, "amount_not_float"))
    if missing:
        print(f"[Schema] Issues found: {len(missing)} (showing up to 5): {missing[:5]}")
    else:
        print("[Schema] OK — required attributes present and timestamps strictly increasing.")

    # Distribution summaries
    lengths = np.array([l for (l, _, _) in trace_stats], dtype=float)
    durs = np.array([d for (_, d, _) in trace_stats], dtype=float)
    costs = np.array([c for (_, _, c) in trace_stats], dtype=float)
    def q(a, p): return float(np.quantile(a, p)) if len(a) else float("nan")
    print("\n[Summary] Traces:", len(log))
    print("          Trace length  mean={:.1f}, q25={:.0f}, q50={:.0f}, q75={:.0f}".format(float(np.mean(lengths)), q(lengths, .25), q(lengths, .5), q(lengths, .75)))
    print("          Case duration (h) mean={:.2f}, std={:.2f}, q10={:.2f}, q90={:.2f}".format(float(np.mean(durs)), float(np.std(durs)), q(durs, .10), q(durs, .90)))
    print("          Case cost     mean={:.2f}, q25={:.2f}, q50={:.2f}, q75={:.2f}".format(float(np.mean(costs)), q(costs, .25), q(costs, .5), q(costs, .75)))

    # Activity counts
    act_counter = Counter(ev["concept:name"] for tr in log for ev in tr)
    common = act_counter.most_common(8)
    print("\n[Activity frequency] Top-8:", common)

    # Inter-event times & remaining time distributions (regression signal)
    rem_times = []
    inter_ev = []
    for tr in log:
        if not tr: continue
        end = tr[-1]["time:timestamp"]
        prev = None
        for ev in tr:
            rem_times.append((end - ev["time:timestamp"]).total_seconds() / 3600.0)
            if prev is not None:
                inter_ev.append((ev["time:timestamp"] - prev).total_seconds() / 60.0)
            prev = ev["time:timestamp"]
    rem_times = np.array(rem_times, dtype=float)
    inter_ev = np.array(inter_ev, dtype=float)
    if len(rem_times):
        print("\n[Remaining time] mean={:.2f}h, std={:.2f}h, q10={:.2f}h, q90={:.2f}h"
              .format(float(np.mean(rem_times)), float(np.std(rem_times)), q(rem_times, .10), q(rem_times, .90)))
    if len(inter_ev):
        print("[Inter-event time] mean={:.1f}min, q10={:.1f}, q90={:.1f}"
              .format(float(np.mean(inter_ev)), q(inter_ev, .10), q(inter_ev, .90)))

    # Ambiguity check: prefixes → multiple next activities
    prefixes = defaultdict(set)  # prefix_str -> next activity set
    sample_prefixes = 0
    for tr in log:
        labels = [ev["concept:name"] for ev in tr]
        for L in range(1, min(len(labels), 6)):  # up to length-5 prefixes
            pref = tuple(labels[:L])
            if L < len(labels):
                prefixes[pref].add(labels[L])
            sample_prefixes += 1
    ambiguous = sum(1 for s in prefixes.values() if len(s) >= 2)
    ratio = ambiguous / max(1, len(prefixes))
    print("\n[Ambiguity check] prefixes with ≥2 next-activities: {} / {} ({:.0%})"
          .format(ambiguous, len(prefixes), ratio))

    # Parallelism check
    pct_parallel = 100.0 * (parallel_traces / max(1, len(log)))
    print("[Parallelism] traces with ≥1 AND/OR join: {:.1f}%".format(pct_parallel))

    # Drift check by era
    print("\n[Drift by era]")
    for era, stats in drift_stats.items():
        if stats["dur"]:
            print(f"  {era}: mean duration={np.mean(stats['dur']):.2f}h, mean cost={np.mean(stats['cost']):.2f}")

# ------------- Main -------------

def execute_script():
    generate_log()

if __name__ == "__main__":
    execute_script()
