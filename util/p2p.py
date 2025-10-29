# p2p_generate.py
# Synthesizes a Procure-to-Pay (P2P) XES log with rich control-flow, timing, resource, and cost patterns.
# Output: p2p.xes.gz (capped at 250 distinct traces)
#
# Requirements: pm4py, numpy (pandas optional for summaries)
#   pip install pm4py numpy pandas

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

from pm4py.objects.log.obj import EventLog, Trace, Event
try:
    # pm4py >= 2.7
    from pm4py.objects.log.exporter.xes import exporter as xes_exporter
    def write_xes(log, path):
        xes_exporter.apply(log, path)
except Exception:
    # pm4py convenience wrapper (older versions)
    import pm4py
    def write_xes(log, path):
        pm4py.write_xes(log, path)

# -----------------------------
# Global configuration / seeds
# -----------------------------
RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

TARGET_DISTINCT_TRACES = 250
MAX_CASES_TO_TRY = 1200    # generate-until-unique safeguard
OUT_PATH = "p2p.xes.gz"

# Time window / eras (concept drift)
START_DATE = datetime(2025, 7, 1, 8, 0, 0)
END_DATE   = datetime(2025, 10, 1, 18, 0, 0)

# Three eras with different parameters/tariffs
ERAS = [
    ("early", datetime(2025, 7, 1), datetime(2025, 8, 1)),
    ("mid",   datetime(2025, 8, 1), datetime(2025, 9, 1)),
    ("late",  datetime(2025, 9, 1), datetime(2025, 10, 1)),
]

# -----------------------------
# Calendars
# -----------------------------
class BusinessCalendar:
    """
    Business calendar: 09:00-17:00 Mon-Fri, lunch 12:00-13:00; weekends closed.
    Supports 'add_work_time' and 'next_open' operations.
    """
    def __init__(self):
        self.day_start = time(9, 0, 0)
        self.day_end = time(17, 0, 0)
        self.lunch_start = time(12, 0, 0)
        self.lunch_end = time(13, 0, 0)

    @staticmethod
    def is_weekend(dt: datetime) -> bool:
        return dt.weekday() >= 5  # 5=Sat, 6=Sun

    def within_lunch(self, dt: datetime) -> bool:
        t = dt.time()
        return self.lunch_start <= t < self.lunch_end

    def within_hours(self, dt: datetime) -> bool:
        if self.is_weekend(dt):
            return False
        t = dt.time()
        return (self.day_start <= t < self.lunch_start) or (self.lunch_end <= t < self.day_end)

    def next_open(self, dt: datetime) -> datetime:
        # If within lunch, jump to lunch end
        if not self.is_weekend(dt):
            if dt.time() < self.day_start:
                return dt.replace(hour=self.day_start.hour, minute=0, second=0, microsecond=0)
            if self.lunch_start <= dt.time() < self.lunch_end:
                return dt.replace(hour=self.lunch_end.hour, minute=0, second=0, microsecond=0)
            if self.day_start <= dt.time() < self.lunch_start or self.lunch_end <= dt.time() < self.day_end:
                return dt
            # after day end → next business day start
        # move to next weekday 09:00
        nd = dt
        while self.is_weekend(nd):
            nd = (nd + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        if nd.time() >= self.day_end:
            nd = (nd + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        # ensure not weekend
        while self.is_weekend(nd):
            nd = (nd + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        return nd.replace(hour=self.day_start.hour, minute=0, second=0, microsecond=0)

    def add_work_time(self, start: datetime, hours: float, allow_overtime: bool=False) -> Tuple[datetime, float]:
        """
        Adds 'hours' of processing, respecting business hours unless allow_overtime.
        Returns (end_time, overtime_hours_used)
        """
        remaining = hours
        cur = start
        overtime_used = 0.0
        while remaining > 1e-9:
            if allow_overtime:
                # Can process continuously; count non-business hours as overtime
                # step in small chunks (15 min)
                step = min(remaining, 0.25)  # 15 min chunk
                next_t = cur + timedelta(hours=step)
                if not self.within_hours(cur):
                    overtime_used += step
                remaining -= step
                cur = next_t
            else:
                cur = self.next_open(cur)
                # compute until next boundary (lunch or day_end)
                t = cur.time()
                boundary = datetime(cur.year, cur.month, cur.day, 12, 0) if t < self.lunch_start else datetime(cur.year, cur.month, cur.day, 17, 0)
                capacity = (boundary - cur).total_seconds() / 3600.0
                take = min(remaining, capacity)
                cur = cur + timedelta(hours=take)
                remaining -= take
                # if we hit a boundary, loop moves to next_open automatically
        return cur, overtime_used

BIZ_CAL = BusinessCalendar()

# -----------------------------
# Activities, families, roles
# -----------------------------
ACTIVITIES = [
    "Start",
    "Create Requisition", "Review Requisition", "Fix Requisition", "Approve Requisition",
    "Select Vendor", "Evaluate Vendor", "Negotiate Terms",
    "OR.Risk Assessment", "OR.Budget Review", "OR.Compliance",  # OR-parallel candidates
    "AND.Join.VendorPrep",
    "Create PO", "Approve PO", "Board Approve", "Send PO",
    "Vendor Confirm", "Fulfill Order", "Ship Goods",
    "Receive Goods", "Inspect Goods",
    "QA Sample", "Join.QA.kofn",
    "3-Way Match",
    "Submit Invoice", "Review Invoice", "Fix Invoice", "Approve Invoice",
    "Payment Batch", "Pay Invoice",
    "Audit",
    "Close Order",
    "Escalate", "Expedite", "Handover", "Wait",
    "Vendor (Outsource)", "Simplify", "Cancel", "RollBack", "Reassign",
    # explicit preemption markers (will use suffixes .start/.resume/.complete dynamically)
]

# Activity family for setup/context switching penalties (broad buckets)
FAMILY = {
    "Create Requisition":"REQ", "Review Requisition":"REQ", "Fix Requisition":"REQ", "Approve Requisition":"REQ",
    "Select Vendor":"VEN", "Evaluate Vendor":"VEN", "Negotiate Terms":"VEN",
    "OR.Risk Assessment":"VEN", "OR.Budget Review":"VEN", "OR.Compliance":"VEN",
    "AND.Join.VendorPrep":"VEN",
    "Create PO":"PO", "Approve PO":"PO", "Board Approve":"PO", "Send PO":"PO",
    "Vendor Confirm":"VEN", "Fulfill Order":"VEN", "Ship Goods":"VEN",
    "Receive Goods":"RCV", "Inspect Goods":"RCV", "QA Sample":"RCV", "Join.QA.kofn":"RCV",
    "3-Way Match":"ACC",
    "Submit Invoice":"INV", "Review Invoice":"INV", "Fix Invoice":"INV", "Approve Invoice":"INV",
    "Payment Batch":"FIN", "Pay Invoice":"FIN", "Audit":"AUD",
    "Close Order":"CLS", "Escalate":"MGT", "Expedite":"MGT", "Handover":"SYS", "Wait":"SYS",
    "Vendor (Outsource)":"VEN", "Simplify":"REQ", "Cancel":"MGT", "RollBack":"FIN", "Reassign":"MGT"
}

# Preferred role per activity (strings; optional attribute, always set org:resource)
ROLE = {
    "Create Requisition":"role:analyst",
    "Review Requisition":"role:analyst",
    "Fix Requisition":"role:analyst",
    "Approve Requisition":"role:manager",
    "Select Vendor":"role:analyst",
    "Evaluate Vendor":"role:analyst",
    "Negotiate Terms":"role:senior",
    "OR.Risk Assessment":"role:senior",
    "OR.Budget Review":"role:analyst",
    "OR.Compliance":"role:auditor",
    "AND.Join.VendorPrep":"role:system",
    "Create PO":"role:clerk",
    "Approve PO":"role:manager",
    "Board Approve":"role:manager",  # multi-resource (manager + senior)
    "Send PO":"role:clerk",
    "Vendor Confirm":"role:vendor",
    "Fulfill Order":"role:vendor",
    "Ship Goods":"role:vendor",
    "Receive Goods":"role:warehouse",
    "Inspect Goods":"role:warehouse",
    "QA Sample":"role:qa",
    "Join.QA.kofn":"role:system",
    "3-Way Match":"role:analyst",
    "Submit Invoice":"role:vendor",
    "Review Invoice":"role:analyst",
    "Fix Invoice":"role:analyst",
    "Approve Invoice":"role:manager",
    "Payment Batch":"role:finance",
    "Pay Invoice":"role:finance",
    "Audit":"role:auditor",
    "Close Order":"role:clerk",
    "Escalate":"role:senior",
    "Expedite":"role:senior",
    "Handover":"role:clerk",
    "Wait":"role:system",
    "Vendor (Outsource)":"role:vendor",
    "Simplify":"role:analyst",
    "Cancel":"role:manager",
    "RollBack":"role:finance",
    "Reassign":"role:clerk"
}

# Fixed-cost baselines per activity (can be 0)
FIXED_COST = {
    "Create Requisition": 8.0,
    "Review Requisition": 6.0,
    "Fix Requisition": 10.0,
    "Approve Requisition": 12.0,
    "Select Vendor": 6.0,
    "Evaluate Vendor": 8.0,
    "Negotiate Terms": 18.0,
    "OR.Risk Assessment": 15.0,
    "OR.Budget Review": 5.0,
    "OR.Compliance": 9.0,
    "AND.Join.VendorPrep": 0.0,
    "Create PO": 7.0,
    "Approve PO": 12.0,
    "Board Approve": 25.0,
    "Send PO": 3.0,
    "Vendor Confirm": 0.0,            # material cost added separately
    "Fulfill Order": 0.0,             # vendor time, capture via rate if assigned
    "Ship Goods": 0.0,                # material shipping cost separate
    "Receive Goods": 4.0,
    "Inspect Goods": 4.0,
    "QA Sample": 3.0,
    "Join.QA.kofn": 0.0,
    "3-Way Match": 6.0,
    "Submit Invoice": 2.0,
    "Review Invoice": 6.0,
    "Fix Invoice": 6.0,
    "Approve Invoice": 12.0,
    "Payment Batch": 1.0,
    "Pay Invoice": 8.0,
    "Audit": 14.0,
    "Close Order": 3.0,
    "Escalate": 50.0,                 # penalty/overhead
    "Expedite": 40.0,                 # expedite fee
    "Handover": 2.0,                  # fee on resource switches
    "Wait": 0.0,
    "Vendor (Outsource)": 0.0,
    "Simplify": 5.0,
    "Cancel": 0.0,
    "RollBack": -30.0,                # credit/refund
    "Reassign": 3.0
}

# Baseline mean durations (in hours) – heavy-tailed sampling will vary
MEAN_HOURS = {
    "Create Requisition": 1.0,
    "Review Requisition": 1.2,
    "Fix Requisition": 1.5,
    "Approve Requisition": 0.8,
    "Select Vendor": 1.0,
    "Evaluate Vendor": 2.0,
    "Negotiate Terms": 3.0,
    "OR.Risk Assessment": 1.5,
    "OR.Budget Review": 1.0,
    "OR.Compliance": 1.2,
    "AND.Join.VendorPrep": 0.2,
    "Create PO": 0.8,
    "Approve PO": 0.5,
    "Board Approve": 1.0,
    "Send PO": 0.2,
    "Vendor Confirm": 4.0,
    "Fulfill Order": 12.0,
    "Ship Goods": 4.0,
    "Receive Goods": 1.0,
    "Inspect Goods": 1.0,
    "QA Sample": 0.5,
    "Join.QA.kofn": 0.1,
    "3-Way Match": 1.0,
    "Submit Invoice": 0.5,
    "Review Invoice": 1.0,
    "Fix Invoice": 0.7,
    "Approve Invoice": 0.3,
    "Payment Batch": 0.1,
    "Pay Invoice": 0.2,
    "Audit": 2.0,
    "Close Order": 0.3,
    "Escalate": 0.2,
    "Expedite": 0.1,
    "Handover": 0.05,
    "Wait": 0.0,
    "Vendor (Outsource)": 6.0,
    "Simplify": 0.5,
    "Cancel": 0.2,
    "RollBack": 0.1,
    "Reassign": 0.2
}

# -----------------------------
# Resources & roles
# -----------------------------
@dataclass
class Resource:
    rid: str
    role: str
    base_rate: float                     # $/hour (regular)
    speed: float                         # lower is faster (multiplier on mean)
    shift_start_h: int
    shift_end_h: int
    next_free: datetime = field(default_factory=lambda: START_DATE)
    skill_counts: Dict[str, int] = field(default_factory=dict)   # family -> count executed
    last_used: Dict[str, datetime] = field(default_factory=dict) # family -> last time executed
    days_off: set = field(default_factory=set)                   # weekday indices off (e.g., {2} for Wed)

    def on_shift(self, t: datetime) -> bool:
        if t.weekday() in self.days_off:
            return False
        th = t.time().hour + t.time().minute/60.0
        return self.shift_start_h <= th < self.shift_end_h and not (12 <= th < 13)

    def next_shift_start(self, t: datetime) -> datetime:
        cur = t
        # jump to today's shift start if before, else to tomorrow
        candidate = cur.replace(hour=self.shift_start_h, minute=0, second=0, microsecond=0)
        if cur.time().hour < self.shift_start_h:
            cur = candidate
        else:
            # move to next day start if outside
            if not self.on_shift(cur):
                cur = (cur + timedelta(days=1)).replace(hour=self.shift_start_h, minute=0, second=0, microsecond=0)
        # ensure not lunch, not day off, not weekend per business hours
        while cur.weekday() in self.days_off or BIZ_CAL.is_weekend(cur):
            cur = (cur + timedelta(days=1)).replace(hour=self.shift_start_h, minute=0, second=0, microsecond=0)
        return cur

# Build 30 resources across roles; heterogeneity in rate/speed and shifts
def build_resources() -> List[Resource]:
    roles_distribution = (
        ["role:analyst"]*8 + ["role:senior"]*3 + ["role:manager"]*5 +
        ["role:finance"]*4 + ["role:warehouse"]*4 + ["role:qa"]*2 +
        ["role:auditor"]*2 + ["role:vendor"]*2
    )
    random.shuffle(roles_distribution)
    resources = []
    for i in range(30):
        role = roles_distribution[i % len(roles_distribution)]
        # base rates by role
        base = {
            "role:analyst": (35, 55),
            "role:senior": (60, 85),
            "role:manager": (55, 80),
            "role:finance": (40, 60),
            "role:warehouse": (20, 35),
            "role:qa": (30, 45),
            "role:auditor": (50, 75),
            "role:vendor": (25, 45)
        }.get(role, (30, 50))
        base_rate = random.uniform(*base)
        speed = random.uniform(0.8, 1.3)  # <1 faster
        # shifts
        shift_start = random.choice([8, 9, 10])
        shift_end = shift_start + 8
        # random weekday off
        day_off = set()
        if random.random() < 0.15:
            day_off.add(random.randint(0, 4))
        r = Resource(
            rid=f"R{i+1}",
            role=role,
            base_rate=base_rate,
            speed=speed,
            shift_start_h=shift_start,
            shift_end_h=shift_end,
            days_off=day_off
        )
        resources.append(r)
    return resources

RESOURCES: List[Resource] = build_resources()

# Helper: candidates by role
def candidates_for_role(role: str) -> List[Resource]:
    return [r for r in RESOURCES if r.role == role]

# -----------------------------
# NHPP arrivals (bursty waves)
# -----------------------------
def hour_intensity(dt: datetime) -> float:
    """Non-homogeneous intensity λ(t) by hour-of-day and day-of-week (waves)."""
    # base diurnal (peaks mid-morning, mid-afternoon)
    h = dt.hour + dt.minute/60.0
    diurnal = 0.7 + 0.6 * math.exp(-((h-10.5)/2.3)**2) + 0.4 * math.exp(-((h-15.0)/2.0)**2)
    # weekday effect: Mon/Tue heavier, Fri lower
    weekday_factor = [1.3, 1.15, 1.0, 1.0, 0.85, 0.3, 0.2][dt.weekday()]
    # month-end push (concept drift also)
    month_end_push = 1.0 + (0.4 if dt.day >= 24 else 0.0)
    return diurnal * weekday_factor * month_end_push

def sample_nhpp_arrivals(start: datetime, end: datetime, target_n: int) -> List[datetime]:
    """Thinning on 15-min grid to collect ~target_n arrivals with waves."""
    grid = []
    cur = start
    while cur < end:
        grid.append(cur)
        cur += timedelta(minutes=15)
    lam_max = max(hour_intensity(t) for t in grid)
    arrivals = []
    for t in grid:
        lam = hour_intensity(t) / lam_max  # normalized ≤1
        # scale to aim for target_n total
        prob = lam * (target_n / len(grid)) * 1.4
        if random.random() < min(prob, 1.0):
            # jitter within the 15-min slot
            jitter = random.randint(0, 14)
            arrivals.append(t + timedelta(minutes=jitter))
    # If too few/many, adjust by sampling
    if len(arrivals) < target_n:
        while len(arrivals) < target_n:
            t = random.choice(grid)
            arrivals.append(t + timedelta(minutes=random.randint(0, 14)))
    arrivals.sort()
    return arrivals[:target_n]

# -----------------------------
# Utility: era, heavy tails, learning/forgetting
# -----------------------------
def era_of(t: datetime) -> str:
    for name, s, e in ERAS:
        if s <= t < e:
            return name
    return "late"

def lognormal_hours(mean_h: float, sigma: float=0.8) -> float:
    """Return a heavy-tailed service time in hours around mean_h using lognormal."""
    # derive mu from mean and sigma: mean = exp(mu + sigma^2/2)
    mu = math.log(max(mean_h, 1e-3)) - 0.5 * sigma * sigma
    return float(np.random.lognormal(mean=mu, sigma=sigma))

def pareto_hours(xm: float=0.2, alpha: float=3.0) -> float:
    return float((np.random.pareto(alpha) + 1) * xm)

def learning_multiplier(resource: Resource, fam: str) -> float:
    """Learning curve: duration multiplier decreases with repetitions."""
    n = resource.skill_counts.get(fam, 0)
    return max(0.75, 1.0 - 0.06 * min(n, 5))  # up to ~25% faster

def forgetting_multiplier(resource: Resource, fam: str, now: datetime) -> float:
    """Forgetting: if long inactivity in family, duration increases."""
    last = resource.last_used.get(fam)
    if not last:
        return 1.0
    days = (now - last).days
    if days >= 10:
        return 1.20
    if days >= 5:
        return 1.10
    return 1.0

def setup_changeover_penalty(prev_fam: Optional[str], cur_fam: str) -> float:
    if prev_fam is None or prev_fam == cur_fam:
        return 0.0
    # sequence-dependent (context switch time in hours)
    return random.uniform(0.1, 0.6)

# -----------------------------
# Cost model
# -----------------------------
def tariff_multiplier_by_era(era: str) -> float:
    return {"early": 1.00, "mid": 1.05, "late": 0.97}[era]

def overtime_multiplier(t: datetime) -> float:
    # nights/weekends are more expensive
    if BIZ_CAL.is_weekend(t) or not BIZ_CAL.within_hours(t):
        return 1.5
    return 1.0

def holding_cost_per_hour(era: str) -> float:
    return {"early": 2.0, "mid": 2.5, "late": 2.2}[era]

# -----------------------------
# Assignment / queues
# -----------------------------
def assignment_policy_mix() -> str:
    # Weighted mixture across policies
    r = random.random()
    if r < 0.5:
        return "skill"
    elif r < 0.75:
        return "least_loaded"
    return "random"

def pick_resource(role: str,
                  ready_time: datetime,
                  expected_hours: float,
                  binding: Optional[Resource]=None,
                  sep_from: Optional[Resource]=None) -> Tuple[List[Resource], datetime, str]:
    """
    Choose 1 or 2 resources (for meetings use twice).
    Returns ([resources], feasible_start_time, policy_used)
    """
    policy = assignment_policy_mix()
    pool = candidates_for_role(role)
    if sep_from:
        pool = [r for r in pool if r.rid != sep_from.rid]
    if binding:
        pool = [binding] + [r for r in pool if r.rid != binding.rid]  # prefer binding first

    if not pool:
        # Fallback to any resource, then to Unknown (will be overridden later)
        pool = RESOURCES

    def availability_time(r: Resource) -> datetime:
        return max(ready_time, r.next_free, r.next_shift_start(ready_time))

    if policy == "least_loaded":
        pool = sorted(pool, key=lambda r: availability_time(r))
        chosen = pool[0]
    elif policy == "skill":
        # minimize expected finish = avail + expected_hours * (speed*learning*forgetting)
        def score(r: Resource):
            fam = None  # fam-only available later; approximate via speed + learning on recent families
            lm = learning_multiplier(r, fam or "GEN")
            fm = forgetting_multiplier(r, fam or "GEN", ready_time)
            dur = expected_hours * r.speed * lm * fm
            return (availability_time(r) + timedelta(hours=dur))
        chosen = min(pool, key=score)
    else:
        chosen = random.choice(pool)

    start = max(ready_time, chosen.next_free, chosen.next_shift_start(ready_time))
    return [chosen], start, policy

def pick_meeting_resources(roles: List[str], ready_time: datetime, expected_hours: float) -> Tuple[List[Resource], datetime]:
    # find earliest overlap of two roles (manager + senior)
    cands = [candidates_for_role(roles[0]), candidates_for_role(roles[1])]
    pairs = [(a,b) for a in cands[0] for b in cands[1] if a.rid != b.rid]
    if not pairs:
        # fallback: choose any two distinct
        pool = [r for r in RESOURCES]
        random.shuffle(pool)
        pairs = [(pool[i], pool[i+1]) for i in range(0, min(len(pool)-1, 10), 2)]
    best_pair = None
    best_start = None
    for a,b in pairs:
        ta = max(ready_time, a.next_free, a.next_shift_start(ready_time))
        tb = max(ready_time, b.next_free, b.next_shift_start(ready_time))
        t = max(ta, tb)
        if best_start is None or t < best_start:
            best_start = t
            best_pair = (a,b)
    return [best_pair[0], best_pair[1]], best_start

# -----------------------------
# Per-activity capacity tracker (virtual queue)
# -----------------------------
class ActivityCapacity:
    """Tracks concurrent tasks per activity to emulate queueing (FIFO/priority/SPT effects)."""
    def __init__(self):
        self.capacity: Dict[str, int] = {}
        self.busy_until: Dict[str, List[datetime]] = {}

    def get_capacity(self, activity: str) -> int:
        if activity not in self.capacity:
            # approximate capacity by number of resources of the role (min 1)
            role = ROLE.get(activity, "role:analyst")
            cap = max(1, len(candidates_for_role(role))//2)
            self.capacity[activity] = cap
        return self.capacity[activity]

    def schedule_start(self, activity: str, ready_time: datetime,
                       predicted_hours: float, queue_discipline: str, priority: int) -> datetime:
        """
        Return start time respecting capacity. For Priority/SPT we reduce the FIFO wait (queue-jump approximation).
        """
        cap = self.get_capacity(activity)
        buf = self.busy_until.setdefault(activity, [])
        # drop finished
        buf = [t for t in buf if t > ready_time]
        self.busy_until[activity] = buf
        if len(buf) < cap:
            return ready_time
        # wait until earliest free
        earliest_free = min(buf)
        # discipline effect: priority gets earlier slot; SPT gets mild earlier slot
        if queue_discipline == "PRIO" and priority > 0:
            return ready_time + (earliest_free - ready_time) * 0.4
        if queue_discipline == "SPT" and predicted_hours < 0.8 * MEAN_HOURS.get(activity, 1.0):
            return ready_time + (earliest_free - ready_time) * 0.7
        return earliest_free

    def occupy(self, activity: str, end_time: datetime):
        self.busy_until.setdefault(activity, []).append(end_time)

ACT_CAP = ActivityCapacity()

# -----------------------------
# Event creation helpers
# -----------------------------
def add_event(trace: Trace, name: str, when: datetime, resource: Optional[str], amount: float, role: Optional[str]=None):
    # small chance of Unknown resource to test defaults
    rid = resource if (resource and random.random() > 0.03) else "Unknown"
    data = {
        "concept:name": name,
        "time:timestamp": when,
        "org:resource": rid,
        "amount": float(amount)
    }
    if role and random.random() < 0.15:
        data["role"] = role  # optional role attr
    trace.append(Event(data))

def ensure_strictly_increasing(trace: Trace):
    # If same timestamp appears, nudge subsequent by microseconds
    last = None
    for e in trace:
        t = e["time:timestamp"]
        if last and t <= last:
            t = last + timedelta(microseconds=1)
            e["time:timestamp"] = t
        last = e["time:timestamp"]

# -----------------------------
# Activity scheduling primitive
# -----------------------------
def schedule_activity(trace: Trace,
                      case_ctx: Dict[str, Any],
                      label: str,
                      ready_time: datetime,
                      bind_to: Optional[Resource]=None,
                      sep_from: Optional[Resource]=None,
                      queue_disc: Optional[str]=None,
                      priority: int=0,
                      allow_preempt: bool=True,
                      meeting_roles: Optional[List[str]]=None,
                      allow_overtime: bool=False) -> Tuple[datetime, Optional[Resource]]:
    """
    Schedule one logical activity into (possibly multiple) events (start/resume/complete).
    Returns (end_time, resource_used)
    """
    era = era_of(ready_time)
    fam = FAMILY.get(label, "GEN")
    role = ROLE.get(label, "role:analyst")
    mean_h = MEAN_HOURS.get(label, 1.0)
    fixed = FIXED_COST.get(label, 0.0) * tariff_multiplier_by_era(era)

    # heavy-tailed service time (mix lognormal + Pareto tail)
    base = 0.7 * lognormal_hours(mean_h, sigma=0.7) + 0.3 * pareto_hours(xm=0.1*mean_h, alpha=3.0)
    # context switching penalty
    base += setup_changeover_penalty(case_ctx.get("last_family"), fam)

    # pick resource(s)
    if meeting_roles:
        res_list, tentative_start = pick_meeting_resources(meeting_roles, ready_time, base)
        resource = res_list[0]  # main
        extra_res = res_list[1]
    else:
        res_list, tentative_start, _ = pick_resource(role, ready_time, base, binding=bind_to, sep_from=sep_from)
        resource = res_list[0]
        extra_res = None

    # learning/forgetting adjust
    lm = learning_multiplier(resource, fam)
    fm = forgetting_multiplier(resource, fam, tentative_start)
    expected = base * resource.speed * lm * fm

    # Queue discipline
    if not queue_disc:
        queue_disc = random.choice(["FIFO", "PRIO", "SPT"])
    start_q = ACT_CAP.schedule_start(label, tentative_start, expected, queue_disc, priority)

    # respect business calendar & resource shift (unless overtime allowed)
    start = max(start_q, resource.next_free)
    if not allow_overtime:
        start = BIZ_CAL.next_open(start)
        start = max(start, resource.next_shift_start(start))
    # compute wait and holding costs
    wait_h = max(0.0, (start - ready_time).total_seconds() / 3600.0)
    if wait_h > 1e-6:
        add_event(trace, "Wait", start, "Unknown", holding_cost_per_hour(era) * wait_h, ROLE["Wait"])

    # preemption decision depends on virtual queue pressure
    cap = ACT_CAP.get_capacity(label)
    q_pressure = max(0, len(ACT_CAP.busy_until.get(label, [])) - cap + 1)
    p_preempt = 0.1 + 0.08 * q_pressure
    will_preempt = allow_preempt and (random.random() < min(0.5, p_preempt))

    # compute end time with/without overtime; overtime allowed can accelerate completion
    allow_ot_now = allow_overtime or (will_preempt and random.random() < 0.5)
    end, ot_used = BIZ_CAL.add_work_time(start, expected, allow_overtime=allow_ot_now)

    # preemption: split into start/resume/complete
    if will_preempt and (end - start).total_seconds() > 3600.0:
        cut = random.uniform(0.2, 0.6) * expected
        cut_end, _ = BIZ_CAL.add_work_time(start, cut, allow_overtime=False)
        add_event(trace, f"{label}.start", start, resource.rid, FIXED_COST.get(label,0.0)*0.2, role)
        # pause (e.g., lunch/day end/other interruption)
        pause = timedelta(hours=random.uniform(1.0, 6.0))
        resume_time = max(cut_end + pause, resource.next_shift_start(cut_end))
        add_event(trace, f"{label}.resume", resume_time, resource.rid, 0.0, role)
        # finish remaining
        remaining_h = max(0.0, expected - cut)
        end, ot_used = BIZ_CAL.add_work_time(resume_time, remaining_h, allow_overtime=allow_ot_now)
        add_event(trace, f"{label}.complete", end, resource.rid,
                  fixed + resource.base_rate * overtime_multiplier(end) * (expected + (ot_used*0.3)),
                  role)
    else:
        add_event(trace, label, end, resource.rid,
                  fixed + resource.base_rate * overtime_multiplier(end) * (expected + (ot_used*0.3)),
                  role)

    # handover event if resource changed from previous event in trace
    prev_res = case_ctx.get("last_resource")
    if prev_res and prev_res != resource.rid and label not in ("Wait","Handover"):
        # insert a small handover right before end (monotonic fix later)
        handover_time = end - timedelta(minutes=random.randint(5, 25))
        if handover_time < start:
            handover_time = start
        add_event(trace, "Handover", handover_time, random.choice([prev_res, resource.rid]),
                  FIXED_COST["Handover"] + 1.5 * random.random(), ROLE["Handover"])

    # Update trackers
    resource.next_free = end + timedelta(minutes=random.randint(1, 15))
    resource.skill_counts[FAMILY.get(label, "GEN")] = resource.skill_counts.get(FAMILY.get(label,"GEN"), 0) + 1
    resource.last_used[FAMILY.get(label, "GEN")] = end
    if extra_res:
        extra_res.next_free = end + timedelta(minutes=random.randint(1, 15))
        extra_res.skill_counts[FAMILY.get(label, "GEN")] = extra_res.skill_counts.get(FAMILY.get(label,"GEN"), 0) + 1
        extra_res.last_used[FAMILY.get(label, "GEN")] = end

    case_ctx["last_family"] = fam
    case_ctx["last_resource"] = resource.rid
    ACT_CAP.occupy(label, end)

    # material cost triggers
    if label in ("Receive Goods","Ship Goods","Vendor Confirm"):
        mat = random.uniform(50, 300) * {"early":1.0,"mid":1.05,"late":0.95}[era]
        add_event(trace, f"{label}.Material", end, resource.rid, float(mat), role)

    return end, resource

# -----------------------------
# Control-flow helpers
# -----------------------------
def maybe_escalate(trace: Trace, case_ctx: Dict[str, Any], ready_time: datetime) -> Optional[datetime]:
    """Deadline/aging: if behind schedule, insert Escalate (penalty + senior routing)."""
    inv_due = case_ctx.get("invoice_due")
    if not inv_due:
        return None
    # if projected remaining slack small, escalate
    slack_h = (inv_due - ready_time).total_seconds() / 3600.0
    if slack_h < random.uniform(4, 24):  # dynamic SLA thresholds
        end, _ = schedule_activity(trace, case_ctx, "Escalate", ready_time, allow_preempt=False)
        # escalation changes future behavior – bias to 'senior'
        case_ctx["escalated"] = True
        return end
    return None

def maybe_expedite(trace: Trace, case_ctx: Dict[str, Any], ready_time: datetime) -> Optional[datetime]:
    """Insert Expedited path to allow queue jump / overtime for next steps."""
    if random.random() < 0.25 or case_ctx.get("escalated"):
        end, _ = schedule_activity(trace, case_ctx, "Expedite", ready_time, allow_preempt=False, allow_overtime=True)
        case_ctx["expedite"] = True
        return end
    return None

def and_parallel(trace: Trace, case_ctx: Dict[str, Any], ready_time: datetime,
                 activities: List[Tuple[str, Dict]]) -> datetime:
    """Run activities in parallel; return join time (max end)."""
    ends = []
    for label, opts in activities:
        e, _ = schedule_activity(trace, case_ctx, label, ready_time, **opts)
        ends.append(e)
    join_ready = max(ends)
    # synchronization latency / handover before join
    delay = timedelta(minutes=random.randint(5, 60))
    join_time = join_ready + delay
    add_event(trace, "AND.Join.VendorPrep", join_time, "Unknown", FIXED_COST["AND.Join.VendorPrep"], ROLE["AND.Join.VendorPrep"])
    case_ctx["last_family"] = FAMILY["AND.Join.VendorPrep"]
    case_ctx["last_resource"] = "Unknown"
    return join_time

def or_parallel_subset(trace: Trace, case_ctx: Dict[str, Any], ready_time: datetime,
                       candidates: List[str]) -> datetime:
    """Choose 1..k of candidates, run in parallel; OR-join at max end."""
    k = random.randint(1, len(candidates))
    choose = random.sample(candidates, k=k)
    return and_parallel(
        trace, case_ctx, ready_time,
        [(label, {}) for label in choose]
    )

def review_fix_loop(trace: Trace, case_ctx: Dict[str, Any], ready_time: datetime,
                    review_label: str, fix_label: str) -> datetime:
    """Geometric rework with heavy-tail link (longer review -> more rework)."""
    e_review, res = schedule_activity(trace, case_ctx, review_label, ready_time)
    # chance of rework increases if review ran long (> mean)
    mean = MEAN_HOURS[review_label]
    long_factor = 0.2 + 0.5 * int((e_review - ready_time).total_seconds()/3600.0 > mean*1.3)
    iters = 0
    while random.random() < (0.25 + long_factor) and iters < 3:
        e_fix, _ = schedule_activity(trace, case_ctx, fix_label, e_review)
        e_review, _ = schedule_activity(trace, case_ctx, review_label, e_fix)
        iters += 1
    return e_review

def qa_k_of_n(trace: Trace, case_ctx: Dict[str, Any], ready_time: datetime, n: int=5, k: int=3) -> datetime:
    """Spawn n QA Sample, proceed after k complete (k-of-n)."""
    finishes = []
    for _ in range(n):
        e, _ = schedule_activity(trace, case_ctx, "QA Sample", ready_time)
        finishes.append(e)
    finishes.sort()
    join_k = finishes[min(k-1, len(finishes)-1)]
    add_event(trace, "Join.QA.kofn", join_k, "Unknown", 0.0, ROLE["Join.QA.kofn"])
    case_ctx["last_family"] = FAMILY["Join.QA.kofn"]
    case_ctx["last_resource"] = "Unknown"
    return join_k

def payment_batch_release(t: datetime, expedited: bool=False) -> datetime:
    """Batching/release: default 14:00 daily; expedite may bypass."""
    if expedited:
        return t
    rel = t.replace(hour=14, minute=0, second=0, microsecond=0)
    if t.time() > time(14, 0):
        rel = rel + timedelta(days=1)
    # if weekend, shift to Monday
    while BIZ_CAL.is_weekend(rel):
        rel = rel + timedelta(days=1)
    return rel

# -----------------------------
# Single case simulation (P2P)
# -----------------------------
def simulate_case(case_id: str, start_at: datetime, case_priority: int) -> Trace:
    """
    Construct one P2P case with composition of templates/patterns.
    Returns a pm4py Trace (with attributes['concept:name'] set).
    """
    trace = Trace()
    trace.attributes["concept:name"] = case_id
    case_ctx: Dict[str, Any] = {"last_family": None, "last_resource": None}

    # case properties / concept drift
    era = era_of(start_at)
    value = random.uniform(100, 5000)
    items = random.randint(1, 8)
    high_value = value > 2000
    rush = (case_priority > 0) or random.random() < 0.2
    case_ctx.update({"value": value, "items": items, "high_value": high_value, "rush": rush})

    add_event(trace, "Start", start_at, "Unknown", 0.0)

    # 1) Requisition
    t = start_at
    t, _ = schedule_activity(trace, case_ctx, "Create Requisition", t)
    t = review_fix_loop(trace, case_ctx, t, "Review Requisition", "Fix Requisition")
    t, _ = schedule_activity(trace, case_ctx, "Approve Requisition", t)

    # optional OR-parallel inserts (risk/budget/compliance)
    if random.random() < 0.6:
        t = or_parallel_subset(trace, case_ctx, t, ["OR.Risk Assessment", "OR.Budget Review", "OR.Compliance"])

    # Long-term dependency: high-value needs board approval later, vendor path biased to in-house vs outsource by budget
    # XOR split: choose vendor path vs outsource vs simplify, influenced by cost so far and rush
    options = ["Select Vendor", "Vendor (Outsource)", "Simplify"]
    probs = [0.55, 0.25, 0.20]
    if high_value:
        probs = [0.65, 0.25, 0.10]
    if rush:
        probs = [0.50, 0.35, 0.15]
    choice = random.choices(options, weights=probs, k=1)[0]
    if choice == "Vendor (Outsource)":
        t, _ = schedule_activity(trace, case_ctx, "Vendor (Outsource)", t, allow_preempt=False)
    elif choice == "Simplify":
        t, _ = schedule_activity(trace, case_ctx, "Simplify", t)

    # 2) Vendor selection / negotiation (AND-parallel)
    if choice != "Vendor (Outsource)":
        join_t = and_parallel(
            trace, case_ctx, t,
            [
                ("Evaluate Vendor", {}),
                ("Negotiate Terms", {})
            ]
        )
        t = join_t

    # 3) Purchase order
    t, _ = schedule_activity(trace, case_ctx, "Create PO", t)
    # Possible Board approval (multi-resource meeting) for high_value or late era
    if high_value or (era == "mid" and random.random() < 0.2):
        res_roles = ["role:manager", "role:senior"]
        # meeting requires overlap
        t, _ = schedule_activity(trace, case_ctx, "Board Approve", t,
                                 meeting_roles=res_roles, allow_preempt=False)
    t, _ = schedule_activity(trace, case_ctx, "Approve PO", t)
    t, _ = schedule_activity(trace, case_ctx, "Send PO", t)

    # 4) Vendor fulfilment (OR/AND choice)
    if choice == "Vendor (Outsource)":
        # slower supplier but cheaper later tariffs (era-based)
        t, _ = schedule_activity(trace, case_ctx, "Vendor Confirm", t, allow_overtime=False)
        # parallel fulfill + ship (AND)
        t = and_parallel(
            trace, case_ctx, t,
            [
                ("Fulfill Order", {}),
                ("Ship Goods", {})
            ]
        )
    else:
        # simpler: confirm then ship
        t, _ = schedule_activity(trace, case_ctx, "Vendor Confirm", t)
        t, _ = schedule_activity(trace, case_ctx, "Ship Goods", t)

    # Occasionally: cancellation & rollback path
    if random.random() < 0.02:
        t, _ = schedule_activity(trace, case_ctx, "Cancel", t, allow_preempt=False)
        t, _ = schedule_activity(trace, case_ctx, "RollBack", t, allow_preempt=False)
        ensure_strictly_increasing(trace)
        return trace  # early termination

    # 5) Receiving & QA (k-of-n)
    t, _ = schedule_activity(trace, case_ctx, "Receive Goods", t)
    # optional inspect followed by QA samples (k-of-n)
    if random.random() < 0.8:
        t, _ = schedule_activity(trace, case_ctx, "Inspect Goods", t)
        t = qa_k_of_n(trace, case_ctx, t, n=min(7, 2+items), k=min(3, 1+items//2))

    # 6) 3-way match
    t, _ = schedule_activity(trace, case_ctx, "3-Way Match", t)

    # 7) Invoicing + review loop (binding-of-duty and separation-of-duty)
    t, _ = schedule_activity(trace, case_ctx, "Submit Invoice", t)
    # case deadline: 14 days from invoice submission
    case_ctx["invoice_due"] = t + timedelta(days=14)
    # binding-of-duty: prefer same resource for review/approve when expedited
    bound_res: Optional[Resource] = None
    t = review_fix_loop(trace, case_ctx, t, "Review Invoice", "Fix Invoice")
    # Maybe escalate if near deadline
    escal_t = maybe_escalate(trace, case_ctx, t)
    if escal_t:
        t = escal_t
    # maybe expedite
    if rush or random.random() < 0.3:
        exp_t = maybe_expedite(trace, case_ctx, t)
        if exp_t:
            t = exp_t
            bound_res = next((r for r in RESOURCES if r.rid == case_ctx.get("last_resource")), None)
    # Approve Invoice (binding-of-duty optional)
    t, approver = schedule_activity(trace, case_ctx, "Approve Invoice", t, bind_to=bound_res)
    # Separation-of-duty with audit
    if random.random() < (0.15 + 0.2*high_value):
        t, _ = schedule_activity(trace, case_ctx, "Audit", t, sep_from=approver, allow_preempt=False)

    # Cooling-off: ensure 1 day between approval and payment (unless expedite)
    min_lag = timedelta(days=1)
    if t + min_lag > t:
        cool = t + min_lag
    else:
        cool = t

    # 8) Payment batching and Pay
    rel = payment_batch_release(cool, expedited=case_ctx.get("expedite", False))
    if rel > t:
        add_event(trace, "Payment Batch", rel, "Unknown", FIXED_COST["Payment Batch"], ROLE["Payment Batch"])
        case_ctx["last_family"] = FAMILY["Payment Batch"]
        case_ctx["last_resource"] = "Unknown"
    t, _ = schedule_activity(trace, case_ctx, "Pay Invoice", max(t, rel),
                             allow_overtime=case_ctx.get("expedite", False))

    # SLA penalty if due exceeded
    if t > case_ctx["invoice_due"]:
        add_event(trace, "Escalate", t, "Unknown", 60.0, ROLE["Escalate"])

    # 9) Close order
    t, _ = schedule_activity(trace, case_ctx, "Close Order", t, allow_preempt=False)

    # Strict order & return
    ensure_strictly_increasing(trace)
    return trace

# -----------------------------
# Distinct-trace control + log build
# -----------------------------
def trace_signature(trace: Trace) -> Tuple[str, ...]:
    # signature over concept:name sequence (keep all labels, incl. start/resume/complete, Wait, etc.)
    return tuple(e["concept:name"] for e in trace)

def execute_script(out_path: str = OUT_PATH, target_distinct: int = TARGET_DISTINCT_TRACES):
    # Generate NHPP case arrivals (more than needed; we'll stop once we have 250 distinct)
    arrivals = sample_nhpp_arrivals(START_DATE, END_DATE, target_n=min(MAX_CASES_TO_TRY, target_distinct*3))
    log = EventLog()
    seen = set()

    # Assign case priorities based on arrival peak (simulate backlog/priority drift)
    priorities = {}
    for idx, t in enumerate(arrivals):
        # higher priority in end-of-month / waves
        pr = 1 if (t.day >= 24 or t.hour in (10, 15)) and random.random() < 0.5 else 0
        priorities[idx] = pr

    case_idx = 0
    attempts = 0
    while case_idx < target_distinct and attempts < len(arrivals):
        start_at = arrivals[attempts]
        cid = f"C{attempts+1:05d}"
        trace = simulate_case(cid, start_at, priorities.get(attempts, 0))
        sig = trace_signature(trace)
        if sig not in seen:
            seen.add(sig)
            log.append(trace)
            case_idx += 1
        attempts += 1

    # -----------------------------
    # Quality & sanity checks
    # -----------------------------
    # 1) schema check + monotonicity
    schema_ok = True
    for tr in log:
        last = None
        for ev in tr:
            if not all(k in ev for k in ("concept:name","time:timestamp","org:resource","amount")):
                schema_ok = False
            if last and ev["time:timestamp"] <= last:
                schema_ok = False
            last = ev["time:timestamp"]

    # 2) summaries
    trace_lengths = [len(tr) for tr in log]
    case_durations_h = [ (tr[-1]["time:timestamp"] - tr[0]["time:timestamp"]).total_seconds()/3600.0 for tr in log ]
    total_amounts = [ sum(float(e["amount"]) for e in tr) for tr in log ]
    # remaining-time distribution (per event)
    rem_all = []
    for tr in log:
        end_t = tr[-1]["time:timestamp"]
        for e in tr:
            rem_all.append((end_t - e["time:timestamp"]).total_seconds()/3600.0)
    # ambiguity check (prefix -> next activity diversity)
    import collections
    nexts_map = collections.defaultdict(set)
    rng_idx = list(range(len(log)))
    random.shuffle(rng_idx)
    sample_traces = rng_idx[:min(100, len(log))]
    for idx in sample_traces:
        tr = log[idx]
        if len(tr) < 3:
            continue
        k = random.randint(1, len(tr)-2)
        prefix = tuple(tr[i]["concept:name"] for i in range(k))
        nexts_map[prefix].add(tr[k]["concept:name"])
    ambiguous = sum(1 for s in nexts_map.values() if len(s) >= 2)
    amb_ratio = ambiguous / max(1, len(nexts_map))

    # parallelism check (presence of AND.Join.VendorPrep or Join.QA.kofn)
    traces_with_and = sum(1 for tr in log if any(e["concept:name"] in ("AND.Join.VendorPrep","Join.QA.kofn") for e in tr))
    pct_parallel = 100.0 * traces_with_and / max(1, len(log))

    # era drift
    def era_of_trace(tr: Trace) -> str:
        return era_of(tr[0]["time:timestamp"])
    era_groups = {}
    for tr in log:
        era = era_of_trace(tr)
        era_groups.setdefault(era, []).append(tr)
    era_stats = {}
    for era, trs in era_groups.items():
        durs = [ (t[-1]["time:timestamp"] - t[0]["time:timestamp"]).total_seconds()/3600.0 for t in trs ]
        costs = [ sum(float(e["amount"]) for e in t) for t in trs ]
        era_stats[era] = {
            "n": len(trs),
            "mean_duration_h": float(np.mean(durs)) if durs else 0.0,
            "mean_cost": float(np.mean(costs)) if costs else 0.0
        }

    # print report
    print("=== Generation Report ===")
    print(f"Target distinct traces : {target_distinct}")
    print(f"Actual distinct traces : {len(log)}")
    print(f"Attempts (cases tried) : {attempts}")
    print(f"Schema OK              : {schema_ok}")
    print(f"Trace length (min/med/max) : {min(trace_lengths)} / {int(np.median(trace_lengths))} / {max(trace_lengths)}")
    print(f"Case duration h (mean/std/q10/q50/q90) : {np.mean(case_durations_h):.2f} / {np.std(case_durations_h):.2f} / "
          f"{np.quantile(case_durations_h,0.1):.2f} / {np.quantile(case_durations_h,0.5):.2f} / {np.quantile(case_durations_h,0.9):.2f}")
    print(f"Total amount per case $ (mean/std)     : {np.mean(total_amounts):.2f} / {np.std(total_amounts):.2f}")
    print(f"Remaining time h across events (mean/std/q10/q50/q90): "
          f"{np.mean(rem_all):.2f} / {np.std(rem_all):.2f} / {np.quantile(rem_all,0.1):.2f} / "
          f"{np.quantile(rem_all,0.5):.2f} / {np.quantile(rem_all,0.9):.2f}")
    print(f"Ambiguity check: {ambiguous} / {len(nexts_map)} sampled prefixes have ≥2 next activities (ratio={amb_ratio:.2f})")
    print(f"Parallelism check: {pct_parallel:.1f}% of traces have an AND/OR join")
    print("Era drift (mean duration h / mean cost):")
    for era, st in era_stats.items():
        print(f"  - {era:>5}: n={st['n']}, dur={st['mean_duration_h']:.2f}, cost={st['mean_cost']:.2f}")

    # -----------------------------
    # Export
    # -----------------------------
    write_xes(log, out_path)
    print(f"\nSaved XES log to: {out_path}")

# -----------------------------
# Minimal path comment (for reference):
# A short case may hit: AND-split (Evaluate/Negotiate), handover event on resource change,
# escalation before Approve Invoice if deadline tight, rework loop on Review/Fix Invoice,
# vendor (outsource) branch with overtime on Pay Invoice.
# -----------------------------

if __name__ == "__main__":
    execute_script(OUT_PATH, TARGET_DISTINCT_TRACES)
