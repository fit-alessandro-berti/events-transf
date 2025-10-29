"""
Close-to-Report synthetic XES log generator (trace-by-trace, no black-box simulator)

- Builds a realistic Close-to-Report (C2R) process log with rich control-flow,
  temporal, resource, and cost patterns.
- Each event has: concept:name, time:timestamp (datetime), org:resource (string), amount (float)
- Trace attribute concept:name is set to the case id so `case:concept:name` is present in PM4Py dataframes.
- Saves to: clos2rep.xes.gz
- Limits the log to 250 traces (exactly 250 distinct cases).

Major features implemented (mix-and-match per case):
  * Linear flow with optional inserts; XOR/OR branching; AND-split/join with sync latency
  * Loops/Rework (Review -> Fix)*; Multi-instance (k-of-n) signoffs; Preemption/resume (".start/.resume/.complete")
  * Long-term dependencies (priority/risk affects later routes); binding/separation of duty
  * Bursty arrivals (month-end waves + intraday peaks), business+shift calendars, lunch break, OT with penalty
  * Resource skills/learning/forgetting; handover delay/fee on resource change
  * Setup/changeover time by activity family; batching/release windows for some tasks
  * Aging/deadlines -> Warn/Escalate; Vendor outsourcing option; Budget-aware branching
  * Cost model: fixed + rate*duration (+holding cost, OT/weekend multipliers, penalties, credits)

Minimal example (embedded in code comments near the bottom) creates a short trace that
hits: AND-split, handover delay, SLA penalty -> Escalate, one rework, and vendor path with overtime.

Dependencies: python stdlib, numpy, (optional pandas for summaries), pm4py
"""

import math
import random
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import pandas as pd  # optional; used for nicer summaries if available
    HAVE_PANDAS = True
except Exception:
    HAVE_PANDAS = False

# PM4Py
from pm4py.objects.log.obj import EventLog, Trace, Event
try:
    import pm4py
    HAVE_PM4PY_WRITE = hasattr(pm4py, "write_xes")
except Exception:
    HAVE_PM4PY_WRITE = False
try:
    from pm4py.objects.log.exporter.xes import exporter as xes_exporter
except Exception:
    xes_exporter = None


# ------------------------------
# Global parameters
# ------------------------------
SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

N_CASES = 250                       # limit required by user
OUT_PATH = "clos2rep.xes.gz"

# Time horizon / eras (simulate concept drift across eras)
ERA_DEFS = [
    # (label, start_date, end_date, param drift)
    ("Era1", datetime(2025, 7, 1), datetime(2025, 8, 15)),
    ("Era2", datetime(2025, 8, 16), datetime(2025, 9, 30)),
    ("Era3", datetime(2025, 10, 1), datetime(2025, 10, 28)),
]

# Work calendar (global business hours; resource shifts refine this)
WEEKDAYS = set([0, 1, 2, 3, 4])     # Mon-Fri
BUSINESS_OPEN = dtime(9, 0)
BUSINESS_CLOSE = dtime(17, 0)
LUNCH_START = dtime(12, 0)
LUNCH_END = dtime(13, 0)

# Batching windows (release times) for certain activities (10:00 and 16:00)
BATCH_TIMES = [dtime(10, 0), dtime(16, 0)]
BATCHED_ACTIVITIES = {
    "CTR_Post_Journal",
    "CTR_Accruals",
    "CTR_Adjustments",
}

# Activity universe (25+ labels; Close-to-Report flavored)
ACTIVITIES = [
    "CTR_Open_Period",
    "CTR_Pre_Close",
    "CTR_Risk_Assessment",
    "CTR_Collect_Data",
    "CTR_Validate_Balances",
    "CTR_Reconcile_GL",
    "CTR_Intercompany_Match",
    "CTR_FX_Revaluation",
    "CTR_Tax_Adjustment",
    "CTR_Post_Journal",
    "CTR_Accruals",
    "CTR_Adjustments",
    "CTR_Consolidate",
    "CTR_Submit",
    "CTR_QA",
    "CTR_Review",
    "CTR_Approve",
    "CTR_Audit",
    "CTR_Publish_Reports",
    "CTR_Close_Period",
    # Control/supportive:
    "CTR_Warn",
    "CTR_Escalate",
    "CTR_Expedite",
    "CTR_Reassign",
    "CTR_Handover",
    "CTR_Wait",
    "CTR_Simplify",
    "CTR_Meeting",
    "CTR_Vendor_Posting",
    "CTR_Subsidiary_Signoff",
    "CTR_Cancel",
    "CTR_Rollback",
    "CTR_Join"
]

# Preferred roles per activity (fallbacks handled in assignment)
PREF_ROLE = {
    "CTR_Open_Period": "role:controller",
    "CTR_Pre_Close": "role:controller",
    "CTR_Risk_Assessment": "role:controller",
    "CTR_Collect_Data": "role:analyst",
    "CTR_Validate_Balances": "role:analyst",
    "CTR_Reconcile_GL": "role:analyst",
    "CTR_Intercompany_Match": "role:analyst",
    "CTR_FX_Revaluation": "role:controller",
    "CTR_Tax_Adjustment": "role:controller",
    "CTR_Post_Journal": "role:analyst",
    "CTR_Accruals": "role:analyst",
    "CTR_Adjustments": "role:analyst",
    "CTR_Consolidate": "role:controller",
    "CTR_Submit": "role:controller",
    "CTR_QA": "role:controller",
    "CTR_Review": "role:senior",
    "CTR_Approve": "role:senior",
    "CTR_Audit": "role:auditor",
    "CTR_Publish_Reports": "role:controller",
    "CTR_Close_Period": "role:controller",
    "CTR_Warn": "role:senior",
    "CTR_Escalate": "role:senior",
    "CTR_Expedite": "role:controller",
    "CTR_Reassign": "role:controller",
    "CTR_Handover": "role:controller",
    "CTR_Wait": "role:controller",
    "CTR_Simplify": "role:controller",
    "CTR_Meeting": "role:senior",
    "CTR_Vendor_Posting": "role:vendor",
    "CTR_Subsidiary_Signoff": "role:analyst",
    "CTR_Cancel": "role:controller",
    "CTR_Rollback": "role:controller",
    "CTR_Join": "role:controller"
}

# Base durations in minutes (rough means for lognormal sampling)
BASE_MEAN_MIN = {
    "CTR_Open_Period": 20,
    "CTR_Pre_Close": 30,
    "CTR_Risk_Assessment": 40,
    "CTR_Collect_Data": 60,
    "CTR_Validate_Balances": 45,
    "CTR_Reconcile_GL": 120,
    "CTR_Intercompany_Match": 90,
    "CTR_FX_Revaluation": 60,
    "CTR_Tax_Adjustment": 50,
    "CTR_Post_Journal": 30,
    "CTR_Accruals": 40,
    "CTR_Adjustments": 45,
    "CTR_Consolidate": 180,
    "CTR_Submit": 20,
    "CTR_QA": 35,
    "CTR_Review": 40,
    "CTR_Approve": 30,
    "CTR_Audit": 90,
    "CTR_Publish_Reports": 40,
    "CTR_Close_Period": 45,
    "CTR_Warn": 5,
    "CTR_Escalate": 10,
    "CTR_Expedite": 5,
    "CTR_Reassign": 15,
    "CTR_Handover": 8,
    "CTR_Wait": 15,  # appears as a buffer activity when needed
    "CTR_Simplify": 25,
    "CTR_Meeting": 30,
    "CTR_Vendor_Posting": 70,
    "CTR_Subsidiary_Signoff": 45,
    "CTR_Cancel": 5,
    "CTR_Rollback": 10,
    "CTR_Join": 3,
}

# Family buckets for setup/context-switch penalties
FAMILY = {
    # Data collection and validation
    "CTR_Collect_Data": "fam:data",
    "CTR_Validate_Balances": "fam:data",
    # Accounting entries
    "CTR_Post_Journal": "fam:entries",
    "CTR_Accruals": "fam:entries",
    "CTR_Adjustments": "fam:entries",
    "CTR_Vendor_Posting": "fam:entries",
    # Reconciliation / interco
    "CTR_Reconcile_GL": "fam:recon",
    "CTR_Intercompany_Match": "fam:recon",
    # Consolidation and reporting
    "CTR_Consolidate": "fam:consol",
    "CTR_Publish_Reports": "fam:consol",
    "CTR_Submit": "fam:consol",
    # Controls and approvals
    "CTR_QA": "fam:control",
    "CTR_Review": "fam:control",
    "CTR_Approve": "fam:control",
    "CTR_Audit": "fam:control",
    # Other
    "CTR_FX_Revaluation": "fam:special",
    "CTR_Tax_Adjustment": "fam:special",
    "CTR_Open_Period": "fam:admin",
    "CTR_Pre_Close": "fam:admin",
    "CTR_Close_Period": "fam:admin",
    "CTR_Risk_Assessment": "fam:admin",
    # pseudo tasks
    "CTR_Warn": "fam:admin",
    "CTR_Escalate": "fam:admin",
    "CTR_Expedite": "fam:admin",
    "CTR_Reassign": "fam:admin",
    "CTR_Handover": "fam:admin",
    "CTR_Wait": "fam:admin",
    "CTR_Simplify": "fam:admin",
    "CTR_Meeting": "fam:admin",
    "CTR_Subsidiary_Signoff": "fam:control",
    "CTR_Cancel": "fam:admin",
    "CTR_Rollback": "fam:admin",
    "CTR_Join": "fam:admin",
}

def mean_minutes(act: str) -> float:
    return BASE_MEAN_MIN.get(act, 30.0)

# Role hourly base rates (USD); era multipliers applied later
ROLE_RATE = {
    "role:analyst": 50.0,
    "role:senior": 80.0,
    "role:controller": 70.0,
    "role:auditor": 90.0,
    "role:vendor": 60.0,
    # catch-all
    "role:unknown": 40.0
}

OVERTIME_MULT = 1.5
WEEKEND_MULT = 2.0
HOLDING_RATE_PER_HOUR = 6.0                 # waiting cost (holding/flow time)
HANDOVER_FEE = 8.0                           # fixed fee at handover pseudo-activity
REWORK_PREMIUM = 1.15                        # geometric increase for rework loops
SLA_PENALTY = 250.0                          # applies on escalate
BUDGET_LIMIT = 4500.0                        # budget-aware branching threshold

UNKNOWN_RESOURCE_FRAC = 0.03                 # small fraction uses "Unknown"

# ------------------------------
# Utilities: dates & calendars
# ------------------------------
def is_weekday(dt: datetime) -> bool:
    return dt.weekday() in WEEKDAYS

def in_interval(dt: datetime, start_t: dtime, end_t: dtime) -> bool:
    t = dt.time()
    return (t >= start_t) and (t < end_t)

@dataclass
class Shift:
    start: dtime
    end: dtime
    lunch_start: dtime = LUNCH_START
    lunch_end: dtime = LUNCH_END
    weekdays: set = None

    def __post_init__(self):
        if self.weekdays is None:
            self.weekdays = WEEKDAYS

    def open_segments_for_day(self, day: datetime) -> List[Tuple[datetime, datetime]]:
        # No work on weekends
        segments = []
        if day.weekday() not in self.weekdays:
            return segments
        d = day.date()
        segs = []
        # morning
        if self.start < self.lunch_start:
            segs.append((datetime.combine(d, self.start), datetime.combine(d, self.lunch_start)))
        # afternoon
        if self.lunch_end < self.end:
            segs.append((datetime.combine(d, self.lunch_end), datetime.combine(d, self.end)))
        return segs

    def is_open(self, ts: datetime) -> bool:
        segs = self.open_segments_for_day(ts)
        for a, b in segs:
            if a <= ts < b:
                return True
        return False

    def next_open(self, ts: datetime) -> datetime:
        if self.is_open(ts):
            return ts
        day = datetime(ts.year, ts.month, ts.day)
        # explore from current ts forward
        for k in range(0, 10):  # look ahead max ~2 weeks
            d = day + timedelta(days=k if ts.time() < self.end else k+1)
            segs = self.open_segments_for_day(d)
            if segs:
                # choose first segment's start
                start = segs[0][0]
                if start > ts:
                    return start
        # fallback: next day at 9am
        return day + timedelta(days=1, hours=BUSINESS_OPEN.hour)

    def add_working_time(self, start: datetime, seconds: float) -> datetime:
        """Adds working time in 'seconds' following shift open segments."""
        t = max(start, self.next_open(start))
        remaining = seconds
        while remaining > 0:
            segs = self.open_segments_for_day(t)
            # Find segment containing or following t
            seg = None
            for a, b in segs:
                if a <= t < b:
                    seg = (a, b)
                    break
                if t < a:
                    seg = (a, b)
                    t = a
                    break
            if seg is None:
                t = self.next_open(t)
                continue
            a, b = seg
            cap = (b - t).total_seconds()
            if remaining <= cap:
                return t + timedelta(seconds=remaining)
            else:
                remaining -= cap
                t = b + timedelta(seconds=1)  # move to next instant
                t = self.next_open(t)
        return t

# ------------------------------
# Resources
# ------------------------------
@dataclass
class Resource:
    rid: str
    role: str
    shift: Shift
    next_free: datetime
    skill_factor_by_family: Dict[str, float]  # <1 faster; >1 slower
    experience: Dict[str, int]                # per-activity count
    last_exec: Dict[str, datetime]            # last time executed act
    last_family: Optional[str] = None
    absent_prob: float = 0.03

def make_resources(global_start: datetime) -> Dict[str, Resource]:
    resources = {}
    # Assign roles roughly: analysts: 18, seniors: 6, controllers: 4, auditors: 2
    role_assign = (["role:analyst"] * 18 +
                   ["role:senior"] * 6 +
                   ["role:controller"] * 4 +
                   ["role:auditor"] * 2)
    rng.shuffle(role_assign)
    shifts = [
        Shift(dtime(9, 0), dtime(17, 0)),
        Shift(dtime(7, 0), dtime(15, 0)),
        Shift(dtime(11, 0), dtime(19, 0)),
    ]
    for i in range(30):
        role = role_assign[i % len(role_assign)]
        shift = random.choice(shifts)
        # family skill heterogeneity
        skills = {fam: max(0.7, min(1.4, rng.lognormal(mean=0.0, sigma=0.25))) for fam in set(FAMILY.values())}
        resources[f"R{i+1}"] = Resource(
            rid=f"R{i+1}",
            role=role,
            shift=shift,
            next_free=global_start,
            skill_factor_by_family=skills,
            experience=defaultdict(int),
            last_exec=defaultdict(lambda: global_start - timedelta(days=30)),
            absent_prob=0.03 if role != "role:senior" else 0.01
        )
    # External vendor (single logical resource string for clarity)
    resources["Vendor"] = Resource(
        rid="Vendor",
        role="role:vendor",
        shift=Shift(dtime(9, 0), dtime(17, 0)),
        next_free=global_start,
        skill_factor_by_family={fam: 1.15 for fam in set(FAMILY.values())},
        experience=defaultdict(int),
        last_exec=defaultdict(lambda: global_start - timedelta(days=30)),
        absent_prob=0.0
    )
    return resources

# ------------------------------
# Arrival process (bursty NHPP-style)
# ------------------------------
def sample_arrival_times(n: int) -> List[datetime]:
    """
    Month-end bursts + intraday waves around 10:00 and 15:00.
    We build dates by mixing Gaussians around month-ends in each era,
    then add intraday peaks, and sort.
    """
    times = []
    for era_label, s, e in ERA_DEFS:
        days = (e - s).days + 1
        # allocate ~ proportionally to days, with more weight to Era3
        weight = days + (10 if era_label == "Era3" else 0) + (5 if era_label == "Era2" else 0)
        times.append((era_label, s, e, weight))
    total_w = sum(w for _, _, _, w in times)
    shares = [max(1, int(round(n * w / total_w))) for _, _, _, w in times]
    # adjust to exact n
    while sum(shares) > n:
        shares[shares.index(max(shares))] -= 1
    while sum(shares) < n:
        shares[shares.index(min(shares))] += 1

    arrivals = []
    for idx, (era_label, s, e, _) in enumerate(times):
        k = shares[idx]
        for _ in range(k):
            # Day selection: biased towards the last week of the month (Close time)
            day_span = (e - s).days
            base_day = rng.integers(0, day_span + 1)
            date = s + timedelta(days=int(base_day))
            # boost toward month-end
            if rng.random() < 0.55:
                # pull towards last 5 days in that month
                last_day = (datetime(date.year + date.month // 12, (date.month % 12) + 1, 1) - timedelta(days=1)).day
                dom = rng.integers(max(1, last_day - 5), last_day + 1)
                date = datetime(date.year, date.month, int(dom))

            # Intraday peaks
            if rng.random() < 0.6:
                hour = 10 + int(np.clip(rng.normal(0, 1.0), -2.5, 2.5))
            else:
                hour = 15 + int(np.clip(rng.normal(0, 1.2), -2.5, 2.5))
            minute = int(np.clip(np.round(rng.normal(0, 10.0)), -25, 25)) + 30
            minute = int(np.clip(minute, 0, 59))
            dt = datetime(date.year, date.month, date.day, int(np.clip(hour, 7, 18)), minute, int(rng.integers(0, 60)))
            # Move to next weekday morning if on weekend
            while dt.weekday() not in WEEKDAYS:
                dt = dt + timedelta(days=1)
                dt = dt.replace(hour=10, minute=30, second=0)
            arrivals.append(dt)
    arrivals.sort()
    return arrivals[:n]

# ------------------------------
# Assignment policy
# ------------------------------
def choose_resource(activity: str, at: datetime, resources: Dict[str, Resource],
                    policy_mix: Tuple[float, float, float] = (0.4, 0.3, 0.3),
                    prefer_role: Optional[str] = None) -> Resource:
    """
    Return a resource per a weighted mixture of:
      - random among role-matching pool (or all)
      - least-loaded (earliest next_free) among role-matching pool
      - skill-based (lowest skill factor for the activity family)
    """
    fam = FAMILY.get(activity, "fam:admin")
    pool = list(resources.values())

    if prefer_role:
        role_pool = [r for r in pool if r.role == prefer_role]
        if role_pool:
            pool = role_pool
    # fallback if role pool empty
    if not pool:
        pool = list(resources.values())

    choice = rng.random()
    if choice < policy_mix[0]:
        # random
        cand = [r for r in pool if rng.random() > r.absent_prob]
        if not cand:
            cand = pool
        return random.choice(cand)
    elif choice < policy_mix[0] + policy_mix[1]:
        # least-loaded
        cand = sorted(pool, key=lambda r: r.next_free)
        for r in cand:
            if rng.random() > r.absent_prob:
                return r
        return cand[0]
    else:
        # skill-based (lower factor is faster)
        cand = sorted(pool, key=lambda r: r.skill_factor_by_family.get(fam, 1.0))
        for r in cand:
            if rng.random() > r.absent_prob:
                return r
        return cand[0]

# ------------------------------
# Duration model (heavy tails + adjustments)
# ------------------------------
def draw_service_seconds(activity: str, resource: Resource, start: datetime,
                         era_label: str, prev_family: Optional[str],
                         learned: int, forgot_days: float) -> float:
    """
    Lognormal base with Pareto-like heavy tail via occasional multiplier.
    Adjust by skill, learning (faster), forgetting (slower), setup/changeover, and era drift.
    """
    base_min = mean_minutes(activity)

    # Era drift: Era2 faster Review/QA, Era3 slower Approve
    era_mult = 1.0
    if era_label == "Era2" and activity in ("CTR_Review", "CTR_QA"):
        era_mult *= 0.85
    if era_label == "Era3" and activity in ("CTR_Approve",):
        era_mult *= 1.15
    if era_label == "Era3" and activity in ("CTR_Vendor_Posting",):
        era_mult *= 0.9  # cheaper vendor later (faster)
    # Resource skill
    fam = FAMILY.get(activity, "fam:admin")
    skill_mult = resource.skill_factor_by_family.get(fam, 1.0)
    # Learning curve: 3% faster per repetition up to 25%
    learn_mult = max(0.75, 1.0 - 0.03 * min(8, learned))
    # Forgetting curve: if inactive > 7 days, 1%/day up to +25%
    forget_mult = min(1.25, 1.0 + max(0.0, (forgot_days - 7.0)) * 0.01)
    # Setup/changeover if switching families
    setup_mult = 1.0
    if prev_family and prev_family != fam:
        setup_mult += rng.uniform(0.05, 0.20)

    # Base lognormal (sigma heavy)
    mu = math.log(max(1.0, base_min * era_mult / 60.0))  # convert to hours for stability
    sigma = 0.70 if activity not in ("CTR_Consolidate", "CTR_Reconcile_GL") else 0.85
    dur_hours = rng.lognormal(mean=mu, sigma=sigma)
    # Occasional Pareto-like tail bump
    if rng.random() < 0.08:
        dur_hours *= rng.pareto(2.5) + 1.0

    total_hours = dur_hours * skill_mult * learn_mult * forget_mult * setup_mult
    seconds = max(60.0, total_hours * 3600.0)  # >= 1 minute
    return float(seconds)

# ------------------------------
# Cost model
# ------------------------------
def hourly_rate(role: str, when: datetime, overtime: bool) -> float:
    base = ROLE_RATE.get(role, ROLE_RATE["role:unknown"])
    rate = base
    if when.weekday() not in WEEKDAYS:
        rate *= WEEKEND_MULT
    if overtime:
        rate *= OVERTIME_MULT
    return rate

def compute_amount(activity: str, role: str, dur_seconds: float, start: datetime,
                   overtime: bool, waiting_seconds: float, rework_iter: int,
                   handover: bool, era_label: str, escalation: bool, material_bump: bool) -> float:
    # fixed component
    fixed = 0.0
    # material costs for some special activities
    if activity in {"CTR_FX_Revaluation", "CTR_Tax_Adjustment"} and material_bump:
        fixed += 30.0 + rng.uniform(0, 40)
    if activity in {"CTR_Post_Journal", "CTR_Accruals", "CTR_Adjustments"}:
        fixed += 8.0 + rng.uniform(0, 12)
    if activity == "CTR_Escalate":
        fixed += SLA_PENALTY
    if handover and activity == "CTR_Handover":
        fixed += HANDOVER_FEE

    # variable (labor) component
    rate = hourly_rate(role, start, overtime)
    variable = (dur_seconds / 3600.0) * rate

    # holding cost for waits
    holding = (waiting_seconds / 3600.0) * HOLDING_RATE_PER_HOUR

    # rework premium grows with iteration
    rework_mult = (REWORK_PREMIUM ** max(0, rework_iter - 1)) if rework_iter >= 1 else 1.0

    # era tariffs (later eras slightly higher controller cost)
    if era_label == "Era3" and role == "role:controller":
        variable *= 1.08

    amt = (fixed + variable + holding) * rework_mult

    # Occasional credits on rollback/cancel
    if activity in {"CTR_Cancel", "CTR_Rollback"}:
        amt += -rng.uniform(20.0, 80.0)

    return float(round(amt, 2))

# ------------------------------
# Batching & release windows
# ------------------------------
def align_to_release(activity: str, candidate: datetime) -> datetime:
    if activity not in BATCHED_ACTIVITIES:
        return candidate
    # Find next batch time today or later
    d = candidate.date()
    times = []
    for bt in BATCH_TIMES:
        dt = datetime.combine(d, bt)
        if dt >= candidate:
            times.append(dt)
    if not times:
        # next day first batch
        nd = d + timedelta(days=1)
        dt = datetime.combine(nd, BATCH_TIMES[0])
        return dt
    return min(times)

# ------------------------------
# Scheduling
# ------------------------------
def schedule_activity(case_id: str, activity: str, ready_time: datetime,
                      resources: Dict[str, Resource], prefer_role: Optional[str],
                      era_label: str, context: dict, prev_resource: Optional[Resource]) -> Tuple[List[Dict], datetime, Resource]:
    """
    Schedules a single logical activity, injecting waits, OT, handover, preemption where applicable.
    Returns: [events], end_time, resource
    """
    events: List[Dict] = []
    # Assignment policy mixture (random/least-loaded/skill)
    res = choose_resource(activity, ready_time, resources, prefer_role=prefer_role)

    # Binding-of-duty: ensure same resource for specific pairs if requested
    if context.get("bind_duty") and activity in {"CTR_Post_Journal", "CTR_Adjustments"} and context.get("bound_resource"):
        res = context["bound_resource"]

    # Separation-of-duty: different resource for Approve vs Audit
    if context.get("sep_duty") and activity == "CTR_Audit" and context.get("approver_resource") == res:
        # intentionally choose someone else of same role if possible
        alt = [r for r in resources.values() if r.role == res.role and r.rid != res.rid]
        if alt:
            res = random.choice(alt)

    # Unknown resource fraction
    use_unknown = (rng.random() < UNKNOWN_RESOURCE_FRAC and res.role != "role:vendor")

    # If resource off-shift, decide to wait or OT or reassign
    allow_ot = False
    if not res.shift.is_open(ready_time):
        if context.get("priority") or context.get("expedite") or rng.random() < 0.25:
            allow_ot = True  # overtime permitted
        elif rng.random() < 0.25:
            # reassign to someone on shift now
            cand = [r for r in resources.values() if r.role == res.role and r.shift.is_open(ready_time)]
            if cand:
                res = min(cand, key=lambda r: r.next_free)

    # Wait until both resource free and within shift (unless OT)
    candidate = max(ready_time, res.next_free)
    if not allow_ot:
        candidate = res.shift.next_open(candidate)

    # Batching
    candidate = align_to_release(activity, candidate)

    waiting_seconds = max(0.0, (candidate - ready_time).total_seconds())

    # Learning / forgetting parameters
    learned = res.experience[activity]
    forgot_days = max(0.0, (candidate - res.last_exec.get(activity, candidate)).total_seconds() / 86400.0)

    # Duration draw
    prev_family = res.last_family
    duration_seconds = draw_service_seconds(activity, res, candidate, era_label, prev_family, learned, forgot_days)

    # Preemption: break long tasks, especially under backlog; create .start/.resume/.complete
    preempted = False
    preempt_labels = []
    backlog_proxy = (candidate - ready_time).total_seconds() / 3600.0  # hours of wait ~ queue pressure
    if duration_seconds > 2.0 * 3600 and (backlog_proxy > 1.0 or rng.random() < 0.25):
        preempted = True
        start_seg = max(15 * 60, duration_seconds * rng.uniform(0.15, 0.35))
        pause_gap = rng.uniform(30 * 60, 2 * 3600)   # 30-120 minutes
        end_seg = max(15 * 60, duration_seconds - start_seg)
        preempt_labels = [f"{activity}.start", f"{activity}.resume", f"{activity}.complete"]
        segs = [start_seg, end_seg]
        # First segment
        start_ts = candidate
        if allow_ot:
            end_ts = start_ts + timedelta(seconds=start_seg)
        else:
            end_ts = res.shift.add_working_time(start_ts, start_seg)
        # record start event
        role = res.role if not use_unknown else "role:unknown"
        e1 = {
            "concept:name": preempt_labels[0],
            "time:timestamp": start_ts,
            "org:resource": res.rid if not use_unknown else "Unknown",
            "org:role": role,
            "amount": compute_amount(activity, res.role, start_seg, start_ts, allow_ot, waiting_seconds, 0, False, era_label, False, True)
        }
        events.append(e1)
        # Pause (no resource held during pause)
        resume_ts = end_ts + timedelta(seconds=pause_gap)
        resume_ts = res.shift.next_open(resume_ts) if not allow_ot else resume_ts
        # Second segment scheduling might collide with res.next_free (set after first seg)
        # Update resource state after first segment
        res.next_free = end_ts
        res.experience[activity] += 1
        res.last_exec[activity] = end_ts
        res.last_family = FAMILY.get(activity, res.last_family)

        # Handover if previous resource differs
        if prev_resource and prev_resource.rid != res.rid:
            # add handover event just before resume
            h_ts = min(resume_ts - timedelta(seconds=1), end_ts + timedelta(seconds=1))
            h_event = {
                "concept:name": "CTR_Handover",
                "time:timestamp": h_ts,
                "org:resource": f"{(prev_resource.rid if prev_resource else 'R?')}->{res.rid}",
                "amount": compute_amount("CTR_Handover", "role:controller", 5*60, h_ts, False, 0.0, 0, True, era_label, False, False)
            }
            events.append(h_event)

        # Second segment
        start_ts2 = max(resume_ts, res.next_free)
        if allow_ot:
            end_ts2 = start_ts2 + timedelta(seconds=end_seg)
        else:
            end_ts2 = res.shift.add_working_time(start_ts2, end_seg)

        e2 = {
            "concept:name": preempt_labels[1],
            "time:timestamp": start_ts2,
            "org:resource": res.rid if not use_unknown else "Unknown",
            "org:role": role,
            "amount": compute_amount(activity, res.role, 60, start_ts2, allow_ot, 0.0, 0, False, era_label, False, False)  # small admin cost for resume
        }
        events.append(e2)
        e3 = {
            "concept:name": preempt_labels[2],
            "time:timestamp": end_ts2,
            "org:resource": res.rid if not use_unknown else "Unknown",
            "org:role": role,
            "amount": compute_amount(activity, res.role, end_seg, start_ts2, allow_ot, 0.0, 0, False, era_label, False, True)
        }
        events.append(e3)

        end_time = end_ts2
        res.next_free = end_time
        res.experience[activity] += 1
        res.last_exec[activity] = end_time
        res.last_family = FAMILY.get(activity, res.last_family)

    else:
        # Simple (non-preempted) case
        start_ts = candidate
        end_ts = start_ts + timedelta(seconds=duration_seconds) if allow_ot else res.shift.add_working_time(start_ts, duration_seconds)
        # Handover pseudo-activity if immediate predecessor was different resource (sequential edges)
        if prev_resource and prev_resource.rid != res.rid:
            h_ts = max(start_ts - timedelta(seconds=5), ready_time + timedelta(seconds=1))
            h_event = {
                "concept:name": "CTR_Handover",
                "time:timestamp": h_ts,
                "org:resource": f"{prev_resource.rid}->{res.rid}",
                "amount": compute_amount("CTR_Handover", "role:controller", 5*60, h_ts, False, 0.0, 0, True, era_label, False, False)
            }
            events.append(h_event)

        role = res.role if not use_unknown else "role:unknown"
        ev = {
            "concept:name": activity,
            "time:timestamp": end_ts,  # choose end moment as event stamp (common for event logs)
            "org:resource": res.rid if not use_unknown else "Unknown",
            "org:role": role,
            "amount": compute_amount(activity, res.role, duration_seconds, start_ts, allow_ot, waiting_seconds, 0, False, era_label, False, True)
        }
        # Optional explicit wait event if significant waiting accumulated
        if waiting_seconds > 15 * 60 and rng.random() < 0.8:
            w_ts = max(ready_time + timedelta(seconds=60), start_ts - timedelta(seconds=60))
            w_event = {
                "concept:name": "CTR_Wait",
                "time:timestamp": w_ts,
                "org:resource": "System",
                "amount": compute_amount("CTR_Wait", "role:controller", 0.0, w_ts, False, waiting_seconds, 0, False, era_label, False, False)
            }
            events.append(w_event)

        events.append(ev)
        end_time = end_ts
        res.next_free = end_ts
        res.experience[activity] += 1
        res.last_exec[activity] = end_ts
        res.last_family = FAMILY.get(activity, res.last_family)

    # Budget-aware routing signal
    context["budget"] += sum(e["amount"] for e in events if "amount" in e)

    # Binding-of-duty anchor
    if activity == "CTR_Post_Journal" and context.get("bind_duty") and "bound_resource" not in context:
        context["bound_resource"] = res

    # Record approver to enforce separation vs audit
    if activity == "CTR_Approve":
        context["approver_resource"] = res

    return events, end_time, res

# ------------------------------
# Case-level skeletons and pattern composition
# ------------------------------
def schedule_and_join(case_id: str, branches: List[List[str]], start_time: datetime,
                      resources: Dict[str, Resource], era_label: str, context: dict,
                      prev_res: Optional[Resource]) -> Tuple[List[Dict], datetime, Optional[Resource], bool]:
    """
    Schedule branches in parallel and join (AND-join). Join event waits for max(children end).
    Adds synchronization latency before the join.
    Returns: events, join_time, synthetic join resource (None), had_and=True
    """
    all_events = []
    end_times = []
    for seq in branches:
        t = start_time
        pr = prev_res
        for act in seq:
            evs, t, pr = schedule_activity(case_id, act, t, resources, PREF_ROLE.get(act), era_label, context, pr)
            all_events.extend(evs)
        end_times.append(t)
    join_time = max(end_times) + timedelta(seconds=rng.uniform(60, 300))  # sync latency
    # Join pseudo-activity
    join_event = {
        "concept:name": "CTR_Join",
        "time:timestamp": join_time,
        "org:resource": "System",
        "amount": compute_amount("CTR_Join", "role:controller", 60, join_time, False, 0.0, 0, False, era_label, False, False)
    }
    all_events.append(join_event)
    return all_events, join_time, None, True

def schedule_or_split(case_id: str, options: List[str], k: int, start_time: datetime,
                      resources: Dict[str, Resource], era_label: str, context: dict,
                      prev_res: Optional[Resource]) -> Tuple[List[Dict], datetime, Optional[Resource]]:
    """
    Choose 1..k of options to run in parallel (OR-join on max end).
    """
    choose = rng.integers(1, min(k, len(options)) + 1)
    chosen = list(rng.choice(options, size=choose, replace=False))
    events, _, _, _ = schedule_and_join(case_id, [[c] for c in chosen], start_time, resources, era_label, context, prev_res)
    join_time = max([e["time:timestamp"] for e in events if e["concept:name"] == "CTR_Join"] + [start_time])
    return events, join_time, None

def review_loop(case_id: str, start: datetime, resources: Dict[str, Resource],
                era_label: str, context: dict, prev_res: Optional[Resource]) -> Tuple[List[Dict], datetime, Resource, int]:
    """
    Geometric rework loop: Review -> (Fix -> Review)* -> pass
    Longer tasks increase probability of rework.
    """
    events = []
    t = start
    pr = prev_res
    iter_count = 0
    while True:
        # Review
        evs, t, pr = schedule_activity(case_id, "CTR_Review", t, resources, "role:senior", era_label, context, pr)
        events.extend(evs)
        # Determine rework probability from effective review time spent (proxy by last duration from cost)
        last_duration_h = np.clip(mean_minutes("CTR_Review") / 60.0 * (1.0 + 0.25 * iter_count), 0.1, 6.0)
        prob_rework = min(0.6, 0.25 + 0.08 * iter_count + 0.05 * (last_duration_h > 0.75))
        # Learning / expedite decreases rework
        if context.get("priority") or context.get("expedite"):
            prob_rework *= 0.85
        if rng.random() < prob_rework and iter_count < 3:
            iter_count += 1
            # Fix
            evs, t, pr = schedule_activity(case_id, "CTR_Adjustments", t, resources, "role:analyst", era_label, context, pr)
            # increase rework premium on these events
            for ev in evs:
                ev["amount"] = round(ev["amount"] * (REWORK_PREMIUM ** (iter_count - 1)), 2)
            events.extend(evs)
            continue
        else:
            break
    return events, t, pr, iter_count

def k_of_n_signoffs(case_id: str, n: int, k: int, start_time: datetime,
                    resources: Dict[str, Resource], era_label: str, context: dict,
                    prev_res: Optional[Resource]) -> Tuple[List[Dict], datetime]:
    """
    Spawn n signoffs; proceed after k complete (k-th order statistic).
    Remaining (n-k) are canceled or finish later; we cancel half with credits.
    """
    branch_ends = []
    branch_ev = []
    durations = []

    for i in range(n):
        act_label = f"CTR_Subsidiary_Signoff"
        evs, end_t, _ = schedule_activity(case_id, act_label, start_time, resources, PREF_ROLE.get(act_label), era_label, context, prev_res)
        branch_ev.extend(evs)
        branch_ends.append(end_t)
        durations.append(end_t)

    sorted_ends = sorted(branch_ends)
    kth_time = sorted_ends[min(k - 1, len(sorted_ends) - 1)]
    # Cancel or keep remaining
    for end_t in branch_ends:
        if end_t > kth_time and rng.random() < 0.5:
            c_ts = end_t + timedelta(seconds=1)
            branch_ev.append({
                "concept:name": "CTR_Cancel",
                "time:timestamp": c_ts,
                "org:resource": "System",
                "amount": compute_amount("CTR_Cancel", "role:controller", 0.0, c_ts, False, 0.0, 0, False, era_label, False, False)
            })
    # Join after k
    join_ts = kth_time + timedelta(seconds=rng.uniform(60, 300))
    branch_ev.append({
        "concept:name": "CTR_Join",
        "time:timestamp": join_ts,
        "org:resource": "System",
        "amount": compute_amount("CTR_Join", "role:controller", 60, join_ts, False, 0.0, 0, False, era_label, False, False)
    })
    return branch_ev, join_ts

def add_warn_escalate_if_deadline(case_id: str, t: datetime, deadline: datetime,
                                  resources: Dict[str, Resource], era_label: str, context: dict,
                                  prev_res: Optional[Resource]) -> Tuple[List[Dict], datetime, Optional[Resource]]:
    if t <= deadline:
        return [], t, prev_res
    events = []
    # Warn then Escalate with senior resource; allow overtime to catch up
    evs, t1, pr = schedule_activity(case_id, "CTR_Warn", t, resources, "role:senior", era_label, context, prev_res)
    events.extend(evs)
    evs, t2, pr = schedule_activity(case_id, "CTR_Escalate", t1, resources, "role:senior", era_label, context, pr)
    events.extend(evs)
    # Escalation cost already includes penalty
    context["priority"] = True  # after escalation, treat as priority
    context["expedite"] = True
    return events, t2, pr

# ------------------------------
# Main case constructor
# ------------------------------
def build_case(case_id: str, arrival: datetime, resources: Dict[str, Resource], era_label: str) -> List[Dict]:
    """
    Build one case trace as a list of event dicts (unsorted). Each dict has required attributes and some extras.
    """
    events: List[Dict] = []
    t = arrival
    prev_res: Optional[Resource] = None

    # Case-level context and pattern bundle
    context = {
        "priority": (rng.random() < 0.15),            # influences queue/branch
        "risk_high": (rng.random() < 0.25),
        "bind_duty": (rng.random() < 0.5),
        "sep_duty": (rng.random() < 0.8),
        "expedite": (rng.random() < 0.10),
        "budget": 0.0,
    }
    # SLA deadline (case-level): arrival + 4-6 business days surrogate
    deadline = arrival + timedelta(days=int(rng.integers(4, 7)))

    # --- START & PRE-CLOSE
    for act in ["CTR_Open_Period", "CTR_Pre_Close", "CTR_Risk_Assessment", "CTR_Collect_Data"]:
        evs, t, prev_res = schedule_activity(case_id, act, t, resources, PREF_ROLE.get(act), era_label, context, prev_res)
        events.extend(evs)

    # Optional validation insert
    if rng.random() < 0.8:
        evs, t, prev_res = schedule_activity(case_id, "CTR_Validate_Balances", t, resources, "role:analyst", era_label, context, prev_res)
        events.extend(evs)

    # XOR branching after collect/validate: choose main path
    xor_options = ["entries", "recon", "special"]
    weights = np.array([0.45, 0.35, 0.20], dtype=float)
    # risk/priority influence
    if context["risk_high"]:
        weights = weights * np.array([0.8, 1.2, 1.4])
    if context["priority"]:
        weights = weights * np.array([1.2, 0.9, 0.9])
    weights = weights / weights.sum()
    branch = rng.choice(xor_options, p=weights)

    # --- MAIN PATHS
    had_and = False

    if branch == "entries":
        seq = ["CTR_Post_Journal", "CTR_Accruals", "CTR_Adjustments"]
        # Budget-aware: after spending too much, simplify or vendor
        for act in seq:
            if context["budget"] > BUDGET_LIMIT and rng.random() < 0.5:
                # vendor post instead of normal posting
                evs, t, prev_res = schedule_activity(case_id, "CTR_Vendor_Posting", t, resources, "role:vendor", era_label, context, prev_res)
                events.extend(evs)
            else:
                evs, t, prev_res = schedule_activity(case_id, act, t, resources, PREF_ROLE.get(act), era_label, context, prev_res)
                events.extend(evs)
        # Batch-induced waits mostly happen in posting/adjustments via align_to_release()

    elif branch == "recon":
        # AND-split: Reconcile_GL || Intercompany_Match
        branches = [
            ["CTR_Reconcile_GL"],
            ["CTR_Intercompany_Match"]
        ]
        evs, t, _, had_and = schedule_and_join(case_id, branches, t, resources, era_label, context, prev_res)
        events.extend(evs)
    else:  # "special"
        # OR-split: choose 1..3 among FX, Tax, Interco
        options = ["CTR_FX_Revaluation", "CTR_Tax_Adjustment", "CTR_Intercompany_Match"]
        evs, t, _ = schedule_or_split(case_id, options, k=3, start_time=t, resources=resources, era_label=era_label, context=context, prev_res=prev_res)
        events.extend(evs)

    # --- Multi-resource meeting (requires two roles concurrently)
    if rng.random() < 0.5:
        # choose a senior and a controller; start when both free
        seniors = [r for r in resources.values() if r.role == "role:senior"]
        controllers = [r for r in resources.values() if r.role == "role:controller"]
        if seniors and controllers:
            sr = min(seniors, key=lambda r: r.next_free)
            cr = min(controllers, key=lambda r: r.next_free)
            mt_start = max(t, sr.next_free, cr.next_free)
            if not sr.shift.is_open(mt_start):
                mt_start = sr.shift.next_open(mt_start)
            if not cr.shift.is_open(mt_start):
                mt_start = cr.shift.next_open(mt_start)
            dur = mean_minutes("CTR_Meeting") * 60
            mt_end = sr.shift.add_working_time(mt_start, dur)
            sr.next_free = mt_end
            cr.next_free = mt_end
            events.append({
                "concept:name": "CTR_Meeting",
                "time:timestamp": mt_end,
                "org:resource": f"{sr.rid}&{cr.rid}",
                "amount": compute_amount("CTR_Meeting", "role:senior", dur, mt_start, False, 0.0, 0, False, era_label, False, False)
            })
            t = mt_end

    # --- k-of-n Subsidiary signoffs
    if rng.random() < 0.7:
        n = int(rng.integers(3, 7))
        k = max(1, int(round(n * rng.uniform(0.5, 0.8))))
        evs, t = k_of_n_signoffs(case_id, n, k, t, resources, era_label, context, prev_res)
        events.extend(evs)

    # --- Review loop (rework)
    r_events, t, prev_res, loops = review_loop(case_id, t, resources, era_label, context, prev_res)
    events.extend(r_events)

    # QA may be skipped with memory: if Review was short (proxied by fewer loops), higher skip prob
    if loops == 0 and rng.random() < 0.35:
        # skip QA (stochastic skip with memory)
        pass
    else:
        evs, t, prev_res = schedule_activity(case_id, "CTR_QA", t, resources, PREF_ROLE.get("CTR_QA"), era_label, context, prev_res)
        events.extend(evs)

    # Cooling-off: enforce min lag between Review and Approve
    t = t + timedelta(minutes=10) if rng.random() < 0.4 else t  # pressure might reduce cooling; insert buffer otherwise
    if rng.random() < 0.5:
        # explicit Wait buffer activity
        evs, t, prev_res = schedule_activity(case_id, "CTR_Wait", t, resources, "role:controller", era_label, context, prev_res)
        events.extend(evs)

    # Deadline check before Approve (may warn/escalate)
    esev, t, prev_res = add_warn_escalate_if_deadline(case_id, t, deadline, resources, era_label, context, prev_res)
    events.extend(esev)

    # Approve
    evs, t, prev_res = schedule_activity(case_id, "CTR_Approve", t, resources, "role:senior", era_label, context, prev_res)
    events.extend(evs)

    # Separation-of-duty: Audit by a different resource
    if rng.random() < 0.6:
        evs, t, prev_res = schedule_activity(case_id, "CTR_Audit", t, resources, "role:auditor", era_label, context, prev_res)
        events.extend(evs)

    # Consolidate (can be preempted)
    evs, t, prev_res = schedule_activity(case_id, "CTR_Consolidate", t, resources, "role:controller", era_label, context, prev_res)
    events.extend(evs)

    # Publish & Close
    if rng.random() < 0.85:
        evs, t, prev_res = schedule_activity(case_id, "CTR_Publish_Reports", t, resources, "role:controller", era_label, context, prev_res)
        events.extend(evs)
    evs, t, prev_res = schedule_activity(case_id, "CTR_Close_Period", t, resources, "role:controller", era_label, context, prev_res)
    events.extend(evs)

    # Budget-aware rollback/credits occasionally
    if context["budget"] > BUDGET_LIMIT * 1.2 and rng.random() < 0.3:
        evs, t, prev_res = schedule_activity(case_id, "CTR_Rollback", t, resources, "role:controller", era_label, context, prev_res)
        events.extend(evs)

    # Mark AND usage
    if had_and:
        context["had_and"] = True
    return events

# ------------------------------
# Era lookup
# ------------------------------
def era_for(ts: datetime) -> str:
    for label, s, e in ERA_DEFS:
        if s <= ts <= e:
            return label
    # default last era
    return ERA_DEFS[-1][0]

# ------------------------------
# Quality / sanity checks and summaries
# ------------------------------
def sort_and_fix_monotonic(events: List[Dict]) -> List[Dict]:
    """Sort by time and guarantee strictly increasing timestamps by +1s nudge if needed."""
    evs = sorted(events, key=lambda e: e["time:timestamp"])
    prev = None
    for i, e in enumerate(evs):
        if prev and e["time:timestamp"] <= prev:
            e["time:timestamp"] = prev + timedelta(seconds=1)
        prev = e["time:timestamp"]
        # Ensure required attributes
        if "org:resource" not in e:
            e["org:resource"] = "Unknown"
        if "amount" not in e:
            e["amount"] = float(rng.uniform(1.0, 50.0))
        # Coerce amount to float
        e["amount"] = float(e["amount"])
    return evs

def compute_trace_stats(tr_events: List[Dict]) -> Dict:
    start = tr_events[0]["time:timestamp"]
    end = tr_events[-1]["time:timestamp"]
    total_cost = sum(e["amount"] for e in tr_events)
    return {
        "n_events": len(tr_events),
        "duration_days": (end - start).total_seconds() / 86400.0,
        "total_cost": total_cost
    }

def prefix_ambiguity(traces: List[List[Dict]], samples: int = 100) -> float:
    """
    Sample prefixes and compute fraction where >= 2 distinct next activities exist.
    """
    # Build prefix->set(next_act) map
    mapping = defaultdict(set)
    for tr in traces:
        names = [e["concept:name"] for e in tr]
        for i in range(1, len(names)):
            prefix = tuple(names[:i])
            mapping[prefix].add(names[i])
    if not mapping:
        return 0.0
    keys = list(mapping.keys())
    rng.shuffle(keys)
    keys = keys[:min(samples, len(keys))]
    amb = [1.0 if len(mapping[k]) >= 2 else 0.0 for k in keys]
    return float(np.mean(amb)) if amb else 0.0

def summaries(traces: List[List[Dict]], had_and_ratio: float):
    lens = [len(tr) for tr in traces]
    costs = [sum(e["amount"] for e in tr) for tr in traces]
    durs = [(tr[-1]["time:timestamp"] - tr[0]["time:timestamp"]).total_seconds()/3600.0 for tr in traces]
    if HAVE_PANDAS:
        df = pd.DataFrame({"len": lens, "cost": costs, "dur_h": durs})
        quant_df = df.quantile([0.1, 0.5, 0.9])
        mean_cost = df["cost"].mean()
        std_cost = df["cost"].std()
        len_quants = {f"{int(q*100)}%": round(quant_df.loc[q, "len"], 2) for q in quant_df.index}
        dur_quants = {f"{int(q*100)}%": round(quant_df.loc[q, "dur_h"], 2) for q in quant_df.index}
        print("\n--- Distribution summaries ---")
        print("Trace length (10/50/90%):", len_quants)
        print("Cost (mean±std):", round(mean_cost, 2), "±", round(std_cost, 2))
        print("Duration hours (10/50/90%):", dur_quants)
    else:
        print("\n--- Distribution summaries (basic) ---")
        print(f"Trace length: mean={np.mean(lens):.2f}, q10={np.quantile(lens,0.1):.1f}, q90={np.quantile(lens,0.9):.1f}")
        print(f"Cost: mean={np.mean(costs):.2f}, q10={np.quantile(costs,0.1):.2f}, q90={np.quantile(costs,0.9):.2f}")
        print(f"Duration(h): mean={np.mean(durs):.2f}, q10={np.quantile(durs,0.1):.2f}, q90={np.quantile(durs,0.9):.2f}")

    # Ambiguity and regression signals
    amb = prefix_ambiguity(traces, 120)
    print(f"Ambiguity check: fraction of sampled prefixes with ≥2 next activities: {amb:.2f}")
    print(f"Parallelism check: % traces with ≥1 AND-split: {had_and_ratio*100:.1f}%")

    # Era-wise drift
    era_stats = defaultdict(lambda: {"dur": [], "cost": []})
    for tr in traces:
        start = tr[0]["time:timestamp"]
        e = era_for(start)
        era_stats[e]["dur"].append((tr[-1]["time:timestamp"] - start).total_seconds()/3600.0)
        era_stats[e]["cost"].append(sum(ev["amount"] for ev in tr))
    print("\n--- Era-wise drift ---")
    for e, s in era_stats.items():
        if s["dur"]:
            print(f"{e}: mean duration(h)={np.mean(s['dur']):.2f}, mean cost={np.mean(s['cost']):.2f}")

# ------------------------------
# Export
# ------------------------------
def export_xes(log: EventLog, path: str):
    if HAVE_PM4PY_WRITE:
        pm4py.write_xes(log, path)
    elif xes_exporter is not None:
        xes_exporter.apply(log, path)
    else:
        raise RuntimeError("No XES exporter found in pm4py.")

# ------------------------------
# Minimal example generation (for reference only; not used in main)
# ------------------------------
def minimal_example_trace(resources: Dict[str, Resource], start: datetime) -> List[Dict]:
    """
    One compact trace that demonstrates:
      - AND-split (Reconcile || Intercompany) with join
      - Handover delay (switch resource)
      - SLA penalty -> Escalate (deadline exceeded)
      - Rework once (Review->Adjustments->Review)
      - Vendor path with overtime
    """
    case_id = "CTR_DEMO"
    era_label = era_for(start)
    t = start
    prev_res = None
    ctx = {"priority": False, "risk_high": True, "bind_duty": True, "sep_duty": True, "expedite": False, "budget": 0.0}
    evs = []
    for a in ["CTR_Open_Period", "CTR_Pre_Close", "CTR_Collect_Data", "CTR_Validate_Balances"]:
        x, t, prev_res = schedule_activity(case_id, a, t, resources, PREF_ROLE.get(a), era_label, ctx, prev_res)
        evs.extend(x)
    # AND
    x, t, _, _ = schedule_and_join(case_id, [["CTR_Reconcile_GL"], ["CTR_Intercompany_Match"]], t, resources, era_label, ctx, prev_res)
    evs.extend(x)
    # Exceed deadline -> escalate
    deadline = start + timedelta(days=1)
    x, t, prev_res = add_warn_escalate_if_deadline(case_id, t + timedelta(days=2), deadline, resources, era_label, ctx, prev_res)
    evs.extend(x)
    # Review -> rework once -> Review
    x, t, prev_res, _ = review_loop(case_id, t, resources, era_label, ctx, prev_res)
    evs.extend(x)
    # Vendor with OT
    x, t, prev_res = schedule_activity(case_id, "CTR_Vendor_Posting", t, resources, "role:vendor", era_label, ctx, prev_res)
    evs.extend(x)
    # Consolidate / Approve / Close
    for a in ["CTR_Consolidate", "CTR_Approve", "CTR_Close_Period"]:
        x, t, prev_res = schedule_activity(case_id, a, t, resources, PREF_ROLE.get(a), era_label, ctx, prev_res)
        evs.extend(x)
    return sort_and_fix_monotonic(evs)

# ------------------------------
# Main
# ------------------------------
def execute_script():
    # Global start anchor for resource next_free initialization
    global_start = ERA_DEFS[0][1]
    resources = make_resources(global_start)

    # Case arrivals (exactly N_CASES)
    arrivals = sample_arrival_times(N_CASES)

    # Build traces case-by-case in arrival order to create realistic resource contention
    traces_events: List[List[Dict]] = []
    had_and_flags = []

    for i, arr in enumerate(arrivals):
        case_id = f"CTR-{i+1:04d}"
        era_label = era_for(arr)
        evs = build_case(case_id, arr, resources, era_label)
        evs = sort_and_fix_monotonic(evs)
        traces_events.append(evs)
        had_and_flags.append(any(e["concept:name"] == "CTR_Join" for e in evs))

    # Sanity checks
    # Schema: all events must have required keys and types
    for tr in traces_events:
        for e in tr:
            assert "concept:name" in e and "time:timestamp" in e and "org:resource" in e and "amount" in e
            assert isinstance(e["time:timestamp"], datetime)
            assert isinstance(e["org:resource"], str)
            assert isinstance(e["amount"], float)

        # strictly increasing timestamps within each trace
        for a, b in zip(tr, tr[1:]):
            assert b["time:timestamp"] > a["time:timestamp"]

    # Build EventLog
    elog = EventLog()
    for i, tr in enumerate(traces_events):
        t = Trace()
        case_id = f"CTR-{i+1:04d}"
        t.attributes["concept:name"] = case_id
        for e in tr:
            # Ensure only required keys are set (plus optional org:role if present)
            ev_dict = {
                "concept:name": e["concept:name"],
                "time:timestamp": e["time:timestamp"],
                "org:resource": e["org:resource"],
                "amount": float(e["amount"])
            }
            if "org:role" in e:
                ev_dict["org:role"] = e["org:role"]
            t.append(Event(ev_dict))
        elog.append(t)

    # Export
    export_xes(elog, OUT_PATH)

    # Print summary
    print(f"\nSaved Close-to-Report log with {len(elog)} traces to: {OUT_PATH}")
    # Summaries
    had_and_ratio = float(np.mean(had_and_flags)) if had_and_flags else 0.0
    summaries(traces_events, had_and_ratio)

    # Extra: regression signal (remaining time distribution spread)
    all_rem = []
    for tr in traces_events:
        end = tr[-1]["time:timestamp"]
        for e in tr[:-1]:
            rem = (end - e["time:timestamp"]).total_seconds() / 3600.0
            all_rem.append(rem)
    if all_rem:
        print("\nRemaining time (hours) across prefixes: mean={:.2f}, std={:.2f}, q10={:.2f}, q90={:.2f}".format(
            float(np.mean(all_rem)), float(np.std(all_rem)),
            float(np.quantile(all_rem, 0.1)), float(np.quantile(all_rem, 0.9))
        ))

if __name__ == "__main__":
    execute_script()
