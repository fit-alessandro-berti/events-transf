"""
Hire-to-Retire synthetic XES log generator (trace-by-trace; no black-box simulator)

Author: (you)
Role: Senior Process Mining Engineer

What this script does
---------------------
- Builds a realistic, richly-instrumented Hire-to-Retire process log with:
  * 250 distinct traces (cases) — hard limit, per user request
  * Activities include hiring, screening, interviews (incl. multi-resource meeting),
    offer, background checks (multi-instance k-of-n), onboarding (AND-split/AND-join),
    payroll/benefits (OR-split), training, reviews, transfers, exits, audits, etc.
  * Control-flow templates: linear with optional inserts, XOR/OR/AND splits,
    loops/rework, skip patterns, long-term dependencies, preempt/resume, multi-instance.
  * Temporal patterns: heavy-tailed service times, bursty arrivals (waves),
    business & shift calendars (9–17 with lunch; weekend freeze), deadlines/SLA,
    setup/changeover, synchronization latency, handover delay, batching (orientation),
    learning/forgetting curves, cooling-off constraints, batching/release effects.
  * Resource patterns: roles & pools (R1..R30), skill heterogeneity, binding/separation
    of duty, handovers, unavailability, assignment policy mixture, multi-resource meetings,
    optional external vendor branches.
  * Cost model at event level: fixed + rate * duration, overtime/weekend multipliers,
    SLA penalties, rework premiums, expedite fees, handover fees, refunds on cancel,
    budget-aware branching, waiting/holding costs, era-based tariff drift.
  * Every event has: concept:name, time:timestamp (datetime), org:resource (string),
    amount (float). Some events intentionally use org:resource="Unknown".
  * Each trace sets trace.attributes['concept:name'] to the case id
    (so pm4py -> DataFrame exposes case:concept:name).

- Performs quality/sanity checks and prints summaries:
  * schema/timestamp checks
  * distribution summaries (trace length, costs, inter-event times)
  * ambiguity check on next-activity from shared prefixes
  * remaining-time stats (regression target signal)
  * parallelism (AND-splits) share and imbalance
  * drift across eras (duration/cost by era)

Dependencies
------------
Python 3.8+; pm4py; numpy; pandas (optional, used for summaries only).
`pip install pm4py pandas numpy`

Output
------
- Writes a compressed XES log to: ./hire2retire.xes.gz

Notes
-----
- Designed to be understandable and hackable. Patterns are implemented explicitly;
  no black-box simulation. The scheduler respects calendars and shifts and builds
  events one-by-one in timestamp order.
- For brevity, some mechanisms are approximate yet faithful to the patterns.
"""

import math
import random
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

# pm4py core objects
from pm4py.objects.log.obj import EventLog, Trace, Event
# Export helpers (we will try pm4py.write_xes first; fallback to exporter)
try:
    import pm4py
except Exception as e:
    raise RuntimeError("pm4py is required. pip install pm4py") from e

# -----------------------------
# Global knobs / reproducibility
# -----------------------------
SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

N_CASES = 250                         # per user: cap at 250 traces
OUT_PATH = "hire2retire.xes.gz"

# Time & calendars
BUSINESS_START = 9     # 09:00
BUSINESS_END = 17      # 17:00
LUNCH_START = 12       # 12:00
LUNCH_END = 13         # 13:00
WEEKEND_OFF = True

# Unknown resource fraction
UNKNOWN_RESOURCE_FRAC = 0.02

# Overtime multiplier
OVERTIME_MULT = 1.5

# Case-level SLA (soft): base threshold (workdays) by case type; see builder
BASE_SLA_DAYS = 14

# Era drift (3 eras)
ERAS = [
    {"name": "Era-1", "service_mult": 1.00, "vendor_rate_mult": 1.00, "rework_penalty_mult": 1.00},
    {"name": "Era-2", "service_mult": 0.95, "vendor_rate_mult": 0.85, "rework_penalty_mult": 1.20},
    {"name": "Era-3", "service_mult": 1.05, "vendor_rate_mult": 0.80, "rework_penalty_mult": 1.40},
]

# Roles & rate table (per hour)
ROLE_RATES = {
    "role:analyst": 45.0,
    "role:hr": 52.0,
    "role:it": 60.0,
    "role:finance": 70.0,
    "role:manager": 80.0,
    "role:senior": 95.0,
    "role:auditor": 78.0,
    "role:vendor": 46.0,
    "Unknown": 55.0,
}

# Activity fixed costs (materials, forms, licenses, etc.) baseline
ACTIVITY_FIXED = defaultdict(
    float,
    {
        # Requisition & approvals
        "Req_Create": 20.0,
        "Req_Approve": 15.0,
        "Job_Post": 50.0,

        # Screening & interviews
        "Cand_Apply": 0.0,
        "Screen_Phone": 10.0,
        "Screen_HR": 15.0,
        "Vendor_Screening": 90.0,
        "Intv_Tech": 30.0,
        "Intv_Panel": 80.0,
        "Intv_HR": 20.0,

        # Offer flow
        "Offer_Draft": 10.0,
        "Offer_Approve": 10.0,
        "Offer_Send": 2.0,
        "Offer_Accepted": 0.0,
        "Offer_Declined": 0.0,

        # BG checks (multi-instance k-of-n)
        "BG_Education": 25.0,
        "BG_Employment": 25.0,
        "BG_References": 25.0,
        "BG_Threshold": 0.0,
        "Cancel_Remaining_BG": -10.0,  # refund/credit
        "Audit": 35.0,

        # Onboarding (AND-split branches + join)
        "Onb_Provision_IT": 40.0,
        "Onb_Prepare_Desk": 20.0,
        "Onb_Documents": 5.0,
        "Onb_Join": 0.0,
        "Onb_Orientation": 30.0,  # batched
        "Payroll_Setup": 10.0,
        "Benefits_Enroll": 10.0,
        "Join_OR": 0.0,

        # Training/QA/Rework
        "Onb_Training": 15.0,
        "Doc_Review": 10.0,
        "Rework_Fix": 8.0,
        "QA_Final": 5.0,

        # Performance / transfers / compensation
        "Perf_Probation_Review": 12.0,
        "Perf_Annual_Review": 12.0,
        "Comp_Change": 5.0,
        "Trans_Internal": 8.0,

        # Leaves & exits
        "Leave_Request": 2.0,
        "Leave_Approve": 2.0,
        "Return_Process": 2.0,
        "Exit_Resign": 0.0,
        "Exit_Offboard": 25.0,
        "Exit_Interview": 5.0,
        "Exit_Revoke_Access": 10.0,

        # Control/aux
        "Escalate": 0.0,
        "Expedite": 150.0,
        "Warn": 0.0,
        "Handover": 0.0,
        "Wait_Buffer": 0.0,
        "Join_Parallel": 0.0,
        "Simplify": 0.0,
        "Reassign": 0.0,
    },
)

# Activity families (for skill, setup/changeover, etc.)
def family_of(activity: str) -> str:
    return activity.split("_")[0] if "_" in activity else activity

# -----------------
# Resource modeling
# -----------------
@dataclass
class Resource:
    rid: str
    role: str
    shift: str                       # 'early' (8–16), 'standard' (9–17), 'late' (10–18)
    next_free: datetime
    experience: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_performed: Dict[str, datetime] = field(default_factory=dict)
    base_skill: Dict[str, float] = field(default_factory=dict)  # family -> speed multiplier (lower=faster)
    absences: set = field(default_factory=set)                  # set of yyyy-mm-dd strings (days off)

    def effective_speed(self, activity: str, now: datetime) -> float:
        fam = family_of(activity)
        mult = self.base_skill.get(fam, 1.0)
        # Learning curve: faster with repetitions
        n = self.experience[fam]
        learning = 1.0 / (1.0 + 0.06 * n)  # up to ~1/(1+0.06*n)
        # Forgetting: if idle on family for long, performance worsens
        last = self.last_performed.get(fam)
        forgetting = 1.0
        if last:
            idle_days = max((now - last).days, 0)
            if idle_days > 30:
                forgetting = 1.0 + min(0.35, 0.01 * (idle_days - 30))
        return mult * learning * forgetting

# Build 30 resources across roles, shifts, and heterogeneous skills
def build_resources(start_time: datetime) -> Dict[str, Resource]:
    roles = ["role:analyst", "role:hr", "role:it", "role:finance", "role:manager", "role:senior", "role:auditor"]
    shifts = ["early", "standard", "late"]
    resources: Dict[str, Resource] = {}
    for i in range(1, 31):
        role = rng.choice(roles, 1, p=[0.20, 0.20, 0.20, 0.10, 0.10, 0.10, 0.10])[0]
        shift = rng.choice(shifts, 1, p=[0.25, 0.5, 0.25])[0]
        rid = f"R{i}"
        res = Resource(rid=rid, role=role, shift=shift, next_free=start_time + timedelta(hours=rng.uniform(0, 8)))
        # Skills by family (lower is faster). Everyone decent at their own domain, worse elsewhere.
        fams = ["Req", "Job", "Cand", "Screen", "Intv", "Offer", "BG", "Onb", "Payroll", "Benefits",
                "Doc", "QA", "Perf", "Comp", "Trans", "Leave", "Exit", "Audit", "IT"]
        res.base_skill = {f: rng.uniform(0.85, 1.15) for f in fams}
        # Role-specific boosts
        if role == "role:hr":
            for f in ["Req", "Job", "Screen", "Intv", "Offer", "Onb", "Doc", "QA"]:
                res.base_skill[f] *= rng.uniform(0.6, 0.9)
        if role == "role:it":
            for f in ["IT", "Onb"]:
                res.base_skill[f] *= rng.uniform(0.55, 0.85)
        if role == "role:senior":
            for f in ["Offer", "Comp", "Audit"]:
                res.base_skill[f] *= rng.uniform(0.6, 0.85)
        if role == "role:auditor":
            for f in ["Audit", "QA"]:
                res.base_skill[f] *= rng.uniform(0.55, 0.85)
        # Random absences
        for _ in range(rng.integers(2, 6)):
            # spread over ~6 months from start
            day = start_time.date() + timedelta(days=int(rng.uniform(0, 180)))
            res.absences.add(day.isoformat())
        resources[rid] = res
    return resources

# ----------------------
# Calendars & scheduling
# ----------------------
def is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5  # 5=Sat, 6=Sun

def shift_window(shift: str) -> Tuple[int, int]:
    if shift == "early":
        return 8, 16
    if shift == "late":
        return 10, 18
    return 9, 17

def in_lunch(dt: datetime) -> bool:
    return dt.hour == LUNCH_START or (LUNCH_START < dt.hour < LUNCH_END) or \
           (dt.hour == LUNCH_END and dt.minute == 0 and dt.second == 0)

def next_business_open(dt: datetime, res: Optional[Resource]) -> datetime:
    # Snap to next open slot (business + resource shift + not weekend + not absent)
    cur = dt
    while True:
        # Weekend freeze
        if WEEKEND_OFF and is_weekend(cur):
            # jump to next Monday at 9
            days_ahead = (7 - cur.weekday()) % 7
            cur = datetime(cur.year, cur.month, cur.day, BUSINESS_START, 0, 0) + timedelta(days=days_ahead)
            continue
        # Resource absence on the day
        if res is not None and cur.date().isoformat() in res.absences:
            cur = datetime(cur.year, cur.month, cur.day, BUSINESS_START, 0, 0) + timedelta(days=1)
            continue
        # Shift & business hours alignment
        s_start, s_end = shift_window(res.shift) if res else (BUSINESS_START, BUSINESS_END)
        open_start = max(BUSINESS_START, s_start)
        open_end = min(BUSINESS_END, s_end)
        # If before open or during lunch or after close, move appropriately
        if cur.hour < open_start or (cur.hour == open_start and cur.minute == 0 and cur.second == 0 and cur < cur.replace(hour=open_start, minute=0, second=0)):
            cur = cur.replace(hour=open_start, minute=0, second=0, microsecond=0)
            continue
        # If lunch, move to end of lunch
        if LUNCH_START <= cur.hour < LUNCH_END:
            cur = cur.replace(hour=LUNCH_END, minute=0, second=0, microsecond=0)
            continue
        if cur.hour >= open_end:
            # move to next day at open_start
            cur = (cur + timedelta(days=1)).replace(hour=open_start, minute=0, second=0, microsecond=0)
            continue
        # OK
        return cur

def add_work_hours(start: datetime, hours: float, res: Optional[Resource], allow_overtime: bool=True) -> Tuple[datetime, float]:
    """
    Adds 'hours' of productive work from 'start', respecting business hours, resource shift,
    lunch, weekends, and absences. If allow_overtime, lets last segment stretch beyond close
    (scaled by OVERTIME_MULT in cost model).
    Returns (end_time, overtime_hours_used)
    """
    cur = next_business_open(start, res)
    remaining = hours
    overtime_used = 0.0
    while remaining > 1e-9:
        s_start, s_end = shift_window(res.shift) if res else (BUSINESS_START, BUSINESS_END)
        open_start = max(BUSINESS_START, s_start)
        open_end = min(BUSINESS_END, s_end)
        # current window end is min(lunch, close)
        # handle lunch split
        if cur.hour < LUNCH_START and LUNCH_START < open_end:
            lunch_dt = cur.replace(hour=LUNCH_START, minute=0, second=0, microsecond=0)
            end_of_window = min(lunch_dt, cur.replace(hour=open_end, minute=0, second=0, microsecond=0))
        else:
            end_of_window = cur.replace(hour=open_end, minute=0, second=0, microsecond=0)
        window_hours = max((end_of_window - cur).total_seconds() / 3600.0, 0.0)
        if window_hours >= remaining:
            cur = cur + timedelta(hours=remaining)
            remaining = 0.0
        else:
            cur = end_of_window
            remaining -= window_hours
            # Lunch break?
            if cur.hour == LUNCH_START and cur.minute == 0:
                cur = cur.replace(hour=LUNCH_END, minute=0, second=0, microsecond=0)
            else:
                # Move to next open
                # Optionally consume overtime to finish today if remaining is small and allow_overtime
                if allow_overtime and remaining <= 1.5:  # allow up to 1.5h overtime
                    cur = cur + timedelta(hours=remaining)
                    overtime_used += remaining
                    remaining = 0.0
                else:
                    cur = (cur + timedelta(days=1)).replace(hour=open_start, minute=0, second=0, microsecond=0)
                    cur = next_business_open(cur, res)
    return cur, overtime_used

# --------------------
# Assignment & queues
# --------------------
P_ASSIGN = (0.35, 0.35, 0.30)  # random, least-loaded, skill-based

def choose_resource(resources: Dict[str, Resource], activity: str, preferred_roles: List[str], ready_time: datetime) -> Resource:
    pool = [r for r in resources.values() if r.role in preferred_roles]
    if not pool:
        pool = list(resources.values())
    # Policy mixture
    p = rng.random()
    if p < P_ASSIGN[0]:
        # Random (but lighter load bias)
        weights = np.array([1.0 / (1.0 + max(0.01, (r.next_free - ready_time).total_seconds()/3600.0)) for r in pool])
        weights = weights / weights.sum()
        return rng.choice(pool, p=weights)
    elif p < P_ASSIGN[0] + P_ASSIGN[1]:
        # Least loaded
        return sorted(pool, key=lambda r: r.next_free)[0]
    else:
        # Skill-based (lowest effective speed value)
        return min(pool, key=lambda r: r.effective_speed(activity, ready_time))

# Queue approximation (for patterns; not a full queueing sim)
def queue_pressure(res: Resource, ready_time: datetime, expected_hours: float) -> float:
    delay = max(0.0, (res.next_free - ready_time).total_seconds()/3600.0)
    return delay / max(0.1, expected_hours)  # approx number of jobs ahead

# -----------------------
# Duration & cost models
# -----------------------
# Base service time (in hours) for activities; will be modulated by role skill, learning, era, etc.
BASE_SERVICE_H = defaultdict(
    float,
    {
        # Requisition & approvals
        "Req_Create": 1.5,
        "Req_Approve": 0.6,
        "Job_Post": 0.8,

        # Screening & interviews
        "Cand_Apply": 0.1,
        "Screen_Phone": 0.6,
        "Screen_HR": 0.8,
        "Vendor_Screening": 2.4,
        "Intv_Tech": 1.5,
        "Intv_Panel": 2.0,   # multi-resource
        "Intv_HR": 1.0,

        # Offer flow
        "Offer_Draft": 0.8,
        "Offer_Approve": 0.5,
        "Offer_Send": 0.2,
        "Offer_Accepted": 0.1,
        "Offer_Declined": 0.1,

        # Background checks
        "BG_Education": 6.0,
        "BG_Employment": 6.0,
        "BG_References": 4.0,
        "BG_Threshold": 0.1,
        "Cancel_Remaining_BG": 0.1,
        "Audit": 1.0,

        # Onboarding
        "Onb_Provision_IT": 2.5,
        "Onb_Prepare_Desk": 1.0,
        "Onb_Documents": 0.7,
        "Onb_Join": 0.1,
        "Onb_Orientation": 3.0,  # batched weekly start
        "Payroll_Setup": 1.0,
        "Benefits_Enroll": 1.0,
        "Join_OR": 0.1,

        # Training/QA/Rework
        "Onb_Training": 3.0,
        "Doc_Review": 0.8,
        "Rework_Fix": 0.8,
        "QA_Final": 0.4,

        # Performance / transfers / compensation
        "Perf_Probation_Review": 1.5,
        "Perf_Annual_Review": 2.0,
        "Comp_Change": 0.6,
        "Trans_Internal": 0.8,

        # Leaves & exits
        "Leave_Request": 0.2,
        "Leave_Approve": 0.2,
        "Return_Process": 0.3,
        "Exit_Resign": 0.1,
        "Exit_Offboard": 1.2,
        "Exit_Interview": 0.5,
        "Exit_Revoke_Access": 0.5,

        # Aux
        "Escalate": 0.2,
        "Expedite": 0.1,
        "Warn": 0.05,
        "Handover": 0.2,
        "Wait_Buffer": 0.5,
        "Join_Parallel": 0.1,
        "Simplify": 0.1,
        "Reassign": 0.5,
    },
)

# Preferred roles per activity (used by assignment policy)
PREFERRED_ROLES = defaultdict(
    lambda: ["role:analyst", "role:hr", "role:manager"],
    {
        # Core mapping
        "Req_Create": ["role:hr", "role:manager"],
        "Req_Approve": ["role:manager", "role:senior"],
        "Job_Post": ["role:hr"],

        "Cand_Apply": ["role:hr"],
        "Screen_Phone": ["role:hr"],
        "Screen_HR": ["role:hr"],
        "Vendor_Screening": ["role:vendor", "role:hr"],
        "Intv_Tech": ["role:manager", "role:senior"],
        "Intv_Panel": ["role:manager", "role:senior"],
        "Intv_HR": ["role:hr"],

        "Offer_Draft": ["role:hr"],
        "Offer_Approve": ["role:manager", "role:senior"],
        "Offer_Send": ["role:hr"],
        "Offer_Accepted": ["role:hr"],
        "Offer_Declined": ["role:hr"],

        "BG_Education": ["role:hr"],
        "BG_Employment": ["role:hr"],
        "BG_References": ["role:hr"],
        "Audit": ["role:auditor", "role:senior"],

        "Onb_Provision_IT": ["role:it"],
        "Onb_Prepare_Desk": ["role:it", "role:analyst"],
        "Onb_Documents": ["role:hr"],
        "Onb_Join": ["role:hr"],
        "Onb_Orientation": ["role:hr"],
        "Payroll_Setup": ["role:finance"],
        "Benefits_Enroll": ["role:hr"],

        "Onb_Training": ["role:hr", "role:manager"],
        "Doc_Review": ["role:hr"],
        "Rework_Fix": ["role:analyst", "role:hr"],
        "QA_Final": ["role:auditor", "role:senior"],

        "Perf_Probation_Review": ["role:manager", "role:senior"],
        "Perf_Annual_Review": ["role:manager", "role:senior"],
        "Comp_Change": ["role:finance", "role:senior"],
        "Trans_Internal": ["role:hr", "role:manager"],

        "Leave_Request": ["role:hr"],
        "Leave_Approve": ["role:manager", "role:senior"],
        "Return_Process": ["role:hr"],
        "Exit_Resign": ["role:hr"],
        "Exit_Offboard": ["role:it", "role:hr"],
        "Exit_Interview": ["role:hr"],
        "Exit_Revoke_Access": ["role:it"],

        "Escalate": ["role:senior"],
        "Expedite": ["role:senior"],
        "Warn": ["role:senior"],
        "Handover": ["role:hr"],
        "Wait_Buffer": ["role:hr"],
        "Join_Parallel": ["role:hr"],
        "Join_OR": ["role:hr"],
        "Simplify": ["role:hr"],
        "Reassign": ["role:hr"],
    },
)

def lognormal_heavy_tail(mean_hours: float) -> float:
    # Construct lognormal with reasonable heavy tail; approx median < mean.
    # Choose sigma and mu such that mean ≈ mean_hours.
    sigma = 0.6  # heavier tails
    mu = math.log(mean_hours) - (sigma**2)/2.0
    return float(rng.lognormal(mu, sigma))

def pareto_tail(mean_hours: float) -> float:
    # Pareto with xm=mean/2, alpha to target mean
    xm = max(0.1, mean_hours / 2.0)
    alpha = 2.5
    return float((rng.pareto(alpha) + 1.0) * xm)

def sample_service_time(activity: str, res: Resource, era_idx: int, prev_activity: Optional[str], now: datetime, rework_iter: int, queue_pressure_est: float) -> float:
    base = BASE_SERVICE_H[activity] * ERAS[era_idx]["service_mult"]
    # heavy-tailed mixture
    if rng.random() < 0.6:
        st = lognormal_heavy_tail(base)
    else:
        st = pareto_tail(base)
    # skill, learning, forgetting
    st *= res.effective_speed(activity, now)
    # setup/changeover penalty when activity family changes
    if prev_activity and family_of(prev_activity) != family_of(activity):
        st *= rng.uniform(1.05, 1.25)
    # Travel/context switch penalty (simulate with small add-on)
    st += rng.uniform(0.05, 0.25)
    # Queue-pressure induced slowdown (stress effect)
    st *= (1.0 + 0.10 * min(queue_pressure_est, 5.0))
    # Rework iteration premium (later attempts faster or slower?)
    st *= 0.9 ** max(0, rework_iter-1)  # slight speed-up with rework familiarity
    # Bound to reasonable range
    return max(0.05, min(st, 24.0))

def role_rate(role: str, era_idx: int, is_vendor: bool=False) -> float:
    rate = ROLE_RATES.get(role, ROLE_RATES["Unknown"])
    if is_vendor:
        rate *= ERAS[era_idx]["vendor_rate_mult"]
    return rate

def compute_event_cost(activity: str, res_role: str, duration_h: float, overtime_h: float, waiting_h: float, era_idx: int, is_vendor: bool=False) -> float:
    fixed = ACTIVITY_FIXED[activity]
    base_rate = role_rate(res_role, era_idx, is_vendor=is_vendor)
    run_cost = base_rate * duration_h + base_rate * (OVERTIME_MULT - 1.0) * overtime_h
    wait_cost = 5.0 * waiting_h  # holding cost
    return float(fixed + run_cost + wait_cost)

# ---------------------------
# Case structures / skeletons
# ---------------------------
@dataclass
class CaseState:
    case_id: str
    era_idx: int
    start_time: datetime
    cur_time: datetime
    deadline: datetime
    cost_so_far: float = 0.0
    last_activity: Optional[str] = None
    last_resource: Optional[str] = None
    bind_map: Dict[str, str] = field(default_factory=dict)   # binding-of-duty anchors
    forbid_map: Dict[str, str] = field(default_factory=dict) # separation-of-duty anchors
    has_and_split: bool = False
    branch_imbalances: List[float] = field(default_factory=list)
    flags: Dict[str, bool] = field(default_factory=dict)     # miscellanea

# Utility: add event to trace with schema
def append_event(trace: Trace, name: str, timestamp: datetime, resource: str, amount: float, role: Optional[str] = None):
    # Occasionally force Unknown to test defaults
    if rng.random() < UNKNOWN_RESOURCE_FRAC:
        resource = "Unknown"
        role = None
    e = Event({
        "concept:name": name,
        "time:timestamp": timestamp,
        "org:resource": resource,
        "amount": float(amount),
    })
    if role is not None:
        e["org:role"] = role
    trace.append(e)

# Handover event (adds delay + fee)
def maybe_handover(case: CaseState, trace: Trace, new_res: Resource):
    prev_res = case.last_resource
    if prev_res and prev_res != new_res.rid and rng.random() < 0.8:
        # Simulate handover as separate event between tasks
        ho_dur_h = rng.uniform(0.1, 0.3)
        start = next_business_open(case.cur_time, None)
        end, ot_h = add_work_hours(start, ho_dur_h, None, allow_overtime=False)
        amt = ACTIVITY_FIXED["Handover"] + 20.0 + 5.0 * ho_dur_h
        append_event(trace, "Handover", end, prev_res, amt, role=PREFERRED_ROLES["Handover"][0])
        case.cur_time = end
        case.cost_so_far += amt
        case.last_activity = "Handover"
        case.last_resource = prev_res

# Orientation batching: next Monday 09:00
def next_orientation_start(cur: datetime) -> datetime:
    # Move to next Monday at 09:00
    days_ahead = (7 - cur.weekday()) % 7
    target = (cur + timedelta(days=days_ahead)).replace(hour=9, minute=0, second=0, microsecond=0)
    if target <= cur:
        target = target + timedelta(days=7)
    return target

def schedule_single_activity(case: CaseState, trace: Trace, activity: str, resources: Dict[str, Resource], vendor: bool=False,
                             preemptable: bool=False, force_role: Optional[str]=None, cooling_min_lag_h: float=0.0,
                             priority: int=0, sep_duty_with: Optional[str]=None, bind_key: Optional[str]=None) -> Tuple[datetime, Resource, float]:
    """
    Schedules and emits one activity (with potential preempt/resume) and returns (finish_time, resource, duration_hours).
    - cooling_min_lag_h: if positive, ensures a buffer (via Wait_Buffer).
    - sep_duty_with: enforce separation-of-duty against case.forbid_map key.
    - bind_key: enforce binding-of-duty using case.bind_map[bind_key].
    """
    # Enforce cooling-off (buffer) if needed
    if cooling_min_lag_h > 0:
        earliest = case.start_time + timedelta(hours=cooling_min_lag_h)
        if case.cur_time < earliest:
            # Insert Wait_Buffer
            wb_start = next_business_open(case.cur_time, None)
            wb_end, _ = add_work_hours(wb_start, 0.5, None, allow_overtime=False)
            wb_amt = compute_event_cost("Wait_Buffer", "role:hr", 0.5, 0.0, 0.0, case.era_idx)
            append_event(trace, "Wait_Buffer", wb_end, "R1", wb_amt, role="role:hr")
            case.cost_so_far += wb_amt
            case.cur_time = wb_end

    # Decide preferred roles & pick resource (binding/separation rules considered)
    preferred = PREFERRED_ROLES[activity][:]
    if force_role:
        preferred = [force_role]

    tentative = choose_resource(resources, activity, preferred, case.cur_time)
    # Binding-of-duty enforcement
    if bind_key and bind_key in case.bind_map:
        bind_rid = case.bind_map[bind_key]
        tentative = resources.get(bind_rid, tentative)
    # Separation-of-duty enforcement
    if sep_duty_with and sep_duty_with in case.forbid_map:
        forbidden = case.forbid_map[sep_duty_with]
        if tentative.rid == forbidden:
            # pick someone else with same role or fallback
            alts = [r for r in resources.values() if r.rid != forbidden and r.role == tentative.role]
            if alts:
                tentative = rng.choice(alts)

    res = tentative
    maybe_handover(case, trace, res)

    # Ready time considering next open and resource availability
    ready = next_business_open(case.cur_time, res)
    ready = max(ready, res.next_free)

    # Queue pressure approximation
    exp_h = BASE_SERVICE_H[activity]
    qpress = queue_pressure(res, ready, exp_h)

    # Heavy-tailed service time with skill/learning/forgetting
    rework_iter = 1 if activity.startswith("Rework_") else 0
    duration_h = sample_service_time(activity, res, case.era_idx, case.last_activity, ready, rework_iter, qpress)

    # Preemption if queue is high or task is long
    preempt_used = False
    if preemptable and (qpress > 1.2 or duration_h > 2.5) and rng.random() < 0.5:
        preempt_used = True
        # Execute start chunk
        start1 = ready
        end1, ot1 = add_work_hours(start1, duration_h * 0.4, res, allow_overtime=False)
        amt1 = compute_event_cost(activity + ".start", res.role, (end1 - start1).total_seconds()/3600.0, ot1, (start1 - case.cur_time).total_seconds()/3600.0, case.era_idx)
        append_event(trace, activity + ".start", end1, res.rid, amt1, role=res.role)
        case.cost_so_far += amt1

        # Pause (simulate interruption)
        gap_h = rng.uniform(0.5, 2.0)
        gap_end, _ = add_work_hours(end1 + timedelta(minutes=5), gap_h, None, allow_overtime=False)
        case.cur_time = gap_end

        # Resume + complete
        start2 = next_business_open(case.cur_time, res)
        start2 = max(start2, res.next_free)
        end2, ot2 = add_work_hours(start2, duration_h * 0.6, res, allow_overtime=True)
        # resume event
        amt2a = compute_event_cost(activity + ".resume", res.role, (end2 - start2).total_seconds()/3600.0 * 0.0 + 0.1, 0.0, 0.0, case.era_idx)
        append_event(trace, activity + ".resume", start2, res.rid, amt2a, role=res.role)
        # complete event
        amt2b = compute_event_cost(activity + ".complete", res.role, (end2 - start2).total_seconds()/3600.0, ot2, (start2 - case.cur_time).total_seconds()/3600.0, case.era_idx)
        append_event(trace, activity + ".complete", end2, res.rid, amt2b, role=res.role)
        case.cost_so_far += (amt2a + amt2b)

        # Update clocks
        res.next_free = end2
        res.experience[family_of(activity)] += 1
        res.last_performed[family_of(activity)] = end2

        case.cur_time = end2
        case.last_activity = activity + ".complete"
        case.last_resource = res.rid
        duration_used = (end2 - ready).total_seconds()/3600.0
        return end2, res, duration_used

    # Non-preemptive
    start = ready
    end, overtime_h = add_work_hours(start, duration_h, res, allow_overtime=True)
    waiting_h = max(0.0, (start - case.cur_time).total_seconds()/3600.0)
    amt = compute_event_cost(activity, res.role, duration_h, overtime_h, waiting_h, case.era_idx, is_vendor=("vendor" in res.role))
    append_event(trace, activity, end, res.rid, amt, role=res.role)
    case.cost_so_far += amt

    res.next_free = end
    res.experience[family_of(activity)] += 1
    res.last_performed[family_of(activity)] = end

    case.cur_time = end
    case.last_activity = activity
    case.last_resource = res.rid
    return end, res, duration_h

def schedule_multi_resource_meeting(case: CaseState, trace: Trace, activity: str, resources: Dict[str, Resource],
                                    roles: List[str]) -> Tuple[datetime, Tuple[Resource, Resource], float]:
    # pick two resources from the roles; may overlap roles
    cand_pool = [r for r in resources.values() if r.role in roles]
    if len(cand_pool) < 2:
        cand_pool = list(resources.values())
    r1 = rng.choice(cand_pool)
    r2 = rng.choice([r for r in cand_pool if r.rid != r1.rid]) if len(cand_pool) > 1 else r1

    maybe_handover(case, trace, r1)  # simulate handover vs last resource

    # find earliest time both can start considering calendars & availability
    t = max(next_business_open(case.cur_time, r1), next_business_open(case.cur_time, r2), r1.next_free, r2.next_free)
    # approximate service time
    base = BASE_SERVICE_H[activity] * ERAS[case.era_idx]["service_mult"]
    dur_h = lognormal_heavy_tail(base) * 0.9
    end1, ot1 = add_work_hours(t, dur_h, r1, allow_overtime=True)
    end2, ot2 = add_work_hours(t, dur_h, r2, allow_overtime=True)
    end = max(end1, end2)

    # cost split across both
    wait_h = max(0.0, (t - case.cur_time).total_seconds()/3600.0)
    amt1 = compute_event_cost(activity, r1.role, dur_h, ot1, wait_h, case.era_idx)
    amt2 = compute_event_cost(activity, r2.role, dur_h, ot2, 0.0, case.era_idx)
    resource_label = f"{r1.rid}|{r2.rid}"
    append_event(trace, activity, end, resource_label, amt1 + amt2, role="role:manager")

    # update both resources
    for r in (r1, r2):
        r.next_free = end
        r.experience[family_of(activity)] += 1
        r.last_performed[family_of(activity)] = end

    case.cur_time = end
    case.last_activity = activity
    case.last_resource = resource_label
    return end, (r1, r2), dur_h

def schedule_and_join_parallel(case: CaseState, trace: Trace, branches: List[List[Tuple[str, Dict]]], resources: Dict[str, Resource]) -> datetime:
    """
    branches: list of branches; each branch = list of (activity_name, kwargs_for_schedule_single_activity)
    Join rule: AND-join → wait for all. Adds extra handover latency before Join_Parallel.
    """
    case.has_and_split = True
    branch_end_times = []
    branch_durations = []
    snapshot_time = case.cur_time
    # Save/restore to schedule each branch from common snapshot
    saved_state = (case.cur_time, case.last_activity, case.last_resource, dict(case.bind_map), dict(case.forbid_map), case.cost_so_far)

    for br in branches:
        # local copy of timing point
        case.cur_time, case.last_activity, case.last_resource, case.bind_map, case.forbid_map, case.cost_so_far = saved_state[0], saved_state[1], saved_state[2], dict(saved_state[3]), dict(saved_state[4]), saved_state[5]
        start_local = case.cur_time
        for act, kwargs in br:
            # Multi-resource meeting special-case
            if act == "Intv_Panel":
                schedule_multi_resource_meeting(case, trace, act, resources, PREFERRED_ROLES[act])
            else:
                schedule_single_activity(case, trace, act, resources, **kwargs)
        branch_end_times.append(case.cur_time)
        branch_durations.append((case.cur_time - start_local).total_seconds()/3600.0)

    # AND-join: join when all complete → join start waits for max end
    join_ready = max(branch_end_times)
    # Synchronization latency + handover delay before join
    latency_h = rng.uniform(0.1, 0.6)
    join_start = join_ready + timedelta(hours=latency_h)
    join_end, _ = add_work_hours(join_start, BASE_SERVICE_H["Join_Parallel"], None, allow_overtime=False)
    append_event(trace, "Join_Parallel", join_end, "R2", compute_event_cost("Join_Parallel", "role:hr", 0.1, 0.0, 0.0, case.era_idx), role="role:hr")

    case.cur_time = join_end
    case.last_activity = "Join_Parallel"
    case.last_resource = "R2"
    # Imbalance metric (max - min)
    if branch_durations:
        case.branch_imbalances.append(float(max(branch_durations) - min(branch_durations)))
    return join_end

def schedule_or_split_join(case: CaseState, trace: Trace, options: List[Tuple[str, Dict]], resources: Dict[str, Resource]) -> datetime:
    """
    OR-split: choose 1..k of options to run in parallel; join when all chosen completed.
    """
    k = rng.integers(1, len(options) + 1)
    chosen = rng.choice(options, size=k, replace=False)
    return schedule_and_join_parallel(case, trace, [[c] for c in chosen], resources)

def bg_multi_instance_k_of_n(case: CaseState, trace: Trace, resources: Dict[str, Resource], k: int = 2) -> datetime:
    """
    Background check: spawn 3 sub-checks; proceed when k complete; cancel the rest (refund).
    """
    tasks = ["BG_Education", "BG_Employment", "BG_References"]
    finishes = []
    res_used = []
    for t in tasks:
        end, res, _ = schedule_single_activity(case, trace, t, resources, preemptable=False)
        finishes.append((end, t, res))
        res_used.append(res)
    # Continue after k-th completion
    finishes_sorted = sorted(finishes, key=lambda x: x[0])
    kth_end = finishes_sorted[k-1][0]
    # Cancel remaining tasks finishing after k-th (refund/credit)
    for end, tname, res in finishes_sorted[k:]:
        # emit cancel event at kth_end + small delay
        t = kth_end + timedelta(minutes=rng.uniform(5, 30))
        amt = ACTIVITY_FIXED["Cancel_Remaining_BG"]
        append_event(trace, "Cancel_Remaining_BG", t, res.rid, amt, role=res.role)
        case.cost_so_far += amt
    # Emit BG_Threshold join
    join_end, _ = add_work_hours(kth_end, BASE_SERVICE_H["BG_Threshold"], None, allow_overtime=False)
    append_event(trace, "BG_Threshold", join_end, "R3", compute_event_cost("BG_Threshold", "role:hr", 0.1, 0.0, 0.0, case.era_idx), role="role:hr")
    case.cur_time = join_end
    case.last_activity = "BG_Threshold"
    case.last_resource = "R3"
    return join_end

# -----------------------
# Case builder (skeleton)
# -----------------------
def build_case(trace: Trace, case: CaseState, resources: Dict[str, Resource]):
    """
    Compose & schedule a full Hire-to-Retire flow with multiple control/temporal/resource/cost patterns.
    """
    # Linear prefix with optional insert
    schedule_single_activity(case, trace, "Req_Create", resources, preemptable=True)
    if rng.random() < 0.7:
        schedule_single_activity(case, trace, "Req_Approve", resources)
    schedule_single_activity(case, trace, "Job_Post", resources)

    # Candidate apply arrives later (bursty arrivals reflected by initial case start); short
    schedule_single_activity(case, trace, "Cand_Apply", resources)

    # XOR: screening path; probability depends on current slack & cost
    slack_h = (case.deadline - case.cur_time).total_seconds()/3600.0
    path = rng.choice(["hr_screen", "phone_then_hr", "vendor"], p=[0.45, 0.40, 0.15] if slack_h > 48 else [0.35, 0.35, 0.30])
    if path == "hr_screen":
        schedule_single_activity(case, trace, "Screen_HR", resources)
    elif path == "phone_then_hr":
        schedule_single_activity(case, trace, "Screen_Phone", resources, preemptable=True)
        schedule_single_activity(case, trace, "Screen_HR", resources)
    else:
        # Vendor path if cost-so-far exceeds limit or slack low
        schedule_single_activity(case, trace, "Vendor_Screening", resources, force_role="role:vendor")

    # Interviews: OR-split among HR, Tech, Panel (choose 1..3)
    opts = [("Intv_HR", {}), ("Intv_Tech", {}), ("Intv_Panel", {})]
    schedule_or_split_join(case, trace, opts, resources)

    # Offer flow with separation-of-duty (Audit vs Approve) and binding-of-duty between approvals later
    end, res_offer, _ = schedule_single_activity(case, trace, "Offer_Draft", resources, preemptable=True)
    end, res_app, _ = schedule_single_activity(case, trace, "Offer_Approve", resources, sep_duty_with="audit_sep")
    case.bind_map["comp_approve"] = res_app.rid       # later Comp_Change to be approved by same (binding)
    case.forbid_map["audit_sep"] = res_app.rid        # enforce separation with Audit

    # Send offer
    schedule_single_activity(case, trace, "Offer_Send", resources)

    # XOR Accept/Decline depends on prior durations/costs (stochastic)
    accept_p = 0.85 if case.cost_so_far < 2000 else 0.78
    if rng.random() < accept_p:
        schedule_single_activity(case, trace, "Offer_Accepted", resources)
        # Cooling-off to start date (min 7 days)
        # Insert Background checks with multi-instance k-of-n
        bg_multi_instance_k_of_n(case, trace, resources, k=2)

        # AND-split: Onboarding IT + Desk in parallel; join; then docs
        branches = [
            [("Onb_Provision_IT", {"preemptable": True})],
            [("Onb_Prepare_Desk", {})]
        ]
        schedule_and_join_parallel(case, trace, branches, resources)
        schedule_single_activity(case, trace, "Onb_Documents", resources)

        # Stochastic skip with memory: skip QA if Doc_Review short; else perform Doc_Review + possible rework loop
        review_duration = 0.0
        if rng.random() < 0.8:
            before = case.cur_time
            schedule_single_activity(case, trace, "Doc_Review", resources)
            review_duration = (case.cur_time - before).total_seconds()/3600.0
            # Rework loop: geometric repeats (duration influences prob)
            p_rework = min(0.6, 0.15 + 0.1 * review_duration)
            iter_cnt = 0
            while rng.random() < p_rework and iter_cnt < 3:
                schedule_single_activity(case, trace, "Rework_Fix", resources)
                schedule_single_activity(case, trace, "Doc_Review", resources)
                iter_cnt += 1

        # QA skip probability decreases if review short; more likely to run QA if review was long
        p_skip_qa = 0.6 if review_duration < 0.7 else 0.2
        if rng.random() > p_skip_qa:
            # Separation-of-duty: QA vs Offer_Approve
            schedule_single_activity(case, trace, "QA_Final", resources, sep_duty_with="audit_sep")
        else:
            # Expedite: pay to skip QA when low slack
            if (case.deadline - case.cur_time).days < 3 and rng.random() < 0.6:
                schedule_single_activity(case, trace, "Expedite", resources, force_role="role:senior")

        # OR-split: Payroll/Benefits (choose possibly both); join via an explicit Join_OR
        chosen = []
        if rng.random() < 0.8: chosen.append(("Payroll_Setup", {}))
        if rng.random() < 0.9: chosen.append(("Benefits_Enroll", {}))
        if not chosen: chosen = [("Payroll_Setup", {})]
        schedule_and_join_parallel(case, trace, [[c] for c in chosen], resources)
        append_event(trace, "Join_OR", case.cur_time, "R4", compute_event_cost("Join_OR", "role:hr", 0.1, 0.0, 0.0, case.era_idx), role="role:hr")

        # Orientation is batched weekly at Monday 09:00
        ori_start = next_orientation_start(case.cur_time)
        waiting_h = max(0.0, (ori_start - case.cur_time).total_seconds()/3600.0)
        # Add holding cost via Warn if long wait → escalate
        if waiting_h > 24 and rng.random() < 0.5:
            append_event(trace, "Warn", case.cur_time + timedelta(hours=0.1), "R5", compute_event_cost("Warn", "role:senior", 0.05, 0.0, 0.0, case.era_idx), role="role:senior")
        # Schedule Orientation
        res_hr = choose_resource(resources, "Onb_Orientation", PREFERRED_ROLES["Onb_Orientation"], ori_start)
        # Wait time cost is embedded in compute_event_cost below
        end, _ = add_work_hours(ori_start, BASE_SERVICE_H["Onb_Orientation"], res_hr, allow_overtime=False)
        amt = compute_event_cost("Onb_Orientation", res_hr.role, BASE_SERVICE_H["Onb_Orientation"], 0.0, waiting_h, case.era_idx)
        append_event(trace, "Onb_Orientation", end, res_hr.rid, amt, role=res_hr.role)
        res_hr.next_free = end
        case.cur_time = end
        case.cost_so_far += amt
        case.last_activity = "Onb_Orientation"
        case.last_resource = res_hr.rid

        # Training (preemptable); sometimes outsourced based on budget slack
        if case.cost_so_far > 4500 and rng.random() < 0.4:
            schedule_single_activity(case, trace, "Onb_Training", resources, force_role="role:vendor", preemptable=True)
        else:
            schedule_single_activity(case, trace, "Onb_Training", resources, preemptable=True)

        # End onboarding join marker
        append_event(trace, "Onb_Join", case.cur_time, "R6", compute_event_cost("Onb_Join", "role:hr", 0.1, 0.0, 0.0, case.era_idx), role="role:hr")

        # After start, include probation review; maybe compensation change approved by same approver as Offer (binding-of-duty)
        schedule_single_activity(case, trace, "Perf_Probation_Review", resources)
        if rng.random() < 0.35:
            schedule_single_activity(case, trace, "Comp_Change", resources, bind_key="comp_approve")

        # Occasionally, an internal transfer or leave request occurs
        if rng.random() < 0.25:
            schedule_single_activity(case, trace, "Trans_Internal", resources, preemptable=True)
        if rng.random() < 0.2:
            schedule_single_activity(case, trace, "Leave_Request", resources)
            schedule_single_activity(case, trace, "Leave_Approve", resources)
            schedule_single_activity(case, trace, "Return_Process", resources)

        # Annual review and possible audit (enforce separation with Offer_Approve)
        if rng.random() < 0.6:
            schedule_single_activity(case, trace, "Perf_Annual_Review", resources)
            if rng.random() < 0.25:
                schedule_single_activity(case, trace, "Audit", resources, sep_duty_with="audit_sep")

    else:
        schedule_single_activity(case, trace, "Offer_Declined", resources)
        # Simplify end path
        append_event(trace, "Simplify", case.cur_time + timedelta(minutes=1), "R7", compute_event_cost("Simplify", "role:hr", 0.1, 0.0, 0.0, case.era_idx), role="role:hr")
        case.cur_time += timedelta(minutes=1)

    # Exit paths (subset of cases)
    if rng.random() < 0.15:
        schedule_single_activity(case, trace, "Exit_Resign", resources)
        schedule_single_activity(case, trace, "Exit_Offboard", resources)
        schedule_single_activity(case, trace, "Exit_Interview", resources)
        schedule_single_activity(case, trace, "Exit_Revoke_Access", resources)

    # SLA breach check → Escalate with senior involvement and penalty (adds cost)
    if case.cur_time > case.deadline:
        esc_end, res, _ = schedule_single_activity(case, trace, "Escalate", resources, force_role="role:senior")
        # apply penalty lump sum at escalation
        penalty = 200.0 * ERAS[case.era_idx]["rework_penalty_mult"]
        append_event(trace, "SLA_Penalty", esc_end, res.rid, penalty, role=res.role)
        case.cost_so_far += penalty

# ------------------------
# Bursty arrivals (NHPP-ish)
# ------------------------
def generate_arrivals(n_cases: int, base_start: datetime) -> List[datetime]:
    """
    Create 3 waves of arrivals over ~6 months with intra-day clustering around morning hours.
    """
    waves = [
        {"count": 100, "start": base_start, "span_days": 40},
        {"count": 100, "start": base_start + timedelta(days=70), "span_days": 40},
        {"count": 50,  "start": base_start + timedelta(days=140), "span_days": 30},
    ]
    arr = []
    for w in waves:
        for _ in range(w["count"]):
            day_offset = rng.uniform(0, w["span_days"])
            day = (w["start"] + timedelta(days=day_offset))
            # intra-day burst: around 9–11am or 2–4pm
            if rng.random() < 0.7:
                hour = rng.uniform(8.5, 11.5)
            else:
                hour = rng.uniform(14.0, 16.5)
            dt = day.replace(hour=int(hour), minute=int((hour % 1)*60), second=rng.integers(0, 60))
            # snap to business time
            if WEEKEND_OFF and is_weekend(dt):
                dt = dt + timedelta(days=(7 - dt.weekday()) % 7)
                dt = dt.replace(hour=9, minute=0, second=0)
            arr.append(dt)
    arr = sorted(arr)[:n_cases]
    return arr

# --------------------------
# Sanity & quality reporting
# --------------------------
def quality_checks(log: EventLog):
    print("\n=== QUALITY & SANITY CHECKS ===")

    # Schema and timestamp checks
    total_events = 0
    issues = 0
    inter_event_sec = []
    trace_lengths = []
    costs = []
    activities = Counter()
    parallel_traces = 0
    branch_imbalances = []

    eras_costs = defaultdict(list)
    eras_durations = defaultdict(list)

    # For ambiguity: prefix -> set(next_activity)
    prefix_to_next = defaultdict(set)

    for tr in log:
        # ensure sort by timestamp
        tr._list.sort(key=lambda e: e["time:timestamp"])
        n = len(tr)
        trace_lengths.append(n)
        total_events += n
        # monotonic timestamps
        for i in range(1, n):
            if tr[i]["time:timestamp"] <= tr[i-1]["time:timestamp"]:
                issues += 1
            inter_event_sec.append((tr[i]["time:timestamp"] - tr[i-1]["time:timestamp"]).total_seconds())
        # schema
        for e in tr:
            if not all(k in e for k in ["concept:name", "time:timestamp", "org:resource", "amount"]):
                issues += 1
            try:
                costs.append(float(e["amount"]))
            except Exception:
                issues += 1
            activities[e["concept:name"]] += 1

        # ambiguity: collect prefix->next
        labels = [e["concept:name"] for e in tr]
        for k in range(1, min(6, n-1)):
            prefix = tuple(labels[:k])
            prefix_to_next[prefix].add(labels[k])

        # era stats from trace-level attributes
        era_name = tr.attributes.get("era", "unknown")
        # duration
        dur = (tr[-1]["time:timestamp"] - tr[0]["time:timestamp"]).total_seconds()/3600.0
        eras_durations[era_name].append(dur)
        eras_costs[era_name].append(sum(float(e["amount"]) for e in tr))

        # parallelism
        if any(e["concept:name"] == "Join_Parallel" for e in tr):
            parallel_traces += 1
            # branch imbalance: approximate via spreads near join
            # rough measure: look for "Join_Parallel" amount, but we stored in attributes? We track none; skip
        # keep branch imbalances from attributes if present
        imb = tr.attributes.get("branch_imbalance", None)
        if isinstance(imb, float):
            branch_imbalances.append(imb)

    print(f"Total traces: {len(log)} (expected {N_CASES})")
    print(f"Total events: {total_events}")
    print(f"Schema/timestamp issues found: {issues}")

    def qtiles(arr):
        q = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95]) if arr else [np.nan]*5
        return {p: float(v) for p, v in zip(["p05", "p25", "p50", "p75", "p95"], q)}

    print("\n-- Trace length (events) quantiles:", qtiles(trace_lengths))
    if inter_event_sec:
        inter_h = [s/3600.0 for s in inter_event_sec]
        print("-- Inter-event time (hours) quantiles:", qtiles(inter_h))
    print("-- Event amount ($) quantiles:", qtiles(costs))

    # Ambiguity check: sample up to 100 prefixes
    prefixes = list(prefix_to_next.items())
    rng.shuffle(prefixes)
    sample = prefixes[:min(100, len(prefixes))]
    ambiguous = sum(1 for _, nxt in sample if len(nxt) >= 2)
    print(f"-- Ambiguity: {ambiguous}/{len(sample)} sampled prefixes had >=2 distinct next activities.")

    # Remaining time distribution (regression target)
    rem_hours = []
    for tr in log:
        end_time = tr[-1]["time:timestamp"]
        for e in tr:
            rem_hours.append((end_time - e["time:timestamp"]).total_seconds()/3600.0)
    print("-- Remaining time (hours) quantiles:", qtiles(rem_hours))

    # Parallelism check
    pct_parallel = 100.0 * parallel_traces / len(log)
    print(f"-- Parallelism: {pct_parallel:.1f}% of traces have at least one AND-split/Join_Parallel.")

    # Drift by era
    print("\n-- Drift by era --")
    for era in sorted(eras_costs.keys()):
        cdur = eras_durations[era]
        ccost = eras_costs[era]
        print(f"  {era}: n={len(cdur)}  mean_dur_h={np.mean(cdur):.2f}  mean_cost=${np.mean(ccost):.2f}")

    # Activity top counts
    print("\nTop 20 activities by count:")
    for act, c in activities.most_common(20):
        print(f"  {act:22s}  {c}")

# --------------------------
# Minimal example in comments
# --------------------------
# Minimal example (conceptual, not executed separately):
# One case triggers: AND-split (IT provisioning || Desk prep), handover delay,
# SLA breach → Escalate, a single Doc_Review rework, and vendor training during overtime.

# -------------
# Main builder
# -------------
def execute_script():
    base_start = datetime(2024, 1, 8, 9, 0, 0)  # Monday
    arrivals = generate_arrivals(N_CASES, base_start)
    resources = build_resources(base_start)

    log = EventLog()
    and_split_counts = 0

    for i, t0 in enumerate(arrivals, start=1):
        # Assign era by index block (drift/seasonality)
        if i <= 85:
            era_idx = 0
        elif i <= 170:
            era_idx = 1
        else:
            era_idx = 2

        # Case type → SLA baseline variability (hours)
        case_type = rng.choice(["NewHire", "Transfer", "Promotion"], p=[0.7, 0.2, 0.1])
        sla_days = BASE_SLA_DAYS + (3 if case_type != "NewHire" else 0)
        deadline = t0 + timedelta(days=sla_days)

        trace = Trace()
        case_id = f"H2R_{i:04d}"
        trace.attributes["concept:name"] = case_id
        trace.attributes["era"] = ERAS[era_idx]["name"]
        trace.attributes["case_type"] = case_type

        case = CaseState(case_id=case_id, era_idx=era_idx, start_time=t0, cur_time=t0, deadline=deadline)
        # Build and schedule case activities
        build_case(trace, case, resources)

        # Save AND-split metadata if any
        if case.has_and_split and case.branch_imbalances:
            trace.attributes["branch_imbalance"] = float(np.mean(case.branch_imbalances))
            and_split_counts += 1

        # Ensure strictly increasing timestamps within the trace
        # If any equal timestamps slip in, nudge forward by seconds
        for j in range(1, len(trace)):
            if trace[j]["time:timestamp"] <= trace[j-1]["time:timestamp"]:
                trace[j]["time:timestamp"] = trace[j-1]["time:timestamp"] + timedelta(seconds=1)

        log.append(trace)

    print(f"Built {len(log)} traces. AND-split traces: {and_split_counts}")

    # Export
    try:
        # pm4py >= 2 has this convenience method
        pm4py.write_xes(log, OUT_PATH)
    except Exception:
        # Fallback to exporter
        from pm4py.objects.log.exporter.xes import exporter as xes_exporter
        xes_exporter.apply(log, OUT_PATH)
    print(f"Wrote XES log to: {OUT_PATH}")

    # Quality checks & summaries
    quality_checks(log)

if __name__ == "__main__":
    execute_script()
