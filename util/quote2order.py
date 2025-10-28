# -*- coding: utf-8 -*-
"""
Quote-to-Order synthetic XES generator (trace-by-trace, pm4py).

This script produces a rich, process-aware event log for a Quote-to-Order process and
exports it to 'quote2order.xes.gz'. It aims to be a compact but expressive sandbox for
process mining experiments: predictive monitoring, prefix classification, drift, etc.

Key features implemented (mix & match per case):
- Control flow: linear-with-optionals, XOR, loops (Review/Fix), AND/OR parallelism with proper join,
  preemption/resume (Negotiate.start/resume/complete), k-of-n multi-instance (vendor bids),
  skip patterns, long-term dependencies (priority/budget affects later steps),
  binding-of-duty and separation-of-duty constraints, explicit Handover pseudo-activity.
- Temporal patterns: NHPP-ish bursty arrivals, business calendars, resource shift calendars
  + overtime, heavy-tailed service times (lognormal), setup/changeover + context switch time,
  queue-induced preemptions, AND-join synchronization latency, batching/release for Book_Order,
  cooling-off constraint (Send_Quote -> Follow_Up), deadlines/SLA → Warn/Escalate.
- Resource patterns: roles & pools (R1..R30) with heterogenous skills, unavailability,
  assignment policy mixture (random / least-loaded / skill-based), multi-resource meeting (as
  two concurrent sub-activities + join), external vendor option via vendor-role resources.
- Cost patterns (event-level 'amount'): fixed + rate * duration, overtime premiums, waiting cost,
  handover fee, rework premium escalation, expedite fee (queue jump), SLA penalties, material costs,
  budget-aware branching, era-based drift of tariffs.
- Quality checks printed after generation (no plots): schema, monotone timestamps, distributions,
  ambiguity of next-activity given prefixes, remaining-time signal, parallelism share, drift checks.

Minimal reference scenario (see code search for '# Minimal reference trace injection'):
Start → ... → AND-split (Credit/Inventory checks) → Join → long wait causes SLA breach → Escalate →
one rework loop → vendor path with overtime.

Author: (You) Senior Process Mining Engineer
"""

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

try:
    import pandas as pd
    HAVE_PANDAS = True
except Exception:
    HAVE_PANDAS = False

# pm4py imports kept minimal
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter


# ------------------------------
# Global configuration
# ------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TARGET_DISTINCT_TRACES = 250          # HARD LIMIT: 250 distinct traces (by activity label sequence)
MAX_GENERATION_ATTEMPTS = 2000        # safety to avoid infinite loops if uniqueness is hard

# Era window (concept drift): generate arrivals across these 2 eras
SIM_START = datetime(2025, 1, 6, 9, 0, 0)  # a Monday 09:00
SIM_DAYS = 90
ERA_CUTOFF = SIM_START + timedelta(days=45)  # two equal halves

# Business calendar (baseline)
BUSINESS_OPEN = 9
BUSINESS_CLOSE = 17
WEEKEND_FREEZE = True

# Resource shifts
SHIFT_EARLY = (9, 17)    # 9-17
SHIFT_LATE = (12, 20)    # 12-20

# Overtime rules
ALLOW_OVERTIME = True
OVERTIME_MULT = 1.4

# Queues/Batching rules
BOOK_ORDER_RELEASE_MINUTES = 30

# SLA thresholds (era drift)
SLA_HOURS_E1 = 72.0
SLA_HOURS_E2 = 60.0  # later era pushes for faster turnaround

# Expedite + escalation knobs
EXPEDITE_WAIT_HOURS = 6.0
EXPEDITE_FEE = 180.0
ESCALATE_FEE_E1 = 350.0
ESCALATE_FEE_E2 = 420.0  # later era: escalation more expensive

# Waiting cost per hour (holding)
WAIT_COST_PER_HOUR_E1 = 5.0
WAIT_COST_PER_HOUR_E2 = 7.0

# Handover fee per occurrence + duration
HANDOVER_FEE = 12.0
HANDOVER_MINUTES = (5, 20)

# Rework premium increment
REWORK_PREMIUM_STEP = 40.0

# Small fraction of Unknown resources
UNKNOWN_RESOURCE_FRAC = 0.02

# Multi-instance k-of-n vendor bids
BIDS_N_RANGE = (3, 5)
BIDS_K = 2

# Preemption for Negotiate based on queue pressure / long duration
NEGOTIATE_PREEMPT_BASE_P = 0.25

# Activity families to model setup/context switch and skills
FAMILIES = [
    "Intake", "Config", "Pricing", "Check", "Review", "Legal", "Finance",
    "Technical", "Approval", "Sales", "Clerk", "Vendor", "System"
]

# Activity catalog (<= 40 labels)
A = dict(
    Start="Start",
    Receive="Receive_RFQ",
    Qualify="Qualify_Lead",
    Precheck="Precheck",
    Scope="Scope",
    KickAna="Kickoff_Meeting_Analyst",
    KickSen="Kickoff_Meeting_Senior",
    KickJoin="Kickoff_Join",
    Configure="Configure_Solution",
    Price="Price_Calc",
    Discount="Discount",
    PrepQuote="Prepare_Quote",
    CheckCredit="Check_Credit",
    CheckInv="Check_Inventory",
    JoinChecks="Consolidate_Checks",
    Legal="Legal_Review",
    Tech="Technical_Review",
    Fin="Finance_Review",
    QA="QA",
    Approve="Approve",
    Send="Send_Quote",
    NegStart="Customer_Negotiate.start",
    NegResume="Customer_Negotiate.resume",
    NegDone="Customer_Negotiate.complete",
    Revise="Revise_Quote",
    Submit="Submit_Quote",
    Follow="Follow_Up",
    Convert="Convert_to_Order",
    Book="Book_Order",
    Notify="Notify_Customer",
    Cancel="Cancel",
    Wait="Wait",
    Handover="Handover",
    Reassign="Reassign",
    Warn="Warn",
    Escalate="Escalate",
    Expedite="Expedite",
    VendorBid="Vendor_Bid",
    BidJoin="Collect_Bids",
    VendorPick="Select_Vendor",
    PO="Place_PO",
    RecPO="Receive_PO",
    Materials="Order_Materials",
    Audit="Audit",
    End="End"
)

# Map activities to families + preferred roles
ACTIVITY_META: Dict[str, Dict[str, Any]] = {
    A["Receive"]:  {"family": "Intake",   "roles": ["role:sales", "role:analyst"]},
    A["Qualify"]:  {"family": "Intake",   "roles": ["role:analyst", "role:sales"]},
    A["Precheck"]: {"family": "Intake",   "roles": ["role:analyst"]},
    A["Scope"]:    {"family": "Intake",   "roles": ["role:analyst", "role:senior"]},
    A["KickAna"]:  {"family": "Intake",   "roles": ["role:analyst"]},
    A["KickSen"]:  {"family": "Intake",   "roles": ["role:senior"]},
    A["KickJoin"]: {"family": "System",   "roles": []},
    A["Configure"]:{"family": "Config",   "roles": ["role:analyst"]},
    A["Price"]:    {"family": "Pricing",  "roles": ["role:analyst"]},
    A["Discount"]: {"family": "Pricing",  "roles": ["role:manager", "role:senior"]},
    A["PrepQuote"]:{"family": "Pricing",  "roles": ["role:analyst"]},
    A["CheckCredit"]:{"family": "Check",  "roles": ["role:analyst"]},
    A["CheckInv"]: {"family": "Check",    "roles": ["role:clerk"]},
    A["JoinChecks"]:{"family": "System",  "roles": []},
    A["Legal"]:    {"family": "Legal",    "roles": ["role:senior"]},
    A["Tech"]:     {"family": "Technical","roles": ["role:analyst"]},
    A["Fin"]:      {"family": "Finance",  "roles": ["role:senior"]},
    A["QA"]:       {"family": "Review",   "roles": ["role:analyst"]},
    A["Approve"]:  {"family": "Approval", "roles": ["role:senior", "role:manager"]},
    A["Send"]:     {"family": "Sales",    "roles": ["role:sales"]},
    A["NegStart"]: {"family": "Sales",    "roles": ["role:sales"]},
    A["NegResume"]:{ "family": "Sales",   "roles": ["role:sales"]},
    A["NegDone"]:  {"family": "Sales",    "roles": ["role:sales"]},
    A["Revise"]:   {"family": "Config",   "roles": ["role:analyst"]},
    A["Submit"]:{  "family": "Sales",     "roles": ["role:sales"]},
    A["Follow"]:{  "family": "Sales",     "roles": ["role:sales"]},
    A["Convert"]:{ "family": "Clerk",     "roles": ["role:clerk", "role:analyst"]},
    A["Book"]:{    "family": "Clerk",     "roles": ["role:clerk"]},
    A["Notify"]:{  "family": "Sales",     "roles": ["role:sales", "role:analyst"]},
    A["Cancel"]:{  "family": "System",    "roles": []},
    A["Wait"]:{    "family": "System",    "roles": []},
    A["Handover"]:{ "family": "System",   "roles": []},
    A["Reassign"]:{ "family": "System",   "roles": []},
    A["Warn"]:{    "family": "System",    "roles": []},
    A["Escalate"]:{ "family": "Approval", "roles": ["role:senior"]},
    A["Expedite"]:{ "family": "Approval", "roles": ["role:senior"]},
    A["VendorBid"]:{ "family": "Vendor",  "roles": ["role:vendor"]},
    A["BidJoin"]:{ "family": "System",    "roles": []},
    A["VendorPick"]:{ "family": "Vendor", "roles": ["role:senior", "role:analyst"]},
    A["PO"]:{      "family": "Vendor",    "roles": ["role:clerk"]},
    A["RecPO"]:{   "family": "Vendor",    "roles": ["role:clerk"]},
    A["Materials"]:{ "family": "Vendor",  "roles": ["role:clerk"]},
    A["Audit"]:{   "family": "Finance",   "roles": ["role:senior"]},
    A["End"]:{     "family": "System",    "roles": []},
}

# Base duration (hours) by family (lognormal mean in hours) and sigma controlling tail
FAMILY_DURATION_PARAMS = {
    "Intake":   (1.0, 0.6),
    "Config":   (3.0, 0.7),
    "Pricing":  (2.0, 0.7),
    "Check":    (2.5, 0.6),
    "Review":   (1.5, 0.6),
    "Legal":    (5.0, 0.8),
    "Finance":  (3.5, 0.7),
    "Technical":(3.5, 0.7),
    "Approval": (1.0, 0.5),
    "Sales":    (2.0, 0.8),
    "Clerk":    (1.2, 0.5),
    "Vendor":   (8.0, 0.9),   # slower mean for vendor work
    "System":   (0.1, 0.2)
}

# Cost base: fixed per activity + hourly rate by role (with era drift)
FIXED_COST_BASE = {
    # key defaults by family if activity not listed
    "DEFAULT": 10.0,
    A["Escalate"]: 0.0,   # penalty added separately
    A["Expedite"]: EXPEDITE_FEE,
    A["Handover"]: 5.0,
    A["Wait"]: 0.0,
    A["Materials"]: 250.0,
    A["VendorPick"]: 20.0,
    A["BidJoin"]: 5.0,
    A["KickJoin"]: 5.0,
}

ROLE_RATE_E1 = {
    "role:analyst": 40.0,
    "role:senior":  65.0,
    "role:manager": 80.0,
    "role:sales":   45.0,
    "role:clerk":   30.0,
    "role:vendor":  35.0
}
ROLE_RATE_E2 = {
    "role:analyst": 42.0,
    "role:senior":  70.0,
    "role:manager": 84.0,
    "role:sales":   46.0,
    "role:clerk":   31.0,
    "role:vendor":  28.0  # vendor cheaper in later era
}


# ------------------------------
# Utilities: Calendars & Time
# ------------------------------
def is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5  # 5=Sat, 6=Sun

def within_hours(dt: datetime, open_h: int, close_h: int) -> bool:
    return open_h <= dt.hour < close_h and not (WEEKEND_FREEZE and is_weekend(dt))

def next_opening(dt: datetime, open_h: int, close_h: int) -> datetime:
    if not WEEKEND_FREEZE:
        if dt.hour < open_h:
            return dt.replace(hour=open_h, minute=0, second=0, microsecond=0)
        # after hours: roll to next day open
        next_day = dt + timedelta(days=1)
        return next_day.replace(hour=open_h, minute=0, second=0, microsecond=0)
    # with weekend freeze
    cur = dt
    while True:
        if not is_weekend(cur):
            if cur.hour < open_h:
                return cur.replace(hour=open_h, minute=0, second=0, microsecond=0)
            # roll to next day if after close
            cur = cur + timedelta(days=1)
            cur = cur.replace(hour=open_h, minute=0, second=0, microsecond=0)
            if not is_weekend(cur):
                return cur
        else:
            # move to next Monday 9am
            days_ahead = (7 - cur.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 1
            cur = (cur + timedelta(days=days_ahead)).replace(hour=open_h, minute=0, second=0, microsecond=0)

def add_working_time(start: datetime, hours: float, shift: Tuple[int, int], allow_overtime=False) -> datetime:
    """
    Add 'hours' of *processing* time respecting shift hours (and weekend freeze).
    If allow_overtime, we allow continuing past close until done (single stretch).
    """
    open_h, close_h = shift
    remaining = hours
    cur = start
    # ensure start is within working time unless overtime is allowed
    if not allow_overtime and not within_hours(cur, open_h, close_h):
        cur = next_opening(cur, open_h, close_h)

    while remaining > 1e-9:
        if allow_overtime:
            # single stretch ignoring close (but still freeze if weekend)
            cur += timedelta(hours=remaining)
            remaining = 0.0
            break
        # compute how much time available today
        if is_weekend(cur) and WEEKEND_FREEZE:
            cur = next_opening(cur, open_h, close_h)
            continue
        day_close = cur.replace(hour=close_h, minute=0, second=0, microsecond=0)
        if cur.hour < open_h:
            cur = cur.replace(hour=open_h, minute=0, second=0, microsecond=0)
        avail = (day_close - cur).total_seconds() / 3600.0
        if avail <= 0:
            cur = next_opening(cur, open_h, close_h)
            continue
        work = min(avail, remaining)
        cur += timedelta(hours=work)
        remaining -= work
        if remaining > 0:
            # move to next opening
            cur = next_opening(cur, open_h, close_h)
    return cur

def ceil_to_release_slot(dt: datetime, minutes: int = 30) -> datetime:
    minute = (dt.minute // minutes) * minutes
    slot = dt.replace(minute=minute, second=0, microsecond=0)
    if slot < dt:
        slot += timedelta(minutes=minutes)
    return slot


# ------------------------------
# Resource model
# ------------------------------
@dataclass
class Resource:
    rid: str
    role: str
    shift: Tuple[int, int]
    next_free: datetime
    skills: Dict[str, float] = field(default_factory=dict)  # family -> speed multiplier (lower=faster)
    last_seen: Dict[str, datetime] = field(default_factory=dict)  # family -> last time executed

    def speed_for(self, family: str) -> float:
        return self.skills.get(family, 1.0)

class ResourcePool:
    def __init__(self, start_time: datetime):
        self.resources: Dict[str, Resource] = {}
        self.by_role: Dict[str, List[str]] = {}
        # create 30 resources with mixed roles and shifts
        roles_all = (["role:analyst"] * 12 +
                     ["role:senior"] * 6 +
                     ["role:sales"] * 4 +
                     ["role:clerk"] * 4 +
                     ["role:manager"] * 2 +
                     ["role:vendor"] * 2)
        random.shuffle(roles_all)
        for i in range(30):
            rid = f"R{i+1}"
            role = roles_all[i]
            shift = SHIFT_EARLY if i < 22 else SHIFT_LATE
            # skill multipliers (lower is better/faster)
            skills = {fam: max(0.6, random.gauss(1.0, 0.15)) for fam in FAMILIES}
            # slightly specialize
            if role == "role:analyst":
                for fam in ["Config", "Pricing", "Technical", "Review"]:
                    skills[fam] *= 0.85
            if role == "role:senior":
                for fam in ["Approval", "Legal", "Finance"]:
                    skills[fam] *= 0.80
            if role == "role:vendor":
                for fam in ["Vendor"]:
                    skills[fam] *= 0.9
            res = Resource(rid, role, shift, start_time, skills)
            self.resources[rid] = res
            self.by_role.setdefault(role, []).append(rid)

        # stochastic absences (per resource, a few random full-day blocks)
        self.absences: Dict[str, List[datetime]] = {}
        for rid in self.resources:
            days_off = random.sample(range(SIM_DAYS), k=random.randint(0, 3))
            self.absences[rid] = [SIM_START + timedelta(days=d) for d in days_off]

    def is_absent(self, rid: str, when: datetime) -> bool:
        days = self.absences.get(rid, [])
        for d in days:
            if when.date() == d.date():
                return True
        return False

    def pick(self, roles_pref: List[str], family: str, when: datetime, policy: str) -> Optional[Resource]:
        # Candidate set
        candidates: List[Resource] = []
        for role in roles_pref:
            for rid in self.by_role.get(role, []):
                r = self.resources[rid]
                if self.is_absent(rid, when):
                    continue
                candidates.append(r)
        if not candidates:
            # back-off: any resource not absent
            for rid, r in self.resources.items():
                if not self.is_absent(rid, when):
                    candidates.append(r)
        if not candidates:
            return None

        if policy == "least_loaded":
            return sorted(candidates, key=lambda r: r.next_free)[0]
        if policy == "skill_based":
            return sorted(candidates, key=lambda r: r.speed_for(family))[0]
        # default random
        return random.choice(candidates)


# ------------------------------
# Case context and helpers
# ------------------------------
@dataclass
class CaseCtx:
    case_id: str
    start_ts: datetime
    era: str
    sla_hours: float
    role_rates: Dict[str, float]
    waiting_cost_per_hour: float
    escalate_fee: float
    priority: bool
    budget_cap: float
    cumulative_cost: float = 0.0
    last_resource: Optional[str] = None
    last_family: Optional[str] = None
    had_and_split: bool = False
    and_branch_imbalances: List[float] = field(default_factory=list)
    rework_count: int = 0
    review_duration_memory: float = 0.0
    binding_map: Dict[str, str] = field(default_factory=dict)  # enforce same resource for certain pairs
    sep_duty_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [(A["Approve"], A["Audit"])])
    used_sep_violation: bool = False
    used_expedite: bool = False
    used_escalate: bool = False


# ------------------------------
# Distributions & durations
# ------------------------------
def lognormal_hours(mean_hours: float, sigma: float) -> float:
    """
    Return a heavy-tailed duration (hours). Calibrate so mean ~ mean_hours.
    """
    # approximate: underlying normal mu such that exp(mu + 0.5*sigma^2) ~= mean
    mu = math.log(max(1e-6, mean_hours)) - 0.5 * (sigma ** 2)
    return float(np.random.lognormal(mean=mu, sigma=sigma))

def family_duration_hours(family: str) -> float:
    m, s = FAMILY_DURATION_PARAMS[family]
    return max(0.05, lognormal_hours(m, s))


# ------------------------------
# Cost model
# ------------------------------
def event_cost(activity: str, role: Optional[str], duration_hours: float, overtime: bool,
               ctx: CaseCtx) -> float:
    fixed = FIXED_COST_BASE.get(activity, FIXED_COST_BASE.get("DEFAULT", 10.0))
    rate = ctx.role_rates.get(role or "role:analyst", 40.0)
    if overtime:
        rate *= OVERTIME_MULT
    return float(fixed + rate * duration_hours)


# ------------------------------
# Arrival generation (bursty NHPP-ish)
# ------------------------------
def generate_case_start_times(n: int) -> List[datetime]:
    """
    Generate start times in bursts across SIM_DAYS using business hours.
    """
    starts: List[datetime] = []
    # define burst windows (days) with higher intensity
    burst_days = set()
    for base in [3, 4, 5, 20, 21, 22, 45, 60, 61, 75]:
        burst_days.add(base)
    day = 0
    while len(starts) < n:
        day_offset = random.randint(0, SIM_DAYS - 1)
        # bias toward bursts
        if day_offset in burst_days or random.random() < 0.35:
            # create a few arrivals on this day during peaks (10-12, 14-16)
            date = SIM_START + timedelta(days=day_offset)
            if is_weekend(date):
                continue
            for _ in range(random.randint(1, 3)):
                hour = random.choice([10, 11, 15, 16])
                minute = random.randint(0, 59)
                starts.append(date.replace(hour=hour, minute=minute, second=0, microsecond=0))
                if len(starts) >= n:
                    break
        else:
            # base single arrival at random hour
            date = SIM_START + timedelta(days=day_offset)
            if is_weekend(date):
                continue
            hour = random.randint(9, 16)
            minute = random.randint(0, 59)
            starts.append(date.replace(hour=hour, minute=minute, second=0, microsecond=0))
    starts = sorted(starts)[:n]
    return starts


# ------------------------------
# Scheduling core
# ------------------------------
def schedule_activity(case_log: List[Dict[str, Any]],
                      ctx: CaseCtx,
                      pool: ResourcePool,
                      activity: str,
                      ready_ts: datetime,
                      policy_mix: Tuple[float, float, float] = (0.34, 0.33, 0.33),
                      enforce_binding_of_duty_for: Optional[Tuple[str, str]] = None,
                      separation_pair: Optional[Tuple[str, str]] = None,
                      prev_activity: Optional[str] = None,
                      may_expedite: bool = True,
                      force_role: Optional[str] = None,
                      force_resource: Optional[str] = None,
                      allow_overtime: Optional[bool] = None
                      ) -> Tuple[datetime, Optional[str], float]:
    """
    Schedule one activity; return (end_ts, resource_id, duration_hours).
    Inserts optional Handover event, Wait event (if large wait), and applies costs.
    """
    meta = ACTIVITY_META.get(activity, {"family": "System", "roles": []})
    family = meta["family"]
    roles_pref = meta["roles"][:]
    if force_role:
        roles_pref = [force_role]

    # binding-of-duty: if binding_map has a key equal to activity, use that resource
    resource_to_bind = ctx.binding_map.get(activity, None)
    if force_resource:
        resource_to_bind = force_resource

    # If separation-of-duty between Approve and Audit, ensure different resource
    sep_enforced_resource = None
    if separation_pair and activity == separation_pair[1]:
        # if we used a specific resource for the first, deny it here
        used = ctx.binding_map.get(separation_pair[0], None)
        if used:
            sep_enforced_resource = used

    # Choose assignment policy
    r = random.random()
    if r < policy_mix[0]:
        policy = "random"
    elif r < policy_mix[0] + policy_mix[1]:
        policy = "least_loaded"
    else:
        policy = "skill_based"

    # Choose/resolve resource
    if resource_to_bind and resource_to_bind in pool.resources:
        res = pool.resources[resource_to_bind]
    else:
        res = pool.pick(roles_pref, family, when=ready_ts, policy=policy)
        if res is None:
            # fallback: Unknown
            res = Resource("Unknown", "role:analyst", SHIFT_EARLY, ready_ts)

    # enforce separation-of-duty if needed
    if sep_enforced_resource and res.rid == sep_enforced_resource:
        # route to Reassign pseudo-activity first
        # small delay + cost
        reass_dur_h = 0.25
        reass_start = max(ready_ts, pool.resources[ctx.binding_map[separation_pair[0]]].next_free)
        reass_end = add_working_time(reass_start, reass_dur_h, SHIFT_EARLY, allow_overtime=True)
        ev = dict(label=A["Reassign"], start=reass_start, end=reass_end, resource="Unknown",
                  role=None, family="System", amount=event_cost(A["Reassign"], None, reass_dur_h, False, ctx))
        case_log.append(ev)
        # pick a different resource
        candidates = [pool.resources[rid] for rid in pool.by_role.get(res.role, []) if rid != sep_enforced_resource]
        if candidates:
            res = random.choice(candidates)
        ctx.used_sep_violation = True

    # compute service time
    base_hours = family_duration_hours(family)
    # skill multiplier (lower faster)
    base_hours *= res.speed_for(family)

    # setup/changeover penalty if family changed for the resource recently
    prev_seen = res.last_seen.get(family, None)
    changeover_hours = 0.0
    if ctx.last_family and ctx.last_family != family:
        changeover_hours += random.uniform(0.1, 0.6)  # context switch
    if prev_seen is None or (ready_ts - prev_seen).total_seconds() > 7*24*3600:
        # forgetting curve
        base_hours *= random.uniform(1.05, 1.25)

    # Decide overtime usage
    if allow_overtime is None:
        allow_ot = ALLOW_OVERTIME and (ctx.priority or activity in (A["Escalate"], A["Expedite"]))
    else:
        allow_ot = allow_overtime

    # compute start time considering resource and shift
    start_cand = ready_ts
    # resource availability
    start_cand = max(start_cand, res.next_free)
    open_h, close_h = res.shift

    # expedite rule: if predicted wait exceeds threshold
    predicted_wait_hours = max(0.0, (start_cand - ready_ts).total_seconds() / 3600.0)
    inserted_expedite = False
    if may_expedite and (ctx.priority or predicted_wait_hours > EXPEDITE_WAIT_HOURS) and activity not in (A["Expedite"], A["Escalate"], A["Wait"]):
        # insert Expedite
        exp_start = ready_ts
        exp_end = add_working_time(exp_start, 0.1, SHIFT_EARLY, allow_overtime=True)
        evx = dict(label=A["Expedite"], start=exp_start, end=exp_end, resource="R1", role="role:senior",
                   family="Approval", amount=event_cost(A["Expedite"], "role:senior", 0.1, True, ctx))
        case_log.append(evx)
        ctx.cumulative_cost += evx["amount"]
        inserted_expedite = True
        ctx.used_expedite = True
        # allow preemption: start right away ignoring queue
        start_cand = exp_end

    # shift opening if not overtime
    if not allow_ot and not within_hours(start_cand, open_h, close_h):
        start_cand = next_opening(start_cand, open_h, close_h)

    # large wait -> add Wait pseudo-event with holding costs
    wait_hours = max(0.0, (start_cand - ready_ts).total_seconds() / 3600.0)
    if wait_hours > 0.25:
        w_start = ready_ts
        w_end = start_cand
        hold_cost = ctx.waiting_cost_per_hour * wait_hours
        evw = dict(label=A["Wait"], start=w_start, end=w_end, resource="Unknown", role=None,
                   family="System", amount=float(hold_cost))
        case_log.append(evw)
        ctx.cumulative_cost += evw["amount"]

    # handover if resource changes from previous event in this case
    if ctx.last_resource and ctx.last_resource != res.rid and activity not in (A["Wait"], A["Handover"], A["Expedite"], A["Escalate"]):
        # schedule handover just before start_cand
        ho_minutes = random.randint(*HANDOVER_MINUTES)
        ho_dur_h = ho_minutes / 60.0
        ho_start = start_cand
        ho_end = add_working_time(ho_start, ho_dur_h, res.shift, allow_overtime=True)
        evh = dict(label=A["Handover"], start=ho_start, end=ho_end, resource="Unknown", role=None,
                   family="System", amount=HANDOVER_FEE + 2.0 * ho_dur_h)
        case_log.append(evh)
        ctx.cumulative_cost += evh["amount"]
        # shift actual start to after handover
        start_cand = ho_end

    # processing with setup/changeover
    proc_hours = max(0.05, base_hours + changeover_hours)

    # Preemption pattern for Negotiate
    overtime_used = allow_ot
    if activity == A["NegStart"]:
        # break Negotiate into start/resume/done
        # small initial slice
        slice1 = min(proc_hours * 0.2, 1.0)
        end1 = add_working_time(start_cand, slice1, res.shift, allow_overtime=allow_ot)
        ev1 = dict(label=A["NegStart"], start=start_cand, end=end1, resource=res.rid, role=res.role,
                   family=family, amount=event_cost(activity, res.role, slice1, allow_ot, ctx))
        case_log.append(ev1)
        ctx.cumulative_cost += ev1["amount"]
        res.next_free = end1
        res.last_seen[family] = end1

        # queue pressure preemption?
        preempt_p = NEGOTIATE_PREEMPT_BASE_P + 0.1 * (wait_hours > 1.0) + (0.05 if inserted_expedite else 0.0)
        if random.random() < preempt_p:
            # some pause, then resume
            pause_h = random.uniform(2.0, 8.0)
            resume_start = add_working_time(end1 + timedelta(hours=pause_h), 0.0, res.shift, allow_overtime=False)
            # "resume" short chunk
            slice2 = min(proc_hours * 0.3, 1.5)
            end2 = add_working_time(resume_start, slice2, res.shift, allow_overtime=False)
            ev2 = dict(label=A["NegResume"], start=resume_start, end=end2, resource=res.rid, role=res.role,
                       family=family, amount=event_cost(A["NegResume"], res.role, slice2, False, ctx))
            case_log.append(ev2)
            ctx.cumulative_cost += ev2["amount"]
            res.next_free = end2
            res.last_seen[family] = end2
            # final complete
            slice3 = max(0.05, proc_hours - slice1 - slice2)
            end3 = add_working_time(end2, slice3, res.shift, allow_overtime=True)
            ev3 = dict(label=A["NegDone"], start=end2, end=end3, resource=res.rid, role=res.role,
                       family=family, amount=event_cost(A["NegDone"], res.role, slice3, True, ctx))
            case_log.append(ev3)
            ctx.cumulative_cost += ev3["amount"]
            res.next_free = end3
            res.last_seen[family] = end3
            ctx.last_resource = res.rid
            ctx.last_family = family
            return end3, res.rid, slice1 + slice2 + slice3
        else:
            # no preemption: compress into start + done
            slice2 = max(0.05, proc_hours - slice1)
            end2 = add_working_time(end1, slice2, res.shift, allow_overtime=allow_ot)
            ev2 = dict(label=A["NegDone"], start=end1, end=end2, resource=res.rid, role=res.role,
                       family=family, amount=event_cost(A["NegDone"], res.role, slice2, allow_ot, ctx))
            case_log.append(ev2)
            ctx.cumulative_cost += ev2["amount"]
            res.next_free = end2
            res.last_seen[family] = end2
            ctx.last_resource = res.rid
            ctx.last_family = family
            return end2, res.rid, slice1 + slice2

    # Normal activity: possibly with overtime if expedite/escalate/priority
    end_ts = add_working_time(start_cand, proc_hours, res.shift, allow_overtime=allow_ot)
    # Cost
    amt = event_cost(activity, res.role if res.rid != "Unknown" else None, proc_hours, overtime=allow_ot, ctx=ctx)
    ev = dict(label=activity, start=start_cand, end=end_ts, resource=res.rid,
              role=(res.role if res.rid != "Unknown" else None),
              family=family, amount=float(amt))
    case_log.append(ev)
    ctx.cumulative_cost += ev["amount"]
    res.next_free = end_ts
    res.last_seen[family] = end_ts

    # enforce binding-of-duty if requested
    if enforce_binding_of_duty_for is not None:
        first, second = enforce_binding_of_duty_for
        if activity == first:
            ctx.binding_map[second] = res.rid

    # separation-of-duty: record who did 'Approve'
    if separation_pair and activity == separation_pair[0]:
        ctx.binding_map[separation_pair[0]] = res.rid

    # update context
    if res.rid != "Unknown" and random.random() < UNKNOWN_RESOURCE_FRAC:
        # intentionally tag a few events with Unknown (to test defaults)
        ev["resource"] = "Unknown"

    ctx.last_resource = ev["resource"]
    ctx.last_family = family

    # SLA checks after each step
    elapsed_h = (end_ts - ctx.start_ts).total_seconds() / 3600.0
    if (not ctx.used_escalate) and elapsed_h > ctx.sla_hours:
        # Warn + Escalate (senior)
        w_start = end_ts
        w_end = add_working_time(w_start, 0.05, SHIFT_EARLY, allow_overtime=True)
        case_log.append(dict(label=A["Warn"], start=w_start, end=w_end, resource="Unknown", role=None,
                             family="System", amount=0.0))
        esc_start = w_end
        esc_end = add_working_time(esc_start, 0.2, SHIFT_EARLY, allow_overtime=True)
        esc_amt = ctx.escalate_fee
        case_log.append(dict(label=A["Escalate"], start=esc_start, end=esc_end, resource="R2", role="role:senior",
                             family="Approval", amount=esc_amt))
        ctx.cumulative_cost += esc_amt
        ctx.used_escalate = True

    return end_ts, res.rid, proc_hours


# ------------------------------
# Control-flow builders
# ------------------------------
def and_parallel(case_log: List[Dict[str, Any]], ctx: CaseCtx, pool: ResourcePool,
                 branches: List[List[str]], ready_ts: datetime) -> Tuple[datetime, List[float]]:
    """
    Schedule branches in parallel; join occurs at max end.
    Return (join_ts, branch_durations_hours)
    """
    ends = []
    branch_durations = []
    for seq in branches:
        cur = ready_ts
        local_start = cur
        for act in seq:
            cur, _, dur = schedule_activity(case_log, ctx, pool, act, cur)
        ends.append(cur)
        branch_durations.append((cur - local_start).total_seconds() / 3600.0)
    join_ts = max(ends)
    # synchronization latency + join handover
    join_end = add_working_time(join_ts, 0.1, SHIFT_EARLY, allow_overtime=True)
    case_log.append(dict(label=A["JoinChecks"] if any(a in (A["CheckCredit"], A["CheckInv"]) for a in sum(branches, [])) else A["KickJoin"],
                         start=join_ts, end=join_end, resource="Unknown", role=None, family="System",
                         amount=FIXED_COST_BASE.get(A["BidJoin"], 5.0)))
    return join_end, branch_durations

def or_parallel(case_log: List[Dict[str, Any]], ctx: CaseCtx, pool: ResourcePool,
                options: List[str], ready_ts: datetime) -> datetime:
    """
    OR-split: choose 1..k of options; run in parallel; OR-join after all chosen complete.
    """
    k = random.randint(1, len(options))
    chosen = random.sample(options, k=k)
    end, _ = and_parallel(case_log, ctx, pool, [[c] for c in chosen], ready_ts)
    return end

def review_fix_loop(case_log: List[Dict[str, Any]], ctx: CaseCtx, pool: ResourcePool,
                    ready_ts: datetime) -> datetime:
    """
    Geometric repeats of {Review -> Fix}. The pass probability depends on review duration memory:
    longer reviews → higher rework probability.
    """
    cur = ready_ts
    # review
    before = cur
    cur, _, dur = schedule_activity(case_log, ctx, pool, A["QA"], cur)
    review_duration = (cur - before).total_seconds() / 3600.0
    ctx.review_duration_memory = review_duration

    # probability to rework increases with longer review
    p_rework = min(0.7, 0.2 + 0.15 * (review_duration / 2.0))
    loops = 0
    while random.random() < p_rework and loops < 3:
        loops += 1
        # Fix (Revise) - binding-of-duty with Configure (same analyst if possible)
        cur, _, _ = schedule_activity(case_log, ctx, pool, A["Revise"], cur,
                                      enforce_binding_of_duty_for=(A["Configure"], A["Revise"]))
        # Review again
        before = cur
        cur, _, dur = schedule_activity(case_log, ctx, pool, A["QA"], cur)
        review_duration = (cur - before).total_seconds() / 3600.0
        ctx.rework_count += 1
        # rework premium
        case_log[-1]["amount"] += REWORK_PREMIUM_STEP * loops
        ctx.cumulative_cost += REWORK_PREMIUM_STEP * loops

    return cur

def multi_instance_bids(case_log: List[Dict[str, Any]], ctx: CaseCtx, pool: ResourcePool,
                        ready_ts: datetime) -> datetime:
    """
    Spawn n vendor bids; proceed after k complete (k-of-n). Cancel the rest with refund.
    """
    n = random.randint(*BIDS_N_RANGE)
    k = min(BIDS_K, n)
    starts = []
    ends = []
    durations = []
    # schedule all
    for i in range(n):
        cur, _, dur = schedule_activity(case_log, ctx, pool, A["VendorBid"], ready_ts, may_expedite=False)
        starts.append(ready_ts)
        ends.append(cur)
        durations.append(dur)
    # proceed after k fastest
    idx_sorted = sorted(range(n), key=lambda i: ends[i])
    kth_end = ends[idx_sorted[k-1]]
    # OR-join-like collector
    join_end = add_working_time(kth_end, 0.1, SHIFT_EARLY, allow_overtime=True)
    case_log.append(dict(label=A["BidJoin"], start=kth_end, end=join_end, resource="Unknown", role=None,
                         family="System", amount=FIXED_COST_BASE.get(A["BidJoin"], 5.0)))
    # cancel slow bidders (refund)
    for i in idx_sorted[k:]:
        cancel_start = ends[i]
        cancel_end = add_working_time(cancel_start, 0.05, SHIFT_EARLY, allow_overtime=True)
        refund = - random.uniform(5.0, 15.0)
        case_log.append(dict(label=A["Cancel"], start=cancel_start, end=cancel_end,
                             resource="Unknown", role=None, family="System", amount=refund))
        ctx.cumulative_cost += refund
    return join_end


# ------------------------------
# Build one case
# ------------------------------
def build_case(case_id: str, start_ts: datetime, pool: ResourcePool) -> Tuple[List[Dict[str, Any]], CaseCtx]:
    # era-specific settings
    if start_ts < ERA_CUTOFF:
        era = "E1"
        sla = SLA_HOURS_E1
        role_rates = ROLE_RATE_E1
        holding = WAIT_COST_PER_HOUR_E1
        esc_fee = ESCALATE_FEE_E1
    else:
        era = "E2"
        sla = SLA_HOURS_E2
        role_rates = ROLE_RATE_E2
        holding = WAIT_COST_PER_HOUR_E2
        esc_fee = ESCALATE_FEE_E2

    # priority & budget
    priority = random.random() < 0.18
    budget_cap = random.uniform(1200, 2200) if not priority else random.uniform(1800, 3000)

    ctx = CaseCtx(case_id=case_id, start_ts=start_ts, era=era, sla_hours=sla,
                  role_rates=role_rates, waiting_cost_per_hour=holding,
                  escalate_fee=esc_fee, priority=priority, budget_cap=budget_cap)

    case_log: List[Dict[str, Any]] = []
    cur = start_ts

    # --- Linear prefix with optional inserts ---
    # Start & Receive
    cur, _, _ = schedule_activity(case_log, ctx, pool, A["Receive"], cur)
    cur, _, _ = schedule_activity(case_log, ctx, pool, A["Qualify"], cur)

    # Optional Precheck and Scope
    if random.random() < 0.6:
        cur, _, _ = schedule_activity(case_log, ctx, pool, A["Precheck"], cur)
    cur, _, _ = schedule_activity(case_log, ctx, pool, A["Scope"], cur)

    # Optional multi-resource kickoff (as two parallel sub-activities + join)
    if random.random() < 0.25:
        ctx.had_and_split = True
        branches = [[A["KickAna"]], [A["KickSen"]]]
        cur, branch_durs = and_parallel(case_log, ctx, pool, branches, cur)
        ctx.and_branch_imbalances.append(abs(branch_durs[0] - branch_durs[1]))

    # Configure & Price (binding-of-duty with Revise, set below)
    cur, rid_conf, _ = schedule_activity(case_log, ctx, pool, A["Configure"], cur,
                                         enforce_binding_of_duty_for=(A["Configure"], A["Revise"]))
    cur, _, _ = schedule_activity(case_log, ctx, pool, A["Price"], cur)
    if random.random() < 0.3:
        cur, _, _ = schedule_activity(case_log, ctx, pool, A["Discount"], cur)
    cur, _, _ = schedule_activity(case_log, ctx, pool, A["PrepQuote"], cur)

    # --- AND-split: checks in parallel ---
    ctx.had_and_split = True
    branches = [[A["CheckCredit"]], [A["CheckInv"]]]
    cur, branch_durs = and_parallel(case_log, ctx, pool, branches, cur)
    ctx.and_branch_imbalances.append(abs(branch_durs[0] - branch_durs[1]))

    # --- OR-split: choose subset of reviews ---
    cur = or_parallel(case_log, ctx, pool, [A["Legal"], A["Tech"], A["Fin"]], cur)

    # --- Stochastic skip with memory: QA less likely if review duration was short ---
    cur = review_fix_loop(case_log, ctx, pool, cur)

    # --- Approval with Separation-of-Duty vs Audit ---
    cur, _, _ = schedule_activity(case_log, ctx, pool, A["Approve"], cur,
                                  separation_pair=(A["Approve"], A["Audit"]))
    # Sometimes audit happens before send (policy constraint)
    if random.random() < 0.4:
        cur, _, _ = schedule_activity(case_log, ctx, pool, A["Audit"], cur,
                                      separation_pair=(A["Approve"], A["Audit"]))

    # --- Send quote ---
    cur, _, _ = schedule_activity(case_log, ctx, pool, A["Send"], cur)

    # Cooling-off constraint: ensure some lag before Follow_Up
    # But we will place Negotiate/Revise/Submit before Follow_Up in most paths

    # --- XOR after Send: Accept fast-track vs Negotiate vs Revise loop ---
    branch_draw = random.random()
    if branch_draw < 0.25 and ctx.cumulative_cost < ctx.budget_cap:
        # Fast-track: customer accepts → Convert
        pass
    elif branch_draw < 0.7:
        # negotiation (with possible preemption)
        cur, _, _ = schedule_activity(case_log, ctx, pool, A["NegStart"], cur)
        # possibly revise once after negotiation
        if random.random() < 0.45:
            cur, _, _ = schedule_activity(case_log, ctx, pool, A["Revise"], cur,
                                          enforce_binding_of_duty_for=(A["Configure"], A["Revise"]))
    else:
        # explicit revise + re-approval
        cur, _, _ = schedule_activity(case_log, ctx, pool, A["Revise"], cur,
                                      enforce_binding_of_duty_for=(A["Configure"], A["Revise"]))
        cur, _, _ = schedule_activity(case_log, ctx, pool, A["Approve"], cur,
                                      separation_pair=(A["Approve"], A["Audit"]))

    # Submit & Follow-up (cooling-off: if too soon, insert Wait)
    cur, _, _ = schedule_activity(case_log, ctx, pool, A["Submit"], cur)
    min_lag = timedelta(hours=random.uniform(1.0, 6.0))
    if cur - start_ts < min_lag:
        # insert explicit Wait buffer to respect cooling-off
        w_end = add_working_time(cur, (min_lag - (cur - start_ts)).total_seconds()/3600.0, SHIFT_EARLY, allow_overtime=False)
        case_log.append(dict(label=A["Wait"], start=cur, end=w_end, resource="Unknown", role=None,
                             family="System", amount=ctx.waiting_cost_per_hour * ((w_end - cur).total_seconds()/3600.0)))
        ctx.cumulative_cost += case_log[-1]["amount"]
        cur = w_end
    cur, _, _ = schedule_activity(case_log, ctx, pool, A["Follow"], cur)

    # Budget-aware branching: if cumulative cost too high, go Vendor path; else in-house
    if ctx.cumulative_cost > ctx.budget_cap and random.random() < 0.7:
        # Vendor k-of-n bids then select vendor, place PO, receive
        cur = multi_instance_bids(case_log, ctx, pool, cur)
        cur, _, _ = schedule_activity(case_log, ctx, pool, A["VendorPick"], cur)
        cur, _, _ = schedule_activity(case_log, ctx, pool, A["PO"], cur)
        cur, _, _ = schedule_activity(case_log, ctx, pool, A["RecPO"], cur)
    else:
        # Internal material order (fixed material cost)
        cur, _, _ = schedule_activity(case_log, ctx, pool, A["Materials"], cur)

    # Convert to order (sometimes Cancel)
    if random.random() < 0.06:
        # cancellation with credit/refund
        c_end = add_working_time(cur, 0.1, SHIFT_EARLY, allow_overtime=True)
        refund = - random.uniform(50.0, 200.0)
        case_log.append(dict(label=A["Cancel"], start=cur, end=c_end, resource="Unknown", role=None,
                             family="System", amount=refund))
        ctx.cumulative_cost += refund
        # finish
        cur = c_end
    else:
        cur, _, _ = schedule_activity(case_log, ctx, pool, A["Convert"], cur)
        # Book order respects release slots (batching)
        slot = ceil_to_release_slot(max(cur, pool.resources[random.choice(list(pool.resources))].next_free),
                                    BOOK_ORDER_RELEASE_MINUTES)
        if slot > cur:
            # holding cost until release
            hold_h = (slot - cur).total_seconds()/3600.0
            case_log.append(dict(label=A["Wait"], start=cur, end=slot, resource="Unknown", role=None,
                                 family="System", amount=ctx.waiting_cost_per_hour * hold_h))
            ctx.cumulative_cost += case_log[-1]["amount"]
            cur = slot
        cur, _, _ = schedule_activity(case_log, ctx, pool, A["Book"], cur)
        cur, _, _ = schedule_activity(case_log, ctx, pool, A["Notify"], cur)

    # Optional post-approval Audit (if not done before)
    if not any(ev["label"] == A["Audit"] for ev in case_log) and random.random() < 0.25:
        cur, _, _ = schedule_activity(case_log, ctx, pool, A["Audit"], cur,
                                      separation_pair=(A["Approve"], A["Audit"]))

    # End marker (system)
    end_end = add_working_time(cur, 0.01, SHIFT_EARLY, allow_overtime=True)
    case_log.append(dict(label=A["End"], start=cur, end=end_end, resource="Unknown", role=None,
                         family="System", amount=0.0))

    return case_log, ctx


# ------------------------------
# Quality checks / summaries
# ------------------------------
def to_eventlog(cases: List[Tuple[str, List[Dict[str, Any]]]]) -> EventLog:
    L = EventLog()
    for cid, evs in cases:
        # sort by timestamp; ensure strict monotonicity by adding microsecond jitter on ties
        evs_sorted = sorted(evs, key=lambda e: (e["start"], e["end"], e["label"]))
        last_ts = None
        tr = Trace()
        tr.attributes["concept:name"] = cid  # so 'case:concept:name' appears in df
        jitter = 0
        for e in evs_sorted:
            st = e["start"]
            en = e["end"]
            if last_ts is not None and st <= last_ts:
                # push by small epsilon
                delta = (last_ts - st).total_seconds()
                bump = max(1, int(delta*1e6) + 1)
                st = st + timedelta(microseconds=bump)
                en = en + timedelta(microseconds=bump)
            last_ts = en
            # build pm4py Event with required attributes
            evt = Event({
                "concept:name": e["label"],
                "time:timestamp": en,  # use end times as event timestamp (finish-to-start semantics)
                "org:resource": e.get("resource", "Unknown"),
                "amount": float(e.get("amount", 0.0))
            })
            # Optional role attribute to aid experiments (not required)
            if e.get("role"):
                evt["org:role"] = e["role"]
            tr.append(evt)
        L.append(tr)
    return L

def check_schema_and_monotonicity(cases: List[Tuple[str, List[Dict[str, Any]]]]) -> Dict[str, Any]:
    issues = []
    lengths = []
    durations = []
    amounts = []
    for _, evs in cases:
        if not evs:
            continue
        # sorted
        evs_sorted = sorted(evs, key=lambda e: (e["start"], e["end"]))
        last = None
        lengths.append(len(evs_sorted))
        for e in evs_sorted:
            for key in ["label", "start", "end"]:
                if key not in e:
                    issues.append(f"Missing {key}")
            if e["end"] <= e["start"]:
                issues.append("Non-positive duration")
            if e.get("resource") is None:
                issues.append("Missing resource")
            if not isinstance(float(e.get("amount", 0.0)), float):
                issues.append("Amount not float")
            if last and e["start"] < last:
                issues.append("Timestamps not increasing within trace")
            last = e["end"]
            durations.append((e["end"] - e["start"]).total_seconds()/3600.0)
            amounts.append(float(e.get("amount", 0.0)))
    return {
        "issues": issues[:10],
        "n_issues": len(issues),
        "trace_len_mean": float(np.mean(lengths)) if lengths else 0.0,
        "trace_len_quantiles": np.quantile(lengths, [0, .25, .5, .75, 1]).tolist() if lengths else [],
        "event_dur_quantiles_h": np.quantile(durations, [0, .25, .5, .75, .9, .99]).tolist() if durations else [],
        "amount_quantiles": np.quantile(amounts, [0, .25, .5, .75, .9, .99]).tolist() if amounts else []
    }

def ambiguity_check(cases: List[Tuple[str, List[Dict[str, Any]]]], samples: int = 100) -> Dict[str, Any]:
    # Map prefix -> next set
    prefix_map: Dict[Tuple[str, ...], set] = {}
    all_positions: List[Tuple[int, int]] = []
    for idx, (_, evs) in enumerate(cases):
        labels = [e["label"] for e in sorted(evs, key=lambda x: (x["start"], x["end"]))]
        for i in range(1, len(labels)):
            pref = tuple(labels[:i])
            nxt = labels[i]
            prefix_map.setdefault(pref, set()).add(nxt)
            all_positions.append((idx, i))
    if not all_positions:
        return {"sampled": 0, "ambiguous_fraction": 0.0}

    sampled_prefixes = random.sample(list(prefix_map.keys()), k=min(samples, len(prefix_map)))
    ambiguous = sum(1 for p in sampled_prefixes if len(prefix_map[p]) >= 2)
    return {
        "sampled": len(sampled_prefixes),
        "ambiguous_fraction": ambiguous / max(1, len(sampled_prefixes))
    }

def remaining_time_signal(cases: List[Tuple[str, List[Dict[str, Any]]]]) -> Dict[str, Any]:
    # compute remaining time from each event to End
    remain_hours = []
    for _, evs in cases:
        evs_sorted = sorted(evs, key=lambda e: (e["start"], e["end"]))
        if not evs_sorted:
            continue
        end_ts = evs_sorted[-1]["end"]
        for e in evs_sorted[:-1]:
            remain_hours.append((end_ts - e["end"]).total_seconds()/3600.0)
    if not remain_hours:
        return {}
    return {
        "mean_h": float(np.mean(remain_hours)),
        "std_h": float(np.std(remain_hours)),
        "quantiles_h": np.quantile(remain_hours, [0, .25, .5, .75, .9, .99]).tolist()
    }

def parallelism_stats(cases: List[Tuple[str, List[Dict[str, Any]]]], ctxs: List[CaseCtx]) -> Dict[str, Any]:
    had_and = sum(1 for c in ctxs if c.had_and_split)
    imb = [x for c in ctxs for x in c.and_branch_imbalances]
    return {
        "share_with_AND": had_and / max(1, len(cases)),
        "avg_branch_imbalance_h": float(np.mean(imb)) if imb else 0.0
    }

def drift_stats(cases: List[Tuple[str, List[Dict[str, Any]]]], ctxs: List[CaseCtx]) -> Dict[str, Any]:
    per_case_dur = []
    per_case_amt = []
    per_case_era = []
    for (cid, evs), ctx in zip(cases, ctxs):
        evs_sorted = sorted(evs, key=lambda e: (e["start"], e["end"]))
        if not evs_sorted:
            continue
        dur_h = (evs_sorted[-1]["end"] - evs_sorted[0]["start"]).total_seconds()/3600.0
        amt = sum(float(e["amount"]) for e in evs_sorted)
        per_case_dur.append(dur_h)
        per_case_amt.append(amt)
        per_case_era.append(ctx.era)
    # group by era
    out = {}
    for metric_name, vals in [("duration_h", per_case_dur), ("amount", per_case_amt)]:
        e1 = [v for v, er in zip(vals, per_case_era) if er == "E1"]
        e2 = [v for v, er in zip(vals, per_case_era) if er == "E2"]
        out[f"{metric_name}_mean_E1"] = float(np.mean(e1)) if e1 else 0.0
        out[f"{metric_name}_mean_E2"] = float(np.mean(e2)) if e2 else 0.0
        out[f"{metric_name}_diff_E2_minus_E1"] = out[f"{metric_name}_mean_E2"] - out[f"{metric_name}_mean_E1"]
    return out


# ------------------------------
# Main: generate unique traces and export
# ------------------------------
def execute_script():
    print("Generating Quote-to-Order log with rich patterns ...")
    pool = ResourcePool(SIM_START)

    # Generate candidate case starts with bursts/seasonality
    start_times = generate_case_start_times(TARGET_DISTINCT_TRACES * 2)  # extra to help uniqueness

    cases: List[Tuple[str, List[Dict[str, Any]]]] = []
    ctxs: List[CaseCtx] = []
    seen_signatures = set()

    attempt = 0
    case_index = 0
    while len(cases) < TARGET_DISTINCT_TRACES and attempt < MAX_GENERATION_ATTEMPTS:
        attempt += 1
        if case_index >= len(start_times):
            # extend if needed
            start_times += generate_case_start_times(TARGET_DISTINCT_TRACES)
        ts = start_times[case_index]
        cid = f"Q2O_{case_index+1:04d}"
        case_index += 1

        evs, ctx = build_case(cid, ts, pool)

        # Signature for "distinct traces": sequence of labels (no timestamps/resources)
        sig = tuple(e["label"] for e in sorted(evs, key=lambda x: (x["start"], x["end"])))
        if sig in seen_signatures:
            continue  # skip duplicates
        seen_signatures.add(sig)
        cases.append((cid, evs))
        ctxs.append(ctx)

    # Cap at TARGET_DISTINCT_TRACES (just in case)
    cases = cases[:TARGET_DISTINCT_TRACES]
    ctxs = ctxs[:TARGET_DISTINCT_TRACES]

    # Convert to pm4py EventLog
    log = to_eventlog(cases)

    # Export to XES (gz)
    out_path = "quote2order.xes.gz"
    try:
        # pm4py's exporter handles gzip when extension is .gz
        xes_exporter.apply(log, out_path)
    except Exception:
        # fallback with explicit param
        xes_exporter.apply(log, out_path, parameters={"gzip": True})
    print(f"Saved: {out_path}")

    # -------------- Sanity & quality checks --------------
    print("\n== Quality & Sanity Checks ==")
    chk = check_schema_and_monotonicity(cases)
    for k, v in chk.items():
        print(f"{k}: {v}")

    amb = ambiguity_check(cases, samples=100)
    print("\nAmbiguity check (prefix -> >=2 distinct next activities):")
    print(amb)

    rem = remaining_time_signal(cases)
    print("\nRemaining time distribution (hours):")
    print(rem)

    par = parallelism_stats(cases, ctxs)
    print("\nParallelism check:")
    print(par)

    drift = drift_stats(cases, ctxs)
    print("\nDrift check (era-wise means, E2 - E1 deltas):")
    print(drift)

    # Optional DataFrame summary (if pandas available)
    try:
        df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
        # ensure 'case:concept:name' present
        if HAVE_PANDAS:
            # quick summaries
            print("\nEvent dataframe head:")
            print(df.head(5).to_string(index=False))
            # trace length distribution
            tl = df.groupby("case:concept:name").size()
            print("\nTrace length quantiles:", tl.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict())
            print("Events total:", len(df))
            print("Traces total:", df["case:concept:name"].nunique())
    except Exception as e:
        print("DataFrame conversion skipped or failed:", str(e))

    print("\nDone.")

# ------------------------------
# Minimal reference trace injection (for comments only)
# Example path hit in many cases:
# Start/Receive → Qualify → Configure/Price/Prepare → AND {Check_Credit || Check_Inventory} → Join
# → OR {Legal, Tech, Finance} subset → QA → Approve → Send → Negotiate (preempted)
# → Revise (binding-of-duty) → Submit → Follow_Up (cooling-off) → Vendor bids (k-of-n)
# → Select_Vendor → Place_PO → Receive_PO → Convert_to_Order → Book_Order (batched) → Notify → End
# Along the way: Handover delays on resource switches; SLA breach triggers Warn/Escalate;
# waiting costs accrue; rework premium if QA loops; overtime for Expedited activities.
# ------------------------------

if __name__ == "__main__":
    execute_script()
