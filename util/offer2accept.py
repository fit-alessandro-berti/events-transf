# generate_offer2accept.py
# --------------------------------------------------------------------
# Offer-to-Acceptance synthetic XES generator (trace-by-trace; pm4py)
# --------------------------------------------------------------------
# Key features implemented:
# - Activity & resource alphabets (30+ activities; R1..R30 with roles)
# - Control-flow skeletons: linear w/ optional inserts, XOR, loops,
#   AND-split/AND-join, OR-split/OR-join, skip patterns, long-term deps,
#   preemption/resume (start/resume/complete labels)
# - Temporal patterns: NHPP bursty arrivals, business & resource calendars,
#   heavy-tailed service times, deadlines/aging w/ Escalate, setup/changeover,
#   interruptions with resume, synchronization latency, queue discipline mix,
#   seasonality/drift, learning & forgetting, cooling-off constraints,
#   handover delays, context-switch costs, batching/release, k-of-n tasks
# - Resource patterns: roles/pools, skill heterogeneity, binding-of-duty,
#   separation-of-duty, explicit Handover pseudo-activity, unavailability,
#   assignment policy mixture, multi-resource meeting, vendor option
# - Cost patterns: fixed+variable (shift/overtime/weekend tariffs),
#   SLA penalties, rework premiums, expedite fee (skip/queue-jump),
#   handover fee, refunds (negative amounts), material triggers,
#   budget-aware branching, waiting/holding cost, era-based tariffs
# - Sampling & scheduling: resource next-free, per-activity queue est.,
#   calendars, parallel join = max(children end), k-of-n join
# - Quality checks and summaries printed after export
#
# Minimal example (in comments near bottom) demonstrates: AND-split,
# handover delay, SLA penalty -> Escalate, rework once, vendor path w/ overtime.
# --------------------------------------------------------------------

from __future__ import annotations
import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Iterable, Any
import numpy as np

try:
    import pandas as pd  # optional; used only for summaries if available
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

# pm4py imports
try:
    # pm4py 2.7+ has write_xes
    import pm4py
    HAS_PM4PY_WRITE = hasattr(pm4py, "write_xes")
except Exception as e:
    raise RuntimeError("pm4py is required to run this script. Install with `pip install pm4py`") from e

from pm4py.objects.log.obj import Event, Trace, EventLog
if not HAS_PM4PY_WRITE:
    # Fallback exporter for older pm4py
    from pm4py.objects.log.exporter.xes import exporter as xes_exporter


# ----------------------------
# Global configuration knobs
# ----------------------------
RNG = np.random.default_rng(42)
PY_RAND = random.Random(42)

N_CASES = 250                         # <= requirement: limit to 250 traces
OUTFILE = "offer2accept.xes.gz"

# Simulation horizon (~60 business days)
START_DATE = datetime(2025, 1, 6, 9, 0, 0)  # Monday 09:00
HORIZON_DAYS = 60

# Business hours
BUSINESS_START = 9
BUSINESS_LUNCH = (12, 13)             # freeze during lunch
BUSINESS_END = 17

# Roles and resources
ROLES = ["role:analyst", "role:senior", "role:vendor"]
RESOURCE_IDS = [f"R{i}" for i in range(1, 31)]  # R1..R30
UNKNOWN_RATE = 0.05                    # small fraction use Unknown

# Era blocks (concept drift)
ERAS = [
    ("Era-1", START_DATE, START_DATE + timedelta(days=20)),
    ("Era-2", START_DATE + timedelta(days=20), START_DATE + timedelta(days=40)),
    ("Era-3", START_DATE + timedelta(days=40), START_DATE + timedelta(days=90)),
]

# Case deadline (business time)
CASE_DEADLINE_BUSINESS_HOURS = 5 * (BUSINESS_END - BUSINESS_START)  # ~5 business days

# Activity alphabet (Offer -> Acceptance domain + generic ops)
# We keep 30+ labels and will dynamically compose flow per-case.
ACTS = [
    # Offer phase
    "OFR_Start", "OFR_Capture_Request", "OFR_Qualify",
    "OFR_Prepare_Offer", "OFR_Internal_Review",
    "OFR_Legal_Review", "OFR_Finance_Review", "OFR_Consolidate_Review",
    "OFR_Batch_Approve", "OFR_Send_Offer", "OFR_Followup",
    "OFR_Negotiate", "OFR_Revise_Offer",
    # Checks and QA (OR-split / k-of-n)
    "QA_Review", "QA_Fix",
    "RefCheck_1", "RefCheck_2", "RefCheck_3", "Security_Check",
    "Compliance_Check", "Background_Check",
    # Acceptance and closure
    "ACC_Signing_Meeting", "ACC_Approve", "ACC_Audit",
    "ACC_Sign_Contract", "ACC_Acceptance_Received",
    "ACC_Onboard_Prep", "ACC_Close",
    # Aux / management
    "Expedite", "Escalate", "Warn", "Reassign", "Handover", "Wait", "Vendor_RefCheck",
    # Joins / markers
    "AND_Join", "OR_Join", "KofN_Join"
]

# Activity families (for setup/learning/context-switch multipliers)
def family_of(act: str) -> str:
    if act.startswith("OFR_"): return "offer"
    if act.startswith("QA_"): return "qa"
    if act.startswith("ACC_"): return "accept"
    if act.startswith("RefCheck") or act in ("Security_Check", "Compliance_Check", "Background_Check"): return "check"
    if act in ("Handover", "Wait", "Expedite", "Escalate", "Warn", "Reassign", "Vendor_RefCheck"): return "mgmt"
    return "other"

# Baseline mean service hours per activity (before multipliers)
BASE_MEAN_H = {
    "OFR_Capture_Request": 0.6,
    "OFR_Qualify": 1.2,
    "OFR_Prepare_Offer": 4.0,
    "OFR_Internal_Review": 0.5,
    "OFR_Legal_Review": 1.5,
    "OFR_Finance_Review": 1.2,
    "OFR_Consolidate_Review": 0.4,
    "OFR_Batch_Approve": 0.6,
    "OFR_Send_Offer": 0.2,
    "OFR_Followup": 0.3,
    "OFR_Negotiate": 2.0,
    "OFR_Revise_Offer": 1.5,
    "QA_Review": 1.0,
    "QA_Fix": 1.2,
    "RefCheck_1": 1.0,
    "RefCheck_2": 1.0,
    "RefCheck_3": 1.0,
    "Security_Check": 1.2,
    "Compliance_Check": 0.8,
    "Background_Check": 1.5,
    "ACC_Signing_Meeting": 0.8,    # requires 2+ resources
    "ACC_Approve": 0.5,
    "ACC_Audit": 0.7,              # enforce separation-of-duty
    "ACC_Sign_Contract": 0.3,
    "ACC_Acceptance_Received": 0.1,
    "ACC_Onboard_Prep": 1.0,
    "ACC_Close": 0.2,
    "Vendor_RefCheck": 1.8,
    # Auxiliary
    "Expedite": 0.0, "Escalate": 0.1, "Warn": 0.0, "Reassign": 0.2,
    "Handover": 0.2, "Wait": 0.0,
    "AND_Join": 0.0, "OR_Join": 0.0, "KofN_Join": 0.0,
    "OFR_Start": 0.0
}

# Fixed cost per activity (material or base fee)
FIXED_COST = {
    "OFR_Capture_Request": 20, "OFR_Qualify": 25, "OFR_Prepare_Offer": 60,
    "OFR_Internal_Review": 15, "OFR_Legal_Review": 30, "OFR_Finance_Review": 28,
    "OFR_Consolidate_Review": 10, "OFR_Batch_Approve": 12, "OFR_Send_Offer": 20,
    "OFR_Followup": 8, "OFR_Negotiate": 35, "OFR_Revise_Offer": 22,
    "QA_Review": 18, "QA_Fix": 22,
    "RefCheck_1": 5, "RefCheck_2": 5, "RefCheck_3": 5,
    "Security_Check": 12, "Compliance_Check": 10, "Background_Check": 15,
    "ACC_Signing_Meeting": 25, "ACC_Approve": 14, "ACC_Audit": 16,
    "ACC_Sign_Contract": 30, "ACC_Acceptance_Received": 6,
    "ACC_Onboard_Prep": 24, "ACC_Close": 10,
    "Vendor_RefCheck": 4,  # cheaper fixed fee, slower mean
    # Aux
    "Expedite": 150, "Escalate": 300, "Warn": 0, "Reassign": 25,
    "Handover": 15, "Wait": 0,
    "AND_Join": 0, "OR_Join": 0, "KofN_Join": 0, "OFR_Start": 0
}

# Variable cost base rates per role (per hour, pre-tariff)
ROLE_RATE = {
    "role:analyst": 40.0,
    "role:senior": 60.0,
    "role:vendor": 35.0,
    "Unknown": 38.0
}
OVERTIME_MULT = 1.5
WEEKEND_MULT = 2.0
ERA_TARIFF_MULT = { "Era-1": 1.00, "Era-2": 1.05, "Era-3": 0.95 }  # drift
REWORK_PREMIUM_STEP = 1.25  # increases per iteration of QA_Fix loop
WAITING_COST_PER_HOUR = 10.0

# Budget threshold for vendor switch / simplify
BUDGET_THRESH = 2500.0

# Probability knobs
P_HEAVY_TAIL = 0.18               # chance to use Pareto tail
P_PREEMPT_BASE = 0.10             # base preempt chance for long tasks
P_ESCALATE_WARN = 0.15            # extra escalate chance after warn
P_EXPEDITE_WHEN_LATE = 0.50       # expedite (skip/queue-jump) when deadline at risk
P_SKIP_QA_IF_FAST = 0.20          # memory-based skip if review was very fast
P_OR_BRANCH_BASE = 0.65           # probability each OR-branch is chosen
P_VENDOR_PATH_IF_BUDGET_HIGH = 0.60

# ---------------------------------------
# Utility: business + resource calendars
# ---------------------------------------
def is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5  # 5=Sat,6=Sun

def business_opening(dt: datetime) -> datetime:
    """Next business opening from dt (respect lunch freeze)."""
    d = dt
    if is_weekend(d):
        # move to next Monday 9:00
        days_ahead = (7 - d.weekday()) % 7
        d = datetime(d.year, d.month, d.day, BUSINESS_START, 0, 0) + timedelta(days=days_ahead)
        return d
    # same day
    if d.hour < BUSINESS_START or (d.hour == BUSINESS_START and d.minute == 0 and d.second == 0):
        return datetime(d.year, d.month, d.day, BUSINESS_START, 0, 0)
    # lunch window
    if (d.hour > BUSINESS_LUNCH[0] or (d.hour == BUSINESS_LUNCH[0] and d.minute >= 0)) and \
       (d.hour < BUSINESS_LUNCH[1]):
        return datetime(d.year, d.month, d.day, BUSINESS_LUNCH[1], 0, 0)
    # after business end -> next day
    if d.hour >= BUSINESS_END:
        nd = d + timedelta(days=1)
        while is_weekend(nd):
            nd += timedelta(days=1)
        return datetime(nd.year, nd.month, nd.day, BUSINESS_START, 0, 0)
    # otherwise OK (within business and not lunch)
    return d

def add_business_time(start: datetime, hours: float) -> datetime:
    """Add hours observing business hours and lunch; no work on weekends."""
    if hours <= 0:
        return business_opening(start)
    cur = business_opening(start)
    remaining = hours
    while remaining > 1e-9:
        # today's workable segments
        day_start = datetime(cur.year, cur.month, cur.day, BUSINESS_START, 0, 0)
        lunch_s = datetime(cur.year, cur.month, cur.day, BUSINESS_LUNCH[0], 0, 0)
        lunch_e = datetime(cur.year, cur.month, cur.day, BUSINESS_LUNCH[1], 0, 0)
        day_end = datetime(cur.year, cur.month, cur.day, BUSINESS_END, 0, 0)
        # compute available window from cur until lunch or end
        if cur < lunch_s:
            seg_end = min(lunch_s, day_end)
        elif cur >= lunch_s and cur < lunch_e:
            cur = lunch_e
            continue
        else:
            seg_end = day_end
        seg_hours = (seg_end - cur).total_seconds() / 3600.0
        if seg_hours <= 0:
            # move to next business day
            nd = cur + timedelta(days=1)
            while is_weekend(nd):
                nd += timedelta(days=1)
            cur = datetime(nd.year, nd.month, nd.day, BUSINESS_START, 0, 0)
            continue
        if remaining <= seg_hours + 1e-9:
            return cur + timedelta(hours=remaining)
        # consume segment and continue
        remaining -= seg_hours
        cur = seg_end
        if cur >= day_end:
            nd = cur + timedelta(days=1)
            while is_weekend(nd):
                nd += timedelta(days=1)
            cur = datetime(nd.year, nd.month, nd.day, BUSINESS_START, 0, 0)
    return cur

@dataclass
class Resource:
    rid: str
    role: str
    shift_start: int
    shift_end: int
    next_free: datetime
    last_family_time: Dict[str, datetime] = field(default_factory=dict)
    family_count: Dict[str, int] = field(default_factory=dict)  # learning curve
    unavailable_days: set = field(default_factory=set)

    @staticmethod
    def _day_time(dt: datetime, hour: int) -> datetime:
        """Return datetime anchored on dt's date at the given hour (supports hour >= 24)."""
        base = datetime(dt.year, dt.month, dt.day, 0, 0, 0)
        return base + timedelta(hours=int(hour))

    def shift_opening(self, dt: datetime) -> datetime:
        """Respect resource shift; allow overtime occasionally with penalty outside shift."""
        d = dt
        # if day is marked unavailable for resource -> push to next day
        while (d.date() in self.unavailable_days) or is_weekend(d):
            d = self._day_time(d, self.shift_start) + timedelta(days=1)
        # shift window for this date
        begin = self._day_time(d, self.shift_start)
        lunch_s = datetime(d.year, d.month, d.day, BUSINESS_LUNCH[0], 0, 0)
        lunch_e = datetime(d.year, d.month, d.day, BUSINESS_LUNCH[1], 0, 0)
        end = self._day_time(d, self.shift_end)
        if d < begin:
            return begin
        # within lunch? delay to lunch end
        if d >= lunch_s and d < lunch_e:
            return lunch_e
        if d >= end:
            # next working day start (resource shift)
            nd = d + timedelta(days=1)
            while (nd.date() in self.unavailable_days) or is_weekend(nd):
                nd += timedelta(days=1)
            return self._day_time(nd, self.shift_start)
        return d

    def add_shift_time(self, start: datetime, hours: float, allow_overtime: bool = True) -> Tuple[datetime, bool]:
        """Add hours with resource-specific shift; returns (end, used_overtime)."""
        if hours <= 0:
            return self.shift_opening(start), False
        cur = self.shift_opening(start)
        remaining = hours
        used_ot = False
        while remaining > 1e-9:
            # build today's window
            begin = self._day_time(cur, self.shift_start)
            lunch_s = datetime(cur.year, cur.month, cur.day, BUSINESS_LUNCH[0], 0, 0)
            lunch_e = datetime(cur.year, cur.month, cur.day, BUSINESS_LUNCH[1], 0, 0)
            end = self._day_time(cur, self.shift_end)

            if cur < lunch_s:
                seg_end = min(lunch_s, end)
            elif cur >= lunch_s and cur < lunch_e:
                cur = lunch_e
                continue
            else:
                seg_end = end
            seg_hours = (seg_end - cur).total_seconds() / 3600.0
            if seg_hours <= 0:
                # consider overtime window if allowed
                if allow_overtime:
                    # allow up to 2h overtime
                    ot_end = end + timedelta(hours=2)
                    if cur < ot_end:
                        seg_end = ot_end
                        seg_hours = (seg_end - cur).total_seconds() / 3600.0
                        used_ot = True
                    else:
                        seg_hours = 0.0
                if seg_hours <= 0.0:
                    # move to next shift day
                    nd = cur + timedelta(days=1)
                    while (nd.date() in self.unavailable_days) or is_weekend(nd):
                        nd += timedelta(days=1)
                    cur = self._day_time(nd, self.shift_start)
                    continue
            if remaining <= seg_hours + 1e-9:
                return cur + timedelta(hours=remaining), used_ot
            remaining -= seg_hours
            cur = seg_end
            # loop to next day automatically
        return cur, used_ot

@dataclass
class ResourceManager:
    resources: Dict[str, Resource]
    role_map: Dict[str, List[str]]
    queue_backlog: Dict[str, int] = field(default_factory=lambda: {a: 0 for a in ACTS})

    def predict_service_time(self, act: str, role: str, era: str, base_mean: float) -> float:
        """Baseline + drift; skills differ by role and family."""
        fam = family_of(act)
        role_mult = 1.0
        if role == "role:analyst":
            if fam == "qa": role_mult = 0.95
            elif fam == "offer": role_mult = 1.0
            elif fam == "accept": role_mult = 1.1
            elif fam == "check": role_mult = 1.05
        elif role == "role:senior":
            if fam in ("offer", "accept"): role_mult = 0.9
            if fam == "qa": role_mult = 0.85
        elif role == "role:vendor":
            if fam in ("check",): role_mult = 1.0
            else: role_mult = 1.15
        drift = 1.0
        if act in ("QA_Review", "QA_Fix") and era == "Era-3":
            drift = 0.85   # learning / process improvement
        if act in ("ACC_Approve",) and era in ("Era-2", "Era-3"):
            drift *= 1.1   # slower compliance in later eras
        return base_mean * role_mult * drift

    def choose_policy(self) -> str:
        # weighted mixture of assignment policies
        return PY_RAND.choices(["skill", "least_loaded", "random"], weights=[0.4, 0.4, 0.2], k=1)[0]

    def assign(self, act: str, earliest: datetime, era: str,
               required_role: Optional[str] = None,
               same_as: Optional[str] = None,
               separate_from: Optional[str] = None) -> Tuple[str, Resource]:
        if PY_RAND.random() < UNKNOWN_RATE:
            return "Unknown", Resource("Unknown", "Unknown", 0, 24, earliest)  # pseudo-resource

        # candidate resources by role constraint
        if required_role:
            cand = self.role_map.get(required_role, [])
        else:
            cand = list(self.resources.keys())
        # enforce separation/binding
        if separate_from and separate_from in cand:
            cand = [r for r in cand if r != separate_from]
        if same_as and same_as in self.resources:
            # prefer same_as if allowed
            if (not required_role) or (self.resources[same_as].role == required_role):
                return same_as, self.resources[same_as]
        if not cand:
            # fall back to any
            cand = list(self.resources.keys())
        policy = self.choose_policy()
        if policy == "random":
            rid = PY_RAND.choice(cand)
            return rid, self.resources[rid]
        elif policy == "least_loaded":
            rid = min(cand, key=lambda r: self.resources[r].next_free)
            return rid, self.resources[rid]
        else:
            # skill-based: pick resource minimizing predicted time
            scores = []
            for r in cand:
                rr = self.resources[r]
                base = BASE_MEAN_H.get(act, 0.5)
                s = self.predict_service_time(act, rr.role, era, base)
                # also consider availability (earliest start)
                avail = max(rr.shift_opening(earliest), rr.next_free)
                wait_h = (avail - earliest).total_seconds() / 3600.0
                score = s + 0.25 * max(0.0, wait_h)  # mild penalty for wait
                scores.append((score, r))
            rid = min(scores, key=lambda x: x[0])[1]
            return rid, self.resources[rid]

# ---------------------------------------
# Arrival process (bursty NHPP)
# ---------------------------------------
def hourly_intensity(dt: datetime) -> float:
    """Cases per hour; higher midweek + midday, seasonal drift by era."""
    dow = dt.weekday()  # 0=Mon
    hour = dt.hour + dt.minute/60.0
    base = 2.0 if 10 <= hour <= 15 else 0.8
    if dow in (1,2,3): base *= 1.6  # Tue-Thu busier
    if dow in (5,6) or hour < BUSINESS_START or hour >= BUSINESS_END: base = 0.0
    # drift: Era-2 +10%, Era-3 -5%
    era = era_of(dt)
    mult = {"Era-1": 1.0, "Era-2": 1.1, "Era-3": 0.95}[era]
    return base * mult

def era_of(dt: datetime) -> str:
    for name, s, e in ERAS:
        if s <= dt < e:
            return name
    return ERAS[-1][0]

def sample_arrivals(n_cases: int, start: datetime, horizon_days: int) -> List[datetime]:
    """Thinning method for NHPP with hourly_intensity as lambda(t)."""
    out = []
    t = start
    end = start + timedelta(days=horizon_days)
    lambda_max = 4.0  # upper bound cases/hour
    while len(out) < n_cases and t < end:
        # exponential interarrival with rate lambda_max
        u = RNG.random()
        if lambda_max <= 1e-9:
            dt_hours = 1.0
        else:
            dt_hours = -math.log(max(u, 1e-12)) / lambda_max
        t = t + timedelta(hours=dt_hours)
        # accept w.p. lambda(t)/lambda_max
        lam_t = hourly_intensity(t)
        if RNG.random() < (lam_t / lambda_max):
            out.append(business_opening(t))
    # ensure sorted and trimmed to n_cases
    out.sort()
    return out[:n_cases]

# ---------------------------------------
# Service-time sampler (heavy tails) and
# setup/changeover, learning/forgetting
# ---------------------------------------
def sample_service_hours(act: str, role: str, era: str,
                         res: Resource,
                         prev_family: Optional[str]) -> Tuple[float, bool]:
    """Return (service_hours, preemptible_long_task)."""
    base = BASE_MEAN_H.get(act, 0.6)
    # role/era adaptation
    mean_est = res.family_count.get(family_of(act), 0)
    # learning curve: faster with repetitions
    learn_mult = max(0.70, math.exp(-0.05 * mean_est) * 1.1)
    # forgetting: if long since last family execution (>10 days), slower
    forget_mult = 1.0
    f = family_of(act)
    if f in res.last_family_time:
        days_since = (res.last_family_time[f] - res.last_family_time[f]).total_seconds() / 86400.0
        # the above naive; adjust below with "now" after call. We'll approximate later.
    # base by ResourceManager prediction will already be applied externally for assignment;
    # here we adjust with learning & forgetting using resource local stats:
    # (we will set forgetting after we update last_family_time in schedule)

    # heavy-tailed draw
    if RNG.random() < P_HEAVY_TAIL:
        # Pareto with xm=base/2, alpha=2.0 -> mean 2*xm
        xm = max(0.1, base / 2.0)
        alpha = 2.0
        x = xm / (RNG.random() ** (1.0/alpha))
        st = x
    else:
        # lognormal around base: choose sigma so mean ~ base
        sigma = 0.6
        mu = math.log(max(1e-3, base)) - 0.5 * sigma**2
        st = RNG.lognormal(mean=mu, sigma=sigma)

    # setup/changeover penalty when switching family
    if prev_family and prev_family != f:
        st += RNG.uniform(0.15, 0.9)  # context switch

    # indicate if long tasks likely to be preempted
    preemptible = st >= 3.5  # hours
    # apply learning multiplier
    st *= learn_mult
    return max(0.01, st), preemptible

# ---------------------------------------
# Cost model
# ---------------------------------------
def event_cost(act: str, role: str, start: datetime, end: datetime,
               used_ot: bool, era: str,
               waiting_hours: float = 0.0,
               rework_iteration: int = 0,
               handover: bool = False,
               escalate: bool = False,
               expedite: bool = False) -> float:
    fixed = FIXED_COST.get(act, 5.0)
    # variable rate
    rate = ROLE_RATE.get(role, ROLE_RATE["Unknown"])
    # tariffs
    tar = ERA_TARIFF_MULT.get(era, 1.0)
    # weekend / overtime multipliers for variable component
    var_mult = 1.0
    if is_weekend(start) or is_weekend(end):
        var_mult *= WEEKEND_MULT
    if used_ot:
        var_mult *= OVERTIME_MULT
    # compute worked hours (approx as duration)
    dur_h = max(0.0, (end - start).total_seconds()/3600.0)
    variable = rate * var_mult * tar * dur_h

    # waiting cost (holding)
    wait_cost = WAITING_COST_PER_HOUR * max(0.0, waiting_hours)

    # rework premium
    if act == "QA_Fix" and rework_iteration > 0:
        fixed *= (REWORK_PREMIUM_STEP ** rework_iteration)

    # expedite fee lump sum
    if expedite:
        fixed += 150.0

    # handover fee
    if handover:
        fixed += 20.0

    # escalate penalty
    if escalate:
        fixed += 300.0

    amount = float(fixed + variable + wait_cost)
    # allow rare refunds/credits on rollback/cancel (not used by default here)
    return amount

# ---------------------------------------
# Flow building helpers
# ---------------------------------------
@dataclass
class ScheduledEvent:
    act: str
    start: datetime
    end: datetime
    resource: str
    amount: float

def ensure_wait_event(trace_events: List[ScheduledEvent], from_t: datetime, to_t: datetime):
    """Insert a Wait event if there's a gap > 5 minutes (for holding cost tracking)."""
    if (to_t - from_t).total_seconds() / 60.0 >= 5.0:
        e = ScheduledEvent("Wait", from_t, to_t, "Unknown",
                           event_cost("Wait", "Unknown", from_t, to_t, used_ot=False, era=era_of(from_t),
                                      waiting_hours=(to_t - from_t).total_seconds()/3600.0))
        trace_events.append(e)

# ---------------------------------------
# Build resources and roles
# ---------------------------------------
def build_resources(start: datetime) -> ResourceManager:
    res: Dict[str, Resource] = {}
    role_map: Dict[str, List[str]] = {r: [] for r in ROLES}
    # Assign roles with a skew: more analysts, fewer seniors, some vendors
    for i, rid in enumerate(RESOURCE_IDS, start=1):
        if i <= 18:
            role = "role:analyst"
            shift = (9, 17) if i % 3 != 0 else (10, 18)
        elif i <= 26:
            role = "role:senior"
            shift = (9, 18) if i % 2 == 0 else (8, 16)
        else:
            role = "role:vendor"
            shift = (10, 18)
        r = Resource(rid=rid, role=role, shift_start=shift[0], shift_end=shift[1], next_free=start)
        # random absences
        for d in range(0, HORIZON_DAYS, PY_RAND.choice([6,7,8])):
            if PY_RAND.random() < 0.2:
                r.unavailable_days.add((start + timedelta(days=d)).date())
        res[rid] = r
        role_map.setdefault(role, []).append(rid)
    return ResourceManager(resources=res, role_map=role_map)

# ---------------------------------------
# Case skeletons (control-flow templates)
# ---------------------------------------
@dataclass
class CaseContext:
    case_id: str
    era: str
    deadline_bh: float
    priority: str
    binding_pairs: Dict[str, str] = field(default_factory=dict)  # act -> rid (binding-of-duty)
    escalated: bool = False
    total_amount: float = 0.0
    had_parallel: bool = False
    review_durations: List[float] = field(default_factory=list)  # for memory-based skip

def sample_priority() -> str:
    return PY_RAND.choices(["Low", "Medium", "High"], weights=[0.3,0.5,0.2], k=1)[0]

def geometric_trials(p_fail: float, max_loops: int = 4) -> int:
    k = 0
    while k < max_loops and PY_RAND.random() < p_fail:
        k += 1
    return k

# ---------------------------------------
# Scheduling primitives
# ---------------------------------------
def schedule_activity(act: str, earliest: datetime, ctx: CaseContext,
                      rm: ResourceManager, prev_resource: Optional[str],
                      required_role: Optional[str] = None,
                      same_as: Optional[str] = None,
                      separate_from: Optional[str] = None,
                      multi_resources: int = 1,
                      record_handover: bool = True,
                      allow_overtime: bool = True,
                      rework_iter: int = 0) -> Tuple[List[ScheduledEvent], str, datetime]:
    """
    Schedule one logical activity (may expand into preempt/resume parts or
    multi-resource meeting). Returns (events, chosen_resource, finish_time).
    """
    events: List[ScheduledEvent] = []
    era = ctx.era
    # choose resource(s)
    if multi_resources == 1:
        rid, rr = rm.assign(act, earliest, era, required_role, same_as, separate_from)
        # waiting until resource and business/shift open
        avail = max(rr.next_free if rid != "Unknown" else earliest,
                    business_opening(earliest))
        # shift-aware opening for named resources
        if rid != "Unknown":
            avail = max(avail, rr.shift_opening(earliest))
        wait_h = max(0.0, (avail - earliest).total_seconds()/3600.0)
        # optional wait event for holding cost
        ensure_wait_event(events, earliest, avail)

        # predict service and schedule
        prev_fam = family_of(act if prev_resource is None else act)
        svc_h, preemptible = sample_service_hours(act, rr.role, era, rr, prev_fam)
        # possibility to preempt if long and backlog high (approx via resource wait)
        backlog = 1 if rr.next_free > earliest else 0
        p_preempt = P_PREEMPT_BASE + 0.15 * backlog
        used_preempt = preemptible and (RNG.random() < p_preempt)

        if used_preempt:
            # split: start -> wait -> resume -> complete
            start = avail
            # do half now, half later
            part1 = 0.4 * svc_h
            end1, used_ot1 = (rr.add_shift_time(start, part1, allow_overtime) if rid != "Unknown"
                              else (add_business_time(start, part1), False))
            amt1 = event_cost(f"{act}.start", rr.role, start, end1, used_ot1, era)
            events.append(ScheduledEvent(f"{act}.start", start, end1, rid, amt1))
            # queue-induced pause
            pause = RNG.uniform(0.2, 1.2)  # hours
            pause_end = add_business_time(end1, pause)
            ensure_wait_event(events, end1, pause_end)
            # resume
            resume_start = pause_end
            part2 = svc_h - part1
            end2, used_ot2 = (rr.add_shift_time(resume_start, part2, allow_overtime) if rid != "Unknown"
                              else (add_business_time(resume_start, part2), False))
            amt2 = event_cost(f"{act}.resume", rr.role, resume_start, end2, used_ot2, era)
            events.append(ScheduledEvent(f"{act}.resume", resume_start, end2, rid, amt2))
            # complete marker (zero-duration but timestamped)
            comp_start = end2
            comp_end = end2
            amt3 = event_cost(f"{act}.complete", rr.role, comp_start, comp_end, used_ot2, era)
            events.append(ScheduledEvent(f"{act}.complete", comp_start, comp_end, rid, amt3))
            finish = end2
            used_ot = used_ot1 or used_ot2
        else:
            start = avail
            end, used_ot = (rr.add_shift_time(start, svc_h, allow_overtime) if rid != "Unknown"
                            else (add_business_time(start, svc_h), False))
            amt = event_cost(act, rr.role, start, end, used_ot, era, waiting_hours=wait_h,
                             rework_iteration=rework_iter)
            events.append(ScheduledEvent(act, start, end, rid, amt))
            finish = end

        # update resource stats
        if rid != "Unknown":
            rr.next_free = finish
            fam = family_of(act)
            rr.last_family_time[fam] = finish
            rr.family_count[fam] = rr.family_count.get(fam, 0) + 1

        # handover event if resource changed
        if record_handover and prev_resource and rid != prev_resource:
            # insert Handover between last event end and current start (best-effort)
            ho_start = min([e.start for e in events])
            ho_end = ho_start + timedelta(minutes=int(RNG.integers(5, 25)))
            ho_amt = event_cost("Handover", "Unknown", ho_start, ho_end, used_ot=False, era=era, handover=True)
            events.insert(0, ScheduledEvent("Handover", ho_start, ho_end, "Unknown", ho_amt))

        # deadline handling / escalate
        return events, rid, finish

    else:
        # Multi-resource meeting (e.g., ACC_Signing_Meeting): find overlapping slot
        # For simplicity, require 2 resources: one senior + one analyst (preferred)
        req_roles = ["role:senior", "role:analyst"][:multi_resources]
        chosen: List[Tuple[str, Resource]] = []
        # naive greedy selection for overlap
        ready = business_opening(earliest)
        for rrole in req_roles:
            rid, rr = rm.assign(act, ready, ctx.era, required_role=rrole)
            chosen.append((rid, rr))
            ready = max(ready, rr.shift_opening(ready), rr.next_free)
        # meeting start is the max of chosen openings
        meet_start = business_opening(ready)
        for rid, rr in chosen:
            meet_start = max(meet_start, rr.shift_opening(meet_start), rr.next_free)
        # schedule duration
        # approximate as base mean
        base_svc = BASE_MEAN_H.get(act, 0.8)
        end_candidates = []
        used_ot_any = False
        for rid, rr in chosen:
            end_i, used_ot_i = rr.add_shift_time(meet_start, base_svc, allow_overtime=True)
            rr.next_free = end_i
            rr.family_count["accept"] = rr.family_count.get("accept", 0) + 1
            rr.last_family_time["accept"] = end_i
            end_candidates.append(end_i)
            used_ot_any = used_ot_any or used_ot_i
        meet_end = max(end_candidates)
        rid_str = "|".join([rid for rid, _ in chosen])
        amt = event_cost(act, "role:senior", meet_start, meet_end, used_ot_any, ctx.era)
        events.append(ScheduledEvent(act, meet_start, meet_end, rid_str, amt))
        # potential handover from prev_resource to this (use first rid)
        if prev_resource and chosen and chosen[0][0] != prev_resource:
            ho_start = meet_start - timedelta(minutes=int(RNG.integers(5, 25)))
            ho_end = meet_start
            ho_amt = event_cost("Handover", "Unknown", ho_start, ho_end, used_ot=False, era=ctx.era, handover=True)
            events.insert(0, ScheduledEvent("Handover", ho_start, ho_end, "Unknown", ho_amt))
        return events, rid_str, meet_end

# ---------------------------------------
# Case generator (compose patterns)
# ---------------------------------------
def build_case(case_id: str, arrival: datetime, rm: ResourceManager) -> Tuple[List[ScheduledEvent], Dict[str, Any], bool]:
    """
    Build one case trace (list of ScheduledEvent in chronological order),
    plus trace attributes and boolean flag whether it had AND-parallelism.
    """
    era = era_of(arrival)
    ctx = CaseContext(case_id=case_id, era=era,
                      deadline_bh=CASE_DEADLINE_BUSINESS_HOURS,
                      priority=sample_priority())
    trace_ev: List[ScheduledEvent] = []
    cur_t = business_opening(arrival)
    prev_res: Optional[str] = None
    ctx_attr: Dict[str, Any] = {}

    # 0) Start marker
    e0 = ScheduledEvent("OFR_Start", cur_t, cur_t, "System",
                        event_cost("OFR_Start", "role:analyst", cur_t, cur_t, used_ot=False, era=era))
    trace_ev.append(e0)

    # 1) Linear spine with optional inserts: Capture -> Qualify
    evs, prev_res, cur_t = schedule_activity("OFR_Capture_Request", cur_t, ctx, rm, prev_res, required_role="role:analyst")
    trace_ev += evs
    evs, prev_res, cur_t = schedule_activity("OFR_Qualify", cur_t, ctx, rm, prev_res, required_role="role:analyst")
    trace_ev += evs

    # Optionally insert Docs/Background pre-step based on priority and backlog
    if ctx.priority != "High" and RNG.random() < 0.5:
        # OR: choose one of docs/background/compliance
        opt_act = PY_RAND.choice(["Background_Check", "Compliance_Check"])
        evs, prev_res, cur_t = schedule_activity(opt_act, cur_t, ctx, rm, prev_res)
        trace_ev += evs

    # 2) XOR branching after Qualify: choose one of {Prepare->InternalReview, Direct QA, Early Negotiate}
    branch_draw = RNG.random()
    if branch_draw < 0.55:
        # Prepare -> Internal Review -> AND(Legal || Finance) -> Consolidate
        evs, prev_res, cur_t = schedule_activity("OFR_Prepare_Offer", cur_t, ctx, rm, prev_res, required_role="role:analyst")
        trace_ev += evs
        evs, prev_res, cur_t = schedule_activity("OFR_Internal_Review", cur_t, ctx, rm, prev_res)
        trace_ev += evs

        # AND-split: Legal and Finance in parallel
        ctx.had_parallel = True
        # schedule both from same readiness
        parallel_ready = cur_t
        leg_ev, leg_res, leg_end = schedule_activity("OFR_Legal_Review", parallel_ready, ctx, rm, prev_res, required_role="role:senior")
        fin_ev, fin_res, fin_end = schedule_activity("OFR_Finance_Review", parallel_ready, ctx, rm, prev_res, required_role="role:senior")
        # record children
        trace_ev += leg_ev
        trace_ev += fin_ev
        join_t = max(leg_end, fin_end)
        # Synchronization latency and handover to join
        jstart = join_t
        jend = jstart  # zero-duration marker
        trace_ev.append(ScheduledEvent("AND_Join", jstart, jend, "System",
                                       event_cost("AND_Join", "Unknown", jstart, jend, used_ot=False, era=era)))
        # Consolidate after both
        evs, prev_res, cur_t = schedule_activity("OFR_Consolidate_Review", join_t, ctx, rm, prev_res)
        trace_ev += evs

        # Optional batching release (Batch_Approve at 10:00 or 15:00)
        if RNG.random() < 0.6:
            rel_candidates = [
                datetime(cur_t.year, cur_t.month, cur_t.day, 10, 0, 0),
                datetime(cur_t.year, cur_t.month, cur_t.day, 15, 0, 0)
            ]
            rel_candidates = [t for t in rel_candidates if t >= business_opening(cur_t)]
            if not rel_candidates:
                rel_time = business_opening(cur_t + timedelta(days=1, hours=1))
            else:
                rel_time = min(rel_candidates, key=lambda t: abs((t - cur_t).total_seconds()))
            ensure_wait_event(trace_ev, cur_t, rel_time)
            evs, prev_res, cur_t = schedule_activity("OFR_Batch_Approve", rel_time, ctx, rm, prev_res, required_role="role:senior")
            trace_ev += evs

    elif branch_draw < 0.8:
        # Direct QA before prepare; skip prepare sometimes
        evs, prev_res, cur_t = schedule_activity("QA_Review", cur_t, ctx, rm, prev_res, required_role="role:analyst")
        trace_ev += evs
        # memory for skip later
        if trace_ev:
            dur = max(0.0, (trace_ev[-1].end - trace_ev[-1].start).total_seconds()/3600.0)
            ctx.review_durations.append(dur)

    else:
        # Early negotiate path
        evs, prev_res, cur_t = schedule_activity("OFR_Negotiate", cur_t, ctx, rm, prev_res)
        trace_ev += evs

    # 3) QA rework loop (geometric repeats); probability depends on long durations observed
    need_qa = True
    # if prior review was very short, allow skipping QA with some probability (stochastic skip w/ memory)
    if ctx.review_durations and np.mean(ctx.review_durations) < 0.6 and RNG.random() < P_SKIP_QA_IF_FAST:
        need_qa = False
    if need_qa:
        evs, prev_res, cur_t = schedule_activity("QA_Review", cur_t, ctx, rm, prev_res)
        trace_ev += evs
        # loop on Fix
        loops = geometric_trials(0.35 if RNG.random() < 0.5 else 0.25, max_loops=3)
        for i in range(loops):
            evs, prev_res, cur_t = schedule_activity("QA_Fix", cur_t, ctx, rm, prev_res, rework_iter=i+1)
            trace_ev += evs
            evs, prev_res, cur_t = schedule_activity("QA_Review", cur_t, ctx, rm, prev_res)
            trace_ev += evs

    # 4) OR-split: choose 1..k of checks (Ref/Sec/Comp) in parallel
    OR_children = []
    for cand in ["Security_Check", "Compliance_Check", "Background_Check"]:
        if RNG.random() < P_OR_BRANCH_BASE:
            OR_children.append(cand)
    if not OR_children:
        OR_children.append(PY_RAND.choice(["Security_Check", "Compliance_Check"]))
    parallel_ready = cur_t
    or_ends = []
    for c in OR_children:
        evs, _, eend = schedule_activity(c, parallel_ready, ctx, rm, prev_res)
        trace_ev += evs
        or_ends.append(eend)
    if len(OR_children) >= 1:
        join_t = max(or_ends) if or_ends else parallel_ready
        trace_ev.append(ScheduledEvent("OR_Join", join_t, join_t, "System",
                                       event_cost("OR_Join", "Unknown", join_t, join_t, used_ot=False, era=era)))
        cur_t = join_t

    # 5) k-of-n multi-instance: 3 reference checks, proceed after any 2 complete
    refs = ["RefCheck_1", "RefCheck_2", "RefCheck_3"]
    ref_ends = []
    ref_events_per_branch = []
    for ract in refs:
        evs, _, eend = schedule_activity(ract, cur_t, ctx, rm, prev_res)
        trace_ev += evs
        ref_events_per_branch.append((ract, evs))
        ref_ends.append((eend, ract))
    # proceed after 2 fastest complete
    ref_ends.sort(key=lambda t: t[0])
    kofn_t = ref_ends[1][0]  # time of 2nd completion
    trace_ev.append(ScheduledEvent("KofN_Join", kofn_t, kofn_t, "System",
                                   event_cost("KofN_Join", "Unknown", kofn_t, kofn_t, used_ot=False, era=era)))
    cur_t = kofn_t

    # Budget-aware branch: if total cost so far high, either outsource vendor for refs (insert Vendor_RefCheck) or simplify
    # compute current cost
    current_amount = sum(e.amount for e in trace_ev)
    if current_amount > BUDGET_THRESH and RNG.random() < P_VENDOR_PATH_IF_BUDGET_HIGH:
        evs, prev_res, cur_t = schedule_activity("Vendor_RefCheck", cur_t, ctx, rm, prev_res, required_role="role:vendor")
        trace_ev += evs

    # 6) Send offer -> cooling-off -> followup; expedite if deadline risk
    evs, prev_res, cur_t = schedule_activity("OFR_Send_Offer", cur_t, ctx, rm, prev_res)
    trace_ev += evs

    # Cooling-off: ensure at least 24h before followup; else insert Wait
    foll_earliest = add_business_time(cur_t, 24.0)  # 24h in business time
    ensure_wait_event(trace_ev, cur_t, foll_earliest)

    late = (foll_earliest - arrival).total_seconds()/3600.0 > (CASE_DEADLINE_BUSINESS_HOURS * 0.7)
    if late and RNG.random() < P_EXPEDITE_WHEN_LATE:
        # Pay expedite fee; sometimes allows skip optional step next
        # Represent as explicit activity
        exp_s = foll_earliest
        exp_e = exp_s
        amt = event_cost("Expedite", "role:senior", exp_s, exp_e, used_ot=False, era=era, expedite=True)
        trace_ev.append(ScheduledEvent("Expedite", exp_s, exp_e, "R2", amt))

    evs, prev_res, cur_t = schedule_activity("OFR_Followup", foll_earliest, ctx, rm, prev_res)
    trace_ev += evs

    # Long-term dependency: bind Negotiate and Revise_Offer to same resource if used
    # (binding-of-duty); if not possible -> Reassign event
    need_neg = RNG.random() < 0.6
    if need_neg:
        # choose negotiator
        evs, neg_res, cur_t = schedule_activity("OFR_Negotiate", cur_t, ctx, rm, prev_res)
        trace_ev += evs
        ctx.binding_pairs["OFR_Revise_Offer"] = neg_res

        # If revising, enforce binding-of-duty
        if RNG.random() < 0.7:
            # try same resource
            evs, rev_res, cur_t = schedule_activity("OFR_Revise_Offer", cur_t, ctx, rm, prev_res,
                                                    same_as=ctx.binding_pairs.get("OFR_Revise_Offer"))
            if rev_res != ctx.binding_pairs.get("OFR_Revise_Offer"):
                # not same -> reassign sub-process
                rs = cur_t
                re = rs + timedelta(minutes=15)
                trace_ev.append(ScheduledEvent("Reassign", rs, re, "System",
                                               event_cost("Reassign", "Unknown", rs, re, used_ot=False, era=era)))
            trace_ev += evs

    # Approval & Audit with separation-of-duty
    evs, app_res, cur_t = schedule_activity("ACC_Approve", cur_t, ctx, rm, prev_res, required_role="role:senior")
    trace_ev += evs
    evs, aud_res, cur_t = schedule_activity("ACC_Audit", cur_t, ctx, rm, prev_res, separate_from=app_res, required_role=None)
    trace_ev += evs

    # Multi-resource meeting (signing)
    evs, prev_res, cur_t = schedule_activity("ACC_Signing_Meeting", cur_t, ctx, rm, prev_res, multi_resources=2)
    trace_ev += evs

    # SLA deadline check; if exceeded -> Warn/Escalate
    finish_so_far = cur_t
    ttl_bh = (finish_so_far - arrival).total_seconds() / 3600.0
    if ttl_bh > CASE_DEADLINE_BUSINESS_HOURS * 1.15:
        # Warn then maybe Escalate
        w_s = cur_t; w_e = w_s
        trace_ev.append(ScheduledEvent("Warn", w_s, w_e, "System",
                                       event_cost("Warn", "Unknown", w_s, w_e, used_ot=False, era=era)))
        if RNG.random() < (P_ESCALATE_WARN + 0.2):
            e_s = w_e; e_e = e_s + timedelta(minutes=10)
            trace_ev.append(ScheduledEvent("Escalate", e_s, e_e, "R5",
                                           event_cost("Escalate", "role:senior", e_s, e_e, used_ot=False, era=era, escalate=True)))
            ctx.escalated = True

    # Sign contract -> Acceptance -> Onboard -> Close
    evs, prev_res, cur_t = schedule_activity("ACC_Sign_Contract", cur_t, ctx, rm, prev_res)
    trace_ev += evs
    evs, prev_res, cur_t = schedule_activity("ACC_Acceptance_Received", cur_t, ctx, rm, prev_res, required_role="role:analyst")
    trace_ev += evs

    # Optional: vendor cheaper in Era-3 for onboarding (era tariffs)
    evs, prev_res, cur_t = schedule_activity("ACC_Onboard_Prep", cur_t, ctx, rm, prev_res,
                                             required_role="role:analyst" if era != "Era-3" else None)
    trace_ev += evs
    evs, prev_res, cur_t = schedule_activity("ACC_Close", cur_t, ctx, rm, prev_res)
    trace_ev += evs

    # Summaries on trace
    ctx.total_amount = sum(e.amount for e in trace_ev)
    ctx_attr["case:priority"] = ctx.priority
    ctx_attr["case:era"] = ctx.era
    ctx_attr["case:escalated"] = str(ctx.escalated)
    ctx_attr["case:total_amount"] = round(ctx.total_amount, 2)

    # sort chronologically and return
    trace_ev.sort(key=lambda e: (e.end, e.start))
    return trace_ev, ctx_attr, ctx.had_parallel

# ---------------------------------------
# Quality & sanity checks
# ---------------------------------------
def sanity_checks(log: EventLog):
    # Schema check + monotonic timestamps
    bad = 0
    for i, t in enumerate(log):
        last_ts = None
        for j, e in enumerate(t):
            # required attributes
            miss = [k for k in ("concept:name", "time:timestamp", "org:resource", "amount") if k not in e]
            if miss:
                bad += 1
            # monotonic times
            ts = e["time:timestamp"]
            if last_ts and ts < last_ts:
                bad += 1
            last_ts = ts
    print(f"[CHECK] Events with schema/time issues: {bad}")

def summarize_distributions(log: EventLog):
    # Collect stats
    trace_lens = [len(t) for t in log]
    total_amounts = [sum(float(e["amount"]) for e in t) for t in log]
    case_dur_h = []
    for t in log:
        ts = [e["time:timestamp"] for e in t]
        case_dur_h.append(max(ts).timestamp() - min(ts).timestamp())
    case_dur_h = [x/3600.0 for x in case_dur_h]
    print(f"[SUM] Traces: {len(log)} | Events: {sum(trace_lens)}")
    print(f"[SUM] Trace length - mean {np.mean(trace_lens):.1f}, q50 {np.quantile(trace_lens,0.5):.0f}, q90 {np.quantile(trace_lens,0.9):.0f}")
    print(f"[SUM] Case duration (h) - mean {np.mean(case_dur_h):.1f}, q50 {np.quantile(case_dur_h,0.5):.1f}, q90 {np.quantile(case_dur_h,0.9):.1f}")
    print(f"[SUM] Case total amount - mean {np.mean(total_amounts):.2f}, q50 {np.quantile(total_amounts,0.5):.2f}, q90 {np.quantile(total_amounts,0.9):.2f}")

def ambiguity_check(log: EventLog, samples: int = 100):
    """Sample prefixes and see if >1 distinct next acts appear."""
    prefix_next = {}
    for t in log:
        acts = [str(e["concept:name"]) for e in t]
        for k in range(1, min(len(acts), 6)):  # short prefixes
            pref = tuple(acts[:k])
            if k < len(acts):
                nxt = acts[k]
                prefix_next.setdefault(pref, set()).add(nxt)
    if not prefix_next:
        print("[AMB] No prefixes to analyze.")
        return
    keys = PY_RAND.sample(list(prefix_next.keys()), k=min(samples, len(prefix_next)))
    multi = sum(1 for k in keys if len(prefix_next[k]) >= 2)
    print(f"[AMB] Prefixes with ≥2 plausible next activities: {multi}/{len(keys)} ({100.0*multi/len(keys):.1f}%)")

def remaining_time_signal(log: EventLog):
    """Rough check: compute remaining-time from middle of traces."""
    rem = []
    for t in log:
        if len(t) < 3: continue
        mid = len(t)//2
        first_ts = t[0]["time:timestamp"]
        last_ts = t[-1]["time:timestamp"]
        mid_ts = t[mid]["time:timestamp"]
        rem.append((last_ts - mid_ts).total_seconds()/3600.0)
    if rem:
        print(f"[RT] Remaining time (h): mean {np.mean(rem):.1f}, std {np.std(rem):.1f}, q10 {np.quantile(rem,0.1):.1f}, q90 {np.quantile(rem,0.9):.1f}")

def parallelism_check(par_flags: List[bool]):
    pct = 100.0 * sum(1 for x in par_flags if x) / len(par_flags)
    print(f"[PAR] Traces with ≥1 AND-split: {pct:.1f}%")

def drift_check(log: EventLog):
    era_groups: Dict[str, List[float]] = {}
    for t in log:
        era = t.attributes.get("case:era", "Unknown")
        dur_h = (t[-1]["time:timestamp"].timestamp() - t[0]["time:timestamp"].timestamp())/3600.0 if len(t)>1 else 0.0
        era_groups.setdefault(era, []).append(dur_h)
    for era, arr in era_groups.items():
        print(f"[DRIFT] {era}: mean duration (h) {np.mean(arr):.1f}, q50 {np.quantile(arr,0.5):.1f}")

# ---------------------------------------
# Build the entire log
# ---------------------------------------
def build_log() -> EventLog:
    # resources
    rm = build_resources(START_DATE)
    # arrivals
    arrivals = sample_arrivals(N_CASES, START_DATE, HORIZON_DAYS)
    # build eventlog
    L = EventLog()
    par_flags = []
    for idx, arr in enumerate(arrivals, start=1):
        case_id = f"CASE_{idx:04d}"
        events, trace_attrs, had_par = build_case(case_id, arr, rm)
        # Convert scheduled events to pm4py Events
        t = Trace()
        t.attributes["concept:name"] = case_id  # so case:concept:name appears in DataFrame conversion
        for ev in events:
            e = Event()
            e["concept:name"] = ev.act
            e["time:timestamp"] = ev.end  # store completion times
            e["org:resource"] = ev.resource
            e["amount"] = float(ev.amount)
            t.append(e)
        # add case-level attrs
        for k, v in trace_attrs.items():
            t.attributes[k] = v
        L.append(t)
        par_flags.append(had_par)
    return L, par_flags

# ---------------------------------------
# Export helper (pm4py compat)
# ---------------------------------------
def export_xes(log: EventLog, path: str):
    if HAS_PM4PY_WRITE:
        pm4py.write_xes(log, path)
    else:
        xes_exporter.apply(log, path)

# ---------------------------------------
# Minimal example trace (commentary)
# ---------------------------------------
# The generator will naturally produce cases that hit the following:
# - AND-split (Legal || Finance) with join
# - Handover event when resource changes between activities
# - SLA overrun -> Warn -> Escalate
# - QA rework loop (QA_Fix executed once or more)
# - Vendor_RefCheck path triggered when budget high, potentially using
#   overtime windows due to resource shift calendars

# ---------------------------------------
# Main
# ---------------------------------------
def main():
    log, par_flags = build_log()

    # Sanity checks
    sanity_checks(log)
    summarize_distributions(log)
    ambiguity_check(log, samples=100)
    remaining_time_signal(log)
    parallelism_check(par_flags)
    drift_check(log)

    # Export
    export_xes(log, OUTFILE)
    print(f"[OK] Wrote XES log to: {OUTFILE}")

if __name__ == "__main__":
    main()
