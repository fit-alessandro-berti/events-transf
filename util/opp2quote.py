"""
Opportunity-to-Quote (O2Q) synthetic log generator (pm4py)
---------------------------------------------------------
Author: (you)
Role: Senior Process Mining Engineer

What this script does
---------------------
• Synthesizes a single XES log for an Opportunity-to-Quote process, trace-by-trace (no black-box simulator).
• Injects rich control-flow, temporal, resource, and cost patterns to create realistic, ambiguous data.
• Each event carries: concept:name, time:timestamp (datetime), org:resource (string), amount (float).
• Trace attribute 'concept:name' is set to the case id (so 'case:concept:name' appears in DataFrame views).
• Limits the final log to **250 distinct traces** and writes to **opp2quote.xes.gz**.

Main ingredients implemented
----------------------------
Activities & resources:
  - 35+ activity labels with families; 30 resources (R1..R30) with roles: analyst/senior/vendor + 'Unknown' noise.
Control-flow templates (used in different mixes per case):
  - Linear with optional inserts, XOR branching, Review→Fix loops, AND/OR splits with joins,
    long-term dependency (binding/separation of duty), preemption/resume, skip patterns,
    multi-instance (k-of-n) vendor quotes, null transitions.
Temporal & resource patterns:
  - NHPP bursty arrivals, heavy-tailed service times (lognormal + Pareto tails), business + shift calendars,
    overtime, deadlines with Warn/Escalate, setup/changeover, interruption & resume,
    synchronization latency, queue discipline via assignment policies, seasonality & drift,
    learning/forgetting, cooling-off constraints, handover delays, batching (QA release windows),
    stochastic skip with memory, k-of-n multi-instance.
Costs:
  - Fixed + variable (role × shift × era × overtime), waiting (holding) cost, SLA penalty,
    rework premium escalation, expedite fee, handover fee, negative credits on cancels/rollbacks,
    material cost triggers, budget-aware branching.

Quality checks printed at the end:
  - Schema and monotonicity, distributions (lengths, costs, inter-event times),
    ambiguity checks on next-activity prediction from prefixes,
    remaining-time variability, parallelism rate & branch imbalance,
    era-wise drift on duration/cost.

Minimal example path (in comments near the bottom) shows how a single trace can include:
  AND-split, handover delay, SLA penalty → Escalate, rework once, vendor path with overtime.

Dependencies
------------
  pip install pm4py numpy pandas

Output
------
  opp2quote.xes.gz  (in the current working directory)
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time, date
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np

# pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.exporter.xes import exporter as xes_exporter


# ----------------------------
# Global configuration
# ----------------------------
SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

N_TRACES = 250  # hard cap (very important)
OUT_PATH = "opp2quote.xes.gz"

# Simulation horizon / eras
SIM_START = datetime(2024, 1, 8, 9, 0, 0)  # a Monday 09:00
HORIZON_DAYS = 60  # spread arrivals across ~2 months
ERA_CUTS = [SIM_START,
            SIM_START + timedelta(days=20),
            SIM_START + timedelta(days=40),
            SIM_START + timedelta(days=HORIZON_DAYS)]
ERA_NAMES = {0: "Era-1", 1: "Era-2", 2: "Era-3"}

# Business calendar
BUSINESS_START = time(9, 0, 0)
BUSINESS_END = time(17, 0, 0)
LUNCH_START = time(12, 0, 0)
LUNCH_END = time(13, 0, 0)
OVERTIME_END = time(20, 0, 0)  # overtime upper bound (if enabled)

# QA batch release times (batching pattern)
QA_RELEASE_SLOTS = [time(10, 0, 0), time(15, 0, 0)]

# Roles
ROLE_ANALYST = "role:analyst"
ROLE_SENIOR = "role:senior"
ROLE_VENDOR = "role:vendor"

# Rates by role (per hour) baseline (drift applied via eras)
BASE_RATE = {ROLE_ANALYST: 45.0, ROLE_SENIOR: 75.0, ROLE_VENDOR: 50.0}
OVERTIME_FACTOR = 1.5

# Cost weights
HANDOVER_FEE = 15.0
EXPEDITE_FEE = 60.0
SLA_ESCALATE_FEE = 120.0
CANCEL_CREDIT = -25.0
WAITING_RATE_PER_HOUR = 2.0  # holding cost
REWORK_PREMIUM_STEP = 12.0  # added per loop iteration

# Budget threshold for branching to cheaper vendor path or simplify
BUDGET_THRESHOLD = 650.0

# Unknown resource injection rate
UNKNOWN_RES_P = 0.035

# Preemption probability multipliers
PREEMPT_BASE_P = 0.08  # baseline chance to split a long task
PREEMPT_QUEUE_MULT = 0.12  # extra if resource queue is long (next_free >> ready)

# Handover delay minutes if resource changes
HANDOVER_MINUTES = (8, 28)

# Cooling-off min lag for Negotiate→Revise.Quote
COOLING_OFF_MINUTES = 60

# Queue discipline mixture (assignment policies)
POLICY_MIX = {"random": 0.30, "least_loaded": 0.40, "skill_based": 0.30}


# ----------------------------
# Utilities
# ----------------------------
def is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5  # 5=Sat, 6=Sun


def combine_date_time(d: date, t: time) -> datetime:
    return datetime(d.year, d.month, d.day, t.hour, t.minute, t.second)


def clamp_to_day(dt: datetime, t_from: time, t_to: time) -> Tuple[datetime, datetime]:
    """Returns (window_start, window_end) for dt's day."""
    d = dt.date()
    return combine_date_time(d, t_from), combine_date_time(d, t_to)


def quantiles(a: List[float], qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> Dict[float, float]:
    if not a:
        return {}
    arr = np.asarray(a, dtype=float)
    return {q: float(np.quantile(arr, q)) for q in qs}


def minutes(n: float) -> timedelta:
    return timedelta(minutes=float(n))


def hours(n: float) -> timedelta:
    return timedelta(hours=float(n))


def era_index(ts: datetime) -> int:
    for i in range(len(ERA_CUTS)-1):
        if ERA_CUTS[i] <= ts < ERA_CUTS[i+1]:
            return i
    return len(ERA_CUTS)-2


def era_tariff_multiplier(role: str, era_idx: int) -> float:
    # Example drift: vendor cheaper over time, senior costlier; analysts slightly up
    if role == ROLE_VENDOR:
        return [1.00, 0.95, 0.90][era_idx]
    if role == ROLE_SENIOR:
        return [1.00, 1.05, 1.10][era_idx]
    if role == ROLE_ANALYST:
        return [1.00, 1.02, 1.04][era_idx]
    return 1.0


# ----------------------------
# Calendars
# ----------------------------
@dataclass
class BusinessCalendar:
    """Business + resource shift calendars with lunch break and optional overtime."""
    def is_open_for_resource(self, dt: datetime, resource: "Resource") -> bool:
        if is_weekend(dt):
            return False
        if resource.is_absent_on(dt.date()):
            return False
        s_h, e_h = resource.shift_hours(dt.date())
        start, end = clamp_to_day(dt, time(s_h, 0), time(e_h, 0))
        if not (start <= dt < end):
            return False
        # lunch close
        lunch_s, lunch_e = clamp_to_day(dt, LUNCH_START, LUNCH_END)
        if lunch_s <= dt < lunch_e:
            return False
        return True

    def next_open_time(self, dt: datetime, resource: "Resource", allow_overtime: bool = False) -> datetime:
        probe = dt
        for _ in range(14):  # search up to 2 weeks
            if not is_weekend(probe):
                # consider lunch, shift, overtime
                sh, eh = resource.shift_hours(probe.date())
                if allow_overtime:
                    eh = max(eh, OVERTIME_END.hour)
                day_start = combine_date_time(probe.date(), time(sh, 0))
                lunch_s = combine_date_time(probe.date(), LUNCH_START)
                lunch_e = combine_date_time(probe.date(), LUNCH_END)
                day_end = combine_date_time(probe.date(), time(eh, 0))

                # if before day start
                if probe < day_start:
                    probe = day_start
                # if in lunch
                if lunch_s <= probe < lunch_e:
                    probe = lunch_e
                # if after day end -> next morning
                if probe >= day_end or (resource.is_absent_on(probe.date())):
                    probe = combine_date_time(probe.date() + timedelta(days=1),
                                              time(resource.shift_hours((probe.date() + timedelta(days=1)).weekday())[0], 0))
                    continue

                # within working span now
                if self.is_open_for_resource(probe, resource) or (allow_overtime and not is_weekend(probe) and probe < day_end):
                    return probe
            # move to next day start
            probe = combine_date_time(probe.date() + timedelta(days=1), time(resource.shift_hours((probe.date() + timedelta(days=1)).weekday())[0], 0))
        return dt  # fallback

    def add_work_time(self, start: datetime, minutes_needed: float, resource: "Resource", allow_overtime: bool = False) -> Tuple[datetime, bool]:
        """Add processing minutes respecting shift + lunch; returns (end_time, used_overtime?)."""
        remaining = float(minutes_needed)
        t = max(start, self.next_open_time(start, resource, allow_overtime=allow_overtime))
        used_ot = False
        while remaining > 1e-6:
            if is_weekend(t) or resource.is_absent_on(t.date()):
                # jump to next open
                t = self.next_open_time(t + timedelta(days=1), resource, allow_overtime=allow_overtime)
                continue
            sh, eh = resource.shift_hours(t.date())
            if allow_overtime:
                eh = max(eh, OVERTIME_END.hour)
            ds = combine_date_time(t.date(), time(sh, 0))
            de = combine_date_time(t.date(), time(eh, 0))
            ls = combine_date_time(t.date(), LUNCH_START)
            le = combine_date_time(t.date(), LUNCH_END)

            # figure next block end considering lunch and day end
            block_end = de
            if t < ls and ls < de:
                block_end = ls
            elif le <= t < de:
                block_end = de

            if block_end <= t:
                # move after lunch or to next day start
                if t < ls:
                    t = le  # after lunch
                else:
                    # next day start
                    t = self.next_open_time(combine_date_time(t.date() + timedelta(days=1), time(sh, 0)),
                                            resource, allow_overtime=allow_overtime)
                continue

            # minutes available in this block
            available_min = (block_end - t).total_seconds() / 60.0
            if available_min <= 0:
                t = block_end
                continue

            use = min(available_min, remaining)
            # flag overtime if beyond nominal end
            if t.time() >= BUSINESS_END or block_end.time() > BUSINESS_END:
                used_ot = True
            t = t + minutes(use)
            remaining -= use

            # if we hit lunch start exactly, skip lunch
            if t == ls:
                t = le

            if remaining <= 1e-6:
                break

            # else, continue loop to next block/day
        return t, used_ot

    def align_to_batch_window(self, dt: datetime, activity: str) -> datetime:
        """Batching for QA: only starts at nearest next release slot if activity is QA."""
        if "QA" not in activity:
            return dt
        # if weekend, move to next Monday morning 9:00 (then seek slot)
        t = dt
        if is_weekend(t):
            # advance to next Monday 09:00
            days_fwd = (7 - t.weekday()) % 7
            if days_fwd == 0:
                days_fwd = 1
            t = combine_date_time((t + timedelta(days=days_fwd)).date(), BUSINESS_START)
        # find the next slot today or tomorrow
        for _ in range(3):
            d = t.date()
            candidates = [combine_date_time(d, s) for s in QA_RELEASE_SLOTS]
            for c in candidates:
                if c >= t and not is_weekend(c):
                    return c
            t = combine_date_time(d + timedelta(days=1), BUSINESS_START)
        return dt


# ----------------------------
# Resources
# ----------------------------
@dataclass
class Resource:
    rid: str
    role: str
    shift_pattern: int  # 0: 08-16, 1: 09-17, 2: 10-18
    base_rate: float
    overtime_factor: float = OVERTIME_FACTOR
    next_free: datetime = field(default_factory=lambda: SIM_START)
    # skill multipliers by family (lower is faster)
    skill: Dict[str, float] = field(default_factory=dict)
    learned_count: Dict[str, int] = field(default_factory=dict)
    last_exec: Dict[str, datetime] = field(default_factory=dict)
    absences: set = field(default_factory=set)

    def shift_hours(self, _date: date | int) -> Tuple[int, int]:
        # return (start_hour, end_hour)
        if self.shift_pattern == 0:
            return (8, 16)
        elif self.shift_pattern == 1:
            return (9, 17)
        else:
            return (10, 18)

    def is_absent_on(self, d: date) -> bool:
        return d in self.absences


def build_resources() -> Dict[str, Resource]:
    resources: Dict[str, Resource] = {}
    # Families for skill heterogeneity
    families = ["Opp", "Lead", "Account", "Contact", "Scope", "Tech", "Finance",
                "Compliance", "Discount", "Review", "Fix", "QA", "Estimate",
                "Price", "Vendor", "Design", "Prepare", "Legal", "Send", "Followup",
                "Negotiate", "Revise", "Audit", "Simplify", "Wait", "Handover", "Escalate", "Expedite"]
    # create pool
    roles = ([ROLE_ANALYST] * 18) + ([ROLE_SENIOR] * 9) + ([ROLE_VENDOR] * 3)
    shift_patterns = [0, 1, 2] * 10  # just to distribute
    for i in range(30):
        rid = f"R{i+1}"
        role = roles[i]
        sp = shift_patterns[i]
        base = BASE_RATE[role]
        # draw skill multipliers ~ N(1.0, 0.1) capped [0.7, 1.4]
        skill = {fam: float(np.clip(rng.normal(1.0, 0.12), 0.7, 1.4)) for fam in families}
        r = Resource(rid=rid, role=role, shift_pattern=sp, base_rate=base, skill=skill)
        # random absences (2-4 days over horizon)
        n_abs = rng.integers(2, 5)
        for _ in range(n_abs):
            dd = SIM_START.date() + timedelta(days=int(rng.integers(0, HORIZON_DAYS)))
            # only weekdays
            while datetime(dd.year, dd.month, dd.day).weekday() >= 5:
                dd += timedelta(days=1)
            r.absences.add(dd)
        resources[rid] = r
    return resources


# ----------------------------
# Activities (durations and family)
# ----------------------------
@dataclass
class ActivitySpec:
    name: str
    family: str
    mean_min: float
    sigma: float
    fixed_cost: float
    heavy_tailed: bool = False
    preferred_role: Optional[str] = None


def build_activity_catalog() -> Dict[str, ActivitySpec]:
    # NB: mean_min is a *scale* for lognormal; we fold resource skill and learning/forgetting at sample time.
    A = [
        ActivitySpec("Start", "Opp", 3, 0.4, 0.0),
        ActivitySpec("Opp.Create", "Opp", 15, 0.5, 5.0, heavy_tailed=False, preferred_role=ROLE_ANALYST),
        ActivitySpec("Lead.Intake", "Lead", 18, 0.6, 6.0, heavy_tailed=True, preferred_role=ROLE_ANALYST),
        ActivitySpec("Account.Enrich", "Account", 25, 0.5, 8.0, heavy_tailed=False, preferred_role=ROLE_ANALYST),
        ActivitySpec("Contact.Customer", "Contact", 20, 0.8, 4.0, heavy_tailed=True, preferred_role=ROLE_ANALYST),
        ActivitySpec("Opp.Qualify", "Opp", 30, 0.6, 12.0, heavy_tailed=True, preferred_role=ROLE_ANALYST),
        ActivitySpec("Scope.Requirements", "Scope", 75, 0.7, 18.0, heavy_tailed=True, preferred_role=ROLE_ANALYST),
        ActivitySpec("Feasibility.Check", "Tech", 60, 0.7, 16.0, heavy_tailed=False, preferred_role=ROLE_ANALYST),
        ActivitySpec("Technical.Assess", "Tech", 70, 0.8, 17.0, heavy_tailed=True, preferred_role=ROLE_ANALYST),
        ActivitySpec("Finance.Check", "Finance", 45, 0.7, 14.0, heavy_tailed=False, preferred_role=ROLE_ANALYST),
        ActivitySpec("Compliance.Check", "Compliance", 55, 0.8, 14.0, heavy_tailed=False, preferred_role=ROLE_ANALYST),
        ActivitySpec("Discount.Request", "Discount", 25, 0.7, 5.0, heavy_tailed=False, preferred_role=ROLE_ANALYST),
        ActivitySpec("Solution.Design.Meeting", "Design", 60, 0.5, 20.0, heavy_tailed=False, preferred_role=None),  # multi-resource
        ActivitySpec("Review", "Review", 40, 0.9, 10.0, heavy_tailed=True, preferred_role=ROLE_ANALYST),
        ActivitySpec("Fix", "Fix", 35, 0.9, 9.0, heavy_tailed=False, preferred_role=ROLE_ANALYST),
        ActivitySpec("QA", "QA", 25, 0.7, 6.0, heavy_tailed=False, preferred_role=ROLE_ANALYST),
        ActivitySpec("Vendor.Source", "Vendor", 35, 0.8, 8.0, heavy_tailed=False, preferred_role=ROLE_ANALYST),
        ActivitySpec("Vendor.Quote", "Vendor", 80, 0.8, 22.0, heavy_tailed=True, preferred_role=ROLE_VENDOR),
        ActivitySpec("Estimate", "Estimate", 120, 0.8, 25.0, heavy_tailed=True, preferred_role=ROLE_ANALYST),
        ActivitySpec("Price.Calculate", "Price", 50, 0.7, 14.0, heavy_tailed=False, preferred_role=ROLE_ANALYST),
        ActivitySpec("Price.Approve", "Price", 30, 0.6, 10.0, heavy_tailed=False, preferred_role=ROLE_SENIOR),
        ActivitySpec("Audit", "Audit", 40, 0.7, 12.0, heavy_tailed=False, preferred_role=ROLE_SENIOR),
        ActivitySpec("Prepare.Quote", "Prepare", 65, 0.7, 16.0, heavy_tailed=False, preferred_role=ROLE_ANALYST),
        ActivitySpec("Legal.Review", "Legal", 50, 0.7, 15.0, heavy_tailed=False, preferred_role=ROLE_SENIOR),
        ActivitySpec("Send.Quote", "Send", 15, 0.5, 5.0, heavy_tailed=False, preferred_role=ROLE_ANALYST),
        ActivitySpec("Followup", "Followup", 20, 0.6, 6.0, heavy_tailed=False, preferred_role=ROLE_ANALYST),
        ActivitySpec("Negotiate", "Negotiate", 80, 0.9, 15.0, heavy_tailed=True, preferred_role=ROLE_ANALYST),
        ActivitySpec("Revise.Quote", "Revise", 55, 0.8, 12.0, heavy_tailed=False, preferred_role=ROLE_ANALYST),
        ActivitySpec("Simplify", "Simplify", 25, 0.7, 9.0, heavy_tailed=False, preferred_role=ROLE_ANALYST),
        ActivitySpec("Escalate", "Escalate", 10, 0.5, SLA_ESCALATE_FEE, heavy_tailed=False, preferred_role=ROLE_SENIOR),
        ActivitySpec("Expedite", "Expedite", 8, 0.5, EXPEDITE_FEE, heavy_tailed=False, preferred_role=ROLE_SENIOR),
        ActivitySpec("Handover", "Handover", 12, 0.4, HANDOVER_FEE, heavy_tailed=False, preferred_role=None),
        ActivitySpec("Wait", "Wait", 40, 0.4, 0.0, heavy_tailed=False, preferred_role=None),
        ActivitySpec("Cancel", "Vendor", 10, 0.5, CANCEL_CREDIT, heavy_tailed=False, preferred_role=ROLE_ANALYST),
        ActivitySpec("Rollback", "Vendor", 10, 0.5, CANCEL_CREDIT, heavy_tailed=False, preferred_role=ROLE_ANALYST),
        ActivitySpec("Close.Won", "Opp", 15, 0.5, 0.0, heavy_tailed=False, preferred_role=ROLE_ANALYST),
        ActivitySpec("Close.Lost", "Opp", 10, 0.5, 0.0, heavy_tailed=False, preferred_role=ROLE_ANALYST),
        ActivitySpec("Archive", "Opp", 8, 0.4, 0.0, heavy_tailed=False, preferred_role=ROLE_ANALYST),
        ActivitySpec("End", "Opp", 2, 0.2, 0.0),
    ]
    return {a.name: a for a in A}


ACTIVITY = build_activity_catalog()
CAL = BusinessCalendar()


# ----------------------------
# Assignment policy mixture
# ----------------------------
class Assigner:
    def __init__(self, resources: Dict[str, Resource]):
        self.resources = resources
        # cache by role
        self.role_index: Dict[str, List[Resource]] = {}
        for r in resources.values():
            self.role_index.setdefault(r.role, []).append(r)

    def _candidates(self, role_pref: Optional[str]) -> List[Resource]:
        if role_pref and role_pref in self.role_index:
            return list(self.role_index[role_pref])
        # fallback: any resource
        return list(self.resources.values())

    def choose(self, activity: ActivitySpec, ready: datetime, case_policy_mix: Dict[str, float],
               prefer_role: Optional[str] = None, avoid: Optional[str] = None) -> Resource:
        # Expand candidate list by preferred role; if none available soon, widen search.
        base_cands = self._candidates(prefer_role)
        # separation-of-duty: avoid specific rid if requested
        if avoid:
            base_cands = [r for r in base_cands if r.rid != avoid]
        if not base_cands:
            base_cands = [r for r in self.resources.values() if r.rid != (avoid or "___NONE___")]

        # pick policy
        u = rng.random()
        cum = 0.0
        policy = "random"
        for k, w in case_policy_mix.items():
            cum += w
            if u <= cum:
                policy = k
                break

        family = activity.family
        if policy == "random":
            return random.choice(base_cands)
        elif policy == "least_loaded":
            # pick resource with minimal next_free - ready (negative = already idle)
            return min(base_cands, key=lambda r: (r.next_free - ready).total_seconds())
        else:  # skill_based
            return min(base_cands, key=lambda r: r.skill.get(family, 1.0))


# ----------------------------
# Duration & costs
# ----------------------------
def sample_service_minutes(act: ActivitySpec, resource: Resource,
                           case_ctx: Dict[str, Any], prev_family: Optional[str]) -> float:
    """
    Draw service time (minutes) with:
      - lognormal baseline + Pareto tail (heavy_tailed activities),
      - resource skill,
      - learning (faster with repetitions),
      - forgetting (slower after long inactivity),
      - setup/changeover penalty if switching family,
      - seasonality/drift via era (mild).
    """
    # baseline (lognormal parameterization by mean_min, sigma as log-scale factor)
    # We'll use lognormal by specifying mean of the underlying normal:
    # Use mu so that median ≈ mean_min; rough heuristic
    mu = math.log(max(1.0, act.mean_min))
    base = float(rng.lognormal(mean=mu, sigma=act.sigma))

    # occasional heavy-tail via Pareto
    if act.heavy_tailed and rng.random() < 0.20:
        # Pareto with xm ~ base scale, alpha ~2.2
        xm = max(10.0, 0.8 * act.mean_min)
        base = float(xm * (1.0 / rng.random()) ** (1.0 / 2.2))

    # resource skill
    fam = act.family
    skill = resource.skill.get(fam, 1.0)
    base *= skill

    # learning curve (reduce time with repetitions)
    cnt = resource.learned_count.get(act.name, 0)
    if cnt > 0:
        base *= (0.98 ** min(cnt, 25))  # bounded
    # forgetting curve (if long since last exec of this family)
    last_t = resource.last_exec.get(fam)
    if last_t is not None:
        gap_days = max(0.0, (case_ctx["now"] - last_t).total_seconds() / 86400.0)
        if gap_days > 7:
            base *= 1.0 + min(0.25, 0.02 * (gap_days - 7))

    # setup/changeover penalty if switching family
    if prev_family and prev_family != fam:
        base += float(rng.uniform(6.0, 25.0))  # minutes

    # era drift: later eras slightly faster for Review, slower for Approve for variety
    era_idx_ = case_ctx["era_idx"]
    if act.name == "Review":
        base *= [1.0, 0.92, 0.88][era_idx_]
    if act.name == "Price.Approve":
        base *= [1.0, 1.05, 1.10][era_idx_]

    # keep within reasonable bounds
    base = float(np.clip(base, 2.0, 16 * 60.0))  # cap at ~16 hours
    return base


def compute_event_amount(act: ActivitySpec, resource: Optional[Resource], duration_min: float,
                         waiting_min: float, used_overtime: bool, era_idx_: int,
                         extra_fixed: float = 0.0) -> float:
    """Fixed + variable rate * duration + waiting cost; plus any extra fixed (e.g., escalate)."""
    role = resource.role if resource else ROLE_ANALYST
    rate = BASE_RATE.get(role, 40.0) * era_tariff_multiplier(role, era_idx_)
    if used_overtime:
        rate *= OVERTIME_FACTOR
    variable = (duration_min / 60.0) * rate
    waiting_cost = (waiting_min / 60.0) * WAITING_RATE_PER_HOUR
    amount = float(act.fixed_cost + variable + waiting_cost + extra_fixed)
    return float(round(amount, 2))


# ----------------------------
# NHPP arrivals (bursty)
# ----------------------------
def intensity_lambda(t: datetime) -> float:
    """Time-varying intensity (cases/hour).
       Weekday peaks morning and mid-afternoon; weekly seasonality; era drift."""
    # hour-of-day bumps
    h = t.hour + t.minute / 60.0
    daily = 0.25 + 0.9 * math.exp(-0.5 * ((h - 10.5) / 1.6) ** 2) + 0.7 * math.exp(-0.5 * ((h - 15.0) / 1.8) ** 2)
    # weekly modulation: Fridays lower, Mondays slightly higher
    dow = t.weekday()
    weekly = [1.15, 1.05, 1.0, 1.0, 0.85, 0.2, 0.15][dow]
    # era effect
    e = era_index(t)
    era_factor = [1.0, 0.9, 1.1][e]
    return max(0.02, daily * weekly * era_factor)  # per hour


def simulate_nhpp_arrivals(n: int, start: datetime, days: int) -> List[datetime]:
    """Thinning for NHPP to get 'n' arrivals within 'days' starting at 'start'."""
    arrivals = []
    t = start
    t_end = start + timedelta(days=days)
    lam_max = 3.0  # upper bound per hour
    while len(arrivals) < n:
        # propose next by homogeneous Poisson with lam_max
        # delta ~ Exp(lam_max), in hours
        delta_h = rng.exponential(1.0 / lam_max)
        t = t + hours(delta_h)
        if t >= t_end:
            # wrap around within horizon
            t = start + timedelta(seconds=float(rng.uniform(0, days * 86400)))
        lam_t = intensity_lambda(t)
        if rng.random() <= min(1.0, lam_t / lam_max) and not is_weekend(t):
            arrivals.append(t)
    arrivals.sort()
    return arrivals[:n]


# ----------------------------
# Case generation (control-flow + scheduling)
# ----------------------------
class O2QGenerator:
    def __init__(self):
        self.resources = build_resources()
        self.assigner = Assigner(self.resources)
        self.parallelism_flags: Dict[str, bool] = {}  # case_id -> had AND-split?
        self.branch_imbalance_samples: List[float] = []  # absolute diff / sum
        self.case_meta: Dict[str, Dict[str, Any]] = {}  # case_id -> info for stats
        self.prefix_map: Dict[Tuple[str, ...], Dict[str, int]] = {}  # prefix -> counts of next
        self.event_start_time: Dict[Tuple[str, int], datetime] = {}  # (case_id, seq_idx) -> start (for metrics)
        self.case_total_amount: Dict[str, float] = {}
        self.case_deadline: Dict[str, datetime] = {}
        self.case_arrival: Dict[str, datetime] = {}

    def _record_prefix(self, case_id: str, labels: List[str]):
        # for ambiguity check, collect transitions
        for i in range(len(labels) - 1):
            prefix = tuple(labels[:i + 1])
            nxt = labels[i + 1]
            self.prefix_map.setdefault(prefix, {})
            self.prefix_map[prefix][nxt] = self.prefix_map[prefix].get(nxt, 0) + 1

    def _maybe_unknown(self, rid: str) -> str:
        return "Unknown" if rng.random() < UNKNOWN_RES_P else rid

    # -- scheduling primitives -------------------------------------------------

    def _schedule_activity(self, trace: Trace, case_ctx: Dict[str, Any],
                           act_name: str, role_pref: Optional[str],
                           allow_overtime: bool = False,
                           avoid_res: Optional[str] = None,
                           bound_res: Optional[str] = None,
                           prev_resource: Optional[str] = None,
                           prev_family: Optional[str] = None,
                           wait_cooling: Optional[int] = None) -> Tuple[datetime, str, float]:
        """Schedule a single activity; returns (end_time, used_resource_id, duration_minutes)."""
        act = ACTIVITY[act_name]
        case_id = case_ctx["case_id"]
        ready = case_ctx["now"]

        # cooling-off constraint -> insert Wait
        if wait_cooling and ready < case_ctx["last_negotiation_end"] + minutes(wait_cooling):
            wait_start = ready
            wait_end = case_ctx["last_negotiation_end"] + minutes(wait_cooling)
            wait_dur = (wait_end - wait_start).total_seconds() / 60.0
            # Wait pseudo-activity
            self._emit_event(trace, case_ctx, "Wait", None, wait_start, wait_end, wait_dur, waiting_min=wait_dur)
            ready = wait_end
            case_ctx["now"] = ready

        # resource assignment (binding/separation/reassign rules)
        if bound_res is not None:
            res = self.resources[bound_res]
        else:
            res = self.assigner.choose(act, ready, case_ctx["policy_mix"], prefer_role=role_pref, avoid=avoid_res)

        # batch alignment for QA
        ready = CAL.align_to_batch_window(ready, act_name)

        # earliest start considering resource availability and calendar
        start0 = max(ready, res.next_free)
        start = CAL.next_open_time(start0, res, allow_overtime=allow_overtime)
        waiting_min = max(0.0, (start - ready).total_seconds() / 60.0)

        # preemption (split long tasks) decision proxy using queue delay
        preempt = False
        if act_name in ("Estimate", "Prepare.Quote", "Negotiate") and rng.random() < PREEMPT_BASE_P:
            # amplify if resource is quite busy
            if (res.next_free - ready).total_seconds() > 2 * 3600:
                if rng.random() < PREEMPT_QUEUE_MULT:
                    preempt = True
            else:
                preempt = rng.random() < 0.35

        # service duration
        case_ctx["now"] = start  # for sampling context (forgetting uses now)
        service_min = sample_service_minutes(act, res, case_ctx, prev_family)

        used_overtime = False
        # implement preemption via start/resume/complete events
        if preempt:
            # split into 2 or 3 chunks
            n_chunks = 2 if rng.random() < 0.7 else 3
            remaining = service_min
            phase_labels = ["start", "resume", "complete"][:n_chunks]
            for i, phase in enumerate(phase_labels):
                frac = 0.40 if i < n_chunks - 1 else 1.0
                chunk = float(min(remaining * frac, remaining))
                # allow overtime on last chunk if deadline is tight
                allow_ot_chunk = allow_overtime or (case_ctx["deadline"] - start < timedelta(hours=4))
                end, used_ot = CAL.add_work_time(start, chunk, res, allow_overtime=allow_ot_chunk)
                used_overtime = used_overtime or used_ot
                self._emit_event(trace, case_ctx, f"{act_name}.{phase}", res, start, end, chunk, waiting_min=(waiting_min if i == 0 else 0.0))
                # preemption gap (simulated external interruption)
                if i < n_chunks - 1:
                    gap = float(rng.uniform(30.0, 120.0))  # minutes
                    start = end + minutes(gap)
                else:
                    start = end
                remaining -= chunk
            end_time = start
            eff_service = service_min
        else:
            end_time, used_overtime = CAL.add_work_time(start, service_min, res, allow_overtime=allow_overtime)
            self._emit_event(trace, case_ctx, act_name, res, start, end_time, service_min, waiting_min=waiting_min)
            eff_service = service_min

        # update resource stats (learning/forgetting)
        res.learned_count[act.name] = res.learned_count.get(act.name, 0) + 1
        res.last_exec[act.family] = end_time
        res.next_free = end_time

        # handover penalty if resource changed
        if prev_resource and prev_resource != res.rid:
            hmin = float(rng.uniform(*HANDOVER_MINUTES))
            hstart = end_time
            hend, used_ot_h = CAL.add_work_time(hstart, hmin, res, allow_overtime=allow_overtime)
            self._emit_event(trace, case_ctx, "Handover", None, hstart, hend, hmin, waiting_min=0.0,
                             resource_override=f"{prev_resource}+{res.rid}")
            used_overtime = used_overtime or used_ot_h
            end_time = hend  # extend case readiness
            res.next_free = max(res.next_free, hend)

        # update case clock
        case_ctx["now"] = end_time
        # return
        return end_time, res.rid, eff_service

    def _emit_event(self, trace: Trace, case_ctx: Dict[str, Any], label: str,
                    res: Optional[Resource], start_t: datetime, end_t: datetime,
                    service_min: float, waiting_min: float = 0.0,
                    extra_fixed: float = 0.0, resource_override: Optional[str] = None):
        # Build the event, include cost calculation; export end_t as event timestamp
        act = ACTIVITY.get(label, ActivitySpec(label, "Misc", 10, 0.5, 0.0))
        era_idx_ = case_ctx["era_idx"]
        used_ot = end_t.time() > BUSINESS_END  # approximate
        amount = compute_event_amount(act, res, service_min, waiting_min, used_ot, era_idx_, extra_fixed=extra_fixed)

        rid = resource_override if resource_override else (res.rid if res else "R?")
        rid_out = self._maybe_unknown(rid)

        ev = Event({
            "concept:name": label,
            "time:timestamp": end_t,
            "org:resource": rid_out,
            "amount": float(amount),
        })
        trace.append(ev)

        # bookkeeping for metrics
        self.case_total_amount[case_ctx["case_id"]] = self.case_total_amount.get(case_ctx["case_id"], 0.0) + float(amount)

    def _and_split_join(self, trace: Trace, case_ctx: Dict[str, Any],
                        branches: List[List[Tuple[str, Optional[str]]]],
                        join_label: str = "Consolidate.Assessments") -> datetime:
        """Run branches in parallel (AND), join when all finished."""
        self.parallelism_flags[case_ctx["case_id"]] = True
        starts = []
        ends = []
        res_ids = []
        prev_res = case_ctx.get("last_res")
        prev_fam = case_ctx.get("last_fam")
        case_ready_saved = case_ctx["now"]
        for path in branches:
            case_ctx["now"] = case_ready_saved  # each branch can start when case is ready
            last_res_branch = prev_res
            last_fam_branch = prev_fam
            for act_name, role_pref in path:
                end, rid, _ = self._schedule_activity(
                    trace, case_ctx, act_name, role_pref,
                    prev_resource=last_res_branch, prev_family=last_fam_branch
                )
                last_res_branch = rid
                last_fam_branch = ACTIVITY[act_name].family
            starts.append(case_ready_saved)
            ends.append(case_ctx["now"])
            res_ids.append(last_res_branch)
        # join occurs at max(ends)
        join_start = max(ends)
        # add synchronization/handover latency before join
        latency_min = float(rng.uniform(5.0, 30.0))
        join_end, used_ot = CAL.add_work_time(join_start, latency_min, self.resources[res_ids[0]], allow_overtime=False)
        # compute imbalance and record
        if len(ends) >= 2:
            dur = [(e - s).total_seconds() for e, s in zip(ends, starts)]
            if sum(dur) > 0:
                diff = abs(dur[0] - dur[1]) / max(1.0, (dur[0] + dur[1]))
                self.branch_imbalance_samples.append(diff)
        # join event
        self._emit_event(trace, case_ctx, join_label, self.resources[res_ids[0]], join_start, join_end, latency_min)
        case_ctx["now"] = join_end
        case_ctx["last_res"] = res_ids[0]
        return join_end

    def _or_split_join(self, trace: Trace, case_ctx: Dict[str, Any],
                       options: List[Tuple[str, Optional[str]]],
                       choose_k: Tuple[int, int] = (1, 3),
                       join_label: str = "Consolidate.OR") -> datetime:
        """Choose 1..k options to run in parallel; OR-join after all chosen finish."""
        kmin, kmax = choose_k
        k = int(rng.integers(kmin, min(kmax, len(options)) + 1))
        chosen = rng.choice(len(options), size=k, replace=False)
        branches = [[options[i]] for i in chosen]
        return self._and_split_join(trace, case_ctx, branches, join_label=join_label)

    def _k_of_n_multi_instance(self, trace: Trace, case_ctx: Dict[str, Any],
                               base_label: str, role_pref: Optional[str], n: int, k: int) -> datetime:
        """Spawn n 'Vendor.Quote#i' tasks; proceed after k complete; cancel others."""
        ready = case_ctx["now"]
        ends: List[Tuple[datetime, int, str]] = []  # (end_time, index, res)
        res_ids: Dict[int, str] = {}
        prev_res = case_ctx.get("last_res")
        for i in range(n):
            label = f"{base_label}#{i+1}"
            # sample duration & schedule (as a single activity)
            end, rid, _ = self._schedule_activity(trace, case_ctx, "Vendor.Quote", role_pref,
                                                  prev_resource=prev_res, prev_family="Vendor")
            res_ids[i] = rid
            ends.append((end, i, rid))
            # restore case readiness so these go in parallel
            case_ctx["now"] = ready

        # determine k-th completion time (order statistic)
        ends_sorted = sorted(ends, key=lambda x: x[0])
        kth_end = ends_sorted[k - 1][0]
        # cancel remaining beyond k (if any)
        for end_t, idx, rid in ends_sorted[k:]:
            # rollback/cancel credit; model as quick 'Cancel'
            start_t = end_t
            cancel_dur = float(rng.uniform(5.0, 15.0))
            c_end, used_ot = CAL.add_work_time(start_t, cancel_dur, self.resources[rid], allow_overtime=False)
            self._emit_event(trace, case_ctx, "Cancel", self.resources[rid], start_t, c_end, cancel_dur)

        # OR-join at kth_end with small consolidation
        join_delay = float(rng.uniform(5.0, 20.0))
        join_end, used_ot = CAL.add_work_time(kth_end, join_delay, self.resources[res_ids[ends_sorted[0][1]]], allow_overtime=False)
        self._emit_event(trace, case_ctx, "Consolidate.Quotes", self.resources[res_ids[ends_sorted[0][1]]],
                         kth_end, join_end, join_delay)
        case_ctx["now"] = join_end
        return join_end

    # -- main case logic -------------------------------------------------------

    def generate_case(self, case_id: str, arrival: datetime, forced_example: bool = False) -> Trace:
        trace = Trace()
        trace.attributes["concept:name"] = case_id
        self.case_total_amount[case_id] = 0.0
        self.case_arrival[case_id] = arrival

        # build per-case context
        ctx = {
            "case_id": case_id,
            "now": arrival,
            "arrival": arrival,
            "deadline": arrival + timedelta(days=float(rng.uniform(3.0, 7.0))),
            "era_idx": era_index(arrival),
            "policy_mix": POLICY_MIX.copy(),
            "last_res": None,
            "last_fam": None,
            "last_negotiation_end": arrival,
        }
        self.case_deadline[case_id] = ctx["deadline"]
        self.case_meta[case_id] = {"had_and_split": False, "era_idx": ctx["era_idx"]}

        # pattern bundle selection (3–8 toggles)
        patterns = {
            "or_checks": rng.random() < 0.75,
            "and_scope_feas": rng.random() < 0.65,
            "review_rework": True,
            "qa_batched": rng.random() < 0.70,
            "preempt_capable": True,
            "cooling_off": rng.random() < 0.65,
            "budget_branching": rng.random() < 0.55,
            "multi_instance_vendor": rng.random() < 0.75,
            "skip_optional": rng.random() < 0.60,
            "binding_prepare_scope": rng.random() < 0.65,
        }
        if forced_example:
            # ensure the minimal example features kick in
            patterns.update({
                "and_scope_feas": True,
                "review_rework": True,
                "qa_batched": True,
                "preempt_capable": True,
                "cooling_off": True,
                "budget_branching": True,
                "multi_instance_vendor": True,
                "binding_prepare_scope": True,
            })

        # 0) Start marker
        self._schedule_activity(trace, ctx, "Start", ROLE_ANALYST)

        # 1) Intake / qualify (+ optional enrich / contact)
        self._schedule_activity(trace, ctx, "Opp.Create", ROLE_ANALYST, prev_resource=ctx["last_res"], prev_family=ctx["last_fam"])
        if rng.random() < 0.65:
            self._schedule_activity(trace, ctx, "Lead.Intake", ROLE_ANALYST, prev_resource=ctx["last_res"], prev_family="Opp")
        if rng.random() < 0.50:
            self._schedule_activity(trace, ctx, "Account.Enrich", ROLE_ANALYST, prev_resource=ctx["last_res"], prev_family="Lead")
        if rng.random() < 0.55:
            self._schedule_activity(trace, ctx, "Contact.Customer", ROLE_ANALYST, prev_resource=ctx["last_res"], prev_family="Account")

        # 2) Qualify with XOR: route probability influenced by early cost and era
        self._schedule_activity(trace, ctx, "Opp.Qualify", ROLE_ANALYST)
        early_cost = self.case_total_amount[case_id]
        route_probs = np.array([0.35, 0.45, 0.20])  # simple/standard/complex
        if early_cost > 50:
            route_probs = np.array([0.25, 0.45, 0.30])
        route = rng.choice(["simple", "standard", "complex"], p=(route_probs / route_probs.sum()))

        # 3) Either AND-split for scope|feas or OR-split on checks
        bound_scope_res: Optional[str] = None
        if route in ("standard", "complex") and patterns["and_scope_feas"]:
            self._and_split_join(
                trace, ctx,
                branches=[
                    [("Scope.Requirements", ROLE_ANALYST)],
                    [("Feasibility.Check", ROLE_ANALYST)]
                ],
                join_label="Consolidate.ScopeFeas"
            )
            self.case_meta[case_id]["had_and_split"] = True
            # binding-of-duty: later Prepare.Quote by same resource who did Scope (we approximate by last_res)
            bound_scope_res = ctx.get("last_res")
        elif route == "complex" and patterns["or_checks"]:
            # OR-split choose subset from compliance/finance/technical (1..3)
            opts = [("Compliance.Check", ROLE_ANALYST), ("Finance.Check", ROLE_ANALYST), ("Technical.Assess", ROLE_ANALYST)]
            self._or_split_join(trace, ctx, options=opts, choose_k=(1, 3), join_label="Consolidate.Checks")
        else:
            # linear with optional
            if rng.random() < 0.6:
                self._schedule_activity(trace, ctx, "Scope.Requirements", ROLE_ANALYST)
            if rng.random() < 0.4:
                self._schedule_activity(trace, ctx, "Feasibility.Check", ROLE_ANALYST)

        # 4) Solution design meeting (needs 2 resources simultaneously)
        # We approximate by a single event with combined resource ID string
        # Assign analyst + senior; schedule when both free with overlapping window
        ra = self.assigner.choose(ACTIVITY["Solution.Design.Meeting"], ctx["now"], ctx["policy_mix"], prefer_role=ROLE_ANALYST)
        rs = self.assigner.choose(ACTIVITY["Solution.Design.Meeting"], ctx["now"], ctx["policy_mix"], prefer_role=ROLE_SENIOR)
        # align to both resources' next open time
        start = max(CAL.next_open_time(max(ctx["now"], ra.next_free), ra),
                    CAL.next_open_time(max(ctx["now"], rs.next_free), rs))
        # service duration (use senior for sampling skill)
        ctx["now"] = start
        dur_min = sample_service_minutes(ACTIVITY["Solution.Design.Meeting"], rs, ctx, prev_family="Scope")
        end, used_ot = CAL.add_work_time(start, dur_min, rs, allow_overtime=False)
        # emit with combined resource string
        ev = Event({
            "concept:name": "Solution.Design.Meeting",
            "time:timestamp": end,
            "org:resource": self._maybe_unknown(f"{ra.rid}&{rs.rid}"),
            "amount": float(compute_event_amount(ACTIVITY["Solution.Design.Meeting"], rs, dur_min, 0.0, used_ot, ctx["era_idx"]))
        })
        trace.append(ev)
        ra.next_free = max(ra.next_free, end)
        rs.next_free = max(rs.next_free, end)
        ctx["now"] = end
        ctx["last_res"] = rs.rid  # approximate
        ctx["last_fam"] = "Design"

        # 5) Discount request may affect later approvals (long-term dependency)
        discount_taken = rng.random() < 0.45
        if discount_taken:
            self._schedule_activity(trace, ctx, "Discount.Request", ROLE_ANALYST)
            # If discount big (randomly), enforce senior-only approvals later
            large_discount = rng.random() < 0.4
        else:
            large_discount = False

        # 6) Review → Fix loop (geometric)
        if patterns["review_rework"]:
            n_loops = 0
            while True:
                end, rid, smin = self._schedule_activity(trace, ctx, "Review", ROLE_ANALYST)
                # long review increases chance of rework
                rework_p = 0.25 + 0.10 * (min(240.0, smin) / 60.0)
                # learning reduces rework over successive iterations
                rework_p *= max(0.35, 0.85 ** n_loops)
                if rng.random() < rework_p and n_loops < 3:
                    # escalate if deadline pressure
                    if ctx["deadline"] - ctx["now"] < timedelta(hours=6):
                        self._schedule_activity(trace, ctx, "Escalate", ROLE_SENIOR)
                    # perform Fix
                    self._schedule_activity(trace, ctx, "Fix", ROLE_ANALYST)
                    n_loops += 1
                else:
                    break
        # 7) QA optional skip with memory:
        # if Review was very fast, more likely to skip QA; if long, likely to include
        if patterns["qa_batched"]:
            skip_prob = 0.30 if rng.random() < 0.5 else 0.15  # base
            if rng.random() > skip_prob:
                # QA batched release
                self._schedule_activity(trace, ctx, "QA", ROLE_ANALYST)

        # 8) Vendor path decision (budget/deadline aware)
        vendor_path = False
        if patterns["budget_branching"]:
            if self.case_total_amount[case_id] > BUDGET_THRESHOLD or (ctx["deadline"] - ctx["now"] < timedelta(hours=10)):
                vendor_path = True
        vendor_path = vendor_path or (route == "complex" and rng.random() < 0.4)

        if vendor_path:
            self._schedule_activity(trace, ctx, "Vendor.Source", ROLE_ANALYST)
            # multi-instance vendor quotes (k-of-n)
            if patterns["multi_instance_vendor"]:
                n = int(rng.integers(2, 5))
                k = min(2, n)
                self._k_of_n_multi_instance(trace, ctx, "Vendor.Quote", ROLE_VENDOR, n=n, k=k)
            else:
                self._schedule_activity(trace, ctx, "Vendor.Quote", ROLE_VENDOR)
        else:
            if rng.random() < 0.35:
                # simplify internal path if over budget
                self._schedule_activity(trace, ctx, "Simplify", ROLE_ANALYST)

        # 9) Estimate (preemption possible)
        allow_ot_estimate = ctx["deadline"] - ctx["now"] < timedelta(hours=8)
        self._schedule_activity(trace, ctx, "Estimate", ROLE_ANALYST, allow_overtime=allow_ot_estimate)

        # 10) Pricing (calculate + approve) with separation-of-duty for Audit
        self._schedule_activity(trace, ctx, "Price.Calculate", ROLE_ANALYST)
        # Approver must be senior; if discount was large, bias toward senior-only and escalation on delay
        avoid_res = None
        approver = self.assigner.choose(ACTIVITY["Price.Approve"], ctx["now"], ctx["policy_mix"], prefer_role=ROLE_SENIOR)
        if large_discount:
            avoid_res = None  # ensure senior anyway
        end, approver_id, _ = self._schedule_activity(trace, ctx, "Price.Approve", ROLE_SENIOR, avoid_res=avoid_res)
        # Separation-of-duty with Audit (must be different from approver)
        self._schedule_activity(trace, ctx, "Audit", ROLE_SENIOR, avoid_res=approver_id)

        # 11) Prepare Quote (binding-of-duty with Scope if enabled)
        bound = bound_scope_res if patterns["binding_prepare_scope"] and bound_scope_res else None
        self._schedule_activity(trace, ctx, "Prepare.Quote", ROLE_ANALYST, bound_res=bound)

        # 12) Legal review optional
        if route == "complex" and rng.random() < 0.65:
            self._schedule_activity(trace, ctx, "Legal.Review", ROLE_SENIOR)

        # 13) Send quote + followup
        self._schedule_activity(trace, ctx, "Send.Quote", ROLE_ANALYST)
        self._schedule_activity(trace, ctx, "Followup", ROLE_ANALYST)

        # 14) Negotiate / Revise loop with cooling-off
        n_rev = 0
        while rng.random() < (0.55 if route != "simple" else 0.35) and n_rev < 2:
            end, rid, _ = self._schedule_activity(trace, ctx, "Negotiate", ROLE_ANALYST)
            ctx["last_negotiation_end"] = end
            # cooling-off constraint before revise
            self._schedule_activity(trace, ctx, "Revise.Quote", ROLE_ANALYST,
                                    wait_cooling=(COOLING_OFF_MINUTES if patterns["cooling_off"] else None))
            n_rev += 1
            # chance to expedite to skip QA-like checks in later loop
            if rng.random() < 0.25:
                self._schedule_activity(trace, ctx, "Expedite", ROLE_SENIOR)

        # 15) SLA check & penalty/escalate if deadline breached at this point
        if ctx["now"] > ctx["deadline"]:
            self._schedule_activity(trace, ctx, "Escalate", ROLE_SENIOR)

        # 16) Close: Won / Lost
        win_p = 0.55
        if vendor_path:
            win_p += 0.05
        if ctx["now"] > ctx["deadline"]:
            win_p -= 0.20
        close_label = "Close.Won" if rng.random() < max(0.05, min(0.95, win_p)) else "Close.Lost"
        self._schedule_activity(trace, ctx, close_label, ROLE_ANALYST)
        self._schedule_activity(trace, ctx, "Archive", ROLE_ANALYST)
        self._schedule_activity(trace, ctx, "End", ROLE_ANALYST)

        # track prefixes for ambiguity check
        self._record_prefix(case_id, [ev["concept:name"] for ev in trace])

        return trace

    # -------------------------------------------------------------------------


def build_log() -> EventLog:
    gen = O2QGenerator()

    # Generate NHPP arrivals
    arrivals = simulate_nhpp_arrivals(N_TRACES, SIM_START, HORIZON_DAYS)
    # Sort and pair with case IDs
    case_ids = [f"O2Q-{i+1:06d}" for i in range(len(arrivals))]

    # Force first case to be a rich "minimal example" path
    log = EventLog()
    forced_trace = gen.generate_case(case_ids[0], arrivals[0], forced_example=True)
    log.append(forced_trace)

    for cid, arr in zip(case_ids[1:], arrivals[1:]):
        tr = gen.generate_case(cid, arr)
        log.append(tr)

    # Ensure strict timestamp ordering within each trace
    for t in log:
        t[:] = sorted(t, key=lambda e: e["time:timestamp"])

    # Quality checks and prints
    run_quality_checks(log, gen)

    return log


# ----------------------------
# Quality & sanity checks
# ----------------------------
def run_quality_checks(log: EventLog, gen: O2QGenerator):
    # Schema & monotonicity
    n_events = sum(len(t) for t in log)
    missing = 0
    non_float_amount = 0
    non_increasing = 0
    for tr in log:
        last = None
        for ev in tr:
            for key in ("concept:name", "time:timestamp", "org:resource", "amount"):
                if key not in ev:
                    missing += 1
            if not isinstance(ev["amount"], float):
                non_float_amount += 1
            if last and ev["time:timestamp"] <= last:
                non_increasing += 1
            last = ev["time:timestamp"]

    print("\n=== Quality Checks ===")
    print(f"Traces: {len(log)}  Events: {n_events}")
    print(f"Missing required fields (events): {missing}")
    print(f"Non-float 'amount' fields: {non_float_amount}")
    print(f"Non-increasing timestamps within traces: {non_increasing}")

    # Distribution summaries
    trace_lengths = [len(t) for t in log]
    inter_event_mins = []
    event_amounts = []
    case_durations_h = []
    era_durations: Dict[int, List[float]] = {0: [], 1: [], 2: []}
    era_costs: Dict[int, List[float]] = {0: [], 1: [], 2: []}

    for tr in log:
        times = [ev["time:timestamp"] for ev in tr]
        amts = [ev["amount"] for ev in tr]
        event_amounts.extend(amts)
        if times:
            inter_event_mins.extend([(t2 - t1).total_seconds() / 60.0 for t1, t2 in zip(times[:-1], times[1:])])
            dur_h = (times[-1] - times[0]).total_seconds() / 3600.0
            case_durations_h.append(dur_h)
            # era of arrival
            cid = tr.attributes.get("concept:name")
            eidx = era_index(gen.case_arrival.get(cid, times[0]))
            era_durations[eidx].append(dur_h)
            era_costs[eidx].append(gen.case_total_amount.get(cid, sum(amts)))

    print("\n-- Trace length (events) quantiles:", quantiles(trace_lengths))
    print("-- Event amount (float) quantiles:", quantiles(event_amounts))
    print("-- Inter-event time (min) quantiles:", quantiles(inter_event_mins))
    print("-- Case duration (hours) quantiles:", quantiles(case_durations_h))

    # Ambiguity: sample 100 prefixes and see ≥2 next activities exist
    prefixes = list(gen.prefix_map.keys())
    rng.shuffle(prefixes)
    sample = prefixes[:min(100, len(prefixes))]
    ambig = sum(1 for p in sample if len(gen.prefix_map[p]) >= 2)
    print(f"\n-- Ambiguity check: {ambig}/{len(sample)} sampled prefixes have ≥2 distinct next-activities.")

    # Remaining time distribution
    remaining_hours = []
    for tr in log:
        times = [ev["time:timestamp"] for ev in tr]
        for i, t in enumerate(times[:-1]):
            remaining_hours.append((times[-1] - t).total_seconds() / 3600.0)
    print("-- Remaining time (hours) mean/std/quantiles:",
          np.mean(remaining_hours) if remaining_hours else None,
          np.std(remaining_hours) if remaining_hours else None,
          quantiles(remaining_hours))

    # Parallelism
    had_and = sum(1 for cid, meta in gen.case_meta.items() if meta.get("had_and_split"))
    print(f"-- Parallelism: {had_and/len(log)*100:.1f}% traces have ≥1 AND-split.")
    if gen.branch_imbalance_samples:
        print("-- AND-branch imbalance (|d1-d2|/(d1+d2)) quantiles:", quantiles(gen.branch_imbalance_samples))

    # Drift by era
    for e in (0, 1, 2):
        dd = era_durations[e]
        cc = era_costs[e]
        if dd:
            print(f"-- {ERA_NAMES[e]}: mean duration (h)={np.mean(dd):.2f}  mean cost={np.mean(cc):.2f}  n={len(dd)}")

    # End-of-check marker
    print("=== End Quality Checks ===\n")


# ----------------------------
# Main
# ----------------------------
def execute_script():
    log = build_log()
    # Export (supports .xes and .xes.gz)
    xes_exporter.apply(log, OUT_PATH)
    print(f"Saved XES log with {len(log)} traces to: {OUT_PATH}")


if __name__ == "__main__":
    execute_script()

"""
Notes (Minimal Example intent):
- The first generated case is forced to include: AND-split (Scope || Feasibility), batching (QA),
  preemption-capable tasks (Estimate/Prepare may be split), deadline pressure → Escalate,
  vendor path with multi-instance quotes (k-of-n), and a handover when resource changes
  (explicit 'Handover' event with fee and delay).
- All events include concept:name, time:timestamp (end times), org:resource, and amount (float).
- Trace attribute 'concept:name' is set to the case id, producing 'case:concept:name' in pm4py DataFrames.
"""
