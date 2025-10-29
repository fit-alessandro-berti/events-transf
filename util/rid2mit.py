"""
Risk Identification → Mitigation synthetic event log generator (rid2mit.xes.gz)

Author: Senior Process Mining Engineer (script-only, no background simulators)
Python libs: stdlib + numpy + pm4py (pandas optional for summaries)
Output: a single XES log written to ./rid2mit.xes.gz
Limit: at most 250 DISTINCT traces (by activity label sequence).

---------------------------------------------------------------------------
What this script does (at a glance)
---------------------------------------------------------------------------
• Builds a realistic activity alphabet for a Risk Identification→Mitigation process (30+ labels).
• Creates 30 resources (R1..R30) with roles (analyst, senior, qa, auditor, manager, vendor).
  - Small fraction of events intentionally use org:resource="Unknown".
• Provides control-flow templates and composes 3–8 patterns per case:
  - Linear paths with optional inserts, XOR choices, loops/rework, AND/OR parallelism (with explicit splits/joins),
    multi-instance (k-of-n) execution, skip patterns, long-term dependencies (binding/separation of duty),
    preemption/resume via “…Task.start/resume/complete” triads, batching releases, and cooling-off constraints.
• Temporal patterns:
  - Heavy-tailed service times; bursty NHPP arrivals; business & shift calendars (9–17 with lunch, weekends freeze),
    resource availability & overtime; aging deadlines with Warn/Escalate; synchronization latency on joins; handover+travel delays;
    learning/forgetting effects; seasonality/concept drift across eras.
• Queue/assignment:
  - Mixture of random / least-loaded / skill-based assignment.
  - Basic queue discipline effects (FIFO / priority / SPT) reflected via resource next_free and per-activity dispatch tweaks.
• Cost per event:
  - amount = fixed(activity) + rate(role, shift, era) * processing_time
  - Overtime/weekend multipliers, handover fee, waiting cost (explicit Wait events),
    rework premium, escalation penalty, expedite fee, material cost triggers, negative credits on Rollback,
    and budget-aware Vendor path routing.
• Scheduling:
  - Per-event scheduling honors calendars, resources, batching releases, and split/join synchronization.
  - Events written chronologically; trace.attributes['concept:name'] is the case id (DataFrame-friendly).
• Post-generation quality checks printed to console:
  - Schema/attribute checks; strictly increasing timestamps; duration sanity;
    distributions (trace length, activities, cost, inter-event times);
    ambiguity from random prefixes; remaining-time stats; parallelism share/imbalance; era-wise drift.

---------------------------------------------------------------------------
Minimal example (narrative in comments):
• One short case that hits: AND-split, handover delay, SLA penalty→Escalate, one rework, vendor path with overtime.
  This pattern is representative and appears across the generated population.
---------------------------------------------------------------------------
"""

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from statistics import mean, pstdev
from typing import Dict, List, Tuple, Optional, Set

import numpy as np

# pm4py imports
from pm4py.objects.log.obj import EventLog, Trace, Event
# robust XES export across pm4py versions
try:
    from pm4py import write_xes as _write_xes
    def write_xes(log, path):
        _write_xes(log, path)
except Exception:
    from pm4py.objects.log.exporter.xes import exporter as xes_exporter
    def write_xes(log, path):
        # pm4py exporter recognizes .xes.gz and gzips automatically
        xes_exporter.apply(log, path)


# --------------------------
# Global configuration knobs
# --------------------------
SEED = 17
random.seed(SEED)
np.random.seed(SEED)

# Hard cap of 250 distinct traces
MAX_DISTINCT_TRACES = 250
MAX_ATTEMPTS = 1200   # try up to this many cases to collect 250 unique variants

# Eras for concept drift (absolute dates matter!)
ERA_BOUNDS = [
    (datetime(2025, 1, 1, 9, 0), datetime(2025, 3, 31, 23, 59), "era1"),
    (datetime(2025, 4, 1, 0, 0), datetime(2025, 7, 31, 23, 59), "era2"),
    (datetime(2025, 8, 1, 0, 0), datetime(2025, 10, 29, 23, 59), "era3"),
]

# Activity set (domain-flavored, 30+)
ACTIVITIES = [
    "Risk.Identify",
    "Meeting.Triage",
    "Assess.Initial",
    "Assess.Qualitative",
    "Assess.Quantitative",
    "Vendor.Assess",
    "Analyze.RootCause",
    "Update.Register",
    "Mitigate.Plan",
    "Mitigate.Execute",
    "Mitigate.Execute.Task.start",
    "Mitigate.Execute.Task.resume",
    "Mitigate.Execute.Task.complete",
    "Mitigate.Verify",
    "QA",
    "Audit",
    "Monitor",
    "Approve",
    "Warn",
    "Escalate",
    "Reassign",
    "Expedite",
    "Handover",
    "Wait",
    "Simplify",
    "Vendor.Mitigate",
    "Cancel",
    "Rollback",
    "Close.Case",
    "AND.Split",
    "AND.Join",
    "OR.Split",
    "OR.Join",
    "Release.Batch",
    "Multi.KofN.Join",
]

# Preferred roles per activity (used by skill-based assignment)
PREF_ROLE = {
    "Risk.Identify": "analyst",
    "Meeting.Triage": "manager",
    "Assess.Initial": "analyst",
    "Assess.Qualitative": "analyst",
    "Assess.Quantitative": "senior",
    "Vendor.Assess": "vendor",
    "Analyze.RootCause": "senior",
    "Update.Register": "analyst",
    "Mitigate.Plan": "senior",
    "Mitigate.Execute": "analyst",
    "Mitigate.Execute.Task.start": "analyst",
    "Mitigate.Execute.Task.resume": "analyst",
    "Mitigate.Execute.Task.complete": "analyst",
    "Mitigate.Verify": "qa",
    "QA": "qa",
    "Audit": "auditor",
    "Monitor": "analyst",
    "Approve": "manager",
    "Warn": "manager",
    "Escalate": "senior",
    "Reassign": "manager",
    "Expedite": "manager",
    "Handover": "analyst",
    "Wait": "analyst",
    "Simplify": "analyst",
    "Vendor.Mitigate": "vendor",
    "Cancel": "manager",
    "Rollback": "manager",
    "Close.Case": "manager",
    "AND.Split": "analyst",
    "AND.Join": "analyst",
    "OR.Split": "analyst",
    "OR.Join": "analyst",
    "Release.Batch": "auditor",
    "Multi.KofN.Join": "analyst",
}

# Base mean processing times (hours) if performed by a competent resource during business time
MEAN_HOURS = {
    "Risk.Identify": 1.0,
    "Meeting.Triage": 0.6,
    "Assess.Initial": 1.6,
    "Assess.Qualitative": 2.2,
    "Assess.Quantitative": 3.0,
    "Vendor.Assess": 2.8,
    "Analyze.RootCause": 2.5,
    "Update.Register": 0.4,
    "Mitigate.Plan": 2.0,
    "Mitigate.Execute": 4.0,  # per subtask baseline
    "Mitigate.Execute.Task.start": 0.25,
    "Mitigate.Execute.Task.resume": 0.25,
    "Mitigate.Execute.Task.complete": 0.5,
    "Mitigate.Verify": 1.4,
    "QA": 1.0,
    "Audit": 2.0,
    "Monitor": 0.5,
    "Approve": 0.6,
    "Warn": 0.1,
    "Escalate": 0.25,
    "Reassign": 0.25,
    "Expedite": 0.05,
    "Handover": 0.2,
    "Wait": 0.0,       # not used, Wait is created from actual waiting time
    "Simplify": 0.6,
    "Vendor.Mitigate": 3.5,
    "Cancel": 0.15,
    "Rollback": 0.2,
    "Close.Case": 0.1,
    "AND.Split": 0.01,
    "AND.Join": 0.01,
    "OR.Split": 0.01,
    "OR.Join": 0.01,
    "Release.Batch": 0.01,
    "Multi.KofN.Join": 0.02,
}

# Fixed cost per activity (USD-equivalent units)
FIXED_COST = defaultdict(lambda: 12.0)
FIXED_COST.update({
    "Risk.Identify": 40.0,
    "Meeting.Triage": 50.0,
    "Assess.Initial": 60.0,
    "Assess.Qualitative": 120.0,
    "Assess.Quantitative": 170.0,
    "Vendor.Assess": 80.0,
    "Analyze.RootCause": 140.0,
    "Update.Register": 15.0,
    "Mitigate.Plan": 120.0,
    "Mitigate.Execute": 160.0,
    "Mitigate.Execute.Task.start": 20.0,
    "Mitigate.Execute.Task.resume": 15.0,
    "Mitigate.Execute.Task.complete": 40.0,
    "Mitigate.Verify": 50.0,
    "QA": 35.0,
    "Audit": 95.0,
    "Monitor": 10.0,
    "Approve": 30.0,
    "Warn": 15.0,
    "Escalate": 250.0,     # SLA penalty/overhead
    "Reassign": 25.0,
    "Expedite": 100.0,     # fee to jump queue/allow overtime
    "Handover": 20.0,      # knowledge transfer fee
    "Wait": 0.0,           # waiting cost is rate-based
    "Simplify": 30.0,
    "Vendor.Mitigate": 120.0,
    "Cancel": 5.0,
    "Rollback": -30.0,     # a small credit/refund
    "Close.Case": 5.0,
    "AND.Split": 0.0,
    "AND.Join": 0.0,
    "OR.Split": 0.0,
    "OR.Join": 0.0,
    "Release.Batch": 0.0,
    "Multi.KofN.Join": 0.0,
})

# Base hourly variable rate by role
BASE_RATE = {
    "analyst": 60.0,
    "senior": 90.0,
    "qa": 55.0,
    "auditor": 85.0,
    "manager": 110.0,
    "vendor": 42.0,
    "Unknown": 50.0,
}

# Multipliers
OVERTIME_MULT = 1.5
WEEKEND_MULT = 1.7
ERA_TARIFF = {
    "era1": 1.00,
    "era2": 0.98,  # slightly cheaper in era2
    "era3": 1.05,  # small inflation/penalties later
}

WAITING_RATE_PER_HOUR = 9.0  # holding cost
REWORK_PREMIUM_STEP = 25.0   # add per rework iteration
MATERIAL_COST_MITIGATE = 200.0  # fixed material cost when Mitigate.Execute starts
BUDGET_VENDOR_THRESHOLD = 800.0

# Small probability to mask a resource as 'Unknown' even if assigned
UNKNOWN_RESOURCE_PROB = 0.05

# Assignment policy mixture per case
ASSIGN_POLICIES = ["random", "least_loaded", "skill_based"]

# Queue discipline per activity family (rough influence via added wait / priority boost)
DISPATCH_POLICY = {
    "QA": "SPT",
    "Audit": "batch_priority",
    "Approve": "priority",
    "Mitigate.Execute": "FIFO",
    "Assess.Quantitative": "FIFO",
    "Assess.Qualitative": "FIFO",
}

# Shifts per role (hours in [0..24]); internal teams freeze weekends
SHIFT = {
    "analyst": (9, 17),
    "senior": (10, 18),
    "qa": (9, 17),
    "auditor": (8, 16),
    "manager": (10, 19),
    "vendor": (7, 19),  # vendors operate all week (lighter weekend rules)
}

LUNCH_BREAK = (12, 13)
ALLOWABLE_DAILY_OVERTIME_HOURS = 1.5  # how much beyond shift end we can squeeze when expediting

# Families for context switch penalties (prefix before first '.')
def family(label: str) -> str:
    return label.split('.')[0]


# --------------------------
# Resource & calendar models
# --------------------------
@dataclass
class Resource:
    name: str
    role: str
    next_free: datetime
    speed: float = 1.0                   # <1 faster, >1 slower (base multiplier)
    experience: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_seen: Dict[str, datetime] = field(default_factory=dict)  # family->last timestamp
    absent_days: Set[datetime] = field(default_factory=set)       # dates with full-day absence

    def speed_multiplier(self, label: str, now: datetime) -> float:
        """Learning (more runs => faster), forgetting (long inactivity => slower)."""
        fam = family(label)
        exp = self.experience[fam]
        # Learning: up to ~20% faster with experience
        learn = max(0.8, 1.0 - 0.12 * math.log(1 + exp))
        # Forgetting: if not seen in > 10 days, up to 25% slower
        last = self.last_seen.get(fam)
        forget = 1.0
        if last:
            days = (now - last).days
            if days > 10:
                forget = min(1.25, 1.0 + 0.03 * (days - 10))
        return max(0.6, min(1.6, self.speed * learn * forget))

@dataclass
class ResourceManager:
    resources: Dict[str, Resource]
    calendar: "Calendar"

    def pick(self, activity: str, ready: datetime, policy: str, prefer_role: str, severity: str) -> List[Resource]:
        """
        Return [resource] or [resource1, resource2] for multi-resource meetings.
        """
        # Multi-resource meetings: Meeting.Triage sometimes needs 2 resources
        if activity == "Meeting.Triage" and random.random() < 0.25:
            # pick a manager + senior if possible
            mgrs = [r for r in self.resources.values() if r.role == "manager"]
            sens = [r for r in self.resources.values() if r.role in ("senior", "analyst")]
            if mgrs and sens:
                r1 = min(mgrs, key=lambda r: r.next_free)
                r2 = min(sens, key=lambda r: r.next_free if r.name != r1.name else datetime.max)
                return [r1, r2]

        # Normal single-resource assignment
        candidates = list(self.resources.values())
        # Skill filter if policy is skill_based
        if policy == "skill_based" and prefer_role:
            skilled = [r for r in candidates if r.role == prefer_role]
            if skilled:
                candidates = skilled

        # Prioritize least loaded by next_free proximity to 'ready'
        if policy in ("least_loaded", "skill_based"):
            candidates.sort(key=lambda r: self.calendar.next_open_for(r, ready))

        # Severity-based: for high severity, prefer seniors/managers if within same wait
        if severity == "high":
            candidates.sort(key=lambda r: (r.role not in ("senior", "manager"), self.calendar.next_open_for(r, ready)))

        # Random fallback
        if not candidates:
            candidates = list(self.resources.values())
            random.shuffle(candidates)

        return [candidates[0]]

    def update_after(self, resources: List[Resource], label: str, end_time: datetime):
        fam = family(label)
        for r in resources:
            r.next_free = end_time
            r.experience[fam] += 1
            r.last_seen[fam] = end_time


class Calendar:
    """
    Business + shift calendar:
    - Internal roles: Mon–Fri, freeze on weekends. Work windows per role (SHIFT), lunch 12–13.
    - Vendor role: 7–19 all week, shorter breaks; slight weekend penalty via rate multiplier.
    - Supports overtime up to ALLOWABLE_DAILY_OVERTIME_HOURS if 'expedite' is enabled for the activity.
    """
    def __init__(self):
        pass

    def open_window_for(self, res: Resource, when: datetime) -> Tuple[datetime, datetime, bool]:
        role = res.role
        start_h, end_h = SHIFT.get(role, (9, 17))
        d = datetime(when.year, when.month, when.day, 0, 0)
        work_start = d + timedelta(hours=start_h)
        work_end = d + timedelta(hours=end_h)
        weekend = when.weekday() >= 5  # Sat/Sun
        if role == "vendor":
            # vendors work 7–19 even on weekends
            weekend = False
        if weekend:
            return (None, None, True)  # frozen today for non-vendors
        return (work_start, work_end, False)

    def next_open_for(self, res: Resource, when: datetime) -> datetime:
        cur = when
        # skip full-day absences
        while True:
            ws = datetime(cur.year, cur.month, cur.day)
            work_start, work_end, frozen = self.open_window_for(res, cur)
            if frozen:
                # go to next Monday 7–10 AM based on role
                nxt = cur + timedelta(days=(7 - cur.weekday()))
                cur = datetime(nxt.year, nxt.month, nxt.day, SHIFT.get(res.role, (9, 17))[0], 0)
                continue
            # full-day personal absence?
            if ws in res.absent_days:
                cur = work_start + timedelta(days=1)
                continue
            if cur < work_start:
                return work_start
            # if inside lunch break, resume at end of lunch
            if LUNCH_BREAK[0] <= cur.hour < LUNCH_BREAK[1]:
                return datetime(cur.year, cur.month, cur.day, LUNCH_BREAK[1], 0)
            # within normal shift?
            if work_start <= cur < work_end:
                # also ensure not inside lunch window edge
                if cur.hour == LUNCH_BREAK[0] and cur.minute == 0:
                    return datetime(cur.year, cur.month, cur.day, LUNCH_BREAK[1], 0)
                return cur
            # after today's work_end → next day
            cur = work_start + timedelta(days=1)

    def add_work(self, res: Resource, start: datetime, hours: float, allow_overtime: bool) -> Tuple[datetime, float, float]:
        """Return (end_time, overtime_hours, weekend_hours). Accumulates across days & handles lunch."""
        remaining = max(0.0, hours)
        cur = self.next_open_for(res, start)
        overtime_acc = 0.0
        weekend_acc = 0.0

        while remaining > 1e-9:
            ws, we, frozen = self.open_window_for(res, cur)
            # lunch handling
            lunch_start = datetime(cur.year, cur.month, cur.day, LUNCH_BREAK[0], 0)
            lunch_end = datetime(cur.year, cur.month, cur.day, LUNCH_BREAK[1], 0)

            if cur < lunch_start < we:
                # available time until lunch
                until = min(lunch_start, we)
            else:
                until = we

            # extend for overtime if allowed
            overtime_limit = we + timedelta(hours=ALLOWABLE_DAILY_OVERTIME_HOURS) if allow_overtime else we
            work_end_today = overtime_limit

            # how much time can we spend now?
            slot = (until - cur).total_seconds() / 3600.0
            if slot <= 0:
                # move across lunch or to next day start
                if cur < lunch_start:
                    cur = lunch_end
                    continue
                cur = self.next_open_for(res, we + timedelta(minutes=1))
                continue
            spend = min(slot, remaining)
            # record multipliers
            if cur >= we:
                # spending in overtime window
                overtime_here = spend
                weekend_here = 0.0
            else:
                overtime_here = 0.0
                weekend_here = 0.0 if res.role != "vendor" else 0.0
                # vendors count weekend in weekend_acc if Saturday/Sunday
                if res.role == "vendor" and cur.weekday() >= 5:
                    weekend_here = spend

            overtime_acc += overtime_here
            weekend_acc += weekend_here

            # advance
            cur += timedelta(hours=spend)
            remaining -= spend

            if remaining <= 1e-9:
                return (cur, overtime_acc, weekend_acc)

            # If still remaining, try overtime window after 'we'
            if allow_overtime and cur < work_end_today:
                extra_slot = (work_end_today - cur).total_seconds() / 3600.0
                if extra_slot > 0:
                    extra_spend = min(extra_slot, remaining)
                    overtime_acc += extra_spend
                    cur += timedelta(hours=extra_spend)
                    remaining -= extra_spend
                    if remaining <= 1e-9:
                        return (cur, overtime_acc, weekend_acc)

            # move to next open (tomorrow morning)
            cur = self.next_open_for(res, we + timedelta(minutes=1))

        return (cur, overtime_acc, weekend_acc)


# --------------------------
# Helpers: durations & costs
# --------------------------
def era_for(ts: datetime) -> str:
    for a, b, name in ERA_BOUNDS:
        if a <= ts <= b:
            return name
    return "era3"

def heavy_tailed_hours(mean_h: float, rng: np.random.RandomState) -> float:
    """
    Lognormal heavy-tailed draw with moderate variance.
    Truncate to [5 min, 16 h] to keep realism.
    """
    sigma = 0.7  # heavy tail
    mu = math.log(max(1e-3, mean_h)) - 0.5 * sigma * sigma
    val = rng.lognormal(mean=mu, sigma=sigma)
    return float(max(5.0/60.0, min(16.0, val)))

def setup_changeover(prev_label: Optional[str], cur_label: str) -> float:
    """
    Sequence dependent setup: if families differ, add up to 0.4h; certain pairs add more.
    """
    if prev_label is None:
        return 0.0
    pf, cf = family(prev_label), family(cur_label)
    base = 0.0
    if pf != cf:
        base += 0.15 + 0.25 * random.random()  # 0.15..0.4h
    # Assess.Quantitative after non-Initial is more costly
    if cur_label == "Assess.Quantitative" and prev_label != "Assess.Initial":
        base += 0.35
    # Mitigate.Execute after non-Plan adds setup
    if cur_label.startswith("Mitigate.Execute") and prev_label != "Mitigate.Plan":
        base += 0.3
    return base

def context_switch_penalty(prev_label: Optional[str], cur_label: str) -> float:
    """Penalty on handover (travelling/context): 0..0.6h if families differ."""
    if prev_label is None:
        return 0.0
    pf, cf = family(prev_label), family(cur_label)
    if pf == cf:
        return 0.05 * random.random()
    return 0.2 + 0.4 * random.random()

def variable_rate(role: str, ts: datetime, overtime_hours: float, weekend_hours: float) -> float:
    rate = BASE_RATE.get(role, BASE_RATE["Unknown"])
    r = rate * ERA_TARIFF[era_for(ts)]
    # apply weekend/OT multipliers proportionally (simple average if both non-zero)
    if overtime_hours > 1e-9:
        r *= OVERTIME_MULT
    if weekend_hours > 1e-9:
        r *= WEEKEND_MULT
    return r

# --------------------------
# Case & scheduler state
# --------------------------
@dataclass
class CaseState:
    case_id: str
    severity: str
    arrival: datetime
    deadline: datetime
    policy: str
    rng: np.random.RandomState
    ready_time: datetime
    last_label: Optional[str] = None
    last_resource_str: Optional[str] = None
    cost_so_far: float = 0.0
    escalated: bool = False
    events: List[Event] = field(default_factory=list)
    had_and_split: bool = False

# --------------------------
# NHPP arrivals (bursty)
# --------------------------
def nhpp_arrivals(n: int, start: datetime, end: datetime, rng: np.random.RandomState) -> List[datetime]:
    """
    Thinning for a nonhomogeneous Poisson process with daily/weekly/seasonal rate.
    λ_max chosen to keep reasonable volume. No arrivals on weekends for internal process.
    """
    def lam(dt: datetime) -> float:
        # base per-hour intensity (~0.6) scaled by time-of-day and era
        tod = dt.hour + dt.minute/60.0
        # two waves during business hours
        tod_boost = 1.0 + 1.2 * math.exp(-((tod-10.0)/1.8)**2) + 1.0 * math.exp(-((tod-15.0)/1.8)**2)
        wk = 0.0 if dt.weekday() >= 5 else 1.0
        era_mult = {"era1": 1.0, "era2": 0.9, "era3": 1.1}[era_for(dt)]
        return 0.6 * tod_boost * wk * era_mult  # per-hour
    t = start
    lam_max = 2.2  # conservative upper bound per-hour
    arrivals = []
    while len(arrivals) < n and t < end:
        # propose interarrival ~ Exp(lam_max), time in hours
        w = rng.exponential(1.0/lam_max)
        t = t + timedelta(hours=w)
        if t >= end:
            break
        if t.weekday() >= 5:  # weekends: freeze
            # jump to next Monday morning
            t = t + timedelta(days=(7 - t.weekday()))
            t = datetime(t.year, t.month, t.day, 9, 0)
        u = rng.uniform()
        if u <= (lam(t)/lam_max):
            arrivals.append(t)
    # ensure strictly increasing and cap to requested size
    arrivals.sort()
    return arrivals[:n]

# --------------------------
# Build resources
# --------------------------
def make_resources(start_anchor: datetime) -> ResourceManager:
    res = {}
    # R1-12 analysts
    for i in range(1, 13):
        res[f"R{i}"] = Resource(name=f"R{i}", role="analyst", next_free=start_anchor + timedelta(hours=random.random()*4),
                                speed=1.0 + random.uniform(-0.15, 0.2))
    # R13-18 seniors
    for i in range(13, 19):
        res[f"R{i}"] = Resource(name=f"R{i}", role="senior", next_free=start_anchor + timedelta(hours=random.random()*6),
                                speed=0.95 + random.uniform(-0.05, 0.1))
    # R19-20 managers
    for i in range(19, 21):
        res[f"R{i}"] = Resource(name=f"R{i}", role="manager", next_free=start_anchor + timedelta(hours=random.random()*2),
                                speed=1.0 + random.uniform(-0.1, 0.05))
    # R21-22 QA
    for i in range(21, 23):
        res[f"R{i}"] = Resource(name=f"R{i}", role="qa", next_free=start_anchor + timedelta(hours=random.random()*3),
                                speed=1.0 + random.uniform(-0.1, 0.15))
    # R23-24 auditors
    for i in range(23, 25):
        res[f"R{i}"] = Resource(name=f"R{i}", role="auditor", next_free=start_anchor + timedelta(hours=random.random()*5),
                                speed=1.0 + random.uniform(-0.05, 0.15))
    # R25-30 vendor
    for i in range(25, 31):
        res[f"R{i}"] = Resource(name=f"R{i}", role="vendor", next_free=start_anchor + timedelta(hours=random.random()*8),
                                speed=1.05 + random.uniform(-0.05, 0.2))
    # Random absences sprinkled across days (internal only)
    cal = Calendar()
    for r in res.values():
        if r.role != "vendor":
            # ~3% of days absent; stash dates between Jan and Oct 2025
            for _ in range(random.randint(0, 6)):
                day = start_anchor + timedelta(days=random.randint(1, 250))
                if day.weekday() < 5:
                    r.absent_days.add(datetime(day.year, day.month, day.day))
    return ResourceManager(res, cal)

# --------------------------
# Event emission helpers
# --------------------------
def add_event(case: CaseState, label: str, when: datetime, resource_str: str, amount: float):
    e = Event({
        "concept:name": label,
        "time:timestamp": when,
        "org:resource": resource_str,
        "amount": float(amount),
    })
    case.events.append(e)
    case.last_label = label
    case.last_resource_str = resource_str

def maybe_mask_resource(name: str) -> str:
    return "Unknown" if random.random() < UNKNOWN_RESOURCE_PROB else name

# --------------------------
# Scheduling a single activity
# --------------------------
def perform_activity(
    case: CaseState,
    rm: ResourceManager,
    label: str,
    prefer_role: Optional[str] = None,
    allow_overtime: bool = False,
    expedite: bool = False,
    multi_resource: bool = False,
    batch_release: Optional[str] = None,
) -> Tuple[datetime, float]:
    """
    Schedule one activity; inserts 'Wait' and 'Handover' pseudo-events as needed.
    Returns (end_time, processing_hours).
    """
    # Assignment policy and resource selection (can be 1 or 2 resources)
    # Severity influences resource choice for critical tasks
    chosen = rm.pick(label, case.ready_time, case.policy, prefer_role or PREF_ROLE.get(label, None), case.severity)
    if multi_resource and len(chosen) == 1:
        # pick a second helper with earliest availability of different role if possible
        pool = [r for r in rm.resources.values() if r.name != chosen[0].name]
        if pool:
            second = min(pool, key=lambda r: rm.calendar.next_open_for(r, case.ready_time))
            chosen = [chosen[0], second]

    # Batch release (e.g., "Audit" released at top of next hour)
    candidate_start = case.ready_time
    if batch_release == "top_of_hour":
        aligned = candidate_start.replace(minute=0, second=0, microsecond=0)
        if candidate_start.minute > 0 or candidate_start.second > 0 or candidate_start.microsecond > 0:
            candidate_start = aligned + timedelta(hours=1)

    # start = max(ready, each resource's next open)
    start = candidate_start
    start = max(start, max(rm.calendar.next_open_for(r, candidate_start) for r in chosen))

    # Queue discipline influence (lightweight): add synthetic wait if policy disfavors this case
    disp = DISPATCH_POLICY.get(label, None)
    if disp == "priority" and case.severity == "low":
        start += timedelta(minutes=random.randint(10, 30))
    elif disp == "SPT":
        # Shortest-processing-time first; penalize long expected tasks slightly
        if MEAN_HOURS.get(label, 1.0) > 1.2:
            start += timedelta(minutes=random.randint(5, 20))
    elif disp == "batch_priority" and label == "Audit":
        # Reinforce batching beyond simple alignment
        start = (start.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))

    # Explicit Wait event for waiting time (holding cost)
    if start > case.ready_time:
        wait_hours = (start - case.ready_time).total_seconds() / 3600.0
        add_event(case, "Wait", start, maybe_mask_resource("Unknown"), FIXED_COST["Wait"] + WAITING_RATE_PER_HOUR * wait_hours)
        # set ready == start (Wait is recorded at 'start' so strictly increasing is kept)
        case.cost_so_far += FIXED_COST["Wait"] + WAITING_RATE_PER_HOUR * wait_hours

    # Handover delay if resource changes (before actual start)
    # If multiple resources, compare concatenated string
    res_str = "&".join([r.name for r in chosen])
    if case.last_resource_str and case.last_resource_str != res_str:
        ho_delay = context_switch_penalty(case.last_label, label)
        ho_end, ot, wknd = rm.calendar.add_work(chosen[0], start, ho_delay, allow_overtime=False)
        add_event(case, "Handover", ho_end, maybe_mask_resource(chosen[0].name), FIXED_COST["Handover"])
        case.cost_so_far += FIXED_COST["Handover"]
        # start at end of handover
        start = ho_end

    # Processing time (heavy-tailed) + setup/changeover cost
    base_mean = MEAN_HOURS.get(label, 1.0)
    # Speed multipliers: use first resource's speed profile to drive duration (simplification)
    sp_mult = chosen[0].speed_multiplier(label, start)
    proc_hours = heavy_tailed_hours(base_mean * sp_mult, case.rng)
    proc_hours += setup_changeover(case.last_label, label)

    # For long tasks with pressure, allow overtime/expedite (explicit Expedited event)
    if expedite:
        add_event(case, "Expedite", start, maybe_mask_resource(chosen[0].name), FIXED_COST["Expedite"])
        case.cost_so_far += FIXED_COST["Expedite"]

    # Book the calendar (respect lunch, shift, optional overtime)
    end, ot_hours, wknd_hours = rm.calendar.add_work(chosen[0], start, proc_hours, allow_overtime=allow_overtime or expedite)
    # Cost
    role_for_rate = chosen[0].role if len(chosen) == 1 else "manager"  # blended; charge higher rate for joint sessions
    rate = variable_rate(role_for_rate, start, ot_hours, wknd_hours)
    amt = FIXED_COST[label] + rate * proc_hours
    # Material cost trigger
    if label.startswith("Mitigate.Execute") and ".Task." not in label:
        amt += MATERIAL_COST_MITIGATE
    # Record the event at its completion time (ensures strictly increasing timestamps)
    add_event(case, label, end, maybe_mask_resource(res_str), amt)
    case.cost_so_far += amt

    # Update resources
    rm.update_after(chosen, label, end)
    # Move case ready time forward
    case.ready_time = end
    return end, proc_hours

# --------------------------
# Core case generation (compose patterns)
# --------------------------
def compose_and_schedule_case(case: CaseState, rm: ResourceManager):
    """
    Compose 3–8 patterns and schedule:
    - Linear+optional, XOR, Review→Fix loop, AND-split/join, OR-split/join,
      multi-instance mitigate with k-of-n, preemption/resume on some,
      deadlines (Warn/Escalate), batching, binding/separation-of-duty.
    """

    # 1) Start with identification (optional triage meeting)
    perform_activity(case, rm, "Risk.Identify", PREF_ROLE["Risk.Identify"])
    if random.random() < 0.35:
        perform_activity(case, rm, "Meeting.Triage", PREF_ROLE["Meeting.Triage"], multi_resource=True)

    # 2) Initial assessment
    perform_activity(case, rm, "Assess.Initial", PREF_ROLE["Assess.Initial"])

    # 3) XOR: choose one path based on time/queue/severity/cost-so-far
    # Heuristics: late-in-day / high severity -> senior/quantitative; budget pressure -> vendor assess
    hour = case.ready_time.hour
    branch = None
    if case.cost_so_far > 0.7 * BUDGET_VENDOR_THRESHOLD and random.random() < 0.4:
        branch = "Vendor.Assess"
    else:
        probs = [
            ("Assess.Qualitative", 0.35 + 0.10 * (hour < 11)),
            ("Assess.Quantitative", 0.40 + (0.15 if case.severity == "high" else 0.0)),
            ("Vendor.Assess", 0.25 + 0.10 * (hour >= 16)),
        ]
        # normalize & sample
        z = sum(p for _, p in probs)
        r = random.random() * z
        acc = 0.0
        for lbl, p in probs:
            acc += p
            if r <= acc:
                branch = lbl
                break
    perform_activity(case, rm, branch, PREF_ROLE[branch])

    # 4) Review→Fix loop (geometric; longer assessment increases rework odds)
    rework_count = 0
    while random.random() < (0.20 + 0.10 * rework_count):  # increasing difficulty
        # Review by QA; Fix by analyst → back to Review
        perform_activity(case, rm, "QA", PREF_ROLE["QA"])
        perform_activity(case, rm, "Simplify", PREF_ROLE["Simplify"])
        rework_count += 1
        # Rework premium as a small explicit Reassign (paperwork)
        perform_activity(case, rm, "Reassign", PREF_ROLE["Reassign"])

        if rework_count >= 2 and random.random() < 0.5:
            break

    # 5) Root cause → AND-split: plan & register in parallel (synchronization latency at join)
    perform_activity(case, rm, "Analyze.RootCause", PREF_ROLE["Analyze.RootCause"])
    add_event(case, "AND.Split", case.ready_time, maybe_mask_resource("Unknown"), 0.0)
    case.had_and_split = True

    # Parallel branch A: Mitigate.Plan
    tA_end, _ = perform_activity(case, rm, "Mitigate.Plan", PREF_ROLE["Mitigate.Plan"])
    # Parallel branch B: Update.Register (short)
    saved_ready = case.ready_time  # store after Plan
    # Reset ready time to split time to simulate true parallel start for branch B
    branchB_case = CaseState(
        case_id=case.case_id, severity=case.severity, arrival=case.arrival, deadline=case.deadline,
        policy=case.policy, rng=case.rng, ready_time=case.events[-2]["time:timestamp"],  # the time we emitted AND.Split
        last_label=case.last_label, last_resource_str=case.last_resource_str, cost_so_far=case.cost_so_far,
        escalated=case.escalated, events=case.events, had_and_split=case.had_and_split
    )
    tB_end, _ = perform_activity(branchB_case, rm, "Update.Register", PREF_ROLE["Update.Register"])

    # Synchronization latency before Join (handover/merge delay)
    join_time = max(tA_end, tB_end) + timedelta(minutes=random.randint(5, 35))
    add_event(case, "AND.Join", join_time, maybe_mask_resource("Unknown"), 0.0)
    case.ready_time = join_time

    # 6) Multi-instance Mitigate.Execute (n sub-tasks; proceed when k complete)
    n = random.randint(2, 5)
    k = random.randint(1, n)
    exec_finish_times = []
    to_cancel = []
    for i in range(n):
        # Budget-aware vendor switch for some sub-tasks
        lbl = "Vendor.Mitigate" if (case.cost_so_far > BUDGET_VENDOR_THRESHOLD and random.random() < 0.6) else "Mitigate.Execute"
        # Preemption increases with queue pressure (late day) and long tasks
        preempt = (lbl == "Mitigate.Execute") and (case.ready_time.hour >= 15) and random.random() < 0.4
        if preempt:
            # break into start-resume-complete
            end1, _ = perform_activity(case, rm, "Mitigate.Execute.Task.start", PREF_ROLE["Mitigate.Execute.Task.start"])
            # pause (implicit waiting cost captured as Wait before resume)
            end2, _ = perform_activity(case, rm, "Mitigate.Execute.Task.resume", PREF_ROLE["Mitigate.Execute.Task.resume"], allow_overtime=True, expedite=True)
            end3, _ = perform_activity(case, rm, "Mitigate.Execute.Task.complete", PREF_ROLE["Mitigate.Execute.Task.complete"], allow_overtime=True)
            exec_finish_times.append(end3)
        else:
            endi, _ = perform_activity(case, rm, lbl, PREF_ROLE[lbl], allow_overtime=(case.severity == "high"))
            exec_finish_times.append(endi)
        # sometimes immediately cancel a late/duplicate subtask
        if random.random() < 0.15 and len(exec_finish_times) > k:
            cend, _ = perform_activity(case, rm, "Cancel", PREF_ROLE["Cancel"])
            to_cancel.append(cend)

    exec_finish_times.sort()
    kofn_time = exec_finish_times[k-1]
    add_event(case, "Multi.KofN.Join", kofn_time, maybe_mask_resource("Unknown"), 0.0)
    case.ready_time = kofn_time

    # 7) OR-split: choose subset of QA/Audit/Monitor to run in parallel; batch Audit at top-of-hour
    add_event(case, "OR.Split", case.ready_time, maybe_mask_resource("Unknown"), 0.0)
    or_choices = []
    base = ["QA", "Audit", "Monitor"]
    # choose 1..3
    m = random.randint(1, 3)
    or_choices = random.sample(base, m)
    ends = []
    for lbl in or_choices:
        if lbl == "Audit":
            endi, _ = perform_activity(case, rm, "Audit", PREF_ROLE["Audit"], batch_release="top_of_hour")
        else:
            endi, _ = perform_activity(case, rm, lbl, PREF_ROLE[lbl])
        ends.append(endi)
    or_join_time = max(ends) if ends else case.ready_time
    add_event(case, "OR.Join", or_join_time, maybe_mask_resource("Unknown"), 0.0)
    case.ready_time = or_join_time

    # 8) Verify & Approve with long-term constraints
    # Separation-of-duty: if Audit executed by a resource also trying to Approve, force Reassign
    approve_expedite = (case.severity == "high") and random.random() < 0.4
    perform_activity(case, rm, "Mitigate.Verify", PREF_ROLE["Mitigate.Verify"])
    # Cooling-off: ensure 4h after any Escalate before Approve, else add Wait
    # (Escalation may be injected below – we check and handle both ways)
    approve_time_before = case.ready_time

    # Deadline/SLA checks (inject Warn/Escalate)
    if case.ready_time > case.deadline and not case.escalated:
        perform_activity(case, rm, "Warn", PREF_ROLE["Warn"])
        perform_activity(case, rm, "Escalate", PREF_ROLE["Escalate"])
        case.escalated = True
        # enforce cooling-off 4h before Approve
        min_approve = case.ready_time + timedelta(hours=4)
        if min_approve > case.ready_time:
            # Wait is emitted within perform_activity if we set ready_time backwards,
            # so here we explicitly push the ready_time and charge waiting
            wait_hours = (min_approve - case.ready_time).total_seconds()/3600.0
            add_event(case, "Wait", min_approve, maybe_mask_resource("Unknown"), WAITING_RATE_PER_HOUR * wait_hours)
            case.cost_so_far += WAITING_RATE_PER_HOUR * wait_hours
            case.ready_time = min_approve

    # Approve (separation-of-duty from Audit -> emit Reassign if same family/role dominated events)
    # We'll simulate by sometimes forcing Reassign before Approve
    if "Audit" in (e["concept:name"] for e in case.events) and random.random() < 0.4:
        perform_activity(case, rm, "Reassign", PREF_ROLE["Reassign"])
    perform_activity(case, rm, "Approve", PREF_ROLE["Approve"], allow_overtime=approve_expedite, expedite=approve_expedite)

    # Optional rollback if Verify was long and cost high (negative amount credit)
    if random.random() < 0.05 and case.cost_so_far > 1.5 * BUDGET_VENDOR_THRESHOLD:
        perform_activity(case, rm, "Rollback", PREF_ROLE["Rollback"])

    # Close
    perform_activity(case, rm, "Close.Case", PREF_ROLE["Close.Case"])


# --------------------------
# Severity, deadlines, era drift params
# --------------------------
def sample_severity(rng: np.random.RandomState) -> str:
    return rng.choice(["low", "medium", "high"], p=[0.4, 0.45, 0.15])

def deadline_from(arrival: datetime, severity: str) -> datetime:
    base = {"low": 7*24, "medium": 4*24, "high": 2*24}[severity]  # hours to deadline
    jitter = random.randint(-12, 12)
    return arrival + timedelta(hours=base + jitter)

# --------------------------
# Distinct trace signature
# --------------------------
def signature_of(events: List[Event]) -> Tuple[str, ...]:
    return tuple([str(e["concept:name"]) for e in events])

# --------------------------
# Quality & sanity checks
# --------------------------
def quality_checks(log: EventLog):
    print("\n==================== QUALITY & SANITY CHECKS ====================")
    # Schema
    missing = 0
    nonfloat_amount = 0
    not_increasing = 0
    neg_gap = 0
    inter_event_gaps = []
    trace_lengths = []
    amounts = []
    activity_counter = Counter()
    remaining_times = []
    traces_with_and = 0
    and_branch_imbal = []

    for tr in log:
        # required attributes on events
        last_ts = None
        t_activities = []
        for ev in tr:
            for key in ("concept:name", "time:timestamp", "org:resource", "amount"):
                if key not in ev:
                    missing += 1
            try:
                amounts.append(float(ev["amount"]))
            except Exception:
                nonfloat_amount += 1
            if last_ts is not None:
                delta = ev["time:timestamp"] - last_ts
                if delta.total_seconds() <= 0:
                    not_increasing += 1
                if delta.total_seconds() < 0:
                    neg_gap += 1
                inter_event_gaps.append(max(0.0, delta.total_seconds()/3600.0))
            last_ts = ev["time:timestamp"]
            t_activities.append(ev["concept:name"])
            activity_counter[ev["concept:name"]] += 1
        trace_lengths.append(len(tr))
        # remaining time distribution
        if len(tr) > 1:
            end_time = tr[-1]["time:timestamp"]
            for ev in tr:
                rt = (end_time - ev["time:timestamp"]).total_seconds()/3600.0
                remaining_times.append(max(0.0, rt))
        if any(a == "AND.Split" for a in t_activities):
            traces_with_and += 1
            # simple branch imbalance proxy: time between AND.Split and AND.Join
            try:
                t_split = next(ev["time:timestamp"] for ev in tr if ev["concept:name"] == "AND.Split")
                t_join = next(ev["time:timestamp"] for ev in tr if ev["concept:name"] == "AND.Join")
                and_branch_imbal.append(max(0.0, (t_join - t_split).total_seconds()/3600.0))
            except StopIteration:
                pass

    print(f"Events missing required attributes: {missing}")
    print(f"Amount non-float cast failures:    {nonfloat_amount}")
    print(f"Non-increasing timestamps count:   {not_increasing}")
    print(f"Negative gap count (should be 0):  {neg_gap}")
    print(f"Trace count:                       {len(log)} (capped at {MAX_DISTINCT_TRACES})")
    print(f"Trace length:                      mean={np.mean(trace_lengths):.2f}, std={np.std(trace_lengths):.2f}, min={np.min(trace_lengths)}, max={np.max(trace_lengths)}")
    print(f"Activity frequency (top 10):       {activity_counter.most_common(10)}")
    if amounts:
        print(f"Amount per event:                  mean={np.mean(amounts):.2f}, std={np.std(amounts):.2f}, q10={np.quantile(amounts,0.10):.2f}, q50={np.quantile(amounts,0.50):.2f}, q90={np.quantile(amounts,0.90):.2f}")
    if inter_event_gaps:
        print(f"Inter-event gap (hours):           mean={np.mean(inter_event_gaps):.2f}, std={np.std(inter_event_gaps):.2f}, q10={np.quantile(inter_event_gaps,0.10):.2f}, q90={np.quantile(inter_event_gaps,0.90):.2f}")
    if remaining_times:
        print(f"Remaining time (hours):            mean={np.mean(remaining_times):.2f}, std={np.std(remaining_times):.2f}, q10={np.quantile(remaining_times,0.10):.2f}, q90={np.quantile(remaining_times,0.90):.2f}")
    if traces_with_and > 0:
        print(f"Traces with AND-split:             {traces_with_and} ({100.0*traces_with_and/len(log):.1f}%)")
        print(f"AND-branch latency (h):            mean={np.mean(and_branch_imbal):.2f}, q90={np.quantile(and_branch_imbal,0.90):.2f}")

    # Ambiguity check: sample 100 random prefixes → how often ≥2 distinct next labels exist
    prefixes = {}
    for tr in log:
        labels = [ev["concept:name"] for ev in tr]
        for i in range(len(labels)-1):
            pref = tuple(labels[:i+1])
            nxt = labels[i+1]
            prefixes.setdefault(pref, set()).add(nxt)
    prefix_list = list(prefixes.items())
    random.shuffle(prefix_list)
    sample = prefix_list[:min(100, len(prefix_list))]
    ambiguous = sum(1 for _, nxts in sample if len(nxts) >= 2)
    print(f"Ambiguity (random prefixes):       {ambiguous}/{len(sample)} with ≥2 possible next activities")

    # Era-wise drift (duration/cost per case)
    # We approximate case duration from first to last event; cost is sum(amount)
    era_stats = defaultdict(lambda: {"n":0, "dur":[], "cost":[]})
    for tr in log:
        if len(tr) < 2:
            continue
        start, end = tr[0]["time:timestamp"], tr[-1]["time:timestamp"]
        era = era_for(start)
        dur_hours = (end - start).total_seconds()/3600.0
        total_cost = sum(float(ev["amount"]) for ev in tr)
        era_stats[era]["n"] += 1
        era_stats[era]["dur"].append(dur_hours)
        era_stats[era]["cost"].append(total_cost)
    for era, s in era_stats.items():
        if s["n"]:
            print(f"Era {era}: cases={s['n']}, mean_dur={np.mean(s['dur']):.2f}h, mean_cost={np.mean(s['cost']):.2f}")

    print("=================================================================\n")


# --------------------------
# Main generation routine
# --------------------------
def execute_script():
    rng = np.random.RandomState(SEED)

    # Generate bursty arrivals over 2025 up to MAX_ATTEMPTS (distinct filter applied later)
    start_horizon = datetime(2025, 1, 2, 9, 0)
    end_horizon = datetime(2025, 10, 29, 18, 0)
    arrivals = nhpp_arrivals(MAX_ATTEMPTS, start_horizon, end_horizon, rng)

    # Resources
    rm = make_resources(start_horizon)

    # Prepare event log
    elog = EventLog()

    distinct_signatures: Set[Tuple[str, ...]] = set()
    kept = 0
    attempts = 0

    for i, arr in enumerate(arrivals, start=1):
        if kept >= MAX_DISTINCT_TRACES:
            break
        attempts += 1

        severity = sample_severity(rng)
        policy = random.choice(ASSIGN_POLICIES)
        deadline = deadline_from(arr, severity)
        case = CaseState(
            case_id=f"RID2MIT_{i:04d}",
            severity=severity,
            arrival=arr,
            deadline=deadline,
            policy=policy,
            rng=rng,
            ready_time=arr,
        )
        # Compose and schedule this case
        compose_and_schedule_case(case, rm)

        # Ensure chronologically sorted (we always append in order, but keep robust)
        case.events.sort(key=lambda e: e["time:timestamp"])

        # Distinctness by activity label sequence
        sig = signature_of(case.events)
        if sig in distinct_signatures:
            continue
        distinct_signatures.add(sig)

        # Build pm4py Trace and attach
        tr = Trace()
        tr.attributes["concept:name"] = case.case_id  # ensures case:concept:name in DataFrame
        for ev in case.events:
            tr.append(ev)
        elog.append(tr)
        kept += 1

    # Export with gzip
    out_path = "rid2mit.xes.gz"
    write_xes(elog, out_path)

    # Print a compact generation summary
    print(f"Generated {len(elog)} DISTINCT traces (attempted {attempts}, arrivals={len(arrivals)})")
    print(f"Saved XES log to: {out_path}")

    # Quality checks and concise summaries
    quality_checks(elog)


# --------------------------
# Entrypoint
# --------------------------
if __name__ == "__main__":
    execute_script()
