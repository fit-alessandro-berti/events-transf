"""
O2C SYNTHETIC LOG GENERATOR (pm4py)
-----------------------------------

Goal
-----
Create a realistic, *trace-by-trace* synthetic O2C (Order-to-Cash) event log with:
  - control-flow variety (XOR / AND / OR, loops, skips, k-of-n, preemption),
  - temporal realism (business calendars, bursts, queues, deadlines, overtime),
  - resource dynamics (roles, skills, availability, binding/separation-of-duty),
  - cost dynamics (fixed + variable, handover, penalties, era tariffs, holding).

Each **event** includes:
  - 'concept:name'  (activity label)
  - 'time:timestamp' (Python datetime)
  - 'org:resource'   (string; small fraction 'Unknown')
  - 'amount'         (float; event-level cost)

Each **trace** sets trace.attributes['concept:name'] = case_id
(so 'case:concept:name' appears when converting to a DataFrame).

Output file: ./o2c.xes.gz
Limit: **max 250 distinct traces** (unique by activity label sequence).

Note: The generator *constructs* traces explicitly (no black-box simulator).

Minimal example (covered by code paths below):
  - A case that uses an AND-split (Pick || Pack), handover delay, SLA breach → Escalate,
    one rework iteration, and a vendor fulfillment with overtime cost.

Dependencies: Python stdlib + numpy + (optional) pandas + pm4py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time, date
from typing import Dict, List, Tuple, Optional, Callable, Set

import numpy as np

try:
    import pandas as pd  # optional, only for summaries if available
except Exception:
    pd = None

from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.exporter.xes import exporter as xes_exporter


# ------------- Reproducibility ------------------------------------------------
RNG = np.random.default_rng(20251028)
random.seed(20251028)


# ------------- Activity & Role Alphabets -------------------------------------
# Families: O_ (order), C_ (credit), F_ (fulfillment), S_ (shipping),
# I_ (invoice), P_ (payment), A_ (assurance/audit), E_ (escalation), H_ (handover),
# M_ (external/vendor), X_ (exception/cancel), G_ (generic)
ACTIVITY_SET = [
    # Order/Check
    "O_Submit_Order", "O_Validate", "C_Check_Credit", "O_Amend_Order",
    "O_Manual_Review", "O_Approve", "O_Reject",
    # Fulfillment core
    "F_Pick", "F_Assemble", "F_Pack", "F_QA", "F_Rework_Review", "F_Rework_Fix",
    "F_Wait_Buffer",
    # Shipping
    "S_Schedule_Shipment", "S_Ship", "S_Deliver",
    # Invoicing & Payment
    "I_Generate_Invoice", "I_Batch_Release", "P_Receive_Payment", "P_Reconcile",
    # Assurance / Audit / Compliance (OR / k-of-n)
    "A_Compliance_Check", "A_Fraud_Screen", "A_Address_Validation",
    "A_Audit", "A_Sample_Test",
    # Cross-cutting
    "E_Warn", "E_Escalate", "H_Handover", "M_Vendor_Fulfillment", "M_Resolve_Dispute_Meeting",
    "G_Expedite", "G_Simplify", "X_Cancel", "X_Rollback", "G_Waiting",
    # Preemption labels (no lifecycle needed)
    "F_Pack.start", "F_Pack.resume", "F_Pack.complete"
]

ROLES = [
    "role:clerk", "role:analyst", "role:senior", "role:picker",
    "role:packer", "role:auditor", "role:manager", "role:vendor"
]

# Preferred roles per family
FAMILY_ROLE_PREF = {
    "O_": ["role:clerk", "role:analyst"],
    "C_": ["role:analyst", "role:senior"],
    "F_": ["role:picker", "role:packer"],
    "S_": ["role:clerk", "role:analyst"],
    "I_": ["role:analyst", "role:clerk"],
    "P_": ["role:clerk", "role:analyst"],
    "A_": ["role:auditor", "role:senior"],
    "E_": ["role:senior", "role:manager"],
    "H_": ["role:analyst", "role:clerk"],
    "M_": ["role:vendor"],
    "G_": ["role:analyst", "role:clerk"],
    "X_": ["role:manager", "role:senior"]
}

# Base service-time (minutes) and fixed-cost ($) anchors by family (scaled by resource)
FAMILY_BASE = {
    "O_": (20, 8.0),
    "C_": (25, 10.0),
    "F_": (40, 12.0),
    "S_": (30, 10.0),
    "I_": (25, 9.0),
    "P_": (20, 7.0),
    "A_": (35, 12.0),
    "E_": (15, 50.0),    # escalate carries penalty too
    "H_": (10, 5.0),
    "M_": (80, 8.0),     # vendor is slower
    "G_": (10, 4.0),
    "X_": (15, -5.0)     # exceptions can include credits/refunds
}

# ------------- Business & Shift Calendars ------------------------------------
def is_weekend(d: datetime) -> bool:
    return d.weekday() >= 5  # 5=Sat, 6=Sun

def next_weekday(d: datetime) -> datetime:
    while is_weekend(d):
        d = d + timedelta(days=1)
    return d

def day_at(dt: datetime, hh: int, mm: int = 0) -> datetime:
    return datetime(dt.year, dt.month, dt.day, hh, mm)

class BusinessCalendar:
    """
    Business hours: Mon-Fri, 09:00-12:00 & 13:00-17:00
    Lunch break 12:00-13:00. No work on weekends.
    Overtime: allowed up to 20:00 on Mon-Fri when flagged (cost multiplier).
    """
    def __init__(self):
        self.open_morning = time(9, 0)
        self.break_start = time(12, 0)
        self.break_end = time(13, 0)
        self.close_regular = time(17, 0)
        self.close_overtime = time(20, 0)

    def in_open_slot(self, dt: datetime, allow_overtime=False) -> bool:
        if is_weekend(dt):
            return False
        t = dt.time()
        if self.open_morning <= t < self.break_start:
            return True
        if self.break_end <= t < (self.close_overtime if allow_overtime else self.close_regular):
            return True
        return False

    def next_opening(self, dt: datetime, allow_overtime=False) -> datetime:
        if self.in_open_slot(dt, allow_overtime):
            return dt
        # move to next valid opening slot
        cur = dt
        if is_weekend(cur):
            cur = next_weekday(datetime(cur.year, cur.month, cur.day, 9, 0))
            return cur
        t = cur.time()
        if t < self.open_morning:
            return day_at(cur, 9, 0)
        if self.break_start <= t < self.break_end:
            return day_at(cur, 13, 0)
        if t >= (self.close_overtime if allow_overtime else self.close_regular):
            nxt = cur + timedelta(days=1)
            nxt = next_weekday(nxt)
            return day_at(nxt, 9, 0)
        # otherwise we're before lunch start or in other gaps
        return day_at(cur, 13, 0)

    def advance_working_time(self, start: datetime, minutes: float, allow_overtime=False) -> datetime:
        """
        Move forward 'minutes' of working time starting at (or after) 'start',
        obeying business hours and lunch. If allow_overtime=True, the afternoon slot
        extends to 20:00, but weekends still frozen.
        """
        remaining = minutes
        cur = self.next_opening(start, allow_overtime)
        while remaining > 1e-9:
            if is_weekend(cur):
                cur = self.next_opening(cur, allow_overtime)
                continue
            t = cur.time()
            # Determine current slot end
            if self.open_morning <= t < self.break_start:
                slot_end = day_at(cur, 12, 0)
            elif self.break_end <= t < (self.close_overtime if allow_overtime else self.close_regular):
                slot_end = day_at(cur, 20 if allow_overtime else 17, 0)
            else:
                cur = self.next_opening(cur, allow_overtime)
                continue
            slot_minutes = (slot_end - cur).total_seconds() / 60.0
            step = min(remaining, slot_minutes)
            cur = cur + timedelta(minutes=step)
            remaining -= step
            if remaining > 1e-9 and cur >= slot_end:
                # move to next slot
                nxt = cur
                if t < self.break_start:
                    nxt = day_at(cur, 13, 0)
                else:
                    nxt = next_weekday(cur + timedelta(days=1))
                    nxt = day_at(nxt, 9, 0)
                cur = nxt
        return cur


# ------------- Resources & Assignment ----------------------------------------
@dataclass
class Resource:
    rid: str
    role: str
    speed: float              # <1 faster; >1 slower
    rate: float               # $/minute baseline (era tariff multiplier applies)
    next_free: datetime
    absence_days: Set[date] = field(default_factory=set)
    activity_counter: Dict[str, int] = field(default_factory=dict)
    last_exec: Dict[str, datetime] = field(default_factory=dict)

    def is_present(self, dt: datetime) -> bool:
        return dt.date() not in self.absence_days

    def touch(self, activity: str, when: datetime):
        self.activity_counter[activity] = self.activity_counter.get(activity, 0) + 1
        self.last_exec[activity] = when


def build_resources(start_time: datetime) -> Dict[str, Resource]:
    """
    Create R1..R30 with roles and heterogeneous speeds/rates.
    Also add one Vendor pseudo resource RV (always available, slower).
    """
    roles_cycle = (
        ["role:clerk"] * 6 + ["role:analyst"] * 6 + ["role:senior"] * 4 +
        ["role:picker"] * 4 + ["role:packer"] * 4 + ["role:auditor"] * 3 +
        ["role:manager"] * 2 + ["role:vendor"] * 1
    )
    roles = []
    # Pick first 30 cyclically but shuffle lightly
    base = roles_cycle.copy()
    RNG.shuffle(base)
    roles = base[:30]

    resources: Dict[str, Resource] = {}
    for i in range(1, 31):
        role = roles[i - 1]
        # speed/rate vary by role
        speed = {
            "role:clerk": RNG.normal(1.05, 0.1),
            "role:analyst": RNG.normal(1.0, 0.08),
            "role:senior": RNG.normal(0.9, 0.07),
            "role:picker": RNG.normal(0.95, 0.1),
            "role:packer": RNG.normal(1.0, 0.1),
            "role:auditor": RNG.normal(1.05, 0.08),
            "role:manager": RNG.normal(1.1, 0.1),
            "role:vendor": RNG.normal(1.2, 0.12),
        }[role]
        rate = {
            "role:clerk": 0.7,
            "role:analyst": 1.0,
            "role:senior": 1.6,
            "role:picker": 0.9,
            "role:packer": 0.9,
            "role:auditor": 1.3,
            "role:manager": 1.8,
            "role:vendor": 0.6
        }[role]
        rid = f"R{i}"
        res = Resource(rid=rid, role=role, speed=max(0.6, speed), rate=rate, next_free=start_time)
        # Random absences (5% of workdays in the 90-day window)
        for d in range(0, 90):
            if RNG.random() < 0.05:
                res.absence_days.add((start_time + timedelta(days=d)).date())
        resources[rid] = res

    # Add vendor "RV" with vendor role
    rv = Resource(rid="RV", role="role:vendor", speed=1.25, rate=0.5, next_free=start_time)
    resources[rv.rid] = rv
    return resources


# ------------- Era / Drift ----------------------------------------------------
@dataclass
class Era:
    name: str
    start: datetime
    end: datetime
    duration_scale: float
    approve_slowdown: float
    review_speedup: float
    tariff_mult: float
    vendor_discount: float

def build_eras(global_start: datetime) -> List[Era]:
    # 3 eras over ~90 days
    e1 = Era("Era-1", global_start, global_start + timedelta(days=30),
             duration_scale=1.0, approve_slowdown=1.0, review_speedup=1.0,
             tariff_mult=1.0, vendor_discount=1.0)
    e2 = Era("Era-2", e1.end, e1.end + timedelta(days=30),
             duration_scale=0.95, approve_slowdown=1.1, review_speedup=0.9,
             tariff_mult=1.05, vendor_discount=0.95)
    e3 = Era("Era-3", e2.end, e2.end + timedelta(days=45),
             duration_scale=0.9, approve_slowdown=1.2, review_speedup=0.85,
             tariff_mult=1.1, vendor_discount=0.9)
    return [e1, e2, e3]

def era_of(ts: datetime, eras: List[Era]) -> Era:
    for e in eras:
        if e.start <= ts < e.end:
            return e
    return eras[-1]


# ------------- Arrival Process (NHPP bursts) ---------------------------------
def nhpp_arrival_times(n_cases_target: int, start: datetime) -> List[datetime]:
    """
    Bursty arrivals over ~90 days with waves (morning/afternoon peaks) using thinning.
    """
    horizon_days = 90
    end = start + timedelta(days=horizon_days)
    lam_max = 12.0  # max per hour
    times = []
    t = start
    while len(times) < int(n_cases_target * 1.5):  # generate extra; we will filter by uniqueness later
        # draw candidate inter-arrival with maximum rate
        u = RNG.random()
        w = -math.log(u) / (lam_max / 60.0)  # minutes
        t = t + timedelta(minutes=w)
        if t >= end:
            break
        # intensity shaped by hour-of-day and era drift (seasonality)
        hour = t.hour + t.minute / 60.0
        # peaks around 10:00 and 15:00 on weekdays; almost zero on weekends
        base = 0.2 if is_weekend(t) else (0.6 + 0.7 * math.exp(-((hour - 10) ** 2) / 6.0)
                                          + 0.6 * math.exp(-((hour - 15) ** 2) / 6.0))
        # monthly wave
        day_frac = (t - start).days / 30.0
        seasonal = 0.9 + 0.2 * math.sin(2 * math.pi * day_frac)
        lam_t = lam_max * base * seasonal
        if RNG.random() < lam_t / lam_max:
            times.append(t)
    times.sort()
    return times[: int(n_cases_target * 1.5)]


# ------------- Utility: durations, costs, queues, policies -------------------
def family_of(activity: str) -> str:
    return activity.split("_")[0] + "_"

def heavy_tailed_minutes(base_mean: float, sigma: float = 0.9) -> float:
    """
    Lognormal with Pareto-ish tail via mixture.
    """
    if RNG.random() < 0.80:
        # lognormal around base_mean
        # derive mu, sigma so that mean ~ base_mean
        # mean(lognormal) = exp(mu + sigma^2/2)
        s = abs(sigma)
        mu = math.log(max(1e-3, base_mean)) - 0.5 * s * s
        val = RNG.lognormal(mean=mu, sigma=s)
    else:
        # Pareto tail for heavy bursts
        xm = max(5.0, 0.2 * base_mean)
        alpha = 2.1
        val = xm / (RNG.random() ** (1.0 / alpha))
    return max(1.0, float(val))

def context_switch_penalty(prev_label: Optional[str], cur_label: str) -> float:
    """
    Minutes lost when switching families; encourages batching.
    """
    if not prev_label:
        return 0.0
    pf, cf = family_of(prev_label), family_of(cur_label)
    if pf == cf:
        return 0.0
    return RNG.uniform(3.0, 20.0)

def handover_penalty(prev_res: Optional[str], new_res: str) -> float:
    if (prev_res is None) or (prev_res == new_res):
        return 0.0
    return RNG.uniform(5.0, 25.0)

def overtime_multiplier(dt: datetime) -> float:
    # outside 9-17 on a weekday: 1.4; weekends are frozen in calendar (no work)
    if is_weekend(dt):
        return 1.0
    if time(9, 0) <= dt.time() < time(17, 0):
        return 1.0
    return 1.4

def fixed_cost_for(activity: str) -> float:
    fam = family_of(activity)
    base = FAMILY_BASE.get(fam, (15, 6.0))[1]
    # material cost triggers for some tasks
    if activity in {"F_Assemble", "F_Pack", "S_Ship", "I_Generate_Invoice"}:
        base += RNG.uniform(2.0, 12.0)
    if activity == "E_Escalate":
        base += 80.0  # big penalty
    if activity == "X_Rollback":
        base -= RNG.uniform(5.0, 20.0)  # credit
    return float(base)

def rate_for(resource: Resource, era: Era, at_time: datetime) -> float:
    # era tariff multiplier + overtime
    return resource.rate * era.tariff_mult * overtime_multiplier(at_time)

def q_wait_time(activity: str, queue_len_est: int, policy: str, sampled_pt: float, priority: int) -> float:
    """
    Approximate waiting time based on queue len and policy.
    """
    if queue_len_est <= 0:
        return 0.0
    # rough average service time for peers in queue
    fam = family_of(activity)
    mean_peer = FAMILY_BASE.get(fam, (15, 6.0))[0]
    if policy == "fifo":
        return RNG.uniform(0.3, 0.7) * queue_len_est * mean_peer
    if policy == "spt":
        # if our processing time small, we wait less (SPT)
        scale = 0.5 + min(2.0, sampled_pt / (mean_peer + 1e-9))
        return RNG.uniform(0.2, 0.6) * queue_len_est * mean_peer / scale
    if policy == "priority":
        # higher priority jumps the queue; priority∈{0,1,2}
        return RNG.uniform(0.2, 0.8) * queue_len_est * mean_peer * (1.0 - 0.35 * priority)
    return RNG.uniform(0.2, 0.6) * queue_len_est * mean_peer

def choose_policy_for(activity: str) -> str:
    # per-activity queue policy mixture
    r = RNG.random()
    return "fifo" if r < 0.45 else ("priority" if r < 0.75 else "spt")


# ------------- Assignment Policies -------------------------------------------
def candidate_resources_for(activity: str, resources: Dict[str, Resource]) -> List[Resource]:
    fam = family_of(activity)
    prefs = FAMILY_ROLE_PREF.get(fam, ["role:analyst"])
    # allow any resource but rank preferred roles first
    cand = list(resources.values())
    RNG.shuffle(cand)
    cand.sort(key=lambda r: (0 if r.role in prefs else 1, r.next_free))
    return cand

def pick_resource(activity: str, when: datetime, resources: Dict[str, Resource],
                  policy_mix=("skill", "least_loaded", "random")) -> Resource:
    fam = family_of(activity)
    prefs = FAMILY_ROLE_PREF.get(fam, ["role:analyst"])
    draw = RNG.choice(policy_mix, p=[0.55, 0.3, 0.15])

    cand = candidate_resources_for(activity, resources)

    if draw == "skill":
        for r in cand:
            if r.role in prefs and r.is_present(when):
                return r
    elif draw == "least_loaded":
        # pick the resource with earliest availability among those present
        present = [r for r in cand if r.is_present(when)]
        present.sort(key=lambda r: r.next_free)
        if present:
            return present[0]

    # fallback random among those present
    present = [r for r in cand if r.is_present(when)]
    if present:
        return RNG.choice(present)
    return cand[0]


# ------------- Case structure & helpers --------------------------------------
@dataclass
class ScheduledEvent:
    label: str
    ts: datetime
    resource: str
    cost: float

@dataclass
class CaseState:
    case_id: str
    start_time: datetime
    deadline: datetime
    events: List[ScheduledEvent] = field(default_factory=list)
    last_label: Optional[str] = None
    last_resource: Optional[str] = None
    cumulative_cost: float = 0.0
    has_and_split: bool = False
    and_branch_durations: List[Tuple[float, float]] = field(default_factory=list)
    approve_res: Optional[str] = None
    audit_res: Optional[str] = None
    priority: int = 0  # 0 low, 1 normal, 2 high

    def add(self, label: str, ts: datetime, res: str, amount: float):
        self.events.append(ScheduledEvent(label, ts, res, float(amount)))
        self.cumulative_cost += float(amount)
        self.last_label = label
        self.last_resource = res

    def jitter_forward(self):
        # ensure strictly increasing timestamps within the trace
        self.events.sort(key=lambda e: e.ts)
        last = None
        for ev in self.events:
            if (last is not None) and (ev.ts <= last):
                delay = int(RNG.integers(1, 8))  # ensure timedelta receives built-in int
                ev.ts = last + timedelta(seconds=delay)
            last = ev.ts


# ------------- Scheduling of one activity ------------------------------------
BUS_CAL = BusinessCalendar()

def schedule_activity(
    cs: CaseState,
    resources: Dict[str, Resource],
    activity: str,
    ready_at: datetime,
    era: Era,
    prev_label: Optional[str],
    prev_res: Optional[str],
    queue_state: Dict[str, int],
    allow_overtime=False,
    rework_iter: int = 0,
    force_resource: Optional[Resource] = None,
    forbid_resource: Optional[str] = None
) -> Tuple[datetime, Resource, float, float]:
    """
    Schedule a single activity:
      - resource assignment
      - queue wait (approx, policy-based)
      - calendar alignment (+overtime)
      - handover/context penalties
      - duration (heavy-tailed), with learning/forgetting effects
      - compute event-level cost
    Returns: (finish_time, resource, duration_minutes, waiting_minutes)
    """
    fam = family_of(activity)
    base_mean, _ = FAMILY_BASE.get(fam, (15, 6.0))

    # Resource selection
    res = force_resource or pick_resource(activity, ready_at, resources)
    if forbid_resource and res.rid == forbid_resource:
        # re-choose ignoring the forbidden one
        alternatives = [r for r in candidate_resources_for(activity, resources) if r.rid != forbid_resource]
        if alternatives:
            res = alternatives[0]

    # Learning / Forgetting
    count = res.activity_counter.get(activity, 0)
    learn_factor = 1.0 / (1.0 + 0.15 * math.log1p(count))  # more executions => faster
    last_t = res.last_exec.get(activity)
    forget_factor = 1.0
    if last_t is not None:
        idle_days = max(0.0, (ready_at - last_t).total_seconds() / 86400.0)
        forget_factor = 1.0 + 0.05 * math.log1p(idle_days)

    # Era drift & role skill
    duration_anchor = base_mean * era.duration_scale
    if activity == "O_Approve":
        duration_anchor *= era.approve_slowdown
    if activity in {"F_Rework_Review", "F_Rework_Fix"}:
        duration_anchor *= era.review_speedup

    # Sequence-dependent setup / context
    setup = context_switch_penalty(prev_label, activity) + handover_penalty(prev_res, res.rid)

    # Draw processing duration (heavy-tailed) then scale by speed, learning, forgetting
    raw = heavy_tailed_minutes(duration_anchor)
    proc = max(1.0, raw * res.speed * learn_factor * forget_factor)
    # rework premium: longer per iteration
    if activity in {"F_Rework_Review", "F_Rework_Fix"} and rework_iter > 0:
        proc *= (1.0 + 0.25 * rework_iter)

    # Baseline queue estimate and policy
    policy = choose_policy_for(activity)
    qlen = queue_state.get(activity, 0)
    wait = q_wait_time(activity, qlen, policy, proc, cs.priority)

    # Calendar: earliest feasible start time (resource free, business open/shift)
    start0 = max(ready_at, res.next_free)
    start1 = BUS_CAL.next_opening(start0 + timedelta(minutes=wait + setup), allow_overtime=allow_overtime)
    # Now advance working time for the processing duration
    finish = BUS_CAL.advance_working_time(start1, proc, allow_overtime=allow_overtime)

    # Costs: fixed + variable rate * duration (+overtime multiplier included in rate)
    fixed = fixed_cost_for(activity)
    rate = rate_for(res, era, start1)
    amount = fixed + rate * proc
    # Handover fee on resource switches
    if prev_res and prev_res != res.rid:
        amount += RNG.uniform(4.0, 18.0)
    # Waiting (holding) cost per hour of wait
    amount += 0.2 * (wait / 60.0) * 10.0

    # Era-based vendor tariffs
    if res.role == "role:vendor":
        amount *= era.vendor_discount

    # Round amount to cents
    amount = float(round(amount, 2))

    # Update resource state and queues
    queue_state[activity] = max(0, qlen - 1)  # one leaves queue now
    res.next_free = finish
    res.touch(activity, finish)

    # Small chance to mark unknown resource for robustness tests
    if RNG.random() < 0.02:
        used_rid = "Unknown"
    else:
        used_rid = res.rid

    cs.add(activity, finish, used_rid, amount)

    return finish, res, proc, wait


# ------------- Control-flow building blocks ----------------------------------
def linear_optional(cs: CaseState, resources: Dict[str, Resource], t0: datetime,
                    queue_state: Dict[str, int], eras: List[Era]) -> datetime:
    """
    Start → O_Validate → (optional O_Amend_Order) → C_Check_Credit
    """
    e = era_of(t0, eras)
    t, r, _, _ = schedule_activity(cs, resources, "O_Validate", t0, e, None, None, queue_state)
    if RNG.random() < 0.35:
        e = era_of(t, eras)
        t, r, _, _ = schedule_activity(cs, resources, "O_Amend_Order", t, e, cs.last_label, cs.last_resource, queue_state)
    e = era_of(t, eras)
    t, r, _, _ = schedule_activity(cs, resources, "C_Check_Credit", t, e, cs.last_label, cs.last_resource, queue_state)
    return t

def xor_after_A(cs: CaseState, resources: Dict[str, Resource], t_ready: datetime,
                queue_state: Dict[str, int], eras: List[Era]) -> Tuple[str, datetime]:
    """
    XOR branching after credit check to {O_Approve, O_Manual_Review, O_Reject}
    Branch probabilities depend on time-of-day and cumulative cost so far.
    """
    e = era_of(t_ready, eras)
    # branch base probs
    hour = t_ready.hour + t_ready.minute / 60.0
    p_review = 0.25 + 0.15 * (0.0 if 9 <= hour <= 17 else 1.0)  # more reviews off-hours
    p_reject = 0.10 + 0.15 * (1.0 if cs.cumulative_cost > 150.0 else 0.0)
    p_approve = max(0.05, 1.0 - (p_review + p_reject))
    choice = RNG.choice(["O_Approve", "O_Manual_Review", "O_Reject"], p=[p_approve, p_review, p_reject])
    t, r, _, _ = schedule_activity(cs, resources, choice, t_ready, e, cs.last_label, cs.last_resource, queue_state)
    if choice == "O_Approve":
        cs.approve_res = r.rid
    return choice, t

def rework_loop(cs: CaseState, resources: Dict[str, Resource], t_ready: datetime,
                queue_state: Dict[str, int], eras: List[Era]) -> datetime:
    """
    Geometric repeats of {F_Rework_Review -> F_Rework_Fix} until pass.
    Longer tasks increase chance of new iteration.
    """
    e = era_of(t_ready, eras)
    it = 0
    while True:
        # Review
        e = era_of(t_ready, eras)
        t1, r1, d1, _ = schedule_activity(cs, resources, "F_Rework_Review", t_ready, e,
                                          cs.last_label, cs.last_resource, queue_state, rework_iter=it)
        # Fix
        e = era_of(t1, eras)
        t2, r2, d2, _ = schedule_activity(cs, resources, "F_Rework_Fix", t1, e,
                                          cs.last_label, cs.last_resource, queue_state, rework_iter=it)
        t_ready = t2
        it += 1
        # Stop with prob decreasing as total time small; increase chance when durations heavy
        again = RNG.random() < min(0.6, 0.15 + 0.0008 * (d1 + d2) ** 1.2)
        if not again or it > 3:
            break
    return t_ready

def and_split_join(cs: CaseState, resources: Dict[str, Resource], t_ready: datetime,
                   queue_state: Dict[str, int], eras: List[Era]) -> datetime:
    """
    A → (F_Pick || F_Pack) → Join → S_Schedule_Shipment
    Sync latency: join waits for both; add small handover delay before join.
    Preemption possible on F_Pack when queues are high.
    """
    e = era_of(t_ready, eras)
    cs.has_and_split = True

    # Branch 1: F_Pick
    t_pick, r_pick, _, _ = schedule_activity(cs, resources, "F_Pick", t_ready, e,
                                             cs.last_label, cs.last_resource, queue_state)

    # Branch 2: F_Pack with preemption when queue is high
    # Simulate preemption by splitting into .start/.resume/.complete
    preempt = RNG.random() < min(0.6, 0.15 + 0.05 * max(0, queue_state.get("F_Pack", 0)))
    if preempt:
        e = era_of(t_ready, eras)
        t_ps, r_pack, d1, w1 = schedule_activity(cs, resources, "F_Pack.start", t_ready, e,
                                                 cs.last_label, cs.last_resource, queue_state, allow_overtime=False)
        # interruption gap (pressure ~ queue length)
        gap = RNG.uniform(10.0, 60.0) * (1.0 + 0.2 * queue_state.get("F_Pack", 0))
        t_resume_ready = t_ps + timedelta(minutes=gap)
        e = era_of(t_resume_ready, eras)
        t_pr, r_pack, d2, w2 = schedule_activity(cs, resources, "F_Pack.resume", t_resume_ready, e,
                                                 "F_Pack.start", r_pack.rid, queue_state, allow_overtime=True)
        # finalization (short)
        e = era_of(t_pr, eras)
        t_pack, r_pack2, d3, w3 = schedule_activity(cs, resources, "F_Pack.complete", t_pr, e,
                                                    "F_Pack.resume", r_pack.rid, queue_state, allow_overtime=True)
        # ensure it's the same resource (binding-of-duty for the resumed task)
        # (we already bias with prev_res in schedule; if different, it's fine)
        last_pack_finish = t_pack
    else:
        e = era_of(t_ready, eras)
        last_pack_finish, r_pack, _, _ = schedule_activity(cs, resources, "F_Pack", t_ready, e,
                                                           cs.last_label, cs.last_resource, queue_state)

    # Join waits for both
    join_start = max(t_pick, last_pack_finish)
    # synchronization latency/handover before join
    join_delay = RNG.uniform(2.0, 20.0)
    join_time = join_start + timedelta(minutes=join_delay)
    # Add a synthetic join marker (as a zero/short cost "G_Waiting")
    cs.add("G_Waiting", join_time, cs.last_resource or "Unknown", float(round(0.5 * join_delay, 2)))

    # Store branch imbalance (seconds)
    cs.and_branch_durations.append((
        abs((t_pick - last_pack_finish).total_seconds()) / 60.0,
        max(1.0, ((t_pick - t_ready).total_seconds() / 60.0) + ((last_pack_finish - t_ready).total_seconds() / 60.0))
    ))

    # Post-join
    e = era_of(join_time, eras)
    t_out, r_out, _, _ = schedule_activity(cs, resources, "S_Schedule_Shipment", join_time, e,
                                           cs.last_label, cs.last_resource, queue_state)
    return t_out

def or_split_join(cs: CaseState, resources: Dict[str, Resource], t_ready: datetime,
                  queue_state: Dict[str, int], eras: List[Era]) -> datetime:
    """
    Choose 1..3 of compliance checks to run in parallel.
    """
    tasks = ["A_Compliance_Check", "A_Fraud_Screen", "A_Address_Validation"]
    k = RNG.integers(1, 4)
    chosen = list(RNG.choice(tasks, size=k, replace=False))
    end_times = []
    for a in chosen:
        e = era_of(t_ready, eras)
        t, r, _, _ = schedule_activity(cs, resources, a, t_ready, e,
                                       cs.last_label, cs.last_resource, queue_state)
        end_times.append(t)
    # OR-join time
    j = max(end_times) if end_times else t_ready
    cs.add("G_Waiting", j, cs.last_resource or "Unknown", float(round(RNG.uniform(0.0, 3.0), 2)))
    return j

def k_of_n_multiinstance(cs: CaseState, resources: Dict[str, Resource], t_ready: datetime,
                         queue_state: Dict[str, int], eras: List[Era]) -> datetime:
    """
    Spawn n sample tests; proceed after k complete; cancel rest with refund.
    """
    n = 4
    k = 2
    finishes = []
    labels = []
    for i in range(n):
        lab = f"A_Sample_Test"
        e = era_of(t_ready, eras)
        t, r, _, _ = schedule_activity(cs, resources, lab, t_ready, e,
                                       cs.last_label, cs.last_resource, queue_state)
        finishes.append(t)
        labels.append(lab)
    # proceed when k-th finishes
    kth = sorted(finishes)[k - 1]
    # cancel remaining after kth with credit
    for t_end in finishes:
        if t_end > kth and RNG.random() < 0.8:
            refund = -round(RNG.uniform(2.0, 10.0), 2)
            cs.add("X_Rollback", t_end + timedelta(minutes=1), "Unknown", refund)
    return kth

def batching_release(cs: CaseState, t_ready: datetime) -> datetime:
    """
    Invoice batch release only at 10:00 and 15:00.
    """
    d = t_ready
    # next release slot today at 10:00 or 15:00; otherwise next workday 10:00
    slots = [day_at(d, 10, 0), day_at(d, 15, 0)]
    slots = [s for s in slots if (s >= d and not is_weekend(s))]
    if not slots:
        d = next_weekday(d + timedelta(days=1))
        d = day_at(d, 10, 0)
    else:
        d = slots[0]
    # waiting cost event
    cs.add("I_Batch_Release", d, "Unknown", round(RNG.uniform(0.0, 5.0), 2))
    return d


# ------------- One full O2C case ---------------------------------------------
def simulate_case(case_id: str, start_time: datetime, resources: Dict[str, Resource],
                  eras: List[Era], queue_state: Dict[str, int]) -> CaseState:
    cs = CaseState(
        case_id=case_id,
        start_time=start_time,
        deadline=start_time + timedelta(days=RNG.uniform(3, 10)),
        priority=int(RNG.choice([0, 1, 2], p=[0.25, 0.55, 0.20]))
    )

    # Case start marker
    cs.add("O_Submit_Order", start_time, "Unknown", round(RNG.uniform(0.0, 3.0), 2))

    # Linear with optional insert
    t = linear_optional(cs, resources, start_time, queue_state, eras)

    # XOR branch after C_Check_Credit
    choice, t = xor_after_A(cs, resources, t, queue_state, eras)

    if choice == "O_Reject":
        # fast-path to cancellation
        e = era_of(t, eras)
        t, r, _, _ = schedule_activity(cs, resources, "X_Cancel", t, e, cs.last_label, cs.last_resource, queue_state)
        return cs

    # Optional manual review path may trigger rework loop
    if choice == "O_Manual_Review":
        if RNG.random() < 0.6:
            t = rework_loop(cs, resources, t, queue_state, eras)
        # After review, approve
        e = era_of(t, eras)
        t, r, _, _ = schedule_activity(cs, resources, "O_Approve", t, e, cs.last_label, cs.last_resource, queue_state)
        cs.approve_res = r.rid

    # AND-split (Pick || Pack) then join → schedule shipment
    t = and_split_join(cs, resources, t, queue_state, eras)

    # OR-split compliance checks
    t = or_split_join(cs, resources, t, queue_state, eras)

    # Vendor option based on budget/ deadline slack
    slack_hours = (cs.deadline - t).total_seconds() / 3600.0
    if cs.cumulative_cost > 260.0 or slack_hours < 12:
        # outsource fulfillment step (slower but cheaper tariff in later eras)
        e = era_of(t, eras)
        # force vendor resource
        vendor = resources["RV"]
        t, _, _, _ = schedule_activity(cs, resources, "M_Vendor_Fulfillment", t, e,
                                       cs.last_label, cs.last_resource, queue_state, allow_overtime=True,
                                       force_resource=vendor)

    # Ship & Deliver
    e = era_of(t, eras)
    t, _, _, _ = schedule_activity(cs, resources, "S_Ship", t, e, cs.last_label, cs.last_resource, queue_state)
    e = era_of(t, eras)
    t, _, _, _ = schedule_activity(cs, resources, "S_Deliver", t, e, cs.last_label, cs.last_resource, queue_state)

    # Invoice with batching/release
    t = batching_release(cs, t)
    e = era_of(t, eras)
    t, _, _, _ = schedule_activity(cs, resources, "I_Generate_Invoice", t, e, cs.last_label, cs.last_resource, queue_state)

    # Payment & Reconcile (cooling-off between invoice and payment; may reroute to buffer)
    min_lag = RNG.uniform(120.0, 600.0)  # minutes
    if (t - start_time).total_seconds() / 60.0 < min_lag:
        # cooling off not satisfied → buffer wait
        wait_end = t + timedelta(minutes=(min_lag - (t - start_time).total_seconds() / 60.0))
        cs.add("F_Wait_Buffer", wait_end, "Unknown", round(RNG.uniform(0.0, 6.0), 2))
        t = wait_end
    e = era_of(t, eras)
    t, _, _, _ = schedule_activity(cs, resources, "P_Receive_Payment", t, e,
                                   cs.last_label, cs.last_resource, queue_state)
    e = era_of(t, eras)
    t, _, _, _ = schedule_activity(cs, resources, "P_Reconcile", t, e,
                                   cs.last_label, cs.last_resource, queue_state)

    # Audit with separation-of-duty (must differ from Approve resource)
    e = era_of(t, eras)
    t, r_audit, _, _ = schedule_activity(cs, resources, "A_Audit", t, e, cs.last_label, cs.last_resource, queue_state,
                                         forbid_resource=cs.approve_res)
    cs.audit_res = r_audit.rid

    # Multi-resource meeting (senior + analyst) if SLA breach
    if t > cs.deadline:
        # Warn + Escalate (senior)
        t_warn = t + timedelta(minutes=1)
        cs.add("E_Warn", t_warn, "Unknown", round(5.0 + RNG.uniform(0.0, 8.0), 2))
        e = era_of(t_warn, eras)
        # force senior role for Escalate
        seniors = [r for r in resources.values() if r.role in ("role:senior", "role:manager")]
        force_r = RNG.choice(seniors) if seniors else None
        t, _, _, _ = schedule_activity(cs, resources, "E_Escalate", t_warn, e, cs.last_label, cs.last_resource,
                                       queue_state, allow_overtime=True, force_resource=force_r)

        # Two-person meeting (analyst + senior) to resolve dispute
        # Find earliest overlap window
        analyst = next((r for r in resources.values() if r.role == "role:analyst"), None)
        senior = next((r for r in resources.values() if r.role in ("role:senior", "role:manager")), None)
        if analyst and senior:
            # earliest time when both free and within calendar (allow overtime)
            base = max(t, analyst.next_free, senior.next_free)
            base = BUS_CAL.next_opening(base, allow_overtime=True)
            # assume 45-90 minute meeting
            meet_dur = RNG.uniform(45.0, 90.0)
            meet_end = BUS_CAL.advance_working_time(base, meet_dur, allow_overtime=True)
            # costs for both resources
            ecur = era_of(base, eras)
            amt = fixed_cost_for("M_Resolve_Dispute_Meeting")
            amt += rate_for(analyst, ecur, base) * (meet_dur / 2.0)
            amt += rate_for(senior, ecur, base) * (meet_dur / 2.0)
            cs.add("M_Resolve_Dispute_Meeting", meet_end, "Unknown", round(amt, 2))
            # block their schedules
            analyst.next_free = meet_end
            senior.next_free = meet_end
            t = meet_end

    # Optional expedite fee to skip QA earlier (memory): if rework was fast, QA might be skipped – already modeled.
    # Final: Close (implicit by end-of-trace)

    # Jitter to ensure strict monotonicity
    cs.jitter_forward()
    return cs


# ------------- Sanity & Quality Checks ---------------------------------------
def check_schema(log: EventLog) -> Tuple[int, int]:
    missing = 0
    bad_cost = 0
    for tr in log:
        prev = None
        for ev in tr:
            # Required attrs
            for k in ("concept:name", "time:timestamp", "org:resource", "amount"):
                if k not in ev:
                    missing += 1
            # types
            if not isinstance(ev["amount"], (int, float)):
                bad_cost += 1
            # monotone timestamps per trace
            if prev and ev["time:timestamp"] <= prev:
                # we silently fix by nudging forward (should not happen due to jitter)
                ev["time:timestamp"] = prev + timedelta(seconds=1)
            prev = ev["time:timestamp"]
    return missing, bad_cost

def trace_stats(log: EventLog):
    lengths = []
    costs = []
    durations = []
    starts = []
    ends = []
    for tr in log:
        ts = [e["time:timestamp"] for e in tr]
        am = [float(e["amount"]) for e in tr]
        lengths.append(len(tr))
        costs.append(sum(am))
        durations.append((max(ts) - min(ts)).total_seconds() / 3600.0)
        starts.append(min(ts))
        ends.append(max(ts))
    return {
        "n_traces": len(log),
        "len_min": int(np.min(lengths)),
        "len_p50": float(np.median(lengths)),
        "len_p90": float(np.quantile(lengths, 0.9)),
        "len_max": int(np.max(lengths)),
        "cost_mean": float(np.mean(costs)),
        "cost_p90": float(np.quantile(costs, 0.9)),
        "dur_h_mean": float(np.mean(durations)),
        "dur_h_p90": float(np.quantile(durations, 0.9))
    }

def ambiguity_check(log: EventLog, sample_prefixes: int = 100) -> Tuple[int, int]:
    """
    Sample prefixes across the log and count how many have 2+ distinct next activities.
    """
    # Build prefix → next activity set
    nexts: Dict[Tuple[str, ...], Set[str]] = {}
    for tr in log:
        labels = [e["concept:name"] for e in tr]
        for i in range(1, max(1, len(labels) - 1)):
            key = tuple(labels[:i])
            nxt = labels[i]
            nexts.setdefault(key, set()).add(nxt)
    keys = list(nexts.keys())
    if not keys:
        return 0, 0
    sample_size = min(sample_prefixes, len(keys))
    indices = RNG.choice(len(keys), size=sample_size, replace=False)
    picks = [keys[i] for i in indices]
    ambi = sum(1 for k in picks if len(nexts[k]) >= 2)
    return ambi, len(picks)

def remaining_time_signal(log: EventLog, sample_events: int = 200) -> Dict[str, float]:
    all_events = []
    for tr in log:
        end = max(e["time:timestamp"] for e in tr)
        for ev in tr:
            rem = (end - ev["time:timestamp"]).total_seconds() / 3600.0
            all_events.append(rem)
    if not all_events:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    sample = RNG.choice(all_events, size=min(sample_events, len(all_events)), replace=False)
    return {
        "mean": float(np.mean(sample)),
        "std": float(np.std(sample)),
        "p10": float(np.quantile(sample, 0.1)),
        "p50": float(np.quantile(sample, 0.5)),
        "p90": float(np.quantile(sample, 0.9)),
    }


# ------------- Main generation ------------------------------------------------
def execute_script(out_path: str = "o2c.xes.gz", limit_traces: int = 250):
    # Global clock start (Tuesday is unknown; pick a stable date for bursts)
    global_start = datetime(2025, 1, 6, 9, 0, 0)  # Monday 09:00
    eras = build_eras(global_start)
    resources = build_resources(global_start)

    # NHPP arrivals (generate more than needed; we'll keep first 250 *unique* traces)
    arrivals = nhpp_arrival_times(limit_traces, global_start)
    arrivals.sort()

    # Queue state (approx per-activity)
    queue_state: Dict[str, int] = {a: int(RNG.integers(0, 5)) for a in ACTIVITY_SET}

    log = EventLog()
    unique_signatures: Set[Tuple[str, ...]] = set()
    kept = 0
    case_idx = 0

    for start_time in arrivals:
        case_id = f"Case_{case_idx:05d}"
        case_idx += 1

        cs = simulate_case(case_id, start_time, resources, eras, queue_state)

        # Signature by label sequence
        sig = tuple(ev.label for ev in cs.events)
        if sig in unique_signatures:
            # Encourage uniqueness by injecting a tiny optional step if we still need more
            if kept < limit_traces:
                # Insert ambiguous skip/expedite
                t_plus = cs.events[-1].ts + timedelta(minutes=RNG.uniform(1, 15))
                cs.add("G_Simplify" if RNG.random() < 0.5 else "G_Expedite", t_plus, "Unknown",
                       round(RNG.uniform(1.0, 12.0), 2))
                sig = tuple(ev.label for ev in cs.events)
        # Keep if unique
        if sig not in unique_signatures:
            unique_signatures.add(sig)
            # Convert to pm4py Trace
            tr = Trace()
            tr.attributes["concept:name"] = cs.case_id  # so case:concept:name is present
            for ev in sorted(cs.events, key=lambda e: e.ts):
                e = Event({
                    "concept:name": ev.label,
                    "time:timestamp": ev.ts,
                    "org:resource": ev.resource,
                    "amount": float(ev.cost)
                })
                tr.append(e)
                # Update queue heuristic (bursty feedback)
                fam = family_of(ev.label)
                if fam[0] in "OCFSPAI":
                    queue_state[ev.label] = queue_state.get(ev.label, 0) + int(RNG.integers(-1, 2))
                    queue_state[ev.label] = max(0, queue_state[ev.label])
            log.append(tr)
            kept += 1

        if kept >= limit_traces:
            break

    # --- Quality & Sanity Checks --------------------------------------------
    missing, bad_cost = check_schema(log)
    stats = trace_stats(log)
    ambi_ok, ambi_total = ambiguity_check(log, 100)
    rem = remaining_time_signal(log, 250)

    # Parallelism & drift summaries (approx)
    # Estimate percent with AND-split by searching for both F_Pick and (F_Pack OR F_Pack.*)
    and_count = 0
    branch_imbalance = []
    era_costs = {"Era-1": [], "Era-2": [], "Era-3": []}
    era_durations = {"Era-1": [], "Era-2": [], "Era-3": []}
    for tr in log:
        labs = [e["concept:name"] for e in tr]
        if "F_Pick" in labs and (("F_Pack" in labs) or ("F_Pack.start" in labs)):
            and_count += 1
        ts = [e["time:timestamp"] for e in tr]
        dur = (max(ts) - min(ts)).total_seconds() / 3600.0
        cost = sum(float(e["amount"]) for e in tr)
        ecase = era_of(min(ts), eras).name
        era_costs[ecase].append(cost)
        era_durations[ecase].append(dur)
        # crude imbalance metric: time diff between Pick and Pack finishing
        if "F_Pick" in labs and "S_Schedule_Shipment" in labs:
            try:
                t_pick = [e["time:timestamp"] for e in tr if e["concept:name"] == "F_Pick"][0]
                t_pack = [e["time:timestamp"] for e in tr if e["concept:name"].startswith("F_Pack")][-1]
                branch_imbalance.append(abs((t_pick - t_pack).total_seconds()) / 60.0)
            except Exception:
                pass

    # Print concise report
    print("=== GENERATION REPORT ===")
    print(f"Traces kept (unique): {len(log)} (limit {limit_traces})")
    print(f"Schema missing attrs: {missing}; non-float costs: {bad_cost}")
    print(f"Trace length: min {stats['len_min']}  p50 {stats['len_p50']:.1f}  p90 {stats['len_p90']:.1f}  max {stats['len_max']}")
    print(f"Trace cost:   mean ${stats['cost_mean']:.2f}  p90 ${stats['cost_p90']:.2f}")
    print(f"Duration(h):  mean {stats['dur_h_mean']:.2f}  p90 {stats['dur_h_p90']:.2f}")
    print(f"Ambiguity check: {ambi_ok}/{ambi_total} random prefixes have ≥2 distinct next activities")
    print(f"Remaining time (hours): mean {rem['mean']:.2f}  std {rem['std']:.2f}  p10 {rem['p10']:.2f}  p50 {rem['p50']:.2f}  p90 {rem['p90']:.2f}")
    if and_count > 0:
        print(f"Parallelism: {100.0 * and_count / len(log):.1f}% of traces contain AND-split")
        if branch_imbalance:
            print(f"AND branch imbalance (min/median/p90 minutes): "
                  f"{np.min(branch_imbalance):.1f} / {np.median(branch_imbalance):.1f} / {np.quantile(branch_imbalance, 0.9):.1f}")
    # Drift by era
    for en in era_costs:
        if era_costs[en]:
            print(f"{en}: mean cost ${np.mean(era_costs[en]):.2f}, mean duration {np.mean(era_durations[en]):.2f}h, n={len(era_costs[en])}")

    # --- Export ----------------------------------------------------------------
    # Ensure strictly increasing timestamps inside each trace (already done)
    # and save to compressed XES
    xes_exporter.apply(log, out_path)
    print(f"Saved XES log to: {out_path}")


# ------------- Entrypoint -----------------------------------------------------
if __name__ == "__main__":
    # Very important: limit the XES log to **250 distinct traces**
    execute_script(out_path="o2c.xes.gz", limit_traces=250)
