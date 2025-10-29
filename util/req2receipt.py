# req2receipt_synth.py
# -------------------------------------------------------------
# Requisition-to-Receipt synthetic event log generator (pm4py)
# -------------------------------------------------------------
# Role: Senior Process Mining Engineer
#
# WHAT THIS SCRIPT DOES
# - Builds a synthetic Requisition-to-Receipt (Req→PO→Receipt→Invoice→Payment) log
# - Injects rich control-flow, temporal, resource and cost patterns (see sections below)
# - Constructs the log trace-by-trace (no black-box simulator)
# - Exports a single XES log with required attributes for every event:
#     * concept:name  (activity label)
#     * time:timestamp (datetime)
#     * org:resource  (string; small fraction 'Unknown')
#     * amount        (float; event-level cost)
# - Sets trace.attributes['concept:name'] = case id (so case:concept:name appears in DF)
# - Limits the log to **250 distinct traces** (very important per request)
#
# MAIN OUTPUT
#   The script writes to: req2receipt.xes.gz
#
# DEPENDENCIES
#   pip install pm4py numpy pandas
#
# MINIMAL EXAMPLE (from the prompt) is also reproduced at the end in comments.
# -------------------------------------------------------------

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd

# pm4py objects
from pm4py.objects.log.obj import Event, Trace, EventLog

# robust XES export (pm4py.write_xes supports .xes.gz; fall back to exporter.apply)
try:
    from pm4py import write_xes as _pm4py_write_xes

    def write_xes_safe(log, path):
        _pm4py_write_xes(log, path)
except Exception:
    from pm4py.objects.log.exporter.xes import exporter as xes_exporter

    def write_xes_safe(log, path):
        xes_exporter.apply(log, path)


def ensure_trace_sorted_by_timestamp(trace: Trace) -> Trace:
    """
    Sort pm4py traces chronologically without assuming list.sort support.
    Returns the sorted trace (which may be the original object).
    """
    key_fn = lambda e: e["time:timestamp"]
    try:
        trace.sort(key=key_fn)  # type: ignore[attr-defined]
        return trace
    except AttributeError:
        sorted_events = sorted(trace, key=key_fn)
        try:
            trace[:] = sorted_events  # type: ignore[index]
            return trace
        except (TypeError, AttributeError):
            # fallback: rebuild trace while preserving metadata
            try:
                while len(trace) > 0:
                    trace.pop()  # type: ignore[attr-defined]
                for ev in sorted_events:
                    trace.append(ev)
                return trace
            except AttributeError:
                new_trace = Trace()
                if hasattr(trace, "attributes"):
                    new_trace.attributes = trace.attributes
                if hasattr(trace, "extensions"):
                    new_trace.extensions = trace.extensions
                if hasattr(trace, "properties"):
                    new_trace.properties = trace.properties
                for ev in sorted_events:
                    new_trace.append(ev)
                return new_trace


# -----------------------------
# GLOBAL CONFIG
# -----------------------------
RNG_SEED = 42
np.random.seed(RNG_SEED)
random.seed(RNG_SEED)

N_TRACES = 250  # hard cap as requested
OUT_PATH = "req2receipt.xes.gz"

# Time horizon for case arrivals
H_START = datetime(2025, 5, 5, 9, 0, 0)  # Monday
H_DAYS = 70  # ~10 weeks
H_END = H_START + timedelta(days=H_DAYS)

# Business calendar (global) Mon-Fri 09:00–17:00 with lunch 12–13; weekends frozen
BUSINESS_START = time(9, 0, 0)
BUSINESS_END = time(17, 0, 0)
LUNCH_START = time(12, 0, 0)
LUNCH_END = time(13, 0, 0)

# Resource shifts (per role; simple template hours)
ROLE_SHIFT = {
    "role:analyst": (time(9, 0), time(17, 0)),
    "role:senior": (time(10, 0), time(18, 0)),
    "role:manager": (time(8, 0), time(16, 0)),
    "role:auditor": (time(9, 30), time(17, 30)),
    "role:clerk": (time(9, 0), time(17, 0)),
    "role:vendor": (time(10, 0), time(19, 0)),
}

# Era drift (three eras across the horizon)
#   - speed_mult: <1.0 faster, >1.0 slower
#   - cost_rate_mult: multiplier on hourly cost rate
ERAS = [
    {"name": "era_early", "split": 1 / 3, "speed_mult": 1.15, "cost_rate_mult": 1.05},
    {"name": "era_mid", "split": 2 / 3, "speed_mult": 0.95, "cost_rate_mult": 1.00},
    {"name": "era_late", "split": 1.00, "speed_mult": 1.05, "cost_rate_mult": 0.92},
]

# Activity alphabet (≈30+ labels; Req→PO→Receipt→Invoice→Payment domain + generic ops)
A = {
    # requisition
    "Start",
    "Create_Requisition",
    "Validate_Requisition",
    "Budget_Check",
    "Approve_L1",
    "Approve_L2",
    # sourcing
    "Select_Vendor",
    "Technical_Evaluation",
    "Legal_Review",
    "Risk_Assessment",
    "Consolidate_Evaluations",
    "Negotiate.start",
    "Negotiate.resume",
    "Negotiate.complete",
    # purchase order
    "Create_PO",
    "Approve_PO",
    "Send_PO",
    "Vendor_Confirm_L1",
    "Vendor_Confirm_L2",
    "Vendor_Confirm_L3",
    "Cancel_Vendor_Confirm",
    # receipt
    "Receive_Goods",
    "Record_Goods_Receipt",
    "Quality_Check",
    "Rework_QC",
    # invoice & matching
    "Invoice_Receive",
    "Invoice_Validate",   # batching (hourly release)
    "Three_Way_Match",    # multi-resource (meeting)
    "Resolve_Mismatch",
    # payment & close
    "Pay_Invoice",
    "Audit",
    "Close_Case",
    "End",
    # controls/patterns
    "Escalate",
    "Expedite",
    "Wait_Buffer",  # cooling-off / explicit holding
    "Handover",
    "Reassign",
    "Simplify_Scope",
    "Vendor_Path",
}

# Family buckets for context-switch overhead
FAMILY = {
    "Create_Requisition": "REQ",
    "Validate_Requisition": "REQ",
    "Budget_Check": "REQ",
    "Approve_L1": "APP",
    "Approve_L2": "APP",
    "Select_Vendor": "SRC",
    "Technical_Evaluation": "SRC",
    "Legal_Review": "SRC",
    "Risk_Assessment": "SRC",
    "Consolidate_Evaluations": "SRC",
    "Negotiate.start": "SRC",
    "Negotiate.resume": "SRC",
    "Negotiate.complete": "SRC",
    "Create_PO": "PO",
    "Approve_PO": "APP",
    "Send_PO": "PO",
    "Vendor_Confirm_L1": "VEN",
    "Vendor_Confirm_L2": "VEN",
    "Vendor_Confirm_L3": "VEN",
    "Cancel_Vendor_Confirm": "VEN",
    "Receive_Goods": "REC",
    "Record_Goods_Receipt": "REC",
    "Quality_Check": "QC",
    "Rework_QC": "QC",
    "Invoice_Receive": "INV",
    "Invoice_Validate": "INV",
    "Three_Way_Match": "INV",
    "Resolve_Mismatch": "INV",
    "Pay_Invoice": "PAY",
    "Audit": "AUD",
    "Close_Case": "CLS",
    "End": "CLS",
    "Escalate": "CTL",
    "Expedite": "CTL",
    "Wait_Buffer": "CTL",
    "Handover": "CTL",
    "Reassign": "CTL",
    "Simplify_Scope": "CTL",
    "Vendor_Path": "CTL",
    "Start": "CTL",
}

# Preferred roles per activity (fallback allowed but slower)
PREF_ROLE = {
    "Create_Requisition": ["role:analyst", "role:clerk"],
    "Validate_Requisition": ["role:analyst"],
    "Budget_Check": ["role:analyst", "role:senior"],
    "Approve_L1": ["role:manager", "role:senior"],
    "Approve_L2": ["role:manager"],
    "Select_Vendor": ["role:analyst", "role:senior"],
    "Technical_Evaluation": ["role:senior"],
    "Legal_Review": ["role:senior", "role:manager"],
    "Risk_Assessment": ["role:senior"],
    "Consolidate_Evaluations": ["role:analyst"],
    "Negotiate.start": ["role:senior", "role:manager"],
    "Negotiate.resume": ["role:senior", "role:manager"],
    "Negotiate.complete": ["role:senior", "role:manager"],
    "Create_PO": ["role:clerk", "role:analyst"],
    "Approve_PO": ["role:manager"],
    "Send_PO": ["role:clerk", "role:analyst"],
    "Vendor_Confirm_L1": ["role:vendor"],
    "Vendor_Confirm_L2": ["role:vendor"],
    "Vendor_Confirm_L3": ["role:vendor"],
    "Cancel_Vendor_Confirm": ["role:manager"],
    "Receive_Goods": ["role:clerk"],
    "Record_Goods_Receipt": ["role:clerk"],
    "Quality_Check": ["role:senior"],
    "Rework_QC": ["role:analyst"],
    "Invoice_Receive": ["role:clerk"],
    "Invoice_Validate": ["role:analyst", "role:senior"],
    "Three_Way_Match": ["role:analyst", "role:senior"],
    "Resolve_Mismatch": ["role:analyst", "role:senior"],
    "Pay_Invoice": ["role:clerk", "role:manager"],
    "Audit": ["role:auditor"],
    "Close_Case": ["role:clerk"],
    "Escalate": ["role:manager"],
    "Expedite": ["role:manager"],
    "Wait_Buffer": ["role:clerk"],
    "Handover": ["role:analyst"],
    "Reassign": ["role:manager"],
    "Simplify_Scope": ["role:manager"],
    "Vendor_Path": ["role:manager"],
    "Start": ["role:clerk"],
    "End": ["role:clerk"],
}

# Base activity parameters (approximate mean service hours & fixed material cost)
ACT_PARAMS = {
    "Create_Requisition": (1.2, 12.0),
    "Validate_Requisition": (0.8, 6.0),
    "Budget_Check": (1.1, 4.0),
    "Approve_L1": (0.5, 0.0),
    "Approve_L2": (0.6, 0.0),
    "Select_Vendor": (1.5, 0.0),
    "Technical_Evaluation": (2.0, 0.0),
    "Legal_Review": (1.7, 0.0),
    "Risk_Assessment": (1.6, 0.0),
    "Consolidate_Evaluations": (0.7, 0.0),
    "Negotiate.start": (0.4, 0.0),
    "Negotiate.resume": (0.3, 0.0),
    "Negotiate.complete": (1.2, 0.0),
    "Create_PO": (0.9, 4.0),
    "Approve_PO": (0.4, 0.0),
    "Send_PO": (0.3, 5.0),
    "Vendor_Confirm_L1": (0.6, 0.0),
    "Vendor_Confirm_L2": (0.6, 0.0),
    "Vendor_Confirm_L3": (0.6, 0.0),
    "Cancel_Vendor_Confirm": (0.2, -5.0),
    "Receive_Goods": (0.8, 20.0),
    "Record_Goods_Receipt": (0.4, 0.0),
    "Quality_Check": (1.0, 0.0),
    "Rework_QC": (0.8, 0.0),
    "Invoice_Receive": (0.4, 0.0),
    "Invoice_Validate": (0.9, 0.0),
    "Three_Way_Match": (1.1, 0.0),
    "Resolve_Mismatch": (1.2, 0.0),
    "Pay_Invoice": (0.4, 0.0),
    "Audit": (0.7, 0.0),
    "Close_Case": (0.2, 0.0),
    "Escalate": (0.1, 60.0),
    "Expedite": (0.05, 80.0),
    "Wait_Buffer": (0.5, 0.0),
    "Handover": (0.1, 8.0),
    "Reassign": (0.2, 0.0),
    "Simplify_Scope": (0.4, -15.0),
    "Vendor_Path": (0.1, 0.0),
    "Start": (0.01, 0.0),
    "End": (0.01, 0.0),
}

# Hourly base rates by role (before era/shift/OT multipliers)
ROLE_RATE = {
    "role:analyst": 40.0,
    "role:senior": 60.0,
    "role:manager": 80.0,
    "role:auditor": 70.0,
    "role:clerk": 30.0,
    "role:vendor": 45.0,
}

# ---------------------------------------
# Helpers: calendars and working time
# ---------------------------------------
def is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5


def within(dt: datetime, start_t: time, end_t: time) -> bool:
    return start_t <= dt.time() < end_t


def next_business_open(dt0: datetime) -> datetime:
    """Align to next business open, respecting lunch gap and weekends."""
    dt = dt0
    # move to next working day if weekend
    while is_weekend(dt):
        dt = datetime(dt.year, dt.month, dt.day, 9, 0) + timedelta(days=1)
    # within day windows
    if dt.time() < BUSINESS_START:
        return datetime(dt.year, dt.month, dt.day, BUSINESS_START.hour, BUSINESS_START.minute)
    if within(dt, LUNCH_START, LUNCH_END):
        return datetime(dt.year, dt.month, dt.day, LUNCH_END.hour, LUNCH_END.minute)
    if dt.time() >= BUSINESS_END:
        # move to next day
        nxt = datetime(dt.year, dt.month, dt.day) + timedelta(days=1)
        return next_business_open(datetime(nxt.year, nxt.month, nxt.day, BUSINESS_START.hour, BUSINESS_START.minute))
    return dt


def align_to_resource_shift(dt0: datetime, shift: Tuple[time, time]) -> datetime:
    """Align a timestamp to the resource's next available shift window (ignoring lunch for simplicity)."""
    dt = dt0
    while True:
        # skip weekends entirely
        while is_weekend(dt):
            dt = datetime(dt.year, dt.month, dt.day, 9, 0) + timedelta(days=1)
        s, e = shift
        if dt.time() < s:
            dt = datetime(dt.year, dt.month, dt.day, s.hour, s.minute)
            break
        if within(dt, s, e):
            break
        if dt.time() >= e:
            # next day at shift start
            dt = datetime(dt.year, dt.month, dt.day, s.hour, s.minute) + timedelta(days=1)
            continue
    # also avoid global lunch if inside
    if within(dt, LUNCH_START, LUNCH_END):
        dt = datetime(dt.year, dt.month, dt.day, LUNCH_END.hour, LUNCH_END.minute)
    return dt


def add_working_time(dt0: datetime, hours: float, allow_overtime: bool = False) -> Tuple[datetime, float]:
    """
    Add 'hours' of processing time to dt0 respecting business hours unless allow_overtime=True.
    Returns (end_time, overtime_hours_used)
    """
    remaining = timedelta(hours=max(0.0, hours))
    dt = dt0
    overtime_used = 0.0

    if allow_overtime:
        # Overtime: we still skip weekends; nights allowed with multiplier
        while is_weekend(dt):
            dt = datetime(dt.year, dt.month, dt.day, 9, 0) + timedelta(days=1)
        end = dt + remaining
        # count overtime hours outside business window for cost accounting
        cursor = dt
        while cursor < end:
            step = min(timedelta(minutes=15), end - cursor)
            in_business = within(cursor, BUSINESS_START, BUSINESS_END) and not within(cursor, LUNCH_START, LUNCH_END)
            if not in_business:
                overtime_used += step.total_seconds() / 3600.0
            cursor += step
        return end, overtime_used

    # No overtime: respect business windows strictly
    dt = next_business_open(dt)
    while remaining > timedelta(0):
        # current day working slot segments: morning [start, lunch_start), afternoon [lunch_end, end)
        day = datetime(dt.year, dt.month, dt.day)
        slots = [
            (datetime(day.year, day.month, day.day, BUSINESS_START.hour, BUSINESS_START.minute),
             datetime(day.year, day.month, day.day, LUNCH_START.hour, LUNCH_START.minute)),
            (datetime(day.year, day.month, day.day, LUNCH_END.hour, LUNCH_END.minute),
             datetime(day.year, day.month, day.day, BUSINESS_END.hour, BUSINESS_END.minute)),
        ]
        # find next slot start >= dt
        slot_found = False
        for s, e in slots:
            if dt <= e:
                st = max(dt, s)
                if st < e:
                    capacity = e - st
                    if remaining <= capacity:
                        return st + remaining, 0.0
                    else:
                        remaining -= capacity
                        dt = e
                        slot_found = True
        if not slot_found or dt.time() >= BUSINESS_END:
            # move to next business day
            dt = next_business_open(day + timedelta(days=1))
    return dt, 0.0


def era_for(t: datetime) -> Dict:
    """Pick parameter regime by absolute time."""
    horizon = (H_END - H_START).total_seconds()
    pos = (t - H_START).total_seconds() / horizon
    for e in ERAS:
        if pos <= e["split"]:
            return e
    return ERAS[-1]


# ---------------------------------------
# NHPP-like bursty arrivals (waves)
# ---------------------------------------
def sample_arrival_times(n_cases: int) -> List[datetime]:
    """
    Construct bursty arrivals by drawing per-day Poisson intensity (weekday > weekend),
    with weekly waves and diurnal peaks (10–12 and 14–16).
    """
    days = pd.date_range(H_START.date(), H_END.date(), freq="D")
    daily_lambda = []
    for d in days:
        dow = d.weekday()
        base = 6.0 if dow < 5 else 1.2  # weekdays vs weekend
        weekly_wave = 1.0 + 0.6 * math.sin(2 * math.pi * (d.day % 28) / 28.0)
        jitter = np.random.uniform(0.8, 1.3)
        daily_lambda.append(base * weekly_wave * jitter)

    # allocate counts per day proportional to lambda
    lam = np.array(daily_lambda)
    probs = lam / lam.sum()
    counts = np.random.multinomial(n_cases, probs)

    arrivals = []
    for d, k in zip(days, counts):
        if k == 0:
            continue
        # diurnal mixture: morning/afternoon peaks
        slots = [
            (time(9, 30), time(12, 0), 0.55),
            (time(13, 30), time(16, 30), 0.45),
        ]
        for _ in range(k):
            bucket = np.random.choice(len(slots), p=np.array([s[2] for s in slots]))
            st, en, _ = slots[bucket]
            dt = datetime(d.year, d.month, d.day, st.hour, st.minute) + timedelta(
                seconds=np.random.uniform(0, (datetime(d.year, d.month, d.day, en.hour, en.minute) -
                                              datetime(d.year, d.month, d.day, st.hour, st.minute)).total_seconds())
            )
            if dt.weekday() >= 5:
                continue
            arrivals.append(dt)
    arrivals = sorted(arrivals)[:n_cases]
    while len(arrivals) < n_cases:
        arrivals.append(next_business_open(arrivals[-1] + timedelta(hours=1)))
    return arrivals[:n_cases]


# ---------------------------------------
# Resource model
# ---------------------------------------
@dataclass
class Resource:
    rid: str
    role: str
    shift: Tuple[time, time]
    next_free: datetime = field(default_factory=lambda: H_START)
    # skill multipliers by family (speed<1 faster; >1 slower)
    skill: Dict[str, float] = field(default_factory=dict)
    # history for learning/forgetting
    hist_count: Dict[str, int] = field(default_factory=dict)
    last_exec: Dict[str, datetime] = field(default_factory=dict)


class ResourceManager:
    def __init__(self):
        self.resources: Dict[str, Resource] = {}
        self._build()

    def _build(self):
        # Map R1..R30 to roles
        mapping = {}
        # analysts: R1-8
        for i in range(1, 9):
            mapping[f"R{i}"] = "role:analyst"
        # seniors: R9-14
        for i in range(9, 15):
            mapping[f"R{i}"] = "role:senior"
        # managers: R15-18
        for i in range(15, 19):
            mapping[f"R{i}"] = "role:manager"
        # auditors: R19-21
        for i in range(19, 22):
            mapping[f"R{i}"] = "role:auditor"
        # clerks: R22-25
        for i in range(22, 26):
            mapping[f"R{i}"] = "role:clerk"
        # vendors: R26-30
        for i in range(26, 31):
            mapping[f"R{i}"] = "role:vendor"

        for rid, role in mapping.items():
            shift = ROLE_SHIFT.get(role, (BUSINESS_START, BUSINESS_END))
            # Heterogeneous skills per family (draw around 1.0; lower=faster)
            skill = {fam: np.random.uniform(0.85, 1.25) for fam in set(FAMILY.values())}
            if role == "role:vendor":
                for fam in skill:
                    if fam in {"APP", "AUD"}:
                        skill[fam] *= 1.3
                    if fam == "VEN":
                        skill[fam] *= 0.9
            self.resources[rid] = Resource(rid=rid, role=role, shift=shift, skill=skill)

    def candidates(self, activity: str) -> List[Resource]:
        preferred = PREF_ROLE.get(activity, [])
        cand = [r for r in self.resources.values() if r.role in preferred]
        if not cand:
            cand = list(self.resources.values())
        return cand

    def least_loaded(self, cand: List[Resource]) -> Resource:
        return min(cand, key=lambda r: r.next_free)

    def skill_based(self, cand: List[Resource], activity: str) -> Resource:
        fam = FAMILY.get(activity, "GEN")
        return min(cand, key=lambda r: r.skill.get(fam, 1.0) * (1.0 + np.random.uniform(-0.05, 0.05)))

    def random_choice(self, cand: List[Resource]) -> Resource:
        return random.choice(cand)

    def assign(self, activity: str, policy: str) -> Resource:
        cand = self.candidates(activity)
        if policy == "least_loaded":
            return self.least_loaded(cand)
        if policy == "skill_based":
            return self.skill_based(cand, activity)
        return self.random_choice(cand)

    def update_next_free(self, rids: Iterable[str], t: datetime):
        for rid in rids:
            self.resources[rid].next_free = t


# ---------------------------------------
# Cost model
# ---------------------------------------
class CostModel:
    def __init__(self):
        self.wait_rate_per_hour = 10.0
        self.handover_fee = 7.5
        self.overtime_mult = 1.5

    def event_amount(
        self,
        activity: str,
        role: str,
        base_dur_h: float,
        era: Dict,
        overtime_hours: float = 0.0,
        material_fix: float = 0.0,
        special_penalty: float = 0.0,
    ) -> float:
        base_fixed = ACT_PARAMS.get(activity, (0.3, 0.0))[1]
        rate = ROLE_RATE.get(role, 35.0) * era["cost_rate_mult"]
        amount = base_fixed + rate * base_dur_h
        if overtime_hours > 0:
            amount += (rate * (self.overtime_mult - 1.0)) * overtime_hours
        amount += material_fix
        amount += special_penalty
        return float(round(amount, 2))

    def waiting_amount(self, wait_hours: float) -> float:
        return float(round(self.wait_rate_per_hour * wait_hours, 2))

    def handover_amount(self) -> float:
        return float(round(self.handover_fee, 2))


# ---------------------------------------
# Duration helpers (heavy-tailed)
# ---------------------------------------
def lognormal_from_mean(mean_h: float, sigma: float = 0.75) -> float:
    # choose mu so mean of lognormal ≈ mean_h
    mu = math.log(max(1e-6, mean_h)) - 0.5 * sigma * sigma
    return float(np.random.lognormal(mean=mu, sigma=sigma))


def service_time_hours(activity: str, resource: Resource, era: Dict, learning: float, forgetting: float) -> float:
    mean_h = ACT_PARAMS.get(activity, (0.5, 0.0))[0]
    fam = FAMILY.get(activity, "GEN")
    base = lognormal_from_mean(mean_h * resource.skill.get(fam, 1.0) * era["speed_mult"], sigma=0.7)
    base *= learning
    base *= forgetting
    if np.random.rand() < 0.05:  # occasional heavy tail
        base *= np.random.uniform(1.8, 3.0)
    return max(0.02, base)  # at least ~1.2 minutes


# ---------------------------------------
# Scheduling primitives
# ---------------------------------------
@dataclass
class EventSpec:
    activity: str
    start: datetime
    end: datetime
    resource: str
    amount: float


@dataclass
class CaseCtx:
    case_id: str
    start_time: datetime
    sla_hours: float
    budget_limit: float
    priority: int  # 1=normal, 2=high
    era: Dict
    cumulative_amount: float = 0.0
    last_resource: Optional[str] = None
    last_family: Optional[str] = None
    has_parallel: bool = False


class Simulator:
    def __init__(self):
        self.rm = ResourceManager()
        self.cm = CostModel()
        self.events: Dict[str, List[EventSpec]] = {}
        self.patterns: Dict[str, List[str]] = {}
        self.prefix_to_next: Dict[Tuple[str, ...], set] = {}

    # ------- utility -------
    def _maybe_unknown(self, rid: str) -> str:
        if np.random.rand() < 0.02:
            return "Unknown"
        return rid

    def _handover(self, ctx: CaseCtx, new_res: str, t_start: datetime) -> Optional[EventSpec]:
        if ctx.last_resource and ctx.last_resource != new_res:
            delay_h = np.random.uniform(5, 20) / 60.0
            handover_end, _ = add_working_time(t_start - timedelta(hours=delay_h), delay_h, allow_overtime=False)
            amt = self.cm.handover_amount()
            res = ctx.last_resource  # previous owner logs handover
            # ensure strictly < next event timestamp
            ts = handover_end - timedelta(seconds=5)
            if ts <= ctx.start_time:
                ts = ctx.start_time + timedelta(seconds=1)
            return EventSpec("Handover", ts, handover_end, self._maybe_unknown(res), amt)
        return None

    def _context_switch_overhead(self, ctx: CaseCtx, activity: str) -> float:
        fam = FAMILY.get(activity, "GEN")
        if ctx.last_resource and ctx.last_family and fam != ctx.last_family:
            return np.random.uniform(0.1, 0.33)  # 6–20 min
        return 0.0

    def _learning_forgetting(self, res: Resource, activity: str, now: datetime) -> Tuple[float, float]:
        fam = FAMILY.get(activity, "GEN")
        count = res.hist_count.get(fam, 0)
        learning = 1.0 / math.sqrt(1 + 0.25 * count)  # more executions => faster (<1)
        last = res.last_exec.get(fam)
        if last is None:
            forgetting = 1.05
        else:
            days_gap = max(0.0, (now - last).total_seconds() / 86400.0)
            forgetting = 1.0 + min(0.35, 0.03 * days_gap)
        return learning, forgetting

    def _queue_length_estimate(self, res: Resource, ready: datetime, mean_service: float) -> int:
        delta_h = max(0.0, (res.next_free - ready).total_seconds() / 3600.0)
        return int(delta_h / max(0.2, mean_service))

    def _assign_resource(self, activity: str, policy_mix: List[str]) -> Resource:
        policy = np.random.choice(policy_mix, p=[0.35, 0.40, 0.25])  # random/least_loaded/skill_based
        return self.rm.assign(activity, policy)

    def _apply_batching_release(self, activity: str, start: datetime) -> datetime:
        if activity == "Invoice_Validate":
            # release at next full hour (batching)
            rounded = datetime(start.year, start.month, start.day, start.hour) + timedelta(hours=1)
            return max(start, rounded)
        return start

    def _cooling_off(self, prev_marker: str, activity: str, start: datetime) -> Tuple[datetime, Optional[EventSpec]]:
        # If prior activity family was 'REC' (after receiving) and we schedule QC, add cooling-off lag
        if (prev_marker in {"Receive_Goods", "Record_Goods_Receipt", "REC"}) and activity == "Quality_Check":
            min_lag = timedelta(hours=np.random.uniform(1.0, 3.0))
            buf_end, _ = add_working_time(start, min_lag.total_seconds() / 3600.0, allow_overtime=False)
            ev = EventSpec("Wait_Buffer", buf_end - timedelta(minutes=1), buf_end, self._maybe_unknown("R22"), self.cm.waiting_amount(min_lag.total_seconds() / 3600.0))
            return buf_end, ev
        return start, None

    def _maybe_escalate(self, ctx: CaseCtx, now: datetime) -> Optional[EventSpec]:
        if (now - ctx.start_time).total_seconds() / 3600.0 > ctx.sla_hours and np.random.rand() < 0.8:
            amt = self.cm.event_amount("Escalate", "role:manager", ACT_PARAMS["Escalate"][0], ctx.era, 0.0, 0.0, special_penalty=40.0)
            return EventSpec("Escalate", now, now + timedelta(minutes=5), self._maybe_unknown("R15"), amt)
        return None

    def _expedite_if_needed(self, ctx: CaseCtx, res: Resource, ready: datetime) -> Tuple[bool, Optional[EventSpec], Optional[Resource]]:
        mean_service = 0.7
        qlen = self._queue_length_estimate(res, ready, mean_service)
        deadline_close = (ctx.start_time + timedelta(hours=ctx.sla_hours) - ready).total_seconds() / 3600.0 < 8.0
        do_expedite = (ctx.priority == 2 and (qlen >= 2 or deadline_close)) and np.random.rand() < 0.6
        if not do_expedite:
            return False, None, None
        ev = EventSpec("Expedite", ready, ready + timedelta(minutes=2), self._maybe_unknown("R16"), ACT_PARAMS["Expedite"][1])
        # pick a senior/manager least loaded
        cands = [r for r in self.rm.resources.values() if r.role in {"role:senior", "role:manager"}]
        if cands:
            new_res = min(cands, key=lambda r: r.next_free)
            return True, ev, new_res
        return True, ev, None

    def _binding_or_separation(self, ctx: CaseCtx, activity: str, assigned: Resource) -> Resource:
        # Binding-of-duty: same resource for Approve_L1 & Approve_L2 (if feasible)
        if activity == "Approve_L2" and ctx.last_resource and FAMILY.get("Approve_L1") == "APP":
            if np.random.rand() < 0.7:
                same = self.rm.resources.get(ctx.last_resource)
                if same and same.role in PREF_ROLE.get(activity, []):
                    return same
        # Separation-of-duty: try to enforce auditors distinct
        if activity == "Audit" and ctx.last_resource:
            if ctx.last_resource == assigned.rid:
                auds = [r for r in self.rm.resources.values() if r.role == "role:auditor" and r.rid != ctx.last_resource]
                if auds:
                    assigned = self.rm.least_loaded(auds)
        return assigned

    def _multi_resource_meeting(self, ready: datetime) -> Tuple[List[str], datetime]:
        """Three_Way_Match needs 2 resources concurrently; compute joint start."""
        cands = [r for r in self.rm.resources.values() if r.role in {"role:analyst", "role:senior"}]
        if len(cands) < 2:
            r = self.rm.least_loaded(list(self.rm.resources.values()))
            start = max(ready, align_to_resource_shift(ready, r.shift), r.next_free)
            return [r.rid], start
        c_sorted = sorted(cands, key=lambda r: r.next_free)
        r1, r2 = c_sorted[0], c_sorted[1]
        s1 = align_to_resource_shift(max(ready, r1.next_free), r1.shift)
        s2 = align_to_resource_shift(max(ready, r2.next_free), r2.shift)
        start = max(ready, s1, s2)
        return [r1.rid, r2.rid], start

    # ------- core scheduling for a single atomic activity -------
    def schedule_activity(
        self,
        ctx: CaseCtx,
        activity: str,
        ready: datetime,
        policy_mix: List[str],
        allow_overtime: bool = False,
        force_resource: Optional[Resource] = None,
    ) -> Tuple[EventSpec, datetime, Optional[EventSpec]]:
        """
        Schedule one activity; returns (event, end_time, optional_aux_event)
        """
        # multi-resource meeting special-case
        if activity == "Three_Way_Match":
            rids, start0 = self._multi_resource_meeting(ready)
            comp_res = "&".join(rids)
            res_obj = self.rm.resources[rids[0]]
            learning, forgetting = self._learning_forgetting(res_obj, activity, start0)
            dur_h = service_time_hours(activity, res_obj, ctx.era, learning, forgetting)
            dur_h += self._context_switch_overhead(ctx, activity)
            # allow overtime if deadline very close
            allow_ot = allow_overtime or ((ctx.start_time + timedelta(hours=ctx.sla_hours) - start0).total_seconds()/3600.0 < 6 and np.random.rand() < 0.5)
            end, ot_hours = add_working_time(start0, dur_h, allow_overtime=allow_ot)
            amt = self.cm.event_amount(activity, res_obj.role, dur_h, ctx.era, overtime_hours=ot_hours)
            ev = EventSpec(activity, start0, end, self._maybe_unknown(comp_res), amt)
            self.rm.update_next_free(rids, end)
            fam = FAMILY.get(activity, "GEN")
            for rid in rids:
                r = self.rm.resources[rid]
                r.hist_count[fam] = r.hist_count.get(fam, 0) + 1
                r.last_exec[fam] = end
            h = self._handover(ctx, rids[0], start0)
            ctx.last_resource = rids[0]
            ctx.last_family = FAMILY.get(activity, "GEN")
            return ev, end, h

        # single-resource path
        res = force_resource if force_resource is not None else self._assign_resource(activity, policy_mix)
        res = self._binding_or_separation(ctx, activity, res)

        # learning & forgetting
        learning, forgetting = self._learning_forgetting(res, activity, res.next_free)

        # consider expedite (may reassign)
        do_exp, ev_exp, new_res = self._expedite_if_needed(ctx, res, ready)
        if do_exp and new_res is not None:
            res = new_res

        # handover event before start (based on resource change)
        h_ev = self._handover(ctx, res.rid, ready)

        # batching/release and alignment to resource shift
        start0 = self._apply_batching_release(activity, max(ready, res.next_free))
        start0 = align_to_resource_shift(start0, res.shift)
        # align to global business hours unless overtime is (later) decided
        start1 = next_business_open(start0)

        # context-switch overhead
        dur_h = service_time_hours(activity, res, ctx.era, learning, forgetting) + self._context_switch_overhead(ctx, activity)

        # cooling-off (e.g., QC shortly after Receipt)
        start2, buf_ev = self._cooling_off(ctx.last_family or "", activity, start1)

        # allow overtime adaptively if deadline or queue is high
        allow_ot = allow_overtime or (
            (ctx.priority == 2 and ((ctx.start_time + timedelta(hours=ctx.sla_hours) - start2).total_seconds()/3600.0 < 6))
            or (self._queue_length_estimate(res, start2, max(0.3, ACT_PARAMS.get(activity, (0.5, 0))[0])) >= 3 and np.random.rand() < 0.4)
        )

        # final end timestamp
        end, ot_hours = add_working_time(start2, dur_h, allow_overtime=allow_ot)

        # explicit holding cost if waited > 3 minutes
        wait_h = max(0.0, (start2 - ready).total_seconds() / 3600.0)
        wb = None
        if wait_h > 0.05:
            wb = EventSpec("Wait_Buffer", start2 - timedelta(minutes=1), start2, self._maybe_unknown("R22"), self.cm.waiting_amount(wait_h))

        # event amount
        amt = self.cm.event_amount(activity, res.role, dur_h, ctx.era, overtime_hours=ot_hours)
        ev = EventSpec(activity, start2, end, self._maybe_unknown(res.rid), amt)

        # update resource calendars & history
        self.rm.update_next_free([res.rid], end)
        fam = FAMILY.get(activity, "GEN")
        res.hist_count[fam] = res.hist_count.get(fam, 0) + 1
        res.last_exec[fam] = end

        # update context
        ctx.last_resource = res.rid
        ctx.last_family = fam

        # pick an auxiliary event to emit near this activity (expedite, buffer, or handover)
        merged_aux = h_ev or ev_exp or buf_ev or wb
        return ev, end, merged_aux

    # ------- higher-level blocks -------
    def _rework_loop(self, ctx: CaseCtx, ready: datetime) -> Tuple[List[EventSpec], datetime]:
        out = []
        ev1, t1, h1 = self.schedule_activity(ctx, "Quality_Check", ready, ["random", "least_loaded", "skill_based"])
        out.extend([x for x in [h1, ev1] if x])
        p = min(0.75, 0.15 + 0.12 * (ev1.end - ev1.start).total_seconds() / 3600.0)  # longer QC => more rework
        iter_no = 0
        t = t1
        while np.random.rand() < p and iter_no < 3:
            ev_fix, t2, h2 = self.schedule_activity(ctx, "Rework_QC", t, ["least_loaded", "skill_based", "random"])
            out.extend([x for x in [h2, ev_fix] if x])
            ev_rev, t3, h3 = self.schedule_activity(ctx, "Quality_Check", t2, ["skill_based", "least_loaded", "random"])
            out.extend([x for x in [h3, ev_rev] if x])
            t = t3
            p *= 0.5
            iter_no += 1
        return out, t

    def _or_split_join(self, ctx: CaseCtx, ready: datetime) -> Tuple[List[EventSpec], datetime]:
        """Choose 1..3 of {Technical_Evaluation, Legal_Review, Risk_Assessment} in parallel; join at Consolidate_Evaluations."""
        opts = ["Technical_Evaluation", "Legal_Review", "Risk_Assessment"]
        k = np.random.randint(1, 4)
        chosen = list(np.random.choice(opts, size=k, replace=False))
        children = []
        ends = []
        for a in chosen:
            ev, t, h = self.schedule_activity(ctx, a, ready, ["skill_based", "least_loaded", "random"])
            children.extend([x for x in [h, ev] if x])
            ends.append(t)
        join_start = max(ends) if ends else ready
        join_start += timedelta(minutes=np.random.uniform(5, 25))  # synchronization latency
        ev_join, t_join, h_join = self.schedule_activity(ctx, "Consolidate_Evaluations", join_start, ["least_loaded", "skill_based", "random"])
        children.extend([x for x in [h_join, ev_join] if x])
        ctx.has_parallel = True
        return children, t_join

    def _and_split_join(self, ctx: CaseCtx, ready: datetime) -> Tuple[List[EventSpec], datetime]:
        """AND-split: (Invoice_Receive || QC/rework) → join to Three_Way_Match (meeting)"""
        children = []
        ev1, t1, h1 = self.schedule_activity(ctx, "Invoice_Receive", ready, ["least_loaded", "random", "skill_based"])
        children.extend([x for x in [h1, ev1] if x])
        block, t2 = self._rework_loop(ctx, ready)
        children.extend(block)
        join_time = max(t1, t2) + timedelta(minutes=np.random.uniform(10, 40))
        ev_join, t_join, h_join = self.schedule_activity(ctx, "Three_Way_Match", join_time, ["skill_based", "least_loaded", "random"])
        children.extend([x for x in [h_join, ev_join] if x])
        ctx.has_parallel = True
        return children, t_join

    def _negotiation_with_preemption(self, ctx: CaseCtx, ready: datetime) -> Tuple[List[EventSpec], datetime]:
        """Interruption & Resume: split one logical task into start/resume/complete."""
        out = []
        ev_s, t1, h1 = self.schedule_activity(ctx, "Negotiate.start", ready, ["skill_based", "least_loaded", "random"])
        out.extend([x for x in [h1, ev_s] if x])
        # preemption probability increases if queue estimate high
        res = self.rm.assign("Negotiate.resume", "least_loaded")
        qlen = self._queue_length_estimate(res, t1, 0.6)
        preempt = np.random.rand() < min(0.8, 0.2 + 0.1 * qlen)
        if preempt:
            wait_h = np.random.uniform(0.2, 1.0)
            t_resume, _ = add_working_time(t1, wait_h, allow_overtime=False)
            ev_r, t2, h2 = self.schedule_activity(ctx, "Negotiate.resume", t_resume, ["least_loaded", "skill_based", "random"])
            out.extend([x for x in [h2, ev_r] if x])
            ev_c, t3, h3 = self.schedule_activity(ctx, "Negotiate.complete", t2, ["skill_based", "least_loaded", "random"])
            out.extend([x for x in [h3, ev_c] if x])
            return out, t3
        else:
            ev_c, t2, h2 = self.schedule_activity(ctx, "Negotiate.complete", t1, ["skill_based", "least_loaded", "random"])
            out.extend([x for x in [h2, ev_c] if x])
            return out, t2

    def _vendor_confirm_k_of_n(self, ctx: CaseCtx, ready: datetime) -> Tuple[List[EventSpec], datetime]:
        """Spawn n=3 confirmations; proceed after k=2 complete; possibly cancel remaining."""
        labels = ["Vendor_Confirm_L1", "Vendor_Confirm_L2", "Vendor_Confirm_L3"]
        evs = []
        fins = []
        for lab in labels:
            ev, t, h = self.schedule_activity(ctx, lab, ready, ["least_loaded", "random", "skill_based"])
            evs.extend([x for x in [h, ev] if x])
            fins.append((t, ev))
        fins_sorted = sorted(fins, key=lambda x: x[0])
        kth_time = fins_sorted[1][0]
        if np.random.rand() < 0.6:
            remain = fins_sorted[2][1]
            cancel_ev = EventSpec(
                "Cancel_Vendor_Confirm",
                kth_time,
                kth_time + timedelta(minutes=2),
                self._maybe_unknown("R15"),
                ACT_PARAMS["Cancel_Vendor_Confirm"][1],
            )
            evs.append(cancel_ev)
        return evs, kth_time

    # ------- main per-case generation -------
    def generate_case(self, case_id: str, case_start: datetime) -> List[EventSpec]:
        ctx = CaseCtx(
            case_id=case_id,
            start_time=case_start,
            sla_hours=np.random.uniform(40, 120),  # 1–3 weeks
            budget_limit=np.random.uniform(700, 2000),
            priority=2 if np.random.rand() < 0.25 else 1,
            era=era_for(case_start),
        )
        out: List[EventSpec] = []

        # Pattern bundle selection (3–8 patterns toggled as "active emphasis")
        all_patterns = [
            "heavy_tail", "bursty", "calendar", "resource_shifts", "deadline", "setup_changeover",
            "preemption", "sync_latency", "queue_discipline", "drift", "learning", "forgetting",
            "cooling_off", "handover_delay", "context_switch", "batch_release",
            "stochastic_skip", "k_of_n", "budget_aware"
        ]
        self.patterns[case_id] = list(np.random.choice(all_patterns, size=np.random.randint(3, 9), replace=False))

        # 0) Start marker
        ev, t, h = self.schedule_activity(ctx, "Start", case_start, ["random", "least_loaded", "skill_based"])
        out.extend([x for x in [h, ev] if x])

        # 1) Requisition core (linear with optional inserts)
        ev1, t1, h1 = self.schedule_activity(ctx, "Create_Requisition", t, ["least_loaded", "random", "skill_based"])
        out.extend([x for x in [h1, ev1] if x])

        # optional Budget_Check (skip patterns)
        if np.random.rand() < 0.75:
            ev2, t2, h2 = self.schedule_activity(ctx, "Budget_Check", t1, ["skill_based", "least_loaded", "random"])
            out.extend([x for x in [h2, ev2] if x])
        else:
            t2 = t1

        ev3, t3, h3 = self.schedule_activity(ctx, "Validate_Requisition", t2, ["least_loaded", "skill_based", "random"])
        out.extend([x for x in [h3, ev3] if x])

        # approvals: L1 always; L2 sometimes
        ev4, t4, h4 = self.schedule_activity(ctx, "Approve_L1", t3, ["least_loaded", "skill_based", "random"])
        out.extend([x for x in [h4, ev4] if x])

        need_L2 = (ctx.priority == 2) or (np.random.rand() < 0.35)
        if need_L2:
            ev5, t5, h5 = self.schedule_activity(ctx, "Approve_L2", t4, ["least_loaded", "skill_based", "random"])
            out.extend([x for x in [h5, ev5] if x])
            t4 = t5

        # XOR branching after approvals based on slack/cost
        cost_so_far = sum(e.amount for e in out)
        slack_h = ctx.sla_hours - ((t4 - ctx.start_time).total_seconds() / 3600.0)
        probs = np.array([0.55, 0.20, 0.25])
        if slack_h < 24:
            probs = np.array([0.40, 0.25, 0.35])
        if cost_so_far > ctx.budget_limit:
            probs = np.array([0.30, 0.50, 0.20])
        branch = np.random.choice(["select_vendor", "simplify", "vendor_direct"], p=probs / probs.sum())

        tX = t4
        if branch == "simplify":
            ev_s, tX, h_s = self.schedule_activity(ctx, "Simplify_Scope", t4, ["manager", "least_loaded", "random"])
            out.extend([x for x in [h_s, ev_s] if x])
        elif branch == "vendor_direct":
            ev_v, tX, h_v = self.schedule_activity(ctx, "Vendor_Path", t4, ["least_loaded", "random", "skill_based"], allow_overtime=True)
            out.extend([x for x in [h_v, ev_v] if x])
        else:
            blk, tX = self._or_split_join(ctx, t4)
            out.extend(blk)
            blk2, tX = self._negotiation_with_preemption(ctx, tX)
            out.extend(blk2)

        # Create PO → Approve PO → Send PO
        for a in ["Create_PO", "Approve_PO", "Send_PO"]:
            ev_p, tX, h_p = self.schedule_activity(ctx, a, tX, ["least_loaded", "skill_based", "random"])
            out.extend([x for x in [h_p, ev_p] if x])

        # Multi-instance vendor confirmations (k-of-n)
        blk3, tX = self._vendor_confirm_k_of_n(ctx, tX)
        out.extend(blk3)

        # Receipt side
        for a in ["Receive_Goods", "Record_Goods_Receipt"]:
            ev_r, tX, h_r = self.schedule_activity(ctx, a, tX, ["random", "least_loaded", "skill_based"])
            out.extend([x for x in [h_r, ev_r] if x])

        # AND-split: (Invoice_Receive || QC/rework) → join to Three_Way_Match (meeting)
        blk4, tX = self._and_split_join(ctx, tX)
        out.extend(blk4)

        # Possible mismatch resolution depending on join latency
        if np.random.rand() < 0.35:
            ev_m, tX, h_m = self.schedule_activity(ctx, "Resolve_Mismatch", tX, ["skill_based", "least_loaded", "random"])
            out.extend([x for x in [h_m, ev_m] if x])

        # Invoice validation (batching release), then payment
        ev_iv, tX, h_iv = self.schedule_activity(ctx, "Invoice_Validate", tX, ["least_loaded", "skill_based", "random"])
        out.extend([x for x in [h_iv, ev_iv] if x])

        # SLA breach check → Escalate
        esc = self._maybe_escalate(ctx, tX)
        if esc:
            out.append(esc)
            tX = esc.end

        ev_pay, tX, h_pay = self.schedule_activity(ctx, "Pay_Invoice", tX, ["least_loaded", "random", "skill_based"])
        out.extend([x for x in [h_pay, ev_pay] if x])

        # Separation-of-duty: Audit (assignment enforces different resource from immediate predecessor where possible)
        if np.random.rand() < 0.55:
            ev_aud, tX, h_aud = self.schedule_activity(ctx, "Audit", tX, ["skill_based", "least_loaded", "random"])
            out.extend([x for x in [h_aud, ev_aud] if x])

        # Close and End
        for a in ["Close_Case", "End"]:
            ev_c, tX, h_c = self.schedule_activity(ctx, a, tX, ["least_loaded", "random", "skill_based"])
            out.extend([x for x in [h_c, ev_c] if x])

        # budget-aware late detour
        total = sum(e.amount for e in out)
        if total > ctx.budget_limit * 1.4 and np.random.rand() < 0.4:
            ev_det, tX, h_det = self.schedule_activity(ctx, "Vendor_Path", tX, ["least_loaded", "random", "skill_based"])
            out.extend([x for x in [h_det, ev_det] if x])

        # prefix → next mapping for ambiguity check
        acts = [e.activity for e in sorted(out, key=lambda x: x.end)]
        for i in range(len(acts) - 1):
            pref = tuple(acts[:i + 1])
            nxt = acts[i + 1]
            self.prefix_to_next.setdefault(pref, set()).add(nxt)

        return sorted(out, key=lambda x: x.end)

    # ------- build complete event log -------
    def build_log(self, arrivals: List[datetime]) -> EventLog:
        log = EventLog()
        for i, t0 in enumerate(arrivals[:N_TRACES]):
            case_id = f"REQ2REC-{i + 1:04d}"
            events = self.generate_case(case_id, t0)
            tr = Trace()
            tr.attributes["concept:name"] = case_id
            for ev in events:
                e = Event({
                    "concept:name": ev.activity,
                    "time:timestamp": ev.end,  # completion time
                    "org:resource": ev.resource,
                    "amount": float(ev.amount),
                })
                tr.append(e)
            # ensure chronological order
            tr = ensure_trace_sorted_by_timestamp(tr)
            log.append(tr)
        return log

    # ------- quality and sanity checks -------
    def quality_report(self, log: EventLog):
        print("\n=== QUALITY & SANITY CHECKS ===")
        # Schema check
        n_ev = 0
        missing = 0
        neg_gap = 0
        lens = []
        amounts = []
        inter_event = []
        parallel_count = 0
        era_stats = {"era_early": [], "era_mid": [], "era_late": []}
        costs_era = {"era_early": [], "era_mid": [], "era_late": []}

        for idx, tr in enumerate(log):
            tr_sorted = ensure_trace_sorted_by_timestamp(tr)
            if tr_sorted is not tr:
                log[idx] = tr_sorted
            tr = tr_sorted
            ts = [e["time:timestamp"] for e in tr]
            lens.append(len(tr))
            for j, e in enumerate(tr):
                n_ev += 1
                for key in ["concept:name", "time:timestamp", "org:resource", "amount"]:
                    if key not in e:
                        missing += 1
                try:
                    amounts.append(float(e["amount"]))
                except Exception:
                    missing += 1
                if j > 0:
                    dt = (tr[j]["time:timestamp"] - tr[j - 1]["time:timestamp"]).total_seconds()
                    if dt <= 0:
                        neg_gap += 1
                    else:
                        inter_event.append(dt / 3600.0)

            # era-wise cycle time and cost
            start_t = tr[0]["time:timestamp"]
            end_t = tr[-1]["time:timestamp"]
            era_name = era_for(start_t)["name"]
            dur_h = (end_t - start_t).total_seconds() / 3600.0
            era_stats[era_name].append(dur_h)
            costs_era[era_name].append(sum(float(e["amount"]) for e in tr))

            # parallel markers
            if any(e["concept:name"] in {"Consolidate_Evaluations", "Three_Way_Match"} for e in tr):
                parallel_count += 1

        print(f"Total events: {n_ev}, traces: {len(log)}")
        print(f"Missing required attrs: {missing}, non-positive inter-event gaps: {neg_gap}")
        print(f"Trace length – mean: {np.mean(lens):.2f}, q25/50/75: {np.percentile(lens, [25,50,75])}")
        print(f"Amount per event – mean: {np.mean(amounts):.2f}, std: {np.std(amounts):.2f}, q05/50/q95: {np.percentile(amounts, [5,50,95])}")
        if inter_event:
            q = np.percentile(inter_event, [5, 25, 50, 75, 95])
            print(f"Inter-event time (h) – mean: {np.mean(inter_event):.2f}, std: {np.std(inter_event):.2f}, q5/25/50/75/95: {q}")
        print(f"Parallelism: {parallel_count/len(log)*100:.1f}% of traces with ≥1 AND/OR-join")

        # Ambiguity check: ≥2 distinct next activities for many prefixes
        prefixes = list(self.prefix_to_next.items())
        np.random.shuffle(prefixes)
        sample = prefixes[: min(100, len(prefixes))]
        ambiguous = sum(1 for _, nxt in sample if len(nxt) >= 2)
        print(f"Ambiguity check: {ambiguous}/{len(sample)} sampled prefixes have ≥2 next activities")

        # Remaining time signal
        rem_times = []
        for tr in log[: min(80, len(log))]:
            for i, e in enumerate(tr[:-1]):
                rem = (tr[-1]["time:timestamp"] - e["time:timestamp"]).total_seconds() / 3600.0
                rem_times.append(rem)
        if rem_times:
            print(f"Remaining time (h) – mean: {np.mean(rem_times):.2f}, std: {np.std(rem_times):.2f}, q10/50/90: {np.percentile(rem_times, [10,50,90])}")

        # Era-wise drift
        for era_name in ["era_early", "era_mid", "era_late"]:
            if era_stats[era_name]:
                print(f"{era_name}: cycle mean {np.mean(era_stats[era_name]):.2f}h, cost mean {np.mean(costs_era[era_name]):.2f}")


def execute_script():
    print("Synthesizing Requisition-to-Receipt XES log (≤250 traces)...")
    sim = Simulator()
    arrivals = sample_arrival_times(N_TRACES)
    log = sim.build_log(arrivals)

    # Defensive schema validation (all required attrs present)
    for idx, tr in enumerate(log):
        tr_sorted = ensure_trace_sorted_by_timestamp(tr)
        if tr_sorted is not tr:
            log[idx] = tr_sorted
        tr = tr_sorted
        for e in tr:
            if "org:resource" not in e or not isinstance(e["org:resource"], str) or e["org:resource"] == "":
                e["org:resource"] = "Unknown"
            if "amount" not in e:
                e["amount"] = float(0.0)
            else:
                e["amount"] = float(e["amount"])

    # Export
    write_xes_safe(log, OUT_PATH)
    print(f"Exported: {OUT_PATH} with {len(log)} traces.")

    # Quality report
    sim.quality_report(log)


if __name__ == "__main__":
    execute_script()


# -------------------------------------------------------------------
# Minimal reference: one short trace hitting AND-split, handover delay,
# SLA penalty → Escalate, rework once, and vendor path with overtime.
# (The full simulator above generates such patterns stochastically.)
#
# from pm4py.objects.log.obj import Event, Trace, EventLog
# from datetime import datetime, timedelta
# L = EventLog(); t = Trace(); t.attributes['concept:name'] = 'REQ2REC-0000'
# t.append(Event({'concept:name':'Start','time:timestamp':datetime.now(),'org:resource':'R22','amount':0.0}))
# t.append(Event({'concept:name':'Create_PO','time:timestamp':datetime.now()+timedelta(hours=2),'org:resource':'R22','amount':50.0}))
# t.append(Event({'concept:name':'Handover','time:timestamp':datetime.now()+timedelta(hours=2, minutes=5),'org:resource':'R22','amount':8.0}))
# t.append(Event({'concept:name':'Invoice_Receive','time:timestamp':datetime.now()+timedelta(hours=4),'org:resource':'R23','amount':25.0}))
# t.append(Event({'concept:name':'Quality_Check','time:timestamp':datetime.now()+timedelta(hours=4, minutes=30),'org:resource':'R10','amount':60.0}))
# t.append(Event({'concept:name':'Rework_QC','time:timestamp':datetime.now()+timedelta(hours=5, minutes=40),'org:resource':'R8','amount':55.0}))
# t.append(Event({'concept:name':'Three_Way_Match','time:timestamp':datetime.now()+timedelta(hours=7),'org:resource':'R9&R10','amount':110.0}))
# t.append(Event({'concept:name':'Escalate','time:timestamp':datetime.now()+timedelta(hours=8),'org:resource':'R15','amount':100.0}))
# t.append(Event({'concept:name':'Vendor_Path','time:timestamp':datetime.now()+timedelta(hours=8, minutes=30),'org:resource':'R26','amount':45.0}))
# t.append(Event({'concept:name':'End','time:timestamp':datetime.now()+timedelta(hours=9),'org:resource':'R22','amount':0.0}))
# L.append(t)
# write_xes_safe(L, 'minimal_req2receipt.xes.gz')
# -------------------------------------------------------------------
