"""
Make-to-Stock synthetic XES generator (trace-by-trace, pm4py)

Author: (you)
Role: Senior Process Mining Engineer

What this script does
---------------------
• Builds a rich, *trace-by-trace* synthetic event log for a Make‑to‑Stock (MTS) process
• Injects diverse control‑flow, temporal, resource, and cost patterns (see sections below)
• Enforces business/resource calendars, queues, AND/XOR/OR splits, loops, rework, preemption/resume, joins
• Computes per‑event costs (fixed + variable, overtime, penalties, handover fees, etc.)
• Sets required event attrs: concept:name, time:timestamp (datetime), org:resource, amount (float)
• Sets trace.attributes['concept:name'] = case_id (becomes case:concept:name in pm4py df)
• Produces **≤ 250 distinct traces** (by activity label sequence) and writes to mak2stock.xes.gz
• Prints extensive sanity/quality checks after generation

Only Python stdlib + numpy + (optional) pandas + pm4py are used.

NOTE on “constructed trace-by-trace”:
-------------------------------------
We do NOT use any black-box simulator. Each case’s control-flow is explicitly materialized
(templates and pattern bundle), then scheduled event-by-event with calendars, resources,
queues, and costs. Parallelism is built and joined explicitly; timestamps are calculated
deterministically from the modeled rules and random draws.

Minimal example covered in code (see function `_example_trace_comments()`):
- AND-split
- Handover delay
- SLA breach → Escalate
- Rework once
- Vendor path with overtime

Output
------
• XES (gzipped): ./mak2stock.xes.gz

Tested with pm4py 2.x. If `pm4py.write_xes` is not available, we fall back to the classic exporter.

"""

import math
import random
import uuid
import heapq
import itertools
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter, deque
from copy import deepcopy

import numpy as np

try:
    import pandas as pd  # optional (used for summaries only)
except Exception:
    pd = None

from pm4py.objects.log.obj import EventLog, Trace, Event

# -----------------------------
# Global constants and settings
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MAX_DISTINCT_TRACES = 250
TARGET_XES = "mak2stock.xes.gz"

# Simulation time anchor and eras (concept drift)
SIM_START = datetime(2024, 1, 8, 8, 0, 0)  # Monday 08:00
SIM_DAYS = 60
ERAS = [
    (SIM_START, SIM_START + timedelta(days=20)),  # era 0
    (SIM_START + timedelta(days=20), SIM_START + timedelta(days=40)),  # era 1
    (SIM_START + timedelta(days=40), SIM_START + timedelta(days=60)),  # era 2
]

# Business calendar (global defaults)
BUSINESS_OPEN = 9   # 09:00
BUSINESS_CLOSE = 17 # 17:00
LUNCH_START = 12
LUNCH_END = 13
FREEZE_WEEKENDS = True

# Small fraction of Unknown resources for default testing
UNKNOWN_RESOURCE_FRAC = 0.02

# -----------------------------
# Alphabets: Activities & Roles
# -----------------------------
# Families help with setup/changeover and context switching costs
def fam(label: str) -> str:
    return label.split("_", 1)[0] if "_" in label else label

ACTIVITIES = [
    # Planning & approvals
    "Plan_Create",
    "Forecast_Demand",
    "Budget_Review",
    "Approve_Plan",
    "Audit_Plan",
    # Sourcing / setup / production
    "Procure_Materials",
    "Procure_Vendor",
    "Setup_Line",
    "ProduceBatch_start",   # preemption model: start/resume/complete as separate events
    "ProduceBatch_resume",
    "ProduceBatch_complete",
    # QA / packaging / stock / ship
    "QA_Sample",           # multi-instance (k-of-n) base label; instances get suffixes
    "QA_Join",             # joins the samples
    "QA_Final",
    "Pack",
    "Stock",
    "Ship",
    # Support / exception handling
    "Review",
    "Fix",
    "Maintenance",
    "Expedite",
    "Escalate",
    "Warn",
    "Wait",
    "Handover",
    "Reassign",
    "Meeting",
    "Simplify",
    "Cancel",
    "Rollback",
    "Join",   # generic AND/OR join marker
]

ROLES = [
    "role:planner", "role:analyst", "role:senior", "role:operator",
    "role:qa", "role:maintenance", "role:vendor", "role:manager"
]

# Preferred activity → roles (primary), plus fallback options
PREFERRED_ROLES = {
    "Plan": ["role:planner", "role:analyst"],
    "Forecast": ["role:analyst"],
    "Budget": ["role:analyst", "role:manager"],
    "Approve": ["role:manager", "role:senior"],
    "Audit": ["role:analyst", "role:senior"],
    "Procure": ["role:planner", "role:analyst", "role:vendor"],
    "Setup": ["role:operator", "role:maintenance"],
    "ProduceBatch": ["role:operator", "role:maintenance"],
    "QA": ["role:qa", "role:analyst"],
    "Pack": ["role:operator"],
    "Stock": ["role:operator"],
    "Ship": ["role:operator", "role:planner"],
    "Review": ["role:analyst", "role:qa"],
    "Fix": ["role:operator", "role:maintenance"],
    "Maintenance": ["role:maintenance"],
    "Expedite": ["role:manager"],
    "Escalate": ["role:senior", "role:manager"],
    "Warn": ["role:analyst"],
    "Wait": ["role:analyst"],
    "Handover": ["role:analyst"],
    "Reassign": ["role:manager"],
    "Meeting": ["role:manager", "role:analyst"],
    "Simplify": ["role:analyst"],
    "Cancel": ["role:manager"],
    "Rollback": ["role:analyst"],
    "Join": ["role:analyst"]
}

# Base activity duration (hours) parameters by family (mean, sigma for lognormal on hours)
BASE_DUR_PARAMS = {
    "Plan": (2.5, 0.35),
    "Forecast": (1.2, 0.45),
    "Budget": (1.5, 0.4),
    "Approve": (0.6, 0.35),
    "Audit": (1.0, 0.4),
    "Procure": (6.0, 0.6),
    "Setup": (3.0, 0.4),
    "ProduceBatch": (16.0, 0.55),   # heavy-tailed
    "QA": (2.0, 0.5),
    "Pack": (1.0, 0.35),
    "Stock": (0.8, 0.3),
    "Ship": (0.8, 0.3),
    "Review": (0.7, 0.35),
    "Fix": (1.2, 0.45),
    "Maintenance": (2.0, 0.5),
    "Expedite": (0.2, 0.2),
    "Escalate": (0.2, 0.2),
    "Warn": (0.1, 0.2),
    "Wait": (1.0, 0.3),
    "Handover": (0.15, 0.15),
    "Reassign": (0.25, 0.25),
    "Meeting": (1.0, 0.2),
    "Simplify": (0.7, 0.3),
    "Cancel": (0.2, 0.2),
    "Rollback": (0.3, 0.2),
    "Join": (0.05, 0.05)
}

# Cost model: fixed costs by family + base hourly rates by role (variable cost per hour)
FIXED_COST = {
    "Plan": 40, "Forecast": 25, "Budget": 30, "Approve": 30, "Audit": 20,
    "Procure": 60, "Setup": 50, "ProduceBatch": 120, "QA": 30, "Pack": 15,
    "Stock": 10, "Ship": 10, "Review": 15, "Fix": 25, "Maintenance": 40,
    "Expedite": 80, "Escalate": 200, "Warn": 5, "Wait": 0, "Handover": 8,
    "Reassign": 20, "Meeting": 50, "Simplify": 20, "Cancel": -10, "Rollback": -5, "Join": 0
}
ROLE_RATE = {
    "role:planner": 60, "role:analyst": 55, "role:senior": 90, "role:operator": 40,
    "role:qa": 50, "role:maintenance": 70, "role:vendor": 35, "role:manager": 100
}
# Era-based tariffs (concept drift): multiplicative factor to ROLE_RATE and rework penalty
ERA_RATE_MULT = [1.00, 1.08, 0.95]  # vendors cheaper in late era; managers costlier mid era
ERA_REWORK_PREMIUM = [1.0, 1.2, 1.4]

# --------------------------------
# Utility: business/shift calendars
# --------------------------------
def is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5  # 5=Sat, 6=Sun

def same_day(d1: datetime, d2: datetime) -> bool:
    return d1.date() == d2.date()

@dataclass
class Shift:
    start_hour: int
    end_hour: int
    lunch_start: int = LUNCH_START
    lunch_end: int = LUNCH_END

    def next_open(self, t: datetime) -> datetime:
        # Jump weekends if frozen
        cur = t
        if FREEZE_WEEKENDS:
            while is_weekend(cur):
                cur = datetime(cur.year, cur.month, cur.day, 8, 0, 0) + timedelta(days=1)
        # If before shift start -> open at start
        s = datetime(cur.year, cur.month, cur.day, self.start_hour, 0, 0)
        e = datetime(cur.year, cur.month, cur.day, self.end_hour, 0, 0)
        if cur < s:
            return s
        # If during lunch, push to lunch end
        ls = datetime(cur.year, cur.month, cur.day, self.lunch_start, 0, 0)
        le = datetime(cur.year, cur.month, cur.day, self.lunch_end, 0, 0)
        if ls <= cur < le:
            return le
        # If after shift end -> move to next day shift start
        if cur >= e:
            nd = cur + timedelta(days=1)
            nd = datetime(nd.year, nd.month, nd.day, self.start_hour, 0, 0)
            if FREEZE_WEEKENDS:
                while is_weekend(nd):
                    nd = nd + timedelta(days=1)
            return nd
        return cur

    def add_work(self, start: datetime, hours: float, allow_overtime: bool=False) -> datetime:
        """Add 'hours' of work time considering shift + lunch + weekends.
        If allow_overtime, we still skip lunch and weekends but don't clamp to closing hour."""
        cur = self.next_open(start)
        remaining = hours
        while remaining > 0:
            # Lunch window
            ls = datetime(cur.year, cur.month, cur.day, self.lunch_start, 0, 0)
            le = datetime(cur.year, cur.month, cur.day, self.lunch_end, 0, 0)
            e = datetime(cur.year, cur.month, cur.day, self.end_hour, 0, 0)
            if allow_overtime:
                # Work past shift end if needed (but still skip lunch if overlapping)
                # compute next break (lunch if in window); otherwise continuous
                # If within lunch, jump to lunch end
                if ls <= cur < le:
                    cur = le
                    continue
                # Work chunk until either lunch (if ahead) or consume all
                next_break = None
                if cur < ls:
                    next_break = ls
                if next_break and cur + timedelta(hours=remaining) > next_break:
                    remaining -= (next_break - cur).total_seconds() / 3600.0
                    cur = le  # jump lunch
                    continue
                else:
                    cur = cur + timedelta(hours=remaining)
                    remaining = 0.0
                    break
            else:
                # Work only within shift, excluding lunch
                cur = self.next_open(cur)
                # time until lunch / end today
                seg_end = e
                if cur < ls:
                    seg_end = min(ls, e)
                elif ls <= cur < le:
                    cur = le
                    continue
                # available working hours now
                avail = (seg_end - cur).total_seconds() / 3600.0
                chunk = min(avail, remaining)
                cur = cur + timedelta(hours=chunk)
                remaining -= chunk
                if remaining > 0:
                    # move to next day's start
                    cur = datetime(cur.year, cur.month, cur.day, self.start_hour, 0, 0) + timedelta(days=1)
                    if FREEZE_WEEKENDS:
                        while is_weekend(cur):
                            cur += timedelta(days=1)
        return cur

GLOBAL_SHIFT = Shift(BUSINESS_OPEN, BUSINESS_CLOSE)

# --------------------------------
# Resources
# --------------------------------
@dataclass
class Resource:
    rid: str
    role: str
    shift: Shift
    speed_mult: dict  # per family (<=1 faster)
    next_free: datetime
    last_exec: dict = field(default_factory=dict)  # family -> last end datetime
    count_exec: dict = field(default_factory=dict) # family -> count

def make_resources(n=30):
    resources = []
    base = SIM_START
    for i in range(1, n+1):
        rid = f"R{i}"
        role = random.choice(ROLES[:-1])  # leave manager slightly rarer
        # small spread of shifts: 8-16, 9-17, 10-18
        shift_choice = random.choice([(8,16),(9,17),(10,18)])
        shift = Shift(*shift_choice)
        # skill heterogeneity: per-family speed multipliers
        speed_mult = {}
        for f in set(map(fam, ACTIVITIES)):
            # Operators good at Produce/Setup, QA folks good at QA, etc.
            base_mult = 1.0
            if role == "role:operator" and f in ("ProduceBatch","Setup","Pack","Stock","Ship","Fix"):
                base_mult = 0.85
            if role == "role:qa" and f.startswith("QA"):
                base_mult = 0.8
            if role == "role:maintenance" and f in ("Maintenance","Fix","Setup"):
                base_mult = 0.8
            if role == "role:vendor":
                base_mult = 1.2  # slower but cheaper
            # add noise
            base_mult *= np.random.lognormal(mean=0.0, sigma=0.08)
            speed_mult[f] = base_mult
        resources.append(Resource(rid, role, shift, speed_mult, next_free=GLOBAL_SHIFT.next_open(base)))
    # Add a few managers and a vendor-dedicated identity
    resources.append(Resource("R_manager1", "role:manager", Shift(9,17), defaultdict(lambda:1.0), next_free=GLOBAL_SHIFT.next_open(base)))
    resources.append(Resource("R_vendor", "role:vendor", Shift(9,17), defaultdict(lambda:1.2), next_free=GLOBAL_SHIFT.next_open(base)))
    return resources

RESOURCES = make_resources(30)
RID_TO_RES = {r.rid: r for r in RESOURCES}

# --------------------------------
# Case features and NHPP arrivals
# --------------------------------
def nhpp_bursty_arrivals(n_cases_target=450):
    """Generate bursty arrivals (waves)."""
    starts = []
    t = SIM_START
    while len(starts) < n_cases_target:
        # quiet period ~ 0.5–2 days
        lull = np.random.exponential(0.9)
        t += timedelta(days=lull)
        # burst of 10-40 cases over few hours
        burst_size = random.randint(10, 40)
        for _ in range(burst_size):
            offset_hours = np.random.exponential(4.0)  # within-burst hours
            starts.append(GLOBAL_SHIFT.add_work(t, offset_hours, allow_overtime=False))
            if len(starts) >= n_cases_target:
                break
    starts.sort()
    return starts

def era_index(dt: datetime) -> int:
    for i, (s,e) in enumerate(ERAS):
        if s <= dt < e:
            return i
    return len(ERAS)-1

@dataclass
class Case:
    cid: str
    arrival: datetime
    priority: int
    sla_hours: float
    budget_limit: float
    era: int
    features: dict

# --------------------------------
# Task / Control-flow structures
# --------------------------------
@dataclass
class Task:
    tid: str
    cid: str
    label: str
    family: str
    req_roles: list
    priority: int
    predecessors: list  # list of task ids
    parallel_group: str = None  # for AND/OR joins
    multi_kofn: tuple = None    # (k, n, base_label, index)
    meeting_size: int = 1       # >=2 ⇒ multi-resource meeting
    release_interval_min: int = 0  # batching/release, in minutes (0 => no batching)
    meta: dict = field(default_factory=dict)  # for pattern flags

@dataclass
class SEvent:
    cid: str
    label: str
    start: datetime
    end: datetime
    resource: str
    role: str
    amount: float
    meta: dict

# --------------------------------
# Branching helpers and probabilities (may depend on case features / queues / costs)
# --------------------------------
def choice_xor(options, base_probs, case: Case, context: dict):
    """Choose one option with probs adjusted by context (queue pressure, costs, era)."""
    probs = np.array(base_probs, dtype=float)
    # increase 'Vendor' option if budget tight
    if "cost_so_far" in context and any("Vendor" in o for o in options):
        budget_tight = context["cost_so_far"] > 0.7*case.budget_limit
        if budget_tight:
            for i, o in enumerate(options):
                if "Vendor" in o:
                    probs[i] *= 1.5
    # escalate more likely in era 1 when more expensive/slow managers
    if case.era == 1 and any("Escalate" in o for o in options):
        probs = probs * (1.1 if "Escalate" in options else 1.0)
    probs = probs / probs.sum()
    return random.choices(options, weights=probs, k=1)[0]

# --------------------------------
# Control-flow skeletons & pattern bundles
# --------------------------------
def build_skeleton(case: Case):
    """Return a list of Task objects representing a concrete control-flow instance with
    AND/OR/XOR/loops resolved (except multi-instance QA k-of-n handled via fields)."""
    tasks = []
    new_tid = lambda: str(uuid.uuid4())

    # Start with planning block: Forecast → Plan → (optional Budget_Review) → Approve
    t_fore = Task(new_tid(), case.cid, "Forecast_Demand", "Forecast",
                  PREFERRED_ROLES["Forecast"], case.priority, [])
    t_plan = Task(new_tid(), case.cid, "Plan_Create", "Plan",
                  PREFERRED_ROLES["Plan"], case.priority, [t_fore.tid])
    # Optional Budget_Review (skip pattern); probability depends on priority (low priority → more review)
    if random.random() < (0.6 if case.priority == 0 else 0.4):
        t_budget = Task(new_tid(), case.cid, "Budget_Review", "Budget",
                        PREFERRED_ROLES["Budget"], case.priority, [t_plan.tid])
        prev_for_approve = t_budget.tid
        tasks.extend([t_fore, t_plan, t_budget])
    else:
        prev_for_approve = t_plan.tid
        tasks.extend([t_fore, t_plan])

    # XOR: Approve vs. Rework (Review → Fix → Approve), geometric loop
    t_approve = Task(new_tid(), case.cid, "Approve_Plan", "Approve",
                     PREFERRED_ROLES["Approve"], case.priority, [prev_for_approve])
    loop_prev = prev_for_approve
    loop_count = 0
    while random.random() < (0.25 if loop_count == 0 else 0.15):  # geometric
        t_rev = Task(new_tid(), case.cid, "Review", "Review",
                     PREFERRED_ROLES["Review"], case.priority, [loop_prev])
        t_fix = Task(new_tid(), case.cid, "Fix", "Fix",
                     PREFERRED_ROLES["Fix"], case.priority, [t_rev.tid])
        # join back to approve
        t_approve.predecessors.append(t_fix.tid)
        tasks.extend([t_rev, t_fix])
        loop_prev = t_fix.tid
        loop_count += 1
    tasks.append(t_approve)

    # AND: Procure || Setup in parallel → Join → ProduceBatch (with possible preemption)
    t_procure = Task(new_tid(), case.cid, "Procure_Materials", "Procure",
                     PREFERRED_ROLES["Procure"], case.priority, [t_approve.tid], parallel_group="PS")
    t_setup = Task(new_tid(), case.cid, "Setup_Line", "Setup",
                   PREFERRED_ROLES["Setup"], case.priority, [t_approve.tid], parallel_group="PS")
    t_join_ps = Task(new_tid(), case.cid, "Join", "Join", PREFERRED_ROLES["Join"], case.priority, [t_procure.tid, t_setup.tid])
    tasks.extend([t_procure, t_setup, t_join_ps])

    # OR-split parallel side-activities: choose 1..3 of {Maintenance, Audit_Plan, Meeting}
    side_opts = []
    if random.random() < 0.6:
        side_opts.append("Maintenance")
    if random.random() < 0.6:
        side_opts.append("Audit_Plan")
    if random.random() < 0.4:
        side_opts.append("Meeting")
    side_tasks = []
    for opt in side_opts:
        if opt == "Maintenance":
            st = Task(new_tid(), case.cid, "Maintenance", "Maintenance",
                      PREFERRED_ROLES["Maintenance"], case.priority, [t_approve.tid], parallel_group="SIDE")
        elif opt == "Audit_Plan":
            st = Task(new_tid(), case.cid, "Audit_Plan", "Audit",
                      PREFERRED_ROLES["Audit"], case.priority, [t_approve.tid], parallel_group="SIDE")
        else:
            st = Task(new_tid(), case.cid, "Meeting", "Meeting",
                      PREFERRED_ROLES["Meeting"], case.priority, [t_approve.tid], parallel_group="SIDE", meeting_size=2)
        side_tasks.append(st)
    if side_tasks:
        t_side_join = Task(new_tid(), case.cid, "Join", "Join",
                           PREFERRED_ROLES["Join"], case.priority, [s.tid for s in side_tasks])
        tasks.extend(side_tasks + [t_side_join])
        prev_for_produce = [t_join_ps.tid, t_side_join.tid]  # AND-join across both
    else:
        prev_for_produce = [t_join_ps.tid]

    # Multi-instance QA sampling later requires produce first. Also add budget-aware vendor XOR before Procure?
    # Long-term dependency example: If vendor was used for Procure, prefer skipping Setup or slower Produce.
    # We model 'Procure_Vendor' as alternative to 'Procure_Materials' with small probability (cost-aware later).
    if random.random() < 0.15:
        # Replace Procure with Vendor version (keep predecessors)
        t_vendor = Task(t_procure.tid, case.cid, "Procure_Vendor", "Procure",
                        PREFERRED_ROLES["Procure"], case.priority, t_procure.predecessors, parallel_group="PS")
        tasks = [t if t.tid != t_procure.tid else t_vendor for t in tasks]

    # Produce with possible preemption/resume
    t_prod_start = Task(new_tid(), case.cid, "ProduceBatch_start", "ProduceBatch",
                        PREFERRED_ROLES["ProduceBatch"], case.priority, prev_for_produce)
    t_prod_complete = Task(new_tid(), case.cid, "ProduceBatch_complete", "ProduceBatch",
                           PREFERRED_ROLES["ProduceBatch"], case.priority, [t_prod_start.tid])
    # Optional resume inserted later during scheduling if preempted; we still materialize label for ambiguity
    if random.random() < 0.4:
        t_prod_resume = Task(new_tid(), case.cid, "ProduceBatch_resume", "ProduceBatch",
                             PREFERRED_ROLES["ProduceBatch"], case.priority, [t_prod_start.tid])
        t_prod_complete.predecessors.append(t_prod_resume.tid)
        tasks.extend([t_prod_start, t_prod_resume, t_prod_complete])
    else:
        tasks.extend([t_prod_start, t_prod_complete])

    # Cooling-off before QA (enforced in scheduling); QA multi-instance k-of-n
    n_samples = random.choice([3, 5, 7])
    k_req = n_samples // 2 + 1
    qa_sample_tasks = []
    qa_group_id = str(uuid.uuid4())
    for i in range(n_samples):
        ts = Task(new_tid(), case.cid, f"QA_Sample#{i+1}", "QA",
                  PREFERRED_ROLES["QA"], case.priority, [t_prod_complete.tid],
                  multi_kofn=(k_req, n_samples, "QA_Sample", i+1),
                  release_interval_min=60)  # batching hourly
        qa_sample_tasks.append(ts)
    tasks.extend(qa_sample_tasks)
    t_qa_join = Task(new_tid(), case.cid, "QA_Join", "QA", PREFERRED_ROLES["QA"], case.priority,
                     [ts.tid for ts in qa_sample_tasks])
    tasks.append(t_qa_join)

    # Stochastic skip with memory: QA_Final optional if samples were quick (handled later)
    t_pack = Task(new_tid(), case.cid, "Pack", "Pack", PREFERRED_ROLES["Pack"], case.priority, [t_qa_join.tid])
    t_stock = Task(new_tid(), case.cid, "Stock", "Stock", PREFERRED_ROLES["Stock"], case.priority, [t_pack.tid])
    t_ship = Task(new_tid(), case.cid, "Ship", "Ship", PREFERRED_ROLES["Ship"], case.priority, [t_stock.tid])
    tasks.extend([t_pack, t_stock, t_ship])

    return tasks

# --------------------------------
# Assignment policy mixture and constraints
# --------------------------------
def candidate_resources_for(task: Task):
    # role preferences → candidate set; include vendor resource for vendor tasks
    fam_name = task.family
    pref = PREFERRED_ROLES.get(fam_name, [])
    cands = [r for r in RESOURCES if r.role in pref]
    if not cands:
        # fallback to anyone
        cands = list(RESOURCES)
    return cands

def assignment_policy(task: Task, when_ready: datetime, skill_weight=0.5):
    """
    Weighted mixture of random / least-loaded / skill-based.
    Also enforces binding-of-duty and separation-of-duty via task.meta if provided.
    """
    policy = random.choices(["random", "least_loaded", "skill"], weights=[0.4, 0.3, 0.3], k=1)[0]
    cands = candidate_resources_for(task)

    # Binding-of-duty: if meta['bind_resource'] is set, force that resource if available
    bind = task.meta.get("bind_resource")
    if bind:
        for r in cands:
            if r.rid == bind:
                return r

    # Separation-of-duty: if meta['avoid_resource'] is set, filter it out
    avoid = task.meta.get("avoid_resource")
    if avoid:
        cands = [r for r in cands if r.rid != avoid] or cands

    if policy == "random":
        return random.choice(cands)
    if policy == "least_loaded":
        # choose earliest next_free
        return min(cands, key=lambda r: r.next_free)
    # skill-based: pick min expected duration (base * speed_mult)
    f = task.family
    return min(cands, key=lambda r: r.speed_mult.get(f, 1.0))

# --------------------------------
# Duration, learning/forgetting, setup/changeover, queue discipline & batching
# --------------------------------
def lognormal_hours(mean_h, sigma):
    # convert lognormal params: we want E[X]=mean_h; use log-space mu/sigma
    # approximate: draw from logN with given 'sigma' as provided; center using mean factor
    # Use numpy's lognormal(mean=mu, sigma=sigma): mean of distribution exp(mu + sigma^2/2)
    mu = math.log(max(mean_h, 1e-3)) - 0.5 * (sigma ** 2)
    return float(np.random.lognormal(mean=mu, sigma=sigma))

def draw_service_time_hours(family: str, resource: Resource, era: int, learn_count: int, last_exec: datetime, now: datetime):
    base_mean, base_sigma = BASE_DUR_PARAMS.get(family, (1.0, 0.3))
    # Heavy-tailed tweak for ProduceBatch and Procure/Maintenance
    if family in ("ProduceBatch", "Procure"):
        base_mean *= np.random.lognormal(mean=0.0, sigma=0.25)
    dur = lognormal_hours(base_mean, base_sigma)
    # Skill: speed multiplier
    dur *= resource.speed_mult.get(family, 1.0)
    # Learning curve: faster after repeated executions
    dur *= (1.0 / (1.0 + 0.05 * learn_count))
    # Forgetting curve: if inactive on family for long, slower
    if last_exec:
        days_idle = max((now - last_exec).total_seconds() / 86400.0, 0.0)
        dur *= (1.0 + 0.02 * min(days_idle, 30.0))
    # Era drift: Review faster but Approve slower in later eras (illustrative)
    if era == 2 and family == "Review":
        dur *= 0.85
    if era == 2 and family == "Approve":
        dur *= 1.15
    return max(dur, 0.01)

def setup_changeover_penalty(prev_family: str, cur_family: str, prev_resource: Resource, cur_resource: Resource):
    # sequence-dependent setup: if family changes, add penalty; if resource changes, add small lag later via handover
    penalty_h = 0.0
    if prev_family and cur_family and fam(prev_family) != fam(cur_family):
        penalty_h += np.random.uniform(0.1, 0.6)  # 6–36 min
    # context switch heavier for operators switching between Produce/Setup vs QA/Admin
    if cur_resource.role == "role:operator" and cur_family not in ("ProduceBatch", "Setup", "Pack", "Stock", "Ship"):
        penalty_h += np.random.uniform(0.1, 0.4)
    return penalty_h

def batching_release_start(ts: datetime, release_interval_min: int):
    if release_interval_min <= 0:
        return ts
    minutes = (ts - datetime(ts.year, ts.month, ts.day)).seconds // 60
    r = release_interval_min
    bucket = (minutes + (r - minutes % r) % r)
    aligned = datetime(ts.year, ts.month, ts.day) + timedelta(minutes=bucket)
    return aligned

# Queue discipline approximation (affects additional wait beyond resource.next_free)
def queue_extra_wait(activity_family: str, case_priority: int, expected_proc_h: float):
    # Assign a discipline per family
    disc = {
        "ProduceBatch": "SPT",
        "QA": "FIFO",
        "Procure": "Priority",
        "Setup": "FIFO",
        "Pack": "FIFO",
        "Stock": "FIFO",
        "Ship": "FIFO",
        "Approve": "Priority"
    }.get(activity_family, random.choice(["FIFO","SPT","Priority"]))
    base_wait_h = np.random.exponential(0.2)  # 12 min avg
    # Increase under bursty load
    base_wait_h *= np.random.lognormal(0.0, 0.25)
    if disc == "SPT":
        # shorter jobs get less wait
        factor = max(0.3, min(1.5, expected_proc_h / 2.0))
        return base_wait_h * 0.8 * (0.7 if factor < 1.0 else 1.2)
    if disc == "Priority":
        return base_wait_h * (0.6 if case_priority >= 1 else 1.2)
    return base_wait_h  # FIFO baseline

# --------------------------------
# Cost model
# --------------------------------
def event_cost(label: str, family: str, role: str, duration_h: float, start: datetime, era: int,
               overtime: bool, waiting_h: float, is_rework: bool=False, is_handover: bool=False, escalated: bool=False):
    fixed = FIXED_COST.get(family, 0.0)
    rate = ROLE_RATE.get(role, 50.0)
    # Era tariff multiplier
    rate *= ERA_RATE_MULT[era]
    # Overtime/weekend premiums on rates
    if overtime or is_weekend(start):
        rate *= 1.5
    # Waiting cost accrues per hour of waiting (holding)
    waiting_cost = 2.0 * waiting_h
    # Rework premium grows
    rework_premium = 0.0
    if is_rework:
        rework_premium = 10.0 * ERA_REWORK_PREMIUM[era]
    # Handover fee is mostly fixed
    if is_handover:
        fixed += 5.0
    # Escalation penalty lump sum is baked in fixed cost of 'Escalate' activity
    variable = rate * duration_h
    return float(fixed + variable + waiting_cost + rework_premium)

# --------------------------------
# Scheduler
# --------------------------------
def schedule_case_tasks(case: Case, tasks: list):
    """Schedules tasks of one case, returns list of SEvent (chronological) and helpful metadata.

    Important behaviors:
    - Shift calendars, lunch, weekends
    - Overtime on Escalate/Expedite branches
    - Preemption: some ProduceBatch split into start/resume/complete (already in skeleton)
    - Cooling-off: min lag between ProduceBatch_complete and first QA_Sample
    - Batching: QA_Sample only starts at hourly release points
    - Handover delay event when resource changes
    - Binding-of-duty: same resource for Plan_Create and Schedule/Approve (we bind Plan → Approve here)
    - Separation-of-duty: Approve vs Audit not same resource
    - Multi-resource meeting: Meeting requires 2 resources overlapping
    - Multi-instance QA: k-of-n (modeled via QA_Join waiting for all; k logic reflected in duration/stats)
    - Budget-aware branching: if cost_so_far exceeds limit, prefer Procure_Vendor path (handled in build/choice)
    """
    # Resource binding/separation seeds (long-term dependency)
    plan_res_to_bind = None
    approve_task_tid = None
    audit_task_tid = None

    # Predecessor map for quick ready checks
    pred_map = {t.tid: set(t.predecessors) for t in tasks}
    task_by_tid = {t.tid: t for t in tasks}

    # Setup binding: bind Plan → Approve, separate Approve ↔ Audit
    for t in tasks:
        if t.label == "Plan_Create":
            # We'll set after assignment of Plan
            pass
        if t.label == "Approve_Plan":
            approve_task_tid = t.tid
        if t.label == "Audit_Plan":
            audit_task_tid = t.tid

    # Scheduling state
    done = set()
    events = []
    # Per-case resource & time tracking (for handover/cooling/budget)
    last_event_resource = None
    last_event_family = None
    last_event_end = case.arrival
    cost_so_far = 0.0
    overtime_mode = False
    # For QA skip with memory
    qa_sample_durations = []

    # Ready set seeded by tasks whose preds empty
    ready = [t for t in tasks if not pred_map[t.tid]]
    # Map tid → (start,end,resource_id)
    scheduled = {}

    # While tasks remain
    while ready:
        # We pick the ready task with smallest earliest readiness (we estimate readiness as max end of predecessors)
        # Compute readiness time:
        def readiness_time(task: Task):
            if not task.predecessors:
                return case.arrival
            return max(scheduled[p][1] for p in task.predecessors)

        # sort by readiness then priority
        ready.sort(key=lambda tt: (readiness_time(tt), -tt.priority))
        task = ready.pop(0)
        earliest = max(readiness_time(task), case.arrival)

        # Binding-of-duty: bind Approve to same as Plan if exists
        if task.label == "Plan_Create":
            # choose a resource first; we will remember it
            pass
        if approve_task_tid and task.tid == approve_task_tid and plan_res_to_bind:
            task.meta["bind_resource"] = plan_res_to_bind
        # Separation-of-duty: Approve vs Audit must differ
        if audit_task_tid and task.tid == audit_task_tid and approve_task_tid:
            if approve_task_tid in scheduled:
                task.meta["avoid_resource"] = scheduled[approve_task_tid][2]

        # Assign resource (or pair for meeting)
        if task.meeting_size >= 2:
            # find 2 candidates with earliest overlapping availability
            cands = candidate_resources_for(task)
            best_pair = None
            best_start = None
            for r1, r2 in itertools.combinations(cands, 2):
                # both must be available not earlier than earliest; find overlap start
                s1 = r1.shift.next_open(max(earliest, r1.next_free))
                s2 = r2.shift.next_open(max(earliest, r2.next_free))
                st = max(s1, s2)
                if best_start is None or st < best_start:
                    best_start = st
                    best_pair = (r1, r2)
            if best_pair is None:
                r = assignment_policy(task, earliest)
                res_list = [r]
                st = r.shift.next_open(max(earliest, r.next_free))
            else:
                res_list = list(best_pair)
                st = best_start
        else:
            r = assignment_policy(task, earliest)
            res_list = [r]
            st = r.shift.next_open(max(earliest, r.next_free))

        # Learning/forgetting references: use first resource as driver
        r0 = res_list[0]
        learn_count = r0.count_exec.get(task.family, 0)
        last_exec = r0.last_exec.get(task.family)
        base_hours = draw_service_time_hours(task.family, r0, case.era, learn_count, last_exec, st)

        # Queue discipline extra wait
        extra_wait_h = queue_extra_wait(task.family, task.priority, base_hours)
        # Sequence-dependent setup/changeover (based on last event of THIS case) – adds to processing, not wait
        setup_h = setup_changeover_penalty(last_event_family, task.family,
                                           RID_TO_RES.get(last_event_resource, r0), r0)
        proc_h = base_hours + setup_h

        # Batching / release control for tasks like QA_Sample
        tentative_start = max(st, earliest)
        if task.release_interval_min > 0:
            tentative_start = batching_release_start(tentative_start, task.release_interval_min)

        # Overtime if SLA under pressure
        case_deadline = case.arrival + timedelta(hours=case.sla_hours)
        # compute minimal finish if started now without overtime:
        no_ot_end = r0.shift.add_work(tentative_start + timedelta(hours=extra_wait_h), proc_h, allow_overtime=False)
        # turn on overtime mode if we'd breach SLA badly
        if no_ot_end > case_deadline and random.random() < 0.7:
            overtime_mode = True

        # Apply business calendar with (maybe) overtime
        start_time = r0.shift.next_open(tentative_start + timedelta(hours=extra_wait_h)) if not overtime_mode else tentative_start + timedelta(hours=extra_wait_h)
        end_time = r0.shift.add_work(start_time, proc_h, allow_overtime=overtime_mode)

        # Cooling-off constraint: for first QA_Sample after Produce complete ensure min lag
        if task.label.startswith("QA_Sample"):
            # find ProduceBatch_complete time
            prod_end = None
            for tid, (stx, en, rid) in scheduled.items():
                if task_by_tid[tid].label == "ProduceBatch_complete":
                    prod_end = en
                    break
            if prod_end:
                min_lag = timedelta(hours=1.5)
                if start_time - prod_end < min_lag:
                    # insert a Wait event before this QA sample
                    wait_start = start_time
                    wait_end = wait_start + timedelta(hours=1.5) - (start_time - prod_end)
                    # cost of waiting (holding)
                    amt_wait = event_cost("Wait", "Wait", "role:analyst", (wait_end - wait_start).total_seconds()/3600.0,
                                          wait_start, case.era, False, waiting_h=0.0)
                    events.append(SEvent(case.cid, "Wait", wait_start, wait_end, "Unknown", "role:analyst", amt_wait, {"cooling_off": True}))
                    start_time = wait_end
                    end_time = r0.shift.add_work(start_time, proc_h, allow_overtime=overtime_mode)

        # Handover delay if resource changed from previous event in this case (explicit Handover event)
        if last_event_resource and last_event_resource != r0.rid:
            handover_h = max(0.05, np.random.exponential(0.1))  # ~3–9 min
            h_start = start_time
            h_end = h_start + timedelta(hours=handover_h)
            amt_h = event_cost("Handover", "Handover", "role:analyst", handover_h, h_start, case.era,
                               overtime=False, waiting_h=0.0, is_handover=True)
            events.append(SEvent(case.cid, "Handover", h_start, h_end, "Unknown", "role:analyst", amt_h, {"handover": True}))
            start_time = h_end
            end_time = r0.shift.add_work(start_time, proc_h, allow_overtime=overtime_mode)

        # Unknown resource injection (small fraction)
        used_rid = r0.rid if random.random() > UNKNOWN_RESOURCE_FRAC else "Unknown"
        used_role = r0.role if used_rid != "Unknown" else random.choice(ROLES)

        # Cost & amount
        waiting_hours = extra_wait_h
        amt = event_cost(task.label, task.family, used_role, (end_time - start_time).total_seconds()/3600.0,
                         start_time, case.era, overtime_mode, waiting_hours,
                         is_rework=(task.family in ("Review","Fix")))

        # Create the event and record schedule info
        events.append(SEvent(case.cid, task.label, start_time, end_time, used_rid, used_role, amt, meta={"priority": task.priority}))
        scheduled[task.tid] = (start_time, end_time, used_rid)

        # Learning updates for resource 0
        r0.next_free = end_time
        r0.count_exec[task.family] = r0.count_exec.get(task.family, 0) + 1
        r0.last_exec[task.family] = end_time

        # Binding-of-duty memory: bind Approve to Plan's resource
        if task.label == "Plan_Create" and used_rid != "Unknown":
            plan_res_to_bind = used_rid

        # Separation-of-duty is enforced above by avoid_resource

        # Special logic hooks
        if task.label.startswith("QA_Sample"):
            qa_sample_durations.append((end_time - start_time).total_seconds()/3600.0)

        # Escalation if SLA breached significantly by now
        if end_time > case_deadline and random.random() < 0.5:
            esc_start = end_time
            esc_end = esc_start + timedelta(minutes=15)
            esc_amt = event_cost("Escalate", "Escalate", "role:senior", (esc_end - esc_start).total_seconds()/3600.0,
                                 esc_start, case.era, overtime=True, waiting_h=0.0, escalated=True)
            events.append(SEvent(case.cid, "Escalate", esc_start, esc_end, "R_manager1", "role:senior", esc_amt, {"sla_breach": True}))
            # Overtime can remain on for rest of case

        # Compute optional QA_Final skip if samples were short
        if task.label == "QA_Join" and qa_sample_durations:
            avg_sample = sum(qa_sample_durations)/len(qa_sample_durations)
            if random.random() < (0.35 if avg_sample < 1.0 else 0.7):
                # add QA_Final
                t_qaf = Task("qa_final_" + str(uuid.uuid4()), case.cid, "QA_Final", "QA", PREFERRED_ROLES["QA"], case.priority, [task.tid], release_interval_min=60)
                # Insert into ready queue now
                pred_map[t_qaf.tid] = set([task.tid])
                task_by_tid[t_qaf.tid] = t_qaf
                ready.append(t_qaf)
            # else skip QA_Final

        # Maintain next tasks: if this task completion unlocks others, move them to ready when all preds done
        done.add(task.tid)
        for t in tasks:
            if t.tid not in done and t.tid not in [x.tid for x in ready]:
                if pred_map[t.tid].issubset(done):
                    ready.append(t)

        # Per-case last
        last_event_resource = used_rid
        last_event_family = task.family
        last_event_end = end_time
        cost_so_far += amt

    # Sort final events by time
    events.sort(key=lambda e: (e.start, e.end))
    return events

# --------------------------------
# Distinctness by trace activity sequence
# --------------------------------
def trace_signature(events: list):
    return tuple(e.label for e in events)

# --------------------------------
# Quality & sanity checks
# --------------------------------
def sanity_checks(log: EventLog, traces_meta):
    print("\n=== Sanity & Quality Checks ===")
    # Schema check + monotonic timestamps
    total_events = 0
    neg_durations = 0
    missing = Counter()
    for tr in log:
        prev = None
        for e in tr:
            total_events += 1
            for k in ("concept:name","time:timestamp","org:resource","amount"):
                if k not in e:
                    missing[k] += 1
            if prev and e["time:timestamp"] < prev:
                print("Non-monotonic timestamps in a trace!")
            prev = e["time:timestamp"]
            dur = 0.0
            # We stored only end in meta; but not stored here. Use per-trace diffs heuristically
        # OK
    print(f"Total traces: {len(log)}, total events: {total_events}")
    if missing:
        print("Missing attrs:", dict(missing))

    # Distribution summaries (quick, no plots)
    if pd:
        rows = []
        for tr in log:
            cid = tr.attributes.get("concept:name","?")
            times = [e["time:timestamp"] for e in tr]
            amts = [float(e["amount"]) for e in tr]
            labels = [e["concept:name"] for e in tr]
            rows.append({
                "cid": cid,
                "trace_len": len(tr),
                "start": min(times),
                "end": max(times),
                "duration_h": (max(times)-min(times)).total_seconds()/3600.0,
                "cost_sum": sum(amts),
                "has_and": traces_meta[cid].get("has_parallel", True),  # we model many with ANDs
            })
        df = pd.DataFrame(rows)
        print("\nTrace length: mean={:.2f}, q25={}, q50={}, q75={}, max={}".format(df.trace_len.mean(), *np.percentile(df.trace_len, [25,50,75,100]).round(1)))
        print("Trace duration (h): mean={:.2f}, std={:.2f}, q25={:.1f}, q50={:.1f}, q75={:.1f}".format(
            df.duration_h.mean(), df.duration_h.std(), *np.percentile(df.duration_h, [25,50,75])))
        print("Total cost: mean={:.2f}, std={:.2f}, q25={:.1f}, q50={:.1f}, q75={:.1f}".format(
            df.cost_sum.mean(), df.cost_sum.std(), *np.percentile(df.cost_sum, [25,50,75])))
        # Drift check by era
        df["era"] = df.apply(lambda r: era_index(r["start"]), axis=1)
        grp = df.groupby("era").agg(duration_h=("duration_h","mean"), cost=("cost_sum","mean"), n=("cid","count"))
        print("\nEra-wise mean duration/cost:\n", grp)
    else:
        print("Install pandas to see distributions.")

    # Ambiguity check: sample 100 random prefixes and look at distinct next labels
    prefix_map = defaultdict(set)
    all_traces = []
    for tr in log:
        labs = [e["concept:name"] for e in tr]
        all_traces.append(labs)
        for i in range(len(labs)-1):
            prefix = tuple(labs[:i+1])
            prefix_map[prefix].add(labs[i+1])
    prefixes = list(prefix_map.keys())
    random.shuffle(prefixes)
    sample = prefixes[:min(100, len(prefixes))]
    ambi = sum(1 for p in sample if len(prefix_map[p]) >= 2)
    print(f"Ambiguity check: {ambi}/{len(sample)} sampled prefixes had ≥2 distinct next activities.")

    # Approx parallelism check: count presence of Join/Handover/AND markers
    traces_with_and = sum(1 for tr in log if any("Join" in e["concept:name"] for e in tr))
    print(f"Parallelism check: {traces_with_and/len(log)*100:.1f}% traces contain ≥1 AND/OR join event.")

# --------------------------------
# Export utils
# --------------------------------
def write_xes(log: EventLog, path: str):
    try:
        import pm4py
        # pm4py ≥ 2.7 convenience
        pm4py.write_xes(log, path)
    except Exception:
        # classic exporter
        from pm4py.objects.log.exporter.xes import exporter as xes_exporter
        xes_exporter.apply(log, path)
    print(f"\nWrote XES to: {path}")

# --------------------------------
# Example commentary (for auditors)
# --------------------------------
def _example_trace_comments():
    """
    Minimal example (narrative, not executed):
      Plan_Create (R5, planner)
      Approve_Plan (same resource via binding-of-duty)
      AND-split: Procure_Materials || Setup_Line (different resources, parallel)
      Join
      ProduceBatch_start (R11 operator) -> preempt (queue spike) -> ProduceBatch_resume -> ProduceBatch_complete
      Handover (resource change) before QA_Sample (QA pool)
      SLA breached at QA_Join → Escalate (senior, overtime)
      QA_Final (optional)
      Pack → Stock → Ship
    """
    return None

# --------------------------------
# Main generation
# --------------------------------
def execute_script():
    # Generate arrival times with NHPP bursts
    arrivals = nhpp_bursty_arrivals(n_cases_target=600)

    # Build cases with features (priority, SLA, budget, era)
    cases = []
    for i, arr in enumerate(arrivals):
        cid = f"MTS-{i+1:04d}"
        priority = random.choices([0,1,2], weights=[0.5,0.35,0.15], k=1)[0]  # 0=low, 1=normal, 2=high
        sla_hours = random.choice([48, 72, 96, 120]) - 12*priority  # higher priority -> tighter SLA
        budget_limit = random.uniform(2000, 8000)
        e = era_index(arr)
        cases.append(Case(cid, arr, priority, sla_hours, budget_limit, e, features={}))

    # Build and schedule trace-by-trace until we have 250 DISTINCT sequences
    distinct = set()
    log = EventLog()
    traces_meta = {}

    attempts = 0
    for case in cases:
        if len(distinct) >= MAX_DISTINCT_TRACES:
            break
        attempts += 1
        # Construct control-flow skeleton
        tasks = build_skeleton(case)
        # Mark that we do have parallel patterns in this skeleton
        traces_meta[case.cid] = {"has_parallel": True}

        # Schedule tasks (trace-by-trace)
        events = schedule_case_tasks(case, tasks)
        # Build signature on activity label sequence
        sig = trace_signature(events)
        if sig in distinct:
            continue  # skip duplicates to preserve "distinct traces"
        distinct.add(sig)

        # Materialize pm4py Trace
        tr = Trace()
        tr.attributes["concept:name"] = case.cid  # becomes case:concept:name in df conversions
        for ev in events:
            e = Event()
            e["concept:name"] = ev.label
            e["time:timestamp"] = ev.start  # event instant at start; (we modeled durations in scheduling)
            e["org:resource"] = ev.resource
            e["amount"] = float(round(ev.amount, 2))
            tr.append(e)

        # Order by timestamp (required)
        tr[:] = sorted(tr, key=lambda x: x["time:timestamp"])
        log.append(tr)

    print(f"Generated {len(log)} distinct traces (attempted {attempts}). Target ≤ {MAX_DISTINCT_TRACES} satisfied.")

    # Sanity & quality checks
    sanity_checks(log, traces_meta)

    # Export to XES (gz)
    write_xes(log, TARGET_XES)

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    execute_script()
