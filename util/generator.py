import os
import subprocess
import threading
import requests


prompt_template = """
Create such a script for the process [[process_name]] and save the resulting XES log to [[xes_gz_path]] ... very important: limit the length of the XES log to 250 distinct traces.

EXAMPLE LOG GENERATION IN PM4PY:
 
from copy import deepcopy
from pm4py.objects.log.obj import EventLog, Trace, Event

def execute_script():
    L = EventLog()
    e1 = Event()
    e1["concept:name"] = "A"
    e2 = Event()
    e2["concept:name"] = "B"
    e3 = Event()
    e3["concept:name"] = "C"
    e4 = Event()
    e4["concept:name"] = "D"
    t = Trace()
    t.append(e1)
    t.append(e2)
    t.append(e3)
    t.append(e4)
    for i in range(10000):
        L.append(deepcopy(t))
    print(len(L))

if **name** == "**main**":
    execute_script()


## Role & Goal

You are a senior process mining engineer. Write a single, self‑contained **Python script** that uses **pm4py** to **synthesize one XES log**, **constructed trace‑by‑trace** (no black‑box simulator), so we can precisely inject patterns.
**each event must include**:

- `concept:name` (activity label)
- `time:timestamp` (Python `datetime`)
- `org:resource` (string)
- `amount` (float; event‑level cost)

Also set the trace attribute `concept:name` to the case id so that `case:concept:name` is present when converting to a DataFrame.

---

## 1. Activity & Resource Alphabets

- Create a generic activity set (e.g., 20–40 labels)
- Create **resource pools** and **roles** (strings only): `R{1..30}`, roles like `role:analyst`, `role:senior`, `role:vendor`. Roles may be an optional attribute; **always** set `org:resource`. Ensure a small fraction uses `Unknown` to test defaults.

---

## 2. Control‑Flow Skeletons (process‑agnostic)

Implement a **library of control‑flow templates**. For each case, sample one or compose a few:

- **Linear with optional inserts:** `Start → A → (optional B) → C → End`
- **XOR branching:** after `A`, pick one of `{C, D, E}`; branch probs may depend on **time, queue length, cost so far, chosen resource earlier**
- **Loops/Rework:** geometric repeats of `{Review → Fix}` until pass
- **AND‑split / AND‑join (parallelism):** `A → (B || C) → Join → D`. The join’s start must wait for both, so the **join time = max(children end)** (important for durations)
- **OR‑split/OR‑join:** choose 1..k of `{X, Y, Z}` to run in parallel
- **Skip patterns:** optional sub‑paths; sometimes null transitions (no operation) to test classification ambiguity
- **Long‑term dependency:** decision at `X` affects allowed/likely paths/resources at `Y` far later (binding‑of‑duty or separation‑of‑duty; see below)
- **Preemption/Resume:** represent as separate activities `Task.start`, `Task.resume`, `Task.complete` (no lifecycle attribute required)

**Requirement:** From the **same prefix**, more than one next activity must be plausible (create label ambiguity) so classification is not trivial.

---

## 3. Temporal Pattern Library (inject many; mix & match)

For each case, **compose 3–8 patterns** below. Each item states the **temporal logic** and its **control‑flow interaction**:

1. **Heavy‑tailed service times:** lognormal / Pareto tails.  
   ↔ Longer tasks increase chance of **rework** or **escalation** branches.
2. **Bursty arrivals (NHPP):** time‑varying inter‑arrival intensity; create case start times in waves.  
   ↔ Queues grow, causing later tasks to **wait** → impacts remaining time.
3. **Work calendars:** business hours (e.g., 9–17, Mon–Fri), lunch breaks; **freeze** on weekends.  
   ↔ Parallel joins cross day boundaries; SLA breaches trigger **Escalate**.
4. **Resource shift calendars:** per‑resource availability windows; overtime allowed with penalty.  
   ↔ If assigned resource is off‑shift, delay or reassign; assignment choice alters next step.
5. **Aging/Deadline timers:** add a deadline at case or activity start; if exceeded, insert `Warn`/`Escalate`.  
   ↔ XOR to escalation sub‑process; increases cost and changes resource role to `senior`.
6. **Setup/Changeover times:** duration of `B` depends on the **previous activity** or **resource** (sequence‑dependent).  
   ↔ Encourages certain activity orders to minimize time; else likely rework.
7. **Interruption & Resume:** random preemptions; split one logical task into `start`/`resume`/`complete`.  
   ↔ Preemption probability increases under high queue lengths.
8. **Synchronization latency:** for AND‑joins, join time = max of predecessors; also add *handover delay* before `Join`.  
   ↔ Longer branch determines remaining time distribution.
9. **Queue discipline:** implement **FIFO**, **priority**, **shortest‑processing‑time first** per activity.  
   ↔ Priority cases may **skip** optional steps (XOR based on priority).
10. **Seasonality & drift:** different parameter regimes by **eras** (e.g., monthly blocks).  
    ↔ In later eras, `Review` becomes faster but `Approve` slower; branch probs shift.
11. **Learning curve:** resource’s service time decreases with repeated execution of the same activity.  
    ↔ Fewer reworks after several repetitions by same resource.
12. **Forgetting curve:** performance worsens after inactivity.  
    ↔ More rework if the last time a resource executed activity `A` was long ago.
13. **Cooling‑off constraint:** enforce a minimum lag between `X` and `Y`.  
    ↔ If violated by schedule pressure, **reroute** to buffer activity `Wait`.
14. **Handover delay:** additional lag whenever **resource changes** between consecutive events.  
    ↔ Induces preference for binding‑of‑duty branches.
15. **Travel/Context switch time:** per‑resource penalty on switching activity types bucketed by label prefix.  
    ↔ Encourages batching (implicit loop) to amortize switch cost.
16. **Batching/Release:** only start `K` tasks of a type at fixed release times.  
    ↔ Produces stepwise waits; parallel branches finish in waves.
17. **Stochastic skip with memory:** probability to skip `QA` decreases if `Review` duration was short.  
    ↔ Long‑term dependency: early speed influences later branch.
18. **Multi‑instance (k‑of‑n) tasks:** spawn n sub‑tasks; proceed after k complete.  
    ↔ Remaining time depends on the k‑th order statistic.

---

## 4. Resource Pattern Library (with control‑flow interplay)

1. **Roles & pools:** map activities to preferred roles; if absent, assign alternative role with slower times.  
   ↔ May trigger `Escalate` if slow path threatens deadline.
2. **Skill heterogeneity:** per‑resource speed and **rework probability** multipliers per activity family.  
   ↔ Low skill increases chance of `Rework` loop.
3. **Binding‑of‑duty:** enforce same resource for `{X, Y}` pairs.  
   ↔ If not possible, route to `Reassign` sub‑process.
4. **Separation‑of‑duty:** enforce different resources for `{Approve, Audit}`.  
   ↔ Violations cause `Reassign` and delay.
5. **Handover‑of‑work patterns:** explicit `Handover` pseudo‑activity on resource changes.  
   ↔ Adds time and cost; also signals control‑flow transitions.
6. **Unavailability:** random absences; force reassign or wait.  
   ↔ Reassignment can switch branch (XOR) to role‑specific path.
7. **Assignment policy mixture:** weighted mixture of **random / least loaded / skill‑based**.  
   ↔ Policy choice interacts with queue discipline and deadlines.
8. **Multi‑resource meetings:** certain activities require 2+ resources concurrently (find overlapping availability).  
   ↔ Start may be delayed; later branches compress to catch up.
9. **External vendor option:** outsource alternative has slower mean but lower fixed cost (or vice versa).  
   ↔ XOR split based on **budget so far** or **deadline slack**.

---

## 5. Cost Pattern Library (event‑level `amount`) + interplay

1. **Fixed + variable:** `amount = fixed(activity) + rate(resource, shift) * duration`.  
   ↔ Overtime and weekends: higher `rate`.
2. **Penalty for SLA breach:** add lump sum on `Escalate` or at `End` if total duration > threshold.  
   ↔ Incentivizes faster branches in XOR.
3. **Rework cost premium:** rework loops have **increasing** fixed cost per iteration.  
   ↔ Geometric loop length affects cumulative cost.
4. **Expedite fee:** paying cost allows **queue jump** or **skip optional step**.  
   ↔ Explicit `Expedite` activity inserted conditionally.
5. **Handover fee:** charge on resource switches (models knowledge transfer).  
   ↔ Encourages binding‑of‑duty branches.
6. **Cancellation charges / refunds:** negative `amount` allowed (credits) on `Cancel`/`Rollback`.  
   ↔ Creates cost variance unrelated to duration.
7. **Material cost triggers:** certain activities add a fixed material cost independent of duration.  
   ↔ May cause budget threshold branch to cheaper path.
8. **Budget‑aware branching:** if cumulative `amount` exceeds a limit, route to `{Simplify, Vendor}`.  
   ↔ Cross‑perspective dependency (cost → control‑flow).
9. **Waiting cost (holding):** accrues per hour of **waiting**, not processing.  
   ↔ Emphasizes queue models and calendars.
10. **Era‑based tariffs:** cost rates drift over time (concept drift).  
    ↔ Later eras: cheaper vendor but higher rework penalty, etc.

---

## 6. Sampling & Scheduling Logic (what to implement)

For each **case**:

1. **Pick a skeleton** (Sec. 2) and a **pattern bundle** (3–8 patterns across Sec. 3–5).
2. **Generate a control‑flow instance:** expanded sequence with explicit **parallel blocks**, **loops**, and **optional steps** already decided. For parallelism, materialize child branches with their own event sequences and then **schedule** them.
3. **Assign resources** per activity using pools, roles, and assignment policies; enforce binding/separation constraints; add **handover events/delays** as required.
4. **Draw processing times** (service time) from per‑activity/resource distributions, then **embed calendars** and queueing:
    - Maintain a per‑resource next‑free timestamp, a per‑activity queue, and the global time cursor.
    - Compute **start time** = max(case‑local readiness, resource next‑free, calendar opening, synchronization constraints).
    - Compute **end time** = start + service time (+ handover/transfer/cooling delays).
    - For parallel blocks, **join** when all required predecessors complete (or k‑of‑n).
5. **Compute event cost** using Sec. 5 formulas; store in `amount`.
6. **Inject deadline/aging** logic to produce **Escalate/Expedite/Reassign** activities where the rules trigger.
7. **(Optional) Concept drift:** partition case starts into **eras** with different parameter sets (durations, branch probs, tariffs).
8. **Emit events** in **chronological order** into a pm4py `Trace`:

    ```python
    from pm4py.objects.log.obj import Event, Trace, EventLog
    e = Event({'concept:name': 'A_Validate',
               'time:timestamp': datetime_obj,
               'org:resource': 'R7',
               'amount': 42.50})
    trace.append(e)
    ```

    Set `trace.attributes['concept:name'] = case_id`. Add all traces to `EventLog`.
9. **Export** with `pm4py.write_xes(log, out_path)` (support `.xes` and `.xes.gz`).

---

## 7. Quality & Sanity Checks (print after generation)

* **Schema check:** all events have the four required attributes; costs are floats.
* **No negative durations**; timestamps strictly increasing per trace.
* **Distribution summaries:** histograms/quantiles of `trace length`, `activity counts`, `amount`, `inter‑event times`.
* **Ambiguity check:** sample 100 random prefixes and verify **≥2 distinct next activities** appear for many prefixes.
* **Regression signal:** distribution of **remaining time** should be wide (report mean/std/quantiles).
* **Parallelism check:** percent of traces with ≥1 AND‑split; average branch imbalance.
* **Drift check:** show era‑wise mean duration/cost difference.

---

## 8. Output

* A **single Python script** that:

  * Builds the log as described,
  * Saves it to a target XES file

**Implementation Notes**

* Use only Python stdlib + `numpy`, `pandas` (optional), and `pm4py`.
* Prefer simple helper classes for **Calendar**, **Queue**, **Assignment**, **CostModel** to keep logic readable.
* Use a small fraction of `org:resource='Unknown'`.
* Ensure costs and times remain in realistic ranges (e.g., minutes to days; costs 1–1000+).
* Order events in each trace by `time:timestamp` before export.

---

## 9. Minimal Example (for reference in the code comments)

Create one short trace that hits: AND‑split, handover delay, SLA penalty → `Escalate`, rework once, and vendor path with overtime.
    """


def generate_and_execute(prompt: str, process_name: str, xes_gz_path: str, py_file_path: str):
    print(f"[{process_name}] Starting generation...")
    
    api_key = os.getenv('GROK_API_KEY')
    if not api_key:
        print(f"[{process_name}] Error: GROK_API_KEY not set.")
        return
    
    try:
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "grok-4-0709",
            "messages": [{"role": "user", "content": prompt}]
        }

        response = requests.post(url, headers=headers, json=data, timeout=20*60)
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code

        result = response.json()
        code = result["choices"][0]["message"]["content"].strip()
        
        # Write the generated code to the target Python file
        with open(py_file_path, 'w', encoding='utf-8') as f:
            f.write(code.split("```python")[-1].split("```")[0])
        
        print(f"[{process_name}] Code generated and saved to {py_file_path}")
        
        # Execute the generated Python file
        result = subprocess.run(['python', py_file_path], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"[{process_name}] Execution failed (exit code {result.returncode}). Cleaning up.")
            print(f"[{process_name}] Stderr: {result.stderr}")
            # Remove both files
            for path in [xes_gz_path, py_file_path]:
                if os.path.exists(path):
                    os.remove(path)
        else:
            print(f"[{process_name}] Execution successful.")
            # Check if both files exist
            if os.path.exists(xes_gz_path) and os.path.exists(py_file_path):
                print(f"[{process_name}] Success! Files kept: {xes_gz_path}, {py_file_path}")
            else:
                print(f"[{process_name}] Execution succeeded but XES.GZ file missing. Cleaning up.")
                # Remove both files
                for path in [xes_gz_path, py_file_path]:
                    if os.path.exists(path):
                        os.remove(path)
                        
    except subprocess.TimeoutExpired:
        print(f"[{process_name}] Execution timed out. Cleaning up.")
        for path in [xes_gz_path, py_file_path]:
            if os.path.exists(path):
                os.remove(path)
    except Exception as e:
        print(f"[{process_name}] Error: {str(e)}. Cleaning up if files exist.")
        for path in [xes_gz_path, py_file_path]:
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    MANUAL = True  # Set to False to run automated threading mode
    # List of tuples for the arguments (process_name, xes_gz_path, py_file_path)
    tasks = [
        ("O2C", "o2c.xes.gz", "o2c.py"),
        ("Procure-to-Pay", "p2p.xes.gz", "p2p.py"),
        ("Lead-to-Opportunity", "lead2opp.xes.gz", "lead2opp.py"),
        ("Opportunity-to-Quote", "opp2quote.xes.gz", "opp2quote.py"),
        ("Campaign-to-Lead", "camp2lead.xes.gz", "camp2lead.py"),
        ("Hire-to-Retire", "hire2retire.xes.gz", "hire2retire.py"),
        ("Offer-to-Acceptance", "offer2accept.xes.gz", "offer2accept.py"),
        ("Disaster-to-Recovery", "di2re.xes.gz", "di2re.py"),
        ("Risk Identification-to-Mitigation", "rid2mit.xes.gz", "rid2mit.py"),
        ("Close-to-Report", "clos2rep.xes.gz", "clos2rep.py"),
        ("Make-to-Stock", "mak2stock.xes.gz", "mak2stock.py"),
        ("Requisition-to-Receipt", "req2receipt.xes.gz", "req2receipt.py"),
        ("Quote-to-Order", "quote2order.xes.gz", "quote2order.py")
    ]
    
    if MANUAL:
        import pyperclip
        for task_args in tasks:
            process_name, xes_gz_path, py_file_path = task_args
            if not os.path.exists(py_file_path) and not os.path.exists(xes_gz_path):
                # Format the prompt with the arguments
                full_prompt = prompt_template.replace("[[process_name]]", process_name).replace("[[xes_gz_path]]", xes_gz_path)
                # Create empty Python file
                with open(py_file_path, 'w') as f:
                    pass  # Empty file
                # Copy prompt to clipboard
                pyperclip.copy(full_prompt)
                print(f"[{process_name}] Empty file created: {py_file_path}")
                print(f"[{process_name}] Prompt copied to clipboard.")
                input(f"[{process_name}] Press Enter to continue to the next configuration...")
            else:
                print(f"[{process_name}] Files already exist, skipping.")
    else:
        threads = []
        for task_args in tasks:
            process_name, xes_gz_path, py_file_path = task_args
            if not os.path.exists(py_file_path) and not os.path.exists(xes_gz_path):
                # Format the prompt with the arguments
                full_prompt = prompt_template.replace("[[process_name]]", process_name).replace("[[xes_gz_path]]", xes_gz_path)
                t = threading.Thread(target=generate_and_execute, args=(full_prompt, process_name, xes_gz_path, py_file_path))
                t.start()
                threads.append(t)
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        print("All tasks completed.")
