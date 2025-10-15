# data_generator.py
import random
import pandas as pd
from datetime import datetime, timedelta, time

class ProcessSimulator:
    """
    Generates event logs from simulated process models with increased complexity and realism.
    - Interdependent attributes (e.g., duration depends on resource).
    - Complex control-flow (parallelism, conditional loops).
    - Realistic timestamp simulation (business hours, weekends).
    - Data-driven decision points.
    """

    def __init__(self, num_cases=500):
        self.num_cases = num_cases
        # Define richer vocabularies for categorical features
        self.vocab = {
            'activity': [
                'Start', 'Submit Request', 'Check Request', 'Approve Request',
                'Analyze Request', 'Request Additional Info', 'Verify Data',
                'Process Payment', 'Notify Customer', 'Archive Case', 'End'
            ],
            'resource': ['Alice', 'Bob', 'Charlie', 'Dana', 'Eve', 'Frank'],
            'department': ['Sales', 'Finance', 'Operations', 'Support']
        }
        self.activity_map = {name: i for i, name in enumerate(self.vocab['activity'])}
        self.resource_map = {name: i for i, name in enumerate(self.vocab['resource'])}
        self.department_map = {name: i for i, name in enumerate(self.vocab['department'])}

        # --- Define attribute correlations and realism parameters ---

        # Base costs per activity
        self.activity_costs = {
            'Submit Request': 5.0, 'Check Request': 10.0, 'Approve Request': 25.0,
            'Analyze Request': 40.0, 'Request Additional Info': 15.0, 'Verify Data': 30.0,
            'Process Payment': 50.0, 'Notify Customer': 8.0, 'Archive Case': 3.0,
            'Start': 0.0, 'End': 0.0
        }

        # Base durations in minutes and resource proficiency multipliers
        self.activity_durations = {
            'Submit Request': (10, 20), 'Check Request': (30, 60), 'Approve Request': (60, 120),
            'Analyze Request': (120, 240), 'Request Additional Info': (20, 40),
            'Verify Data': (90, 180), 'Process Payment': (40, 80), 'Notify Customer': (10, 15),
            'Archive Case': (5, 10), 'Start': (0, 0), 'End': (0, 0)
        }
        self.resource_skills = {
            'Alice': {'Analyze Request': 0.7, 'Process Payment': 1.0}, # Expert analyst
            'Bob': {'Check Request': 0.8, 'Verify Data': 1.2},
            'Charlie': {'Approve Request': 0.9, 'Process Payment': 0.8}, # Finance expert
            'Dana': {'Request Additional Info': 1.0, 'Notify Customer': 0.7}, # Communications expert
            'Eve': {'Submit Request': 1.1, 'Check Request': 1.1}, # Trainee
            'Frank': {'Verify Data': 0.8, 'Archive Case': 1.0}
        }

    def _get_next_timestamp(self, current_time, duration_minutes):
        """Simulates realistic time progression, accounting for business hours and weekends."""
        next_time = current_time + timedelta(minutes=duration_minutes)

        # Simulate non-business hours (events happen between 8 AM and 6 PM)
        if next_time.time() < time(8, 0) or next_time.time() > time(18, 0):
            if next_time.hour >= 18:
                next_time = next_time.replace(hour=8, minute=0, second=0) + timedelta(days=1)
            else:
                next_time = next_time.replace(hour=8, minute=random.randint(0, 30))

        # Simulate weekend delays
        if next_time.weekday() >= 5: # Saturday or Sunday
            next_time += timedelta(days=(7 - next_time.weekday()))
            next_time = next_time.replace(hour=8, minute=random.randint(0, 30))

        return next_time

    def _generate_trace(self, case_id, model_logic):
        """Generates a single, complex trace by executing the model logic."""
        trace = []
        # Case-level attributes that can influence the process
        case_state = {
            'id': case_id,
            'current_time': datetime(2025, 1, 1) + timedelta(days=random.randint(0, 60)),
            'assigned_resource': random.choice(self.vocab['resource']),
            'department': random.choice(self.vocab['department']),
            'value': random.uniform(100, 5000) # e.g., order value
        }

        path_generator = model_logic(case_state)

        for activity_name in path_generator:
            # Handle parallel activities, which are yielded as a tuple
            if isinstance(activity_name, tuple):
                parallel_events = list(activity_name)
                random.shuffle(parallel_events) # Execute in random order
                for act in parallel_events:
                    self._create_event(trace, case_state, act)
                continue

            # Standard sequential activity
            self._create_event(trace, case_state, activity_name)

        # Post-process time features after the trace is complete
        if not trace: return []
        start_time = trace[0]['timestamp']
        for i in range(len(trace)):
            trace[i]['time_from_start'] = trace[i]['timestamp'] - start_time
            if i > 0:
                trace[i]['time_from_previous'] = trace[i]['timestamp'] - trace[i-1]['timestamp']

        return trace

    def _create_event(self, trace, case_state, activity_name):
        """Creates a single event dictionary and updates the case state."""
        resource = case_state['assigned_resource']

        # Calculate duration based on activity and resource skill
        base_min, base_max = self.activity_durations.get(activity_name, (5, 15))
        skill_multiplier = self.resource_skills.get(resource, {}).get(activity_name, 1.0)
        duration = random.uniform(base_min, base_max) * skill_multiplier
        # Add random noise/outliers
        if random.random() < 0.02: # 2% chance of a major delay
            duration *= random.uniform(3, 10)

        case_state['current_time'] = self._get_next_timestamp(case_state['current_time'], duration)

        # Calculate cost
        cost = self.activity_costs.get(activity_name, 0.0) + random.uniform(-2, 2)

        event = {
            'case_id': case_state['id'],
            'activity': self.activity_map[activity_name],
            'timestamp': case_state['current_time'].timestamp(),
            'resource': self.resource_map[resource],
            'department': self.department_map[case_state['department']],
            'cost': max(0.0, round(cost, 2)),
            'time_from_start': 0, # Placeholder
            'time_from_previous': 0, # Placeholder
        }
        trace.append(event)


    # --- Process Model Definitions ---

    def _model_a_logic(self, state):
        """Simple linear process for standard requests."""
        yield 'Start'
        yield 'Submit Request'
        yield 'Check Request'
        yield 'Approve Request'
        yield 'Process Payment'
        yield 'Notify Customer'
        yield 'Archive Case'
        yield 'End'

    def _model_b_logic(self, state):
        """Process with a data-driven choice and potential rework loop."""
        yield 'Start'
        yield 'Submit Request'
        yield 'Analyze Request'
        # Data-driven choice: High-value cases require extra verification
        if state['value'] > 2500:
            yield 'Verify Data'
        else:
            yield 'Check Request'

        # Conditional rework loop
        rework_cycles = 0
        while random.random() < 0.3 and rework_cycles < 2: # 30% chance of rework
            yield 'Request Additional Info'
            yield 'Check Request'
            rework_cycles += 1

        yield 'Approve Request'
        yield 'Process Payment'
        yield 'Notify Customer'
        yield 'Archive Case'
        yield 'End'

    def _model_c_logic(self, state):
        """Process with parallel activities (AND-split/join)."""
        yield 'Start'
        yield 'Submit Request'
        yield 'Check Request'
        # Parallel execution of payment and verification
        yield ('Process Payment', 'Verify Data')
        yield 'Approve Request'
        yield 'Notify Customer'
        yield 'Archive Case'
        yield 'End'

    def _model_d_logic_unseen(self, state):
        """A different process for testing, combining choices and loops."""
        yield 'Start'
        yield 'Analyze Request'
        # Exclusive choice
        if state['department'] == 'Finance':
            yield 'Process Payment'
            yield 'Verify Data'
        else:
            yield 'Request Additional Info'

        if random.random() < 0.5:
            yield 'Approve Request'

        yield 'Notify Customer'
        yield 'End'

    def generate_data_for_model(self, model_type='A'):
        """Generates a full event log for a specific process model."""
        log = []
        logic_map = {
            'A': self._model_a_logic,
            'B': self._model_b_logic,
            'C': self._model_c_logic,
            'D_unseen': self._model_d_logic_unseen
        }
        if model_type not in logic_map:
            raise ValueError(f"Unknown model type: {model_type}")

        model_func = logic_map[model_type]
        for i in range(self.num_cases):
            trace = self._generate_trace(f"{model_type}_{i}", model_func)
            if trace:
                log.append(trace)
        return log


def get_task_data(log, task_type, max_seq_len=10):
    """
    Creates subsequences and corresponding labels for a given task.
    (This function remains unchanged but is included for completeness)
    """
    tasks = []
    for trace in log:
        if len(trace) < 3: continue  # Need at least a prefix and a next event

        for i in range(1, len(trace) - 1):
            prefix = trace[:i + 1]
            if len(prefix) > max_seq_len:
                prefix = prefix[-max_seq_len:]

            if task_type == 'classification':
                # Predict the next activity
                label = trace[i + 1]['activity']
                tasks.append((prefix, label))
            elif task_type == 'regression':
                # Predict remaining time until case end
                last_event_time = trace[-1]['timestamp']
                current_event_time = prefix[-1]['timestamp']
                label = (last_event_time - current_event_time) / 3600.0  # in hours
                tasks.append((prefix, label))

    return tasks

# --- Direct Execution Block ---
if __name__ == '__main__':
    print("Generating a sample of complex process data...")
    simulator = ProcessSimulator(num_cases=10)

    # Invert maps for readable output
    rev_activity_map = {v: k for k, v in simulator.activity_map.items()}
    rev_resource_map = {v: k for k, v in simulator.resource_map.items()}
    rev_department_map = {v: k for k, v in simulator.department_map.items()}

    # Generate data from a complex model with choices and loops
    log_data = simulator.generate_data_for_model('B')

    # Flatten the log and convert to a DataFrame for pretty printing
    flat_log = [event for trace in log_data for event in trace]
    df = pd.DataFrame(flat_log)

    # Convert numeric IDs back to human-readable names
    df['activity'] = df['activity'].map(rev_activity_map)
    df['resource'] = df['resource'].map(rev_resource_map)
    df['department'] = df['department'].map(rev_department_map)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    print(f"\n--- Generated Log Summary ---")
    print(f"Total Cases: {df['case_id'].nunique()}")
    print(f"Total Events: {len(df)}")
    print(f"Average Trace Length: {df.groupby('case_id').size().mean():.2f} events")

    print("\n--- Sample Event Log (first 20 events) ---")
    print(df.head(20).to_string())
