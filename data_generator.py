# data_generator.py
import random
import pandas as pd
from datetime import datetime, timedelta


class ProcessSimulator:
    """
    Generates event logs from different simulated process models.
    Each event has an activity, timestamp, and other categorical/numerical attributes.
    """

    def __init__(self, num_cases=500):
        self.num_cases = num_cases
        # Define vocabularies for categorical features
        self.vocab = {
            'activity': ['Start', 'Activity A', 'Activity B', 'Activity C', 'Activity D', 'Activity E', 'End'],
            'resource': ['Resource 1', 'Resource 2', 'Resource 3', 'Resource 4']
        }
        self.activity_map = {name: i for i, name in enumerate(self.vocab['activity'])}
        self.resource_map = {name: i for i, name in enumerate(self.vocab['resource'])}

    def _generate_trace(self, case_id, model_logic):
        """Generates a single trace based on the provided model logic."""
        trace = []
        current_time = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 30))
        path = model_logic()

        for i, activity_name in enumerate(path):
            time_delta_minutes = random.uniform(10, 120)
            current_time += timedelta(minutes=time_delta_minutes)

            event = {
                'case_id': case_id,
                'activity': self.activity_map[activity_name],
                'timestamp': current_time.timestamp(),
                'resource': self.resource_map[random.choice(self.vocab['resource'])],
                'cost': round(random.uniform(5.0, 50.0), 2),
                'time_from_start': 0,
                'time_from_previous': 0,
            }
            trace.append(event)

        # Post-process time features
        start_time = trace[0]['timestamp']
        for i in range(len(trace)):
            trace[i]['time_from_start'] = trace[i]['timestamp'] - start_time
            if i > 0:
                trace[i]['time_from_previous'] = trace[i]['timestamp'] - trace[i - 1]['timestamp']

        return trace

    def _model_a_logic(self):
        """Simple linear process: Start -> A -> B -> C -> End"""
        return ['Start', 'Activity A', 'Activity B', 'Activity C', 'End']

    def _model_b_logic(self):
        """Process with a choice: Start -> A -> (B or C) -> D -> End"""
        path = ['Start', 'Activity A']
        path.append(random.choice(['Activity B', 'Activity C']))
        path.extend(['Activity D', 'End'])
        return path

    def _model_c_logic(self):
        """Process with a loop: Start -> A -> B -> C -> B -> D -> End"""
        return ['Start', 'Activity A', 'Activity B', 'Activity C', 'Activity B', 'Activity D', 'End']

    def _model_d_logic_unseen(self):
        """A different process for testing: Start -> E -> (A or D) -> C -> End"""
        path = ['Start', 'Activity E']
        path.append(random.choice(['Activity A', 'Activity D']))
        path.extend(['Activity C', 'End'])
        return path

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
            raise ValueError("Unknown model type")

        for i in range(self.num_cases):
            log.append(self._generate_trace(f"{model_type}_{i}", logic_map[model_type]))
        return log


def get_task_data(log, task_type, max_seq_len=10):
    """
    Creates subsequences and corresponding labels for a given task.

    Args:
        log (list): A list of traces.
        task_type (str): 'classification' or 'regression'.
        max_seq_len (int): The maximum length of a subsequence.

    Returns:
        list: A list of tuples (subsequence, label).
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
