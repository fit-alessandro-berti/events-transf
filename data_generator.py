# /io_transformer/data_generator.py

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


class ProcessModel:
    """Defines a simple process model to generate traces."""

    def __init__(self, transitions, activities, cat_attrs, num_attrs):
        self.transitions = transitions
        self.activities = activities
        self.cat_attrs = cat_attrs
        self.num_attrs = num_attrs

    def generate_trace(self, case_id):
        trace = []
        current_activity = 'START'
        timestamp = datetime(2025, 1, 1)
        event_idx = 0

        while current_activity != 'END' and event_idx < 50:  # Max trace length
            if current_activity not in self.transitions:
                break

            possible_next_activities = self.transitions[current_activity]
            next_activity = random.choice(possible_next_activities)

            # Simulate time delta
            time_delta_hours = np.random.lognormal(mean=1.0, sigma=0.5)
            timestamp += timedelta(hours=time_delta_hours)

            # Simulate other attributes
            event = {
                'case_id': case_id,
                'activity': next_activity,
                'timestamp': timestamp,
                'resource': random.choice(self.cat_attrs['resource']),
                'cost': np.random.uniform(10, 100)
            }

            if next_activity != 'END':
                trace.append(event)

            current_activity = next_activity
            event_idx += 1

        return trace


def get_process_models():
    """Returns a list of predefined process models."""
    activities = ['Activity A', 'Activity B', 'Activity C', 'Activity D', 'Activity E']
    cat_attrs = {'resource': [f'Resource {i}' for i in range(1, 4)]}
    num_attrs = ['cost']

    # Model 1: Simple linear sequence
    model1 = ProcessModel(
        transitions={
            'START': ['Activity A'],
            'Activity A': ['Activity B'],
            'Activity B': ['Activity C'],
            'Activity C': ['Activity D'],
            'Activity D': ['Activity E'],
            'Activity E': ['END']
        },
        activities=activities, cat_attrs=cat_attrs, num_attrs=num_attrs
    )

    # Model 2: Sequence with a choice
    model2 = ProcessModel(
        transitions={
            'START': ['Activity A'],
            'Activity A': ['Activity B', 'Activity C'],
            'Activity B': ['Activity D'],
            'Activity C': ['Activity D'],
            'Activity D': ['Activity E'],
            'Activity E': ['END']
        },
        activities=activities, cat_attrs=cat_attrs, num_attrs=num_attrs
    )

    # Model 3: Sequence with a loop
    model3 = ProcessModel(
        transitions={
            'START': ['Activity A'],
            'Activity A': ['Activity B'],
            'Activity B': ['Activity C', 'Activity D'],
            'Activity C': ['Activity B'],  # Loop back
            'Activity D': ['Activity E'],
            'Activity E': ['END']
        },
        activities=activities, cat_attrs=cat_attrs, num_attrs=num_attrs
    )

    return [model1, model2, model3]


def generate_event_log(num_cases):
    """Generates a complete event log by sampling from different models."""
    models = get_process_models()
    all_traces = []

    for i in range(num_cases):
        model = random.choice(models)
        trace = model.generate_trace(case_id=f'case_{i}')
        if trace:
            all_traces.append(pd.DataFrame(trace))

    return all_traces


if __name__ == '__main__':
    event_log = generate_event_log(num_cases=10)
    print("Generated Event Log Example (first case):")
    print(event_log[0])
    print(f"\nTotal cases generated: {len(event_log)}")
