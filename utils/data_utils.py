# File: utils/data_utils.py
import pandas as pd
import random
import numpy as np
from collections import defaultdict

# --- Import from project files ---
from time_transf import transform_time


def get_task_data(log, task_type, max_seq_len=10):
    """
    Converts a processed log (list of traces) into a list of (prefix, label, case_id) tasks.
    """
    tasks = []
    if not log: return tasks
    for trace in log:
        if len(trace) < 3: continue

        # --- FIX: Get the case_id for the whole trace ---
        # All events in a trace share the same case_id
        case_id = trace[0]['case_id']

        for i in range(1, len(trace) - 1):
            prefix = trace[:i + 1]
            if len(prefix) > max_seq_len: prefix = prefix[-max_seq_len:]

            # The activity_id can now be None if get() fails, so we check for that
            next_event_activity_id = trace[i + 1]['activity_id']

            if task_type == 'classification':
                if next_event_activity_id is not None:
                    # --- FIX: Append (prefix, label, case_id) ---
                    tasks.append((prefix, next_event_activity_id, case_id))
            elif task_type == 'regression':
                remaining_time = (trace[-1]['timestamp'] - prefix[-1]['timestamp']) / 3600.0
                # --- FIX: Append (prefix, label, case_id) ---
                tasks.append((prefix, transform_time(remaining_time), case_id))
    return tasks


def create_episode(task_pool, num_shots_range, num_queries_per_class, num_ways_range=(2, 5), shuffle_labels=False):
    """
    Creates a single meta-learning episode. Can optionally shuffle labels
    within the episode to force in-context learning.

    (Moved from training.py)
    """
    class_dict = defaultdict(list)

    # --- FIX ---
    # task_pool now contains (seq, label, case_id) tuples.
    # We unpack all three but only use seq and label for training.
    for seq, label, case_id in task_pool:
        # --- END FIX ---
        class_dict[label].append((seq, label))

    num_ways = random.randint(num_ways_range[0], num_ways_range[1])
    num_shots = random.randint(num_shots_range[0], num_shots_range[1])
    available_classes = [c for c, items in class_dict.items() if len(items) >= num_shots + num_queries_per_class]
    if len(available_classes) < num_ways: return None
    episode_classes = random.sample(available_classes, num_ways)

    # --- NEW: Episodic Label Shuffle Logic ---
    label_map = {}
    if shuffle_labels:
        shuffled_classes = random.sample(episode_classes, len(episode_classes))
        label_map = {original: shuffled for original, shuffled in zip(episode_classes, shuffled_classes)}
    # If not shuffling, the map will be empty.

    support_set, query_set = [], []
    for cls in episode_classes:
        # Use the shuffled label if the map exists, otherwise use the original
        mapped_label = label_map.get(cls, cls)

        samples = random.sample(class_dict[cls], num_shots + num_queries_per_class)

        # Append samples with the potentially shuffled label
        for s in samples[:num_shots]:
            support_set.append((s[0], mapped_label))  # s[0] is the sequence
        for s in samples[num_shots:]:
            query_set.append((s[0], mapped_label))

    random.shuffle(support_set)
    random.shuffle(query_set)
    return support_set, query_set
