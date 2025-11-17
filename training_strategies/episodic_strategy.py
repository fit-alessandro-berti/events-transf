# File: training_strategies/episodic_strategy.py
import random
import torch.nn.functional as F

# --- Import from project files ---
from utils.data_utils import create_episode


def run_episodic_step(model, task_data_pool, task_type, config, should_shuffle_labels):
    """
    Runs a single standard meta-learning (episodic) training step.

    Returns:
        (torch.Tensor or None): The computed loss for the step, or None if step failed.
        (str): A string descriptor for the task (e.g., "classification").
    """
    progress_bar_task = task_type
    episode = None

    if task_type == 'classification':
        episode = create_episode(
            task_data_pool, config['num_shots_range'], config['num_queries'],
            num_ways_range=(3, 10), shuffle_labels=should_shuffle_labels
        )
    else:  # Regression
        if len(task_data_pool) < config['num_shots_range'][1] + config['num_queries']:
            episode = None
        else:
            random.shuffle(task_data_pool)
            num_shots = random.randint(config['num_shots_range'][0], config['num_shots_range'][1])
            support_set_raw = task_data_pool[:num_shots]
            query_set_raw = task_data_pool[num_shots: num_shots + config['num_queries']]
            support_set = [(s[0], s[1]) for s in support_set_raw]
            query_set = [(q[0], q[1]) for q in query_set_raw]
            episode = (support_set, query_set)

    if episode is None or not episode[0] or not episode[1]:
        return None, progress_bar_task

    support_set, query_set = episode
    predictions, true_labels = model(support_set, query_set, task_type)

    if predictions is None:
        return None, progress_bar_task

    if task_type == 'classification':
        loss = F.cross_entropy(predictions, true_labels, ignore_index=-100, label_smoothing=0.05)
    else:
        loss = F.huber_loss(predictions.squeeze(), true_labels)

    return loss, progress_bar_task
