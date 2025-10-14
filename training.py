# training.py
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from data_generator import get_task_data


def create_episode(task_pool, num_shots_range, num_queries):
    """Creates a single meta-learning episode (support set + query set)."""
    # 1. Select a task
    task_data = random.choice(task_pool)
    random.shuffle(task_data)

    # 2. Select a number of shots (K) for this episode
    num_shots = random.randint(num_shots_range[0], num_shots_range[1])

    # 3. Create support and query sets
    if len(task_data) < num_shots + num_queries:
        return None  # Not enough data for this task

    support_set = task_data[:num_shots]
    query_set = task_data[num_shots: num_shots + num_queries]

    return support_set, query_set


def train(model, training_tasks, config):
    """
    Main training loop.

    Args:
        model: The meta-learner model.
        training_tasks (dict): Dict with 'classification' and 'regression' task pools.
        config (dict): Training configuration.
    """
    print("🚀 Starting meta-training...")
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(range(config['episodes_per_epoch']), desc=f"Epoch {epoch + 1}/{config['epochs']}")
        for _ in progress_bar:
            # Randomly pick a task type for the episode
            task_type = random.choice(['classification', 'regression'])

            episode = create_episode(
                training_tasks[task_type],
                config['num_shots_range'],
                config['num_queries']
            )
            if episode is None:
                continue

            support_set, query_set = episode

            optimizer.zero_grad()

            predictions, true_labels = model(support_set, query_set, task_type)

            if task_type == 'classification':
                loss = F.nll_loss(predictions, true_labels)
            else:  # regression
                loss = F.mse_loss(predictions, true_labels)

            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}", task=task_type)

        avg_loss = total_loss / config['episodes_per_epoch']
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

    print("✅ Meta-training complete.")
