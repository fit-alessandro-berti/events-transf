# training.py
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
import os

# NEW: Import the learning rate scheduler
from torch.optim import lr_scheduler


def create_episode(task_pool, num_shots_range, num_queries_per_class, num_ways_range=(2, 5)):
    """
    Creates a single meta-learning episode using N-way K-shot sampling.
    This prevents the KeyError by ensuring query labels are present in the support set.

    Args:
        task_pool (list): A list of (sequence, label) tuples for a single task/process.
        num_shots_range (tuple): Min and max K-shots per class.
        num_queries_per_class (int): Number of query examples per class.
        num_ways_range (tuple): Min and max N-ways (classes) for the episode.
    """
    # 1. Group data by class
    class_dict = defaultdict(list)
    for seq, label in task_pool:
        class_dict[label].append((seq, label))

    # 2. Decide on N-ways and K-shots for this episode
    num_ways = random.randint(num_ways_range[0], num_ways_range[1])
    num_shots = random.randint(num_shots_range[0], num_shots_range[1])

    # 3. Filter out classes that don't have enough examples for support + query
    available_classes = [c for c, items in class_dict.items() if len(items) >= num_shots + num_queries_per_class]
    if len(available_classes) < num_ways:
        return None  # Not enough classes to form an episode

    # 4. Sample N classes for the episode
    episode_classes = random.sample(available_classes, num_ways)

    support_set = []
    query_set = []

    # 5. Sample K-shot support and Q-shot query examples for each class
    for cls in episode_classes:
        samples = random.sample(class_dict[cls], num_shots + num_queries_per_class)
        support_set.extend(samples[:num_shots])
        query_set.extend(samples[num_shots:])

    random.shuffle(support_set)
    random.shuffle(query_set)

    return support_set, query_set


def train(model, training_tasks, config):
    """
    Main training loop.
    """
    print("🚀 Starting meta-training...")
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])

    # NEW: Initialize a learning rate scheduler.
    # This will reduce the learning rate by a factor of 0.5 every epoch.
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(range(config['episodes_per_epoch']), desc=f"Epoch {epoch + 1}/{config['epochs']}")
        for _ in progress_bar:
            task_type = random.choice(['classification', 'regression'])

            task_data_pool = random.choice(training_tasks[task_type])

            if task_type == 'classification':
                episode = create_episode(
                    task_data_pool,
                    config['num_shots_range'],
                    config['num_queries']
                )
            else:
                if len(task_data_pool) < config['num_shots_range'][1] + config['num_queries']:
                    episode = None
                else:
                    random.shuffle(task_data_pool)
                    num_shots = random.randint(config['num_shots_range'][0], config['num_shots_range'][1])
                    support_set = task_data_pool[:num_shots]
                    query_set = task_data_pool[num_shots: num_shots + config['num_queries']]
                    episode = (support_set, query_set)

            if episode is None or not episode[0] or not episode[1]:
                continue

            support_set, query_set = episode

            optimizer.zero_grad()

            predictions, true_labels = model(support_set, query_set, task_type)

            if predictions is None:
                continue

            if task_type == 'classification':
                loss = F.cross_entropy(predictions, true_labels, ignore_index=-100)
            else:  # regression
                # NEW: Scale the regression loss to balance it with the classification loss.
                # This factor is a hyperparameter you can tune.
                regression_loss_scale = 0.01
                loss = F.huber_loss(predictions.squeeze(), true_labels) * regression_loss_scale

            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}", task=task_type)

        avg_loss = total_loss / config['episodes_per_epoch'] if config['episodes_per_epoch'] > 0 else 0

        # NEW: Print the current learning rate at the end of the epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f} | Current LR: {current_lr:.6f}")

        # NEW: Step the scheduler at the end of each epoch
        scheduler.step()

        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Model checkpoint saved to {checkpoint_path}")

    print("✅ Meta-training complete.")
