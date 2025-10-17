# training.py
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
import os
import itertools
import numpy as np

# --- Import from project files ---
from torch.optim import lr_scheduler
from data_generator import XESLogLoader

# Use a try-except block for the optional dependency
try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    levenshtein_distance = None


def evaluate_embedding_quality(model, loader: XESLogLoader):
    """
    Calculates and prints metrics about the learned activity embeddings.
    This function is only active for the 'learned' strategy.
    """
    if model.strategy != 'learned':
        return

    if levenshtein_distance is None:
        print("\n⚠️ Skipping embedding evaluation: `pip install python-Levenshtein` to enable.")
        return

    print("\n📊 Evaluating Learned Embedding Quality...")

    # Get embeddings and vocabulary
    embeddings = model.embedder.activity_embedding.weight.data.clone().detach()
    embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
    id_to_name = {v: k for k, v in loader.activity_to_id.items()}

    # Filter out special tokens like <PAD> and <UNK>
    valid_ids = [i for i, name in id_to_name.items() if name not in [loader.PAD_TOKEN, loader.UNK_TOKEN]]

    if len(valid_ids) < 2:
        print("  - Not enough activities in vocabulary to evaluate.")
        return

    # Calculate string distances and cosine similarities for all pairs
    pairs = []
    for id1, id2 in itertools.combinations(valid_ids, 2):
        name1, name2 = id_to_name[id1], id_to_name[id2]

        # Normalized Levenshtein distance (0=identical, 1=completely different)
        str_dist = levenshtein_distance(name1, name2) / max(len(name1), len(name2))

        # Cosine similarity
        cos_sim = np.dot(embeddings[id1], embeddings[id2])

        pairs.append({'str_dist': str_dist, 'cos_sim': cos_sim})

    if not pairs:
        return

    # Sort pairs by string distance
    pairs.sort(key=lambda x: x['str_dist'])

    # Get top 5 most similar and dissimilar by name
    num_pairs_to_show = min(5, len(pairs))
    similar_by_name = pairs[:num_pairs_to_show]
    dissimilar_by_name = pairs[-num_pairs_to_show:]

    avg_sim_for_similar_names = np.mean([p['cos_sim'] for p in similar_by_name])
    avg_sim_for_dissimilar_names = np.mean([p['cos_sim'] for p in dissimilar_by_name])

    print(f"  - Avg. Cosine Sim for Top {num_pairs_to_show} Similar Names:   {avg_sim_for_similar_names:.4f}")
    print(f"  - Avg. Cosine Sim for Top {num_pairs_to_show} Dissimilar Names: {avg_sim_for_dissimilar_names:.4f}")
    print("-" * 30)


def create_episode(task_pool, num_shots_range, num_queries_per_class, num_ways_range=(2, 5)):
    """
    Creates a single meta-learning episode using N-way K-shot sampling.
    """
    # This function remains unchanged
    class_dict = defaultdict(list)
    for seq, label in task_pool:
        class_dict[label].append((seq, label))
    num_ways = random.randint(num_ways_range[0], num_ways_range[1])
    num_shots = random.randint(num_shots_range[0], num_shots_range[1])
    available_classes = [c for c, items in class_dict.items() if len(items) >= num_shots + num_queries_per_class]
    if len(available_classes) < num_ways: return None
    episode_classes = random.sample(available_classes, num_ways)
    support_set, query_set = [], []
    for cls in episode_classes:
        samples = random.sample(class_dict[cls], num_shots + num_queries_per_class)
        support_set.extend(samples[:num_shots])
        query_set.extend(samples[num_shots:])
    random.shuffle(support_set)
    random.shuffle(query_set)
    return support_set, query_set


def train(model, training_tasks, loader, config):
    """
    Main training loop. Now receives the loader to access vocabulary for evaluation.
    """
    print("🚀 Starting meta-training...")
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    cls_task_pools = [pool for pool in training_tasks['classification'] if pool]
    reg_task_pools = [pool for pool in training_tasks['regression'] if pool]

    if not cls_task_pools and not reg_task_pools:
        print("❌ Error: No valid training tasks available. Aborting training.")
        return

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(range(config['episodes_per_epoch']), desc=f"Epoch {epoch + 1}/{config['epochs']}")
        for _ in progress_bar:
            task_type = random.choice(['classification', 'regression'])

            if task_type == 'classification' and cls_task_pools:
                task_data_pool = random.choice(cls_task_pools)
            elif task_type == 'regression' and reg_task_pools:
                task_data_pool = random.choice(reg_task_pools)
            else:
                task_type = 'regression' if reg_task_pools else 'classification'
                task_data_pool = random.choice(reg_task_pools if reg_task_pools else cls_task_pools)

            if not task_data_pool: continue

            if task_type == 'classification':
                episode = create_episode(task_data_pool, config['num_shots_range'], config['num_queries'],
                                         num_ways_range=(2, 7))
            else:  # Regression
                if len(task_data_pool) < config['num_shots_range'][1] + config['num_queries']:
                    episode = None
                else:
                    random.shuffle(task_data_pool)
                    num_shots = random.randint(config['num_shots_range'][0], config['num_shots_range'][1])
                    support_set = task_data_pool[:num_shots]
                    query_set = task_data_pool[num_shots: num_shots + config['num_queries']]
                    episode = (support_set, query_set)

            if episode is None or not episode[0] or not episode[1]: continue

            support_set, query_set = episode
            optimizer.zero_grad(set_to_none=True)
            predictions, true_labels = model(support_set, query_set, task_type)

            if predictions is None: continue

            if task_type == 'classification':
                loss = F.cross_entropy(predictions, true_labels, ignore_index=-100, label_smoothing=0.05)
            else:
                loss = F.huber_loss(predictions.squeeze(), true_labels)

            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}", task=task_type)

        avg_loss = total_loss / config['episodes_per_epoch'] if config['episodes_per_epoch'] > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch + 1} finished. Average Loss: {avg_loss:.4f} | Current LR: {current_lr:.6f}")

        # --- NEW: Evaluate embedding quality at the end of the epoch ---
        evaluate_embedding_quality(model, loader)

        scheduler.step()
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Model checkpoint saved to {checkpoint_path}")

    print("✅ Meta-training complete.")
