# File: training.py
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
# 🔻🔻🔻 MODIFIED IMPORTS 🔻🔻🔻
# from collections import defaultdict # No longer needed here
import os
import itertools
import numpy as np
# 🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺

# --- Import from project files ---
from torch.optim import lr_scheduler
# 🔺 Import the new scheduler
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from data_generator import XESLogLoader
# 🔻🔻🔻 NEW IMPORT 🔻🔻🔻
from utils.data_utils import create_episode
# 🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺

# Use a try-except block for the optional dependency
try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    levenshtein_distance = None


def evaluate_embedding_quality(model, loader: XESLogLoader):
    # This function remains unchanged from the previous version
    if model.strategy != 'learned':
        return
    if levenshtein_distance is None:
        print("\n⚠️ Skipping embedding evaluation: `pip install python-Levenshtein` to enable.")
        return
    print("\n📊 Evaluating Learned Embedding Quality...")
    activity_names = loader.training_activity_names
    if len(activity_names) < 2:
        print("  - Not enough activities in vocabulary to evaluate.")
        return
    with torch.no_grad():
        model.eval()
        embeddings = model.embedder.char_embedder(activity_names, model.embedder.char_to_id)
        model.train()
    embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
    pairs = []
    for i, j in itertools.combinations(range(len(activity_names)), 2):
        name1, name2 = activity_names[i], activity_names[j]
        str_dist = levenshtein_distance(name1, name2) / max(len(name1), len(name2))
        cos_sim = np.dot(embeddings[i], embeddings[j])
        pairs.append({'str_dist': str_dist, 'cos_sim': cos_sim})
    if not pairs:
        return
    pairs.sort(key=lambda x: x['str_dist'])
    num_pairs_to_show = min(5, len(pairs))
    similar_by_name = pairs[:num_pairs_to_show]
    dissimilar_by_name = pairs[-num_pairs_to_show:]
    avg_sim_for_similar_names = np.mean([p['cos_sim'] for p in similar_by_name])
    avg_sim_for_dissimilar_names = np.mean([p['cos_sim'] for p in dissimilar_by_name])
    print(f"  - Avg. Cosine Sim for Top {num_pairs_to_show} Similar Names:   {avg_sim_for_similar_names:.4f}")
    print(f"  - Avg. Cosine Sim for Top {num_pairs_to_show} Dissimilar Names: {avg_sim_for_dissimilar_names:.4f}")
    print("-" * 30)


# 🔻🔻🔻
# REMOVED create_episode function.
# It has been moved to utils/data_utils.py
# 🔺🔺🔺


def train(model, training_tasks, loader, config):
    """
    Main training loop.
    """
    print("🚀 Starting meta-training...")
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])

    # 🔺 Use CosineAnnealingLR for a smooth decay over all epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    cls_task_pools = [pool for pool in training_tasks['classification'] if pool]
    reg_task_pools = [pool for pool in training_tasks['regression'] if pool]

    if not cls_task_pools and not reg_task_pools:
        print("❌ Error: No valid training tasks available. Aborting training.")
        return

    # Get the shuffle strategy string from the config
    shuffle_strategy = str(config.get('episodic_label_shuffle', 'no')).lower()
    print(f"✅ Episodic Label Shuffle strategy set to: '{shuffle_strategy}'")

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0

        # --- NEW: Determine shuffle status for THIS epoch ---
        should_shuffle_labels = False
        if shuffle_strategy == 'yes':
            should_shuffle_labels = True
        elif shuffle_strategy == 'mixed':
            should_shuffle_labels = (epoch % 2 == 0)  # Shuffle on even epochs, not on odd

        epoch_desc = f"Epoch {epoch + 1}/{config['epochs']}"
        if shuffle_strategy != 'no':
            epoch_desc += f" (Shuffle: {'ON' if should_shuffle_labels else 'OFF'})"
        # --- END NEW LOGIC ---

        progress_bar = tqdm(range(config['episodes_per_epoch']), desc=epoch_desc)

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
                # Pass the dynamically set shuffle flag
                episode = create_episode(
                    task_data_pool, config['num_shots_range'], config['num_queries'],
                    # 🔺 Make the task harder: 3-way to 10-way
                    num_ways_range=(3, 10), shuffle_labels=should_shuffle_labels
                )
            else:
                if len(task_data_pool) < config['num_shots_range'][1] + config['num_queries']:
                    episode = None
                else:
                    random.shuffle(task_data_pool)
                    num_shots = random.randint(config['num_shots_range'][0], config['num_shots_range'][1])

                    # --- FIX ---
                    # task_data_pool contains (seq, label, case_id) tuples.
                    # We must select the raw tuples, then re-format them to
                    # (seq, label) tuples for the model's forward pass.
                    support_set_raw = task_data_pool[:num_shots]
                    query_set_raw = task_data_pool[num_shots: num_shots + config['num_queries']]

                    support_set = [(s[0], s[1]) for s in support_set_raw]
                    query_set = [(q[0], q[1]) for q in query_set_raw]
                    # --- END FIX ---

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

        evaluate_embedding_quality(model, loader)

        scheduler.step()
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Model checkpoint saved to {checkpoint_path}")

    print("✅ Meta-training complete.")
