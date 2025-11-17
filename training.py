# File: training.py
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import itertools
import numpy as np

# --- Import from project files ---
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from data_generator import XESLogLoader
from utils.data_utils import create_episode

# Use a try-except block for the optional dependency
try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    levenshtein_distance = None


def evaluate_embedding_quality(model, loader: XESLogLoader):
    # ... (this function is unchanged) ...
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


def train(model, training_tasks, loader, config):
    """
    Main training loop.
    """
    print("🚀 Starting meta-training...")
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    cls_task_pools = [pool for pool in training_tasks['classification'] if pool]
    reg_task_pools = [pool for pool in training_tasks['regression'] if pool]

    if not cls_task_pools and not reg_task_pools:
        print("❌ Error: No valid training tasks available. Aborting training.")
        return

    # --- 🔻 NEW: Training Strategy Setup 🔻 ---
    training_strategy = config.get('training_strategy', 'episodic')
    retrieval_k_train = config.get('retrieval_train_k', 5)
    retrieval_batch_size = config.get('retrieval_train_batch_size', 64)
    print(f"✅ Training Strategy: '{training_strategy}'")
    if training_strategy in ['retrieval', 'mixed']:
        print(f"  - Retrieval k (train): {retrieval_k_train}")
        print(f"  - Retrieval batch size (train): {retrieval_batch_size}")
    # --- 🔺 END NEW 🔺 ---

    shuffle_strategy = str(config.get('episodic_label_shuffle', 'no')).lower()
    print(f"✅ Episodic Label Shuffle strategy set to: '{shuffle_strategy}'")

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0

        should_shuffle_labels = False
        if shuffle_strategy == 'yes':
            should_shuffle_labels = True
        elif shuffle_strategy == 'mixed':
            should_shuffle_labels = (epoch % 2 == 0)

        epoch_desc = f"Epoch {epoch + 1}/{config['epochs']}"
        if shuffle_strategy != 'no':
            epoch_desc += f" (Shuffle: {'ON' if should_shuffle_labels else 'OFF'})"

        progress_bar = tqdm(range(config['episodes_per_epoch']), desc=epoch_desc)

        for step in progress_bar:
            # --- 🔻 NEW: Determine which strategy to use for this step 🔻 ---
            current_train_mode = training_strategy
            if training_strategy == 'mixed':
                current_train_mode = 'retrieval' if step % 2 == 0 else 'episodic'
            # --- 🔺 END NEW 🔺 ---

            # --- Pick task type and data pool (common to both strategies) ---
            task_type = random.choice(['classification', 'regression'])
            if task_type == 'classification' and cls_task_pools:
                task_data_pool = random.choice(cls_task_pools)
            elif task_type == 'regression' and reg_task_pools:
                task_data_pool = random.choice(reg_task_pools)
            else:
                task_type = 'regression' if reg_task_pools else 'classification'
                task_data_pool = random.choice(reg_task_pools if reg_task_pools else cls_task_pools)

            if not task_data_pool: continue

            optimizer.zero_grad(set_to_none=True)

            # --- 🔻 STRATEGY 1: Standard 'Episodic' Training 🔻 ---
            if current_train_mode == 'episodic':
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

                if episode is None or not episode[0] or not episode[1]: continue

                support_set, query_set = episode
                predictions, true_labels = model(support_set, query_set, task_type)

                if predictions is None: continue

                if task_type == 'classification':
                    loss = F.cross_entropy(predictions, true_labels, ignore_index=-100, label_smoothing=0.05)
                else:
                    loss = F.huber_loss(predictions.squeeze(), true_labels)

                progress_bar_task = task_type

            # --- 🔻 STRATEGY 2: New 'Retrieval' Training 🔻 ---
            elif current_train_mode == 'retrieval':
                if len(task_data_pool) < retrieval_batch_size:
                    continue

                # 1. Sample a large batch from the pool
                batch_tasks_raw = random.sample(task_data_pool, retrieval_batch_size)
                batch_prefixes = [t[0] for t in batch_tasks_raw]
                batch_labels = np.array([t[1] for t in batch_tasks_raw])
                batch_case_ids = np.array([t[2] for t in batch_tasks_raw])

                # 2. Embed the entire batch
                with torch.no_grad():  # Embeddings are detached for k-NN search
                    all_embeddings = model._process_batch(batch_prefixes)
                    all_embeddings_norm = F.normalize(all_embeddings, p=2, dim=1)

                # 3. Re-attach embeddings for gradient flow
                # We do this by calling _process_batch again, but this time
                # gradients will flow *only* from the loss calculation.
                # This is a trick to avoid OOM by not detaching.
                # A simpler way: just let gradients flow from the start.
                all_embeddings = model._process_batch(batch_prefixes)

                device = all_embeddings.device
                total_loss_for_batch = 0.0
                queries_processed = 0

                # 4. Iterate through batch, treating each as a query
                for i in range(retrieval_batch_size):
                    query_label = batch_labels[i]
                    query_case_id = batch_case_ids[i]
                    query_embedding = all_embeddings[i:i + 1]  # (1, D)

                    # --- Build Support Set (k-NN + 1 Guaranteed Positive) ---
                    # a) Find one guaranteed positive (same label, different case)
                    positive_mask = (batch_labels == query_label) & (batch_case_ids != query_case_id)
                    positive_indices = np.where(positive_mask)[0]

                    if len(positive_indices) == 0:
                        continue  # No other-case positive in batch, skip query

                    chosen_positive_idx = random.choice(positive_indices)

                    # b) Find (k-1) nearest neighbors (can be anything)
                    with torch.no_grad():  # k-NN search doesn't need gradients
                        query_embedding_norm = all_embeddings_norm[i:i + 1]
                        sims = query_embedding_norm @ all_embeddings_norm.T  # (1, B)

                        # Mask out self, the chosen positive, and all same-case items
                        same_case_mask = (batch_case_ids == query_case_id)
                        indices_to_mask = np.where(same_case_mask)[0]
                        mask_tensor = torch.from_numpy(indices_to_mask).to(device)

                        sims[0, mask_tensor] = -float('inf')
                        sims[0, chosen_positive_idx] = -float('inf')
                        # sims[0, i] = -float('inf') # Already masked by same_case_mask

                        valid_neighbor_count = (sims[0] > -float('inf')).sum().item()
                        num_neighbors_to_find = min(retrieval_k_train - 1, valid_neighbor_count)

                    if num_neighbors_to_find < 0: continue

                    neighbor_indices = torch.topk(sims.squeeze(0), num_neighbors_to_find).indices

                    # c) Combine to form final support set
                    support_indices = torch.cat([neighbor_indices, torch.tensor([chosen_positive_idx], device=device)])
                    support_embeddings = all_embeddings[support_indices]
                    support_labels_list = batch_labels[support_indices.cpu().numpy()]  # Get labels as numpy/list

                    # 5. Calculate Loss using the head
                    if task_type == 'classification':
                        support_labels_tensor = torch.LongTensor(support_labels_list).to(device)
                        logits, proto_classes = model.proto_head.forward_classification(
                            support_embeddings, support_labels_tensor, query_embedding
                        )
                        if logits is None: continue

                        label_map = {orig.item(): new for new, orig in enumerate(proto_classes)}
                        mapped_label = torch.tensor([label_map.get(query_label, -100)], device=device, dtype=torch.long)

                        if mapped_label.item() == -100:
                            continue  # Should not happen due to guaranteed positive, but a safe check

                        loss = F.cross_entropy(logits, mapped_label, label_smoothing=0.05)

                    else:  # Regression
                        support_labels_tensor = torch.as_tensor(support_labels_list, dtype=torch.float32, device=device)
                        query_label_tensor = torch.as_tensor([query_label], dtype=torch.float32, device=device)
                        prediction = model.proto_head.forward_regression(
                            support_embeddings, support_labels_tensor, query_embedding
                        )
                        loss = F.huber_loss(prediction.squeeze(), query_label_tensor.squeeze())

                    if not torch.isnan(loss):
                        total_loss_for_batch = total_loss_for_batch + loss
                        queries_processed += 1

                # 6. Average loss for the batch and backprop
                if queries_processed > 0:
                    loss = total_loss_for_batch / queries_processed
                else:
                    loss = None  # No valid queries found

                progress_bar_task = f"retrieval_{task_type}"

            # --- 🔻 COMMON: Loss Backward and Step 🔻 ---
            else:
                loss = None
                progress_bar_task = "skip"

            if loss is not None and not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}" if loss else "N/A", task=progress_bar_task)
            # --- 🔺 END OF LOOP 🔺 ---

        avg_loss = total_loss / config['episodes_per_epoch'] if config['episodes_per_epoch'] > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch + 1} finished. Average Loss: {avg_loss:.4f} | Current LR: {current_lr:.6f}")

        evaluate_embedding_quality(model, loader)

        scheduler.step()
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Model checkpoint saved to {checkpoint_path}")

    print("✅ Meta-training complete.")
