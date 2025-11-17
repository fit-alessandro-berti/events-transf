# File: training_strategies/retrieval_strategy.py
import random
import torch
import torch.nn.functional as F
import numpy as np

# --- Import from project files ---
from utils.retrieval_utils import find_knn_indices


def run_retrieval_step(model, task_data_pool, task_type, config):
    """
    Runs a single retrieval-based (hard negative mining) training step.

    Returns:
        (torch.Tensor or None): The computed loss for the step, or None if step failed.
        (str): A string descriptor for the task (e.g., "retrieval_classification").
    """
    progress_bar_task = f"retrieval_{task_type}"
    retrieval_k_train = config.get('retrieval_train_k', 5)
    retrieval_batch_size = config.get('retrieval_train_batch_size', 64)

    if len(task_data_pool) < retrieval_batch_size:
        return None, progress_bar_task

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

            # Mask out self, the chosen positive, and all same-case items
            same_case_mask = (batch_case_ids == query_case_id)
            same_case_indices = np.where(same_case_mask)[0]

            # Combine all indices to mask
            indices_to_mask_np = np.append(same_case_indices, chosen_positive_idx)
            mask_tensor = torch.from_numpy(indices_to_mask_np).to(device)

            num_neighbors_to_find = retrieval_k_train - 1

        if num_neighbors_to_find < 0:
            neighbor_indices = torch.tensor([], dtype=torch.long, device=device)
        else:
            neighbor_indices = find_knn_indices(
                query_embedding_norm,
                all_embeddings_norm,
                k=num_neighbors_to_find,
                indices_to_mask=mask_tensor
            )

        # c) Combine to form final support set
        support_indices = torch.cat([neighbor_indices, torch.tensor([chosen_positive_idx], device=device)])

        support_embeddings = all_embeddings[support_indices]
        support_labels_list = batch_labels[support_indices.cpu().numpy()]  # Get labels as numpy/list

        # 5. Calculate Loss using the head
        if task_type == 'classification':
            support_labels_tensor = torch.LongTensor(support_labels_list).to(device)
            # 🔻 MODIFIED: Unpack confidence (even if unused) 🔻
            logits, proto_classes, _ = model.proto_head.forward_classification(
                support_embeddings, support_labels_tensor, query_embedding
            )
            # 🔺 END MODIFIED 🔺
            if logits is None: continue

            label_map = {orig.item(): new for new, orig in enumerate(proto_classes)}
            mapped_label = torch.tensor([label_map.get(query_label, -100)], device=device, dtype=torch.long)

            if mapped_label.item() == -100:
                continue

            loss = F.cross_entropy(logits, mapped_label, label_smoothing=0.05)

        else:  # Regression
            support_labels_tensor = torch.as_tensor(support_labels_list, dtype=torch.float32, device=device)
            query_label_tensor = torch.as_tensor([query_label], dtype=torch.float32, device=device)
            # 🔻 MODIFIED: Unpack confidence (even if unused) 🔻
            prediction, _ = model.proto_head.forward_regression(
                support_embeddings, support_labels_tensor, query_embedding
            )
            # 🔺 END MODIFIED 🔺
            loss = F.huber_loss(prediction.squeeze(), query_label_tensor.squeeze())

        if not torch.isnan(loss):
            total_loss_for_batch = total_loss_for_batch + loss
            queries_processed += 1

    # 6. Average loss for the batch
    if queries_processed > 0:
        loss = total_loss_for_batch / queries_processed
    else:
        loss = None  # No valid queries found

    return loss, progress_bar_task
