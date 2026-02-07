import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from utils.retrieval_utils import find_knn_indices


def _sample_balanced_classification_batch(task_pool, batch_size, min_per_class=2):
    """
    Ensure selected classes appear at least min_per_class times, ideally across cases.
    Fall back to random sampling when constraints cannot be met.
    """
    by_label = defaultdict(list)
    for seq, label, case_id in task_pool:
        if label is None:
            continue
        if int(label) == -100:
            continue
        by_label[int(label)].append((seq, int(label), case_id))

    eligible = []
    for label, items in by_label.items():
        if len(items) < min_per_class:
            continue
        if len({cid for _, _, cid in items}) < 2:
            continue
        eligible.append(label)

    if not eligible:
        return random.sample(task_pool, batch_size)

    num_classes = min(len(eligible), max(1, batch_size // min_per_class))
    chosen_labels = random.sample(eligible, num_classes)

    batch = []
    for label in chosen_labels:
        items = by_label[label]
        by_case = defaultdict(list)
        for item in items:
            by_case[item[2]].append(item)
        cases = list(by_case.keys())
        random.shuffle(cases)

        for case_id in cases[:min_per_class]:
            batch.append(random.choice(by_case[case_id]))

        while sum(1 for item in batch if item[1] == label) < min_per_class:
            batch.append(random.choice(items))

    while len(batch) < batch_size:
        batch.append(random.choice(task_pool))

    return batch[:batch_size]


def run_retrieval_step(model, task_data_pool, task_type, config):
    progress_bar_task = f"retrieval_{task_type}"
    retrieval_k_train = int(config.get("retrieval_train_k", 5))
    retrieval_batch_size = int(config.get("retrieval_train_batch_size", 64))

    if len(task_data_pool) < retrieval_batch_size:
        return None, progress_bar_task

    if task_type == "classification":
        min_per_class = int(config.get("retrieval_min_per_class", 2))
        batch_tasks_raw = _sample_balanced_classification_batch(
            task_data_pool, retrieval_batch_size, min_per_class=min_per_class
        )
    else:
        batch_tasks_raw = random.sample(task_data_pool, retrieval_batch_size)

    batch_prefixes = [t[0] for t in batch_tasks_raw]
    batch_labels = np.array([t[1] for t in batch_tasks_raw])
    batch_case_ids = np.array([t[2] for t in batch_tasks_raw], dtype=object)

    all_embeddings = model._process_batch(batch_prefixes)
    device = all_embeddings.device

    all_embeddings_norm = F.normalize(all_embeddings, p=2, dim=1)
    all_embeddings_norm_detached = all_embeddings_norm.detach()

    total_loss_for_batch = 0.0
    queries_processed = 0

    for i in range(retrieval_batch_size):
        query_label = batch_labels[i]
        query_case_id = batch_case_ids[i]
        query_embedding = all_embeddings[i : i + 1]

        with torch.no_grad():
            query_embedding_norm = all_embeddings_norm_detached[i : i + 1]
            same_case_indices = np.where(batch_case_ids == query_case_id)[0]
            mask_tensor = torch.from_numpy(same_case_indices).to(device)

        if task_type == "classification":
            if int(query_label) == -100:
                continue

            positive_mask = (batch_labels == query_label) & (batch_case_ids != query_case_id)
            positive_indices = np.where(positive_mask)[0]
            if len(positive_indices) == 0:
                continue

            chosen_positive_idx = int(random.choice(positive_indices))
            with torch.no_grad():
                mask_plus_pos = torch.cat(
                    [mask_tensor, torch.tensor([chosen_positive_idx], device=device)]
                )

            neighbor_indices = find_knn_indices(
                query_embedding_norm,
                all_embeddings_norm_detached,
                k=max(0, retrieval_k_train - 1),
                indices_to_mask=mask_plus_pos,
            )
            support_indices = torch.cat(
                [neighbor_indices, torch.tensor([chosen_positive_idx], device=device)]
            )

            support_embeddings = all_embeddings[support_indices]
            support_labels_list = batch_labels[support_indices.cpu().numpy()]
            support_labels_tensor = torch.as_tensor(support_labels_list, dtype=torch.long, device=device)

            logits, proto_classes, _ = model.proto_head.forward_classification(
                support_embeddings, support_labels_tensor, query_embedding
            )
            if logits is None:
                continue

            label_map = {orig.item(): new for new, orig in enumerate(proto_classes)}
            mapped_label = torch.tensor(
                [label_map.get(int(query_label), -100)], device=device, dtype=torch.long
            )
            if mapped_label.item() == -100:
                continue

            loss = F.cross_entropy(logits, mapped_label, label_smoothing=0.05)

        else:
            neighbor_indices = find_knn_indices(
                query_embedding_norm,
                all_embeddings_norm_detached,
                k=retrieval_k_train,
                indices_to_mask=mask_tensor,
            )
            if neighbor_indices.numel() == 0:
                continue

            support_embeddings = all_embeddings[neighbor_indices]
            support_labels_list = batch_labels[neighbor_indices.cpu().numpy()]
            support_labels_tensor = torch.as_tensor(support_labels_list, dtype=torch.float32, device=device)
            query_label_tensor = torch.as_tensor([query_label], dtype=torch.float32, device=device)

            prediction, _ = model.proto_head.forward_regression(
                support_embeddings, support_labels_tensor, query_embedding
            )
            loss = F.huber_loss(prediction.squeeze(), query_label_tensor.squeeze())

        if not torch.isnan(loss):
            total_loss_for_batch = total_loss_for_batch + loss
            queries_processed += 1

    if queries_processed > 0:
        loss_out = total_loss_for_batch / queries_processed
    else:
        loss_out = None

    return loss_out, progress_bar_task
