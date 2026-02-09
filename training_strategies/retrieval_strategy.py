import random
from contextlib import nullcontext
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from utils.retrieval_utils import find_knn_indices


def _encode_case_ids_to_int(case_ids_np: np.ndarray) -> torch.Tensor:
    """case_ids may be strings/objects -> map to contiguous ints for tensor ops."""
    uniq = list(dict.fromkeys(case_ids_np.tolist()))
    mapping = {cid: i for i, cid in enumerate(uniq)}
    return torch.tensor([mapping[cid] for cid in case_ids_np.tolist()], dtype=torch.long)


def _autocast_disabled_for(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=False)
    return nullcontext()


def _sample_balanced_classification_batch(
    task_pool, batch_size, min_per_class=2, max_classes=None
):
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
    if max_classes is not None:
        num_classes = min(num_classes, max(1, int(max_classes)))
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

    if chosen_labels:
        label_cycle = chosen_labels[:]
        random.shuffle(label_cycle)
        cycle_idx = 0
        while len(batch) < batch_size:
            lbl = label_cycle[cycle_idx % len(label_cycle)]
            batch.append(random.choice(by_label[lbl]))
            cycle_idx += 1

    return batch[:batch_size]


def _supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    case_ids_int: torch.Tensor,
    temperature: float = 0.07,
):
    """
    Supervised contrastive loss.
    Positives are same-label, different-case pairs.
    Same-case pairs are removed from denominator by heavy negative masking.
    """
    device = z.device
    temp = max(float(temperature), 1e-6)

    with _autocast_disabled_for(device):
        z = F.normalize(z.float(), p=2, dim=1)
        batch_size = z.size(0)

        logits = (z @ z.t()) / temp

        self_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        same_case = case_ids_int.view(-1, 1).eq(case_ids_int.view(1, -1))
        ignore = self_mask | same_case

        # Use a large finite negative value to avoid fp16 overflow / NaNs in AMP paths.
        logits = logits.masked_fill(ignore, -1e4)

        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.t()) & (~ignore)
        pos_counts = pos_mask.sum(dim=1)
        valid = pos_counts > 0
        if not valid.any():
            return None

        log_prob = F.log_softmax(logits, dim=1)
        loss_per = -(log_prob * pos_mask.float()).sum(dim=1) / pos_counts.clamp_min(1).float()
        return loss_per[valid].mean()


def _regression_neighbor_contrastive(
    z: torch.Tensor,
    y: torch.Tensor,
    case_ids_int: torch.Tensor,
    temperature: float = 0.07,
    pos_k: int = 2,
):
    """
    Target-neighborhood contrastive objective for regression.
    Positives per anchor are nearest labels in target space (excluding same case).
    """
    device = z.device
    temp = max(float(temperature), 1e-6)
    pos_k = max(int(pos_k), 1)

    with _autocast_disabled_for(device):
        z = F.normalize(z.float(), p=2, dim=1)
        y = y.float().view(-1)
        batch_size = z.size(0)

        logits = (z @ z.t()) / temp

        self_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        same_case = case_ids_int.view(-1, 1).eq(case_ids_int.view(1, -1))
        ignore = self_mask | same_case
        logits = logits.masked_fill(ignore, -1e4)

        log_prob = F.log_softmax(logits, dim=1)

        losses = []
        for i in range(batch_size):
            candidates = (~ignore[i]).nonzero(as_tuple=False).squeeze(1)
            if candidates.numel() == 0:
                continue
            diffs = (y[candidates] - y[i]).abs()
            k_eff = min(pos_k, int(candidates.numel()))
            positives = candidates[torch.topk(diffs, k_eff, largest=False).indices]
            losses.append(-log_prob[i, positives].mean())

        if not losses:
            return None
        return torch.stack(losses).mean()


def _nca_knn_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    case_ids_int: torch.Tensor,
    temperature: float = 0.07,
):
    """Supervised NCA-style objective: maximize same-label probability mass."""
    device = z.device
    temp = max(float(temperature), 1e-6)

    with _autocast_disabled_for(device):
        z = F.normalize(z.float(), p=2, dim=1)
        batch_size = z.size(0)
        logits = (z @ z.t()) / temp

        self_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        same_case = case_ids_int.view(-1, 1).eq(case_ids_int.view(1, -1))
        ignore = self_mask | same_case
        logits = logits.masked_fill(ignore, -1e4)

        log_prob = F.log_softmax(logits, dim=1)
        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.t()) & (~ignore)
        pos_counts = pos_mask.sum(dim=1)
        valid = pos_counts > 0
        if not valid.any():
            return None

        log_p_pos = torch.logsumexp(log_prob.masked_fill(~pos_mask, -1e4), dim=1)
        return (-log_p_pos[valid]).mean()


def _variance_loss(z: torch.Tensor, eps: float = 1e-4, target_std: float = 1.0):
    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(target_std - std))


def _covariance_loss(z: torch.Tensor):
    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)
    n, d = z.shape
    if n <= 1:
        return torch.tensor(0.0, device=z.device)
    cov = (z.t() @ z) / (n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).sum() / d


def run_retrieval_step(model, task_data_pool, task_type, config):
    progress_bar_task = f"retrieval_{task_type}"
    retrieval_k_train = int(config.get("retrieval_train_k", 5))
    retrieval_batch_size = int(config.get("retrieval_train_batch_size", 64))

    if len(task_data_pool) < retrieval_batch_size:
        return None, progress_bar_task

    if task_type == "classification":
        min_per_class = int(config.get("retrieval_min_per_class", 2))
        max_classes = config.get("retrieval_train_max_classes", None)
        batch_tasks_raw = _sample_balanced_classification_batch(
            task_data_pool,
            retrieval_batch_size,
            min_per_class=min_per_class,
            max_classes=max_classes,
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
    cls_pos_k_cfg = int(config.get("retrieval_cls_pos_k", 2))
    neg_pool_factor = max(1, int(config.get("retrieval_neg_pool_factor", 4)))
    neg_random_frac = float(config.get("retrieval_neg_random_frac", 0.25))
    neg_random_frac = min(max(neg_random_frac, 0.0), 1.0)
    pos_use_nearest_cfg = config.get("retrieval_pos_use_nearest", True)
    if isinstance(pos_use_nearest_cfg, str):
        pos_use_nearest = pos_use_nearest_cfg.strip().lower() in {"1", "true", "yes", "y", "on"}
    else:
        pos_use_nearest = bool(pos_use_nearest_cfg)

    contrastive_w = float(config.get("retrieval_contrastive_weight", 0.2))
    contrastive_temp = float(config.get("retrieval_contrastive_temp", 0.07))
    knn_aux_w = float(config.get("retrieval_knn_aux_weight", 0.0))
    contrastive_loss = None
    nca_loss = None
    labels_t = None
    case_ids_int = None

    if contrastive_w > 0 or (task_type == "classification" and knn_aux_w > 0):
        case_ids_int = _encode_case_ids_to_int(batch_case_ids).to(device)

    if task_type == "classification":
        labels_t = torch.as_tensor(batch_labels, dtype=torch.long, device=device)

    if contrastive_w > 0:
        if task_type == "classification":
            contrastive_loss = _supcon_loss(
                all_embeddings, labels_t, case_ids_int, temperature=contrastive_temp
            )
        else:
            y_t = torch.as_tensor(batch_labels, dtype=torch.float32, device=device)
            pos_k = int(config.get("retrieval_regression_pos_k", 2))
            contrastive_loss = _regression_neighbor_contrastive(
                all_embeddings, y_t, case_ids_int, temperature=contrastive_temp, pos_k=pos_k
            )

    if task_type == "classification" and knn_aux_w > 0 and labels_t is not None and case_ids_int is not None:
        nca_loss = _nca_knn_loss(
            all_embeddings,
            labels_t,
            case_ids_int,
            temperature=contrastive_temp,
        )

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
            if positive_indices.size == 0:
                continue

            with torch.no_grad():
                sims = (query_embedding_norm @ all_embeddings_norm_detached.t()).squeeze(0)

            pos_k = min(cls_pos_k_cfg, int(positive_indices.size), max(1, retrieval_k_train - 1))
            if pos_use_nearest:
                pos_candidates = torch.from_numpy(positive_indices).to(device)
                pos_sims = sims[pos_candidates]
                pos_rel = torch.topk(pos_sims, k=pos_k, largest=True).indices
                pos_tensor = pos_candidates[pos_rel]
            else:
                chosen_pos_idx = np.random.choice(positive_indices, size=pos_k, replace=False)
                pos_tensor = torch.from_numpy(chosen_pos_idx).to(device)

            neg_k = retrieval_k_train - pos_k
            if neg_k <= 0:
                continue

            with torch.no_grad():
                neg_mask = (batch_case_ids == query_case_id) | (batch_labels == query_label)
                neg_mask_idx = torch.from_numpy(np.where(neg_mask)[0]).to(device)

            neg_available_np = np.where(~neg_mask)[0]
            if neg_available_np.size == 0:
                continue

            hard_pool_k = min(int(neg_available_np.size), max(neg_k, neg_k * neg_pool_factor))
            hard_pool = find_knn_indices(
                query_embedding_norm,
                all_embeddings_norm_detached,
                k=hard_pool_k,
                indices_to_mask=neg_mask_idx,
            )

            rand_k = int(round(neg_k * neg_random_frac))
            if neg_k > 1:
                rand_k = min(rand_k, neg_k - 1)
            else:
                rand_k = 0
            hard_k = neg_k - rand_k

            hard_pool_np = hard_pool.detach().cpu().numpy() if hard_pool.numel() > 0 else np.array([], dtype=np.int64)
            if hard_pool_np.size < hard_k:
                hard_k = int(hard_pool_np.size)
                rand_k = neg_k - hard_k

            if hard_k > 0:
                hard_sel_np = np.random.choice(hard_pool_np, size=hard_k, replace=False)
            else:
                hard_sel_np = np.array([], dtype=np.int64)

            remaining_for_random = np.setdiff1d(neg_available_np, hard_sel_np, assume_unique=False)
            rand_k_eff = min(rand_k, int(remaining_for_random.size))
            if rand_k_eff > 0:
                rand_sel_np = np.random.choice(remaining_for_random, size=rand_k_eff, replace=False)
            else:
                rand_sel_np = np.array([], dtype=np.int64)

            neg_sel_np = np.concatenate([hard_sel_np, rand_sel_np]).astype(np.int64, copy=False)
            if neg_sel_np.size < neg_k:
                remaining_fill = np.setdiff1d(neg_available_np, neg_sel_np, assume_unique=False)
                need = neg_k - int(neg_sel_np.size)
                fill_k = min(need, int(remaining_fill.size))
                if fill_k > 0:
                    fill_np = np.random.choice(remaining_fill, size=fill_k, replace=False)
                    neg_sel_np = np.concatenate([neg_sel_np, fill_np]).astype(np.int64, copy=False)

            if neg_sel_np.size == 0:
                continue

            neg_indices = torch.from_numpy(neg_sel_np).to(device)

            support_indices = torch.cat([pos_tensor, neg_indices])[:retrieval_k_train]

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

    loss_out = None
    if queries_processed > 0:
        loss_out = total_loss_for_batch / queries_processed
        if contrastive_loss is not None:
            loss_out = loss_out + (contrastive_w * contrastive_loss)
    elif contrastive_loss is not None:
        loss_out = contrastive_w * contrastive_loss

    if nca_loss is not None and knn_aux_w > 0:
        if loss_out is None:
            loss_out = knn_aux_w * nca_loss
        else:
            loss_out = loss_out + (knn_aux_w * nca_loss)

    var_w = float(config.get("retrieval_var_weight", 0.0))
    cov_w = float(config.get("retrieval_cov_weight", 0.0))
    if loss_out is not None and (var_w > 0 or cov_w > 0):
        with _autocast_disabled_for(device):
            reg = torch.tensor(0.0, device=device)
            if var_w > 0:
                reg = reg + (var_w * _variance_loss(all_embeddings))
            if cov_w > 0:
                reg = reg + (cov_w * _covariance_loss(all_embeddings))
        loss_out = loss_out + reg

    return loss_out, progress_bar_task
