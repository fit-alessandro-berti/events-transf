# File: evaluation/eval_retrieval.py
import torch
import torch.nn.functional as F
import random
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tqdm import tqdm

# --- Import from project files ---
from time_transf import inverse_transform_time
# 🔻 NEW IMPORT 🔻
from utils.retrieval_utils import find_knn_indices


# 🔺 END NEW 🔺


def _report_similarity_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    max_knn_queries=1000,
    knn_k_list=(5, 10),
    label: str = None
):
    """
    Reports quick, mean-aggregated similarity metrics for classification embeddings.
    Uses mean-centered cosine similarity to reduce anisotropy effects.
    """
    if embeddings.numel() == 0 or labels.numel() == 0:
        print("  - Similarity metrics: skipped (empty embeddings/labels).")
        return

    device = embeddings.device
    labels = labels.to(device)

    # Mean-center then re-normalize to reduce cosine anisotropy
    mean_emb = embeddings.mean(dim=0, keepdim=True)
    centered = embeddings - mean_emb
    centered = F.normalize(centered, p=2, dim=1)

    # Build centroids via scatter-add for speed
    unique_labels, inverse = torch.unique(labels, sorted=True, return_inverse=True)
    num_classes = unique_labels.numel()
    if num_classes < 2:
        print(f"  - Similarity metrics: skipped (num_classes={num_classes}).")
        return

    n, d = centered.shape
    centroids = torch.zeros((num_classes, d), device=device)
    counts = torch.zeros((num_classes,), device=device)
    centroids.scatter_add_(0, inverse[:, None].expand(-1, d), centered)
    counts.scatter_add_(0, inverse, torch.ones((n,), device=device))
    centroids = centroids / counts[:, None].clamp_min(1.0)
    centroids = F.normalize(centroids, p=2, dim=1)

    # 1) Intra-class cohesion: cosine to own centroid
    sims_to_own = (centered * centroids[inverse]).sum(dim=1)
    intra_mean = sims_to_own.mean().item()

    # 2) Inter-class centroid cosine (mean over pairs)
    centroid_sims = centroids @ centroids.T
    triu_idx = torch.triu_indices(num_classes, num_classes, offset=1, device=device)
    inter_mean = centroid_sims[triu_idx[0], triu_idx[1]].mean().item()

    # 3) Class-level margin: centroid self-sim (1.0) minus max sim to other centroids
    centroid_sims_offdiag = centroids @ centroids.T
    centroid_sims_offdiag.fill_diagonal_(-float('inf'))
    max_other_centroid_sim = centroid_sims_offdiag.max(dim=1).values
    margin_mean = (1.0 - max_other_centroid_sim).mean().item()

    # 4) kNN purity (sampled for speed)
    knn_purity = {}
    max_queries = min(max_knn_queries, n)
    if max_queries < n:
        query_idx = torch.randperm(n, device=device)[:max_queries]
    else:
        query_idx = torch.arange(n, device=device)

    query_emb = centered[query_idx]
    query_labels = labels[query_idx]
    sims = query_emb @ centered.T
    sims[torch.arange(max_queries, device=device), query_idx] = -float('inf')

    for k in knn_k_list:
        k_eff = min(k, n - 1)
        if k_eff <= 0:
            knn_purity[k] = float('nan')
            continue
        topk_idx = torch.topk(sims, k_eff, dim=1).indices
        neighbor_labels = labels[topk_idx]
        purity = (neighbor_labels == query_labels[:, None]).float().mean().item()
        knn_purity[k] = purity

    label_prefix = f"[{label}] " if label else ""
    print(
        "  - "
        + label_prefix
        + "Similarity metrics (mean, mean-centered cosine): "
        f"intra_centroid_cos={intra_mean:.4f} | "
        f"inter_centroid_cos={inter_mean:.4f} | "
        f"centroid_margin={margin_mean:.4f} | "
        + " ".join([f"knn_purity@{k}={knn_purity[k]:.4f}" for k in knn_k_list])
    )


def _report_inter_expert_metrics(expert_task_embeddings, task_type: str):
    """
    Reports pairwise (dis)similarity between experts' embeddings.
    Metrics:
      - mean_cos: mean cosine similarity of aligned sample embeddings
      - mean_l2: mean L2 distance of aligned sample embeddings
      - centroid_cos_mean (classification only): mean cosine similarity of per-class centroids
    """
    expert_names = list(expert_task_embeddings.keys())
    if len(expert_names) < 2:
        return

    pairs = []
    for i in range(len(expert_names)):
        for j in range(i + 1, len(expert_names)):
            pairs.append((expert_names[i], expert_names[j]))

    for expert_a, expert_b in pairs:
        data_a = expert_task_embeddings[expert_a].get(task_type)
        data_b = expert_task_embeddings[expert_b].get(task_type)
        if data_a is None or data_b is None:
            continue

        emb_a, labels_a, _ = data_a
        emb_b, labels_b, _ = data_b

        if emb_a.shape[0] != emb_b.shape[0]:
            print(
                f"  - Inter-expert metrics skipped for {expert_a} vs {expert_b} "
                f"({task_type}): sample count mismatch."
            )
            continue

        # Mean cosine similarity of aligned samples (embeddings already normalized)
        mean_cos = (emb_a * emb_b).sum(dim=1).mean().item()
        mean_l2 = torch.norm(emb_a - emb_b, dim=1).mean().item()

        if task_type == 'classification':
            # Compute per-class centroids for each expert and compare
            unique_labels = torch.unique(labels_a, sorted=True)
            if unique_labels.numel() < 2:
                centroid_cos_mean = float('nan')
            else:
                centroids_a = []
                centroids_b = []
                for lbl in unique_labels:
                    mask_a = labels_a == lbl
                    mask_b = labels_b == lbl
                    if not mask_a.any() or not mask_b.any():
                        continue
                    ca = emb_a[mask_a].mean(dim=0, keepdim=True)
                    cb = emb_b[mask_b].mean(dim=0, keepdim=True)
                    ca = F.normalize(ca, p=2, dim=1)
                    cb = F.normalize(cb, p=2, dim=1)
                    centroids_a.append(ca)
                    centroids_b.append(cb)
                if not centroids_a:
                    centroid_cos_mean = float('nan')
                else:
                    centroids_a = torch.cat(centroids_a, dim=0)
                    centroids_b = torch.cat(centroids_b, dim=0)
                    centroid_cos_mean = (centroids_a * centroids_b).sum(dim=1).mean().item()

            print(
                f"  - Inter-expert metrics ({task_type}) {expert_a} vs {expert_b}: "
                f"mean_cos={mean_cos:.4f} | mean_l2={mean_l2:.4f} | "
                f"centroid_cos_mean={centroid_cos_mean:.4f}"
            )
        else:
            print(
                f"  - Inter-expert metrics ({task_type}) {expert_a} vs {expert_b}: "
                f"mean_cos={mean_cos:.4f} | mean_l2={mean_l2:.4f}"
            )


def _report_rf_metrics(expert_name: str, task_type: str, embeddings: torch.Tensor, labels: torch.Tensor):
    """
    Train/test split (80/20) and evaluate RandomForest on expert embeddings.
    """
    x = embeddings.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()

    num_samples = x.shape[0]
    if num_samples < 2:
        print(f"  - [{expert_name}] RF metrics skipped ({task_type}): not enough samples.")
        return

    if task_type == 'classification':
        unique_labels, counts = np.unique(y, return_counts=True)
        if unique_labels.size < 2:
            print(f"  - [{expert_name}] RF metrics skipped (classification): only one class.")
            return
        # Stratify only if every class has at least 2 samples
        stratify = y if counts.min() >= 2 else None
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42, stratify=stratify
            )
        except ValueError:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42, stratify=None
            )

        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)
        acc = accuracy_score(y_test, preds)
        print(f"  - [{expert_name}] RF (classification, 80/20): Accuracy={acc:.4f} (n={len(y_test)})")
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        reg = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )
        reg.fit(x_train, y_train)
        preds = reg.predict(x_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        print(
            f"  - [{expert_name}] RF (regression, 80/20): "
            f"MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f} (n={len(y_test)})"
        )


def _get_all_test_embeddings(model, test_tasks_list, batch_size=64):
    """
    Helper function to compute embeddings for all (prefix, label, case_id) tuples.
    (Moved from testing.py)

    *** ASSUMPTION ***: This function assumes test_tasks_list contains tuples of
    (prefix, label, case_id) as generated by a modified get_task_data function.

    - Calls `model._process_batch` to build embeddings.
      For MoEModel, this returns average embeddings; for MetaLearner,
      it returns the expert's own embeddings.
    """
    all_embeddings = []
    all_labels = []
    all_case_ids = []  # <-- For contamination fix

    device = next(model.parameters()).device
    model.eval()

    try:
        # Check if data has 3 items (prefix, label, case_id)
        _ = test_tasks_list[0][2]
    except (IndexError, TypeError):
        print("\n" + "=" * 50)
        print("❌ ERROR in _get_all_test_embeddings:")
        print("Test data does not contain case_ids.")
        print("Please modify get_task_data in data_generator.py to return:")
        print("(prefix, label, case_id) tuples.")
        print("Aborting retrieval-augmented evaluation.")
        print("=" * 50 + "\n")
        return None, None, None

    with torch.no_grad():
        for i in tqdm(range(0, len(test_tasks_list), batch_size), desc="Pre-computing test embeddings"):
            batch_tasks = test_tasks_list[i:i + batch_size]
            sequences = [t[0] for t in batch_tasks]
            labels = [t[1] for t in batch_tasks]
            case_ids = [t[2] for t in batch_tasks]  # <-- Get case_ids

            if not sequences: continue

            # Use the model's internal processing function
            # For MoEModel, this returns the average embedding
            encoded_batch = model._process_batch(sequences)

            all_embeddings.append(encoded_batch.cpu())
            all_labels.extend(labels)
            all_case_ids.extend(case_ids)  # <-- Store case_ids

    if not all_embeddings:
        return None, None, None

    all_embeddings_tensor = torch.cat(all_embeddings, dim=0).to(device)
    all_labels_tensor = torch.as_tensor(all_labels, device=device)
    all_case_ids_array = np.array(all_case_ids)  # Use numpy for easy string comparison

    return all_embeddings_tensor, all_labels_tensor, all_case_ids_array


def evaluate_retrieval_augmented(
    model,
    test_tasks,
    num_retrieval_k_list,
    num_test_queries=200,
    candidate_percentages=None
):
    """
    Retrieval-Augmented evaluation.
    (Moved from testing.py)

    1. Computes all test embeddings.
    2. For each query, finds k-NN *from other cases* to form the support set.
    3. Runs predictions per expert (MoE) or single model (non-MoE).

    candidate_percentages: list of percentages of the candidate pool to sample
    for k-NN (100 = full pool).
    """
    print("\n🔬 Starting Retrieval-Augmented Evaluation...")
    model.eval()

    if not candidate_percentages:
        candidate_percentages = [100]

    if hasattr(model, "experts"):
        num_experts = len(model.experts)
    else:
        num_experts = 1

    if num_experts > 1:
        experts_to_eval = [(f"Expert {i}", model.experts[i]) for i in range(num_experts)]
        print(f"  - (MoE) Running k-NN eval for all {num_experts} experts.")
    else:
        experts_to_eval = [("Expert 0", model)]

    expert_task_embeddings = {}

    # --- 1. Pre-compute all embeddings ---
    for expert_name, expert in experts_to_eval:
        expert_task_embeddings[expert_name] = {}

        for task_type, task_data in test_tasks.items():
            if not task_data:
                print(f"Skipping {task_type}: No test data available.")
                continue

            embeddings, labels, case_ids = _get_all_test_embeddings(expert, task_data)

            if embeddings is None:  # Error already printed in helper
                return

            # --- Quick sanity checks ---
            try:
                has_nan = torch.isnan(embeddings).any().item()
                all_finite = torch.isfinite(embeddings).all().item()
                print(f"  - [{expert_name}] Embedding sanity: has_nan={has_nan}, all_finite={all_finite}")
            except Exception as e:
                print(f"  - [{expert_name}] Embedding sanity check failed: {e}")

            if case_ids is not None:
                unique_cases, case_counts = np.unique(case_ids, return_counts=True)
                print(f"  - [{expert_name}] Case ID stats: unique_cases={len(unique_cases)}, total_tasks={len(case_ids)}")
                if len(unique_cases) > 0:
                    top_k = min(5, len(unique_cases))
                    top_idx = np.argsort(case_counts)[-top_k:][::-1]
                    top_cases = [(unique_cases[i], int(case_counts[i])) for i in top_idx]
                    print(f"  - [{expert_name}] Top case_id counts: {top_cases}")
            # --- End sanity checks ---

            # L2-normalize for efficient cosine similarity
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Report quick similarity metrics for classification only
            if task_type == 'classification':
                _report_similarity_metrics(embeddings, labels, label=expert_name)

            expert_task_embeddings[expert_name][task_type] = (embeddings, labels, case_ids)
            print(f"  - [{expert_name}] Pre-computed {embeddings.shape[0]} embeddings for {task_type}.")

    # Report inter-expert (dis)similarity metrics once embeddings are ready
    for task_type in test_tasks.keys():
        _report_inter_expert_metrics(expert_task_embeddings, task_type)

    # --- 2. Evaluate using k-NN retrieval ---
    for task_type in test_tasks.keys():
        available_experts = []
        for expert_name, expert in experts_to_eval:
            if task_type in expert_task_embeddings.get(expert_name, {}):
                embeddings, labels, case_ids = expert_task_embeddings[expert_name][task_type]
                available_experts.append((expert_name, expert, embeddings, labels, case_ids))

        if not available_experts:
            continue

        base_num_samples = available_experts[0][2].shape[0]
        if base_num_samples < 2:
            print(f"Skipping {task_type}: Not enough samples to evaluate.")
            continue

        num_queries = min(num_test_queries, base_num_samples)
        base_query_indices = random.sample(range(base_num_samples), num_queries)

        candidate_pool_masks = {}
        for pct in candidate_percentages:
            if pct >= 100:
                candidate_pool_masks[pct] = None
            elif pct <= 0:
                candidate_pool_masks[pct] = np.arange(base_num_samples)
            else:
                sample_size = int(np.ceil(base_num_samples * (pct / 100.0)))
                sample_size = max(1, min(sample_size, base_num_samples))
                if sample_size == base_num_samples:
                    candidate_pool_masks[pct] = None
                else:
                    candidate_pool_indices = np.random.choice(
                        base_num_samples,
                        size=sample_size,
                        replace=False
                    )
                    mask_np = np.ones(base_num_samples, dtype=bool)
                    mask_np[candidate_pool_indices] = False
                    candidate_pool_masks[pct] = np.where(mask_np)[0]

        for expert_name, expert, all_embeddings, all_labels, all_case_ids in available_experts:
            print(f"\n--- Evaluating task: {task_type} | {expert_name} ---")

            num_total_samples = all_embeddings.shape[0]
            if num_total_samples < 2:
                print(f"Skipping {task_type} | {expert_name}: Not enough samples to evaluate.")
                continue

            if num_total_samples != base_num_samples:
                print(
                    f"  - Warning: {expert_name} has {num_total_samples} samples; "
                    f"expected {base_num_samples}. Re-sampling queries for this expert."
                )
                num_queries = min(num_test_queries, num_total_samples)
                query_indices = random.sample(range(num_total_samples), num_queries)
                expert_candidate_pool_masks = {}
                for pct in candidate_percentages:
                    if pct >= 100:
                        expert_candidate_pool_masks[pct] = None
                    elif pct <= 0:
                        expert_candidate_pool_masks[pct] = np.arange(num_total_samples)
                    else:
                        sample_size = int(np.ceil(num_total_samples * (pct / 100.0)))
                        sample_size = max(1, min(sample_size, num_total_samples))
                        if sample_size == num_total_samples:
                            expert_candidate_pool_masks[pct] = None
                        else:
                            candidate_pool_indices = np.random.choice(
                                num_total_samples,
                                size=sample_size,
                                replace=False
                            )
                            mask_np = np.ones(num_total_samples, dtype=bool)
                            mask_np[candidate_pool_indices] = False
                            expert_candidate_pool_masks[pct] = np.where(mask_np)[0]
            else:
                query_indices = base_query_indices
                expert_candidate_pool_masks = candidate_pool_masks

            for pct in candidate_percentages:
                print(f"\n  - Candidate pool sampling: {pct}%")

                non_candidate_indices_np = expert_candidate_pool_masks.get(pct)
                if non_candidate_indices_np is None:
                    non_candidate_indices_tensor = None
                else:
                    non_candidate_indices_tensor = torch.from_numpy(non_candidate_indices_np).to(
                        all_embeddings.device
                    )

                if non_candidate_indices_tensor is not None:
                    print(f"  - Candidate pool size: {num_total_samples - len(non_candidate_indices_np)} / {num_total_samples}")

                for k in num_retrieval_k_list:
                    if k >= num_total_samples:
                        print(f"Skipping [{expert_name} | k={k} | pct={pct}%]: k is larger than total samples.")
                        continue

                    all_preds, all_true_labels, all_confidences = [], [], []

                    for query_idx in query_indices:
                        query_embedding = all_embeddings[query_idx:query_idx + 1]  # [1, D]
                        query_label = all_labels[query_idx]

                        query_case_id = all_case_ids[query_idx]
                        same_case_indices_np = np.where(all_case_ids == query_case_id)[0]

                        if non_candidate_indices_tensor is None:
                            if same_case_indices_np.size == 0:
                                mask_tensor = None
                            else:
                                mask_tensor = torch.from_numpy(same_case_indices_np).to(all_embeddings.device)
                        else:
                            if same_case_indices_np.size == 0:
                                mask_tensor = non_candidate_indices_tensor
                            else:
                                same_case_tensor = torch.from_numpy(same_case_indices_np).to(all_embeddings.device)
                                mask_tensor = torch.cat([non_candidate_indices_tensor, same_case_tensor])

                        top_k_indices = find_knn_indices(
                            query_embedding,
                            all_embeddings,
                            k=k,
                            indices_to_mask=mask_tensor
                        )

                        if top_k_indices.numel() == 0:
                            continue  # Not enough valid support items found

                        support_embeddings = all_embeddings[top_k_indices]  # [k, D]
                        support_labels = all_labels[top_k_indices]  # [k]

                        with torch.no_grad():
                            if task_type == 'classification':
                                logits, proto_classes, confidence = expert.proto_head.forward_classification(
                                    support_embeddings, support_labels, query_embedding
                                )

                                if logits is None:
                                    continue

                                pred_label_idx = torch.argmax(logits, dim=1).item()
                                pred_confidence = confidence[0, pred_label_idx].item()
                                predicted_class_label = proto_classes[pred_label_idx].item()

                                all_preds.append(predicted_class_label)
                                all_true_labels.append(query_label.item())
                                all_confidences.append(pred_confidence)

                            else:  # Regression
                                prediction, confidence = expert.proto_head.forward_regression(
                                    support_embeddings, support_labels.float(), query_embedding
                                )

                                all_preds.append(prediction[0].item())
                                all_true_labels.append(query_label.item())
                                all_confidences.append(confidence[0].item())

                    if not all_true_labels:
                        print(f"Skipping [{expert_name} | k={k} | pct={pct}%]: no valid queries.")
                        continue

                    if task_type == 'classification':
                        avg_conf = np.mean(all_confidences)
                        print(
                            f"[{expert_name} | {k}-NN | pct={pct}%] "
                            f"Retrieval Accuracy: {accuracy_score(all_true_labels, all_preds):.4f} | "
                            f"Avg. Confidence: {avg_conf:.4f} (on {len(all_true_labels)} queries)"
                        )
                    else:
                        preds_np = np.array(all_preds)
                        labels_np = np.array(all_true_labels)
                        avg_conf = np.mean(all_confidences)

                        preds = inverse_transform_time(preds_np)
                        preds[preds < 0] = 0
                        labels = inverse_transform_time(labels_np)
                        print(
                            f"[{expert_name} | {k}-NN | pct={pct}%] "
                            f"Retrieval MAE: {mean_absolute_error(labels, preds):.4f} | "
                            f"R-squared: {r2_score(labels, preds):.4f} | "
                            f"Avg. Confidence: {avg_conf:.4f}"
                        )

            # After k-NN eval, run RandomForest on 80/20 split
            _report_rf_metrics(expert_name, task_type, all_embeddings, all_labels)
