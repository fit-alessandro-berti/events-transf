# File: evaluation/eval_meta.py
import torch
import torch.nn.functional as F
import random
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from collections import defaultdict
from tqdm import tqdm

# --- Import from project files ---
from time_transf import inverse_transform_time


# --- 🔻 NEW HELPER FUNCTION (Copied from eval_retrieval.py) 🔻 ---
def _get_all_test_embeddings(model, test_tasks_list, batch_size=64):
    """
    Helper function to compute embeddings for all (prefix, label, case_id) tuples.
    This is for pre-computing features for the meta-learning evaluation.

    MODIFIED for MoE:
    - Calls `model._process_batch`, which in `MoEModel` will return
      the *average* embedding from all experts.
    """
    all_embeddings = []
    all_labels = []

    device = next(model.parameters()).device
    model.eval()

    try:
        # Check if data has 3 items (prefix, label, case_id)
        _ = test_tasks_list[0][2]
    except (IndexError, TypeError):
        print("\n" + "=" * 50)
        print("❌ ERROR in _get_all_test_embeddings (meta-eval):")
        print("Test data does not contain case_ids.")
        print("This evaluation requires (prefix, label, case_id) tuples.")
        print("=" * 50 + "\n")
        return None, None

    with torch.no_grad():
        for i in tqdm(range(0, len(test_tasks_list), batch_size), desc="Pre-computing test embeddings (for meta-eval)"):
            batch_tasks = test_tasks_list[i:i + batch_size]
            sequences = [t[0] for t in batch_tasks]
            labels = [t[1] for t in batch_tasks]

            if not sequences: continue

            # Use the model's internal processing function
            # For MoEModel, this returns the average embedding
            encoded_batch = model._process_batch(sequences)

            all_embeddings.append(encoded_batch.cpu())
            all_labels.extend(labels)

    if not all_embeddings:
        return None, None

    all_embeddings_tensor = torch.cat(all_embeddings, dim=0).to(device)
    all_labels_tensor = torch.as_tensor(all_labels, device=device)
    return all_embeddings_tensor, all_labels_tensor


# --- 🔺 END NEW HELPER 🔺 ---


def evaluate_model(model, test_tasks, num_shots_list, num_test_episodes=100):
    """
    Standard episodic meta-learning evaluation.

    --- 🚀 OPTIMIZED VERSION 🚀 ---
    This version pre-computes all embeddings for the test set *once*.
    The episodic loop then samples *embeddings* and feeds them
    directly to the model's prototypical heads, bypassing the
    expensive encoder.
    """
    print("\n🔬 Starting meta-testing on the Transformer-based Meta-Learner (Optimized)...")
    model.eval()

    # --- 1. Pre-compute all embeddings for all tasks ---
    task_embeddings = {}
    with torch.no_grad():
        model.eval()  # Ensure model is in eval mode for embedding
        for task_type, task_data in test_tasks.items():
            if not task_data:
                print(f"Skipping {task_type}: No test data available.")
                continue

            print(f"Pre-computing embeddings for meta-eval {task_type}...")
            embeddings, labels = _get_all_test_embeddings(model, task_data)

            if embeddings is None:
                return  # Error already printed in helper

            task_embeddings[task_type] = (embeddings, labels)

    # --- 2. Run episodic evaluation using pre-computed embeddings ---
    for task_type, data in task_embeddings.items():
        print(f"\n--- Evaluating task: {task_type} ---")

        all_embeddings, all_labels = data
        all_labels_np = all_labels.cpu().numpy()
        task_indices = list(range(len(all_labels_np)))

        # Build class_dict mapping label -> list of indices
        if task_type == 'classification':
            class_dict = defaultdict(list)
            for i, label in enumerate(all_labels_np):
                class_dict[label].append(i)  # Store the index

            class_dict = {c: items for c, items in class_dict.items() if len(items) >= max(num_shots_list) + 1}
            available_classes = list(class_dict.keys())
            if len(available_classes) < 2:
                print("Classification test skipped: Need at least 2 classes with sufficient examples.")
                continue
            N_WAYS_TEST = min(len(available_classes), 7)
            print(f"Running classification test as a {N_WAYS_TEST}-way task.")

        for k in num_shots_list:
            all_preds, all_labels_in_test, all_confidences = [], [], []

            # Re-filter class_dict for the current k
            if task_type == 'classification':
                ep_class_dict = {c: items for c, items in class_dict.items() if len(items) >= k + 1}
                if len(ep_class_dict) < N_WAYS_TEST:
                    print(f"[{k}-shot] Skipping: Not enough classes with k+1 samples.")
                    continue

            for _ in range(num_test_episodes):
                support_indices, query_indices = [], []

                if task_type == 'classification':
                    eligible_classes = list(ep_class_dict.keys())
                    if len(eligible_classes) < N_WAYS_TEST: continue
                    episode_classes = random.sample(eligible_classes, N_WAYS_TEST)

                    for cls in episode_classes:
                        samples_indices = random.sample(ep_class_dict[cls], k + 1)
                        support_indices.extend(samples_indices[:k])
                        query_indices.append(samples_indices[k])

                else:  # Regression
                    if len(task_indices) < k + 1: continue
                    random.shuffle(task_indices)
                    support_indices = task_indices[:k]
                    query_indices = task_indices[k:k + 1]  # Original logic used 1 query

                if not support_indices or not query_indices: continue

                # --- 3. Get features & labels from pre-computed tensors ---
                support_features = all_embeddings[support_indices]
                query_features = all_embeddings[query_indices]

                support_labels_torch = all_labels[support_indices]
                query_labels_torch = all_labels[query_indices]

                # --- 4. MANUALLY run MoE inference (fast) ---
                expert_outputs = []
                with torch.no_grad():
                    for expert in model.experts:
                        proto_head = expert.proto_head

                        if task_type == 'classification':
                            preds, proto_classes, confs = proto_head.forward_classification(
                                support_features, support_labels_torch, query_features
                            )
                            # 'preds' are logits [N_q, C], 'confs' are softmax [N_q, C]
                            expert_outputs.append((preds, query_labels_torch, confs))

                        else:  # Regression
                            preds, confs = proto_head.forward_regression(
                                support_features, support_labels_torch.float(), query_features
                            )
                            # 'preds' are values [N_q], 'confs' are weights [N_q]
                            expert_outputs.append((preds, query_labels_torch, confs))

                if not expert_outputs: continue

                # Use the MoE model's built-in aggregation logic
                predictions, true_labels, confidence = model._aggregate_outputs(
                    expert_outputs, task_type, query_labels_torch
                )
                # --- 5. End manual inference ---

                if predictions is None or true_labels is None: continue

                if task_type == 'classification':
                    # 'predictions' from aggregate_outputs are summed confidences [N_q, C]
                    all_preds.extend(torch.argmax(predictions, dim=1).cpu().numpy())
                    # 'true_labels' is just query_labels_torch, but we need to
                    # map them to the *new* indices (0, 1, ... C-1)
                    label_map = {orig_label.item(): new_label for new_label, orig_label in enumerate(proto_classes)}
                    mapped_labels = torch.tensor([label_map.get(l.item(), -100) for l in true_labels], dtype=torch.long)
                    all_labels_in_test.extend(mapped_labels.cpu().numpy())

                    # 'confidence' from aggregate_outputs is the max of the normalized sums [N_q]
                    all_confidences.extend(confidence.cpu().numpy())

                else:  # Regression
                    # 'predictions' is the weighted average [N_q]
                    all_preds.extend(predictions.view(-1).cpu().tolist())
                    all_labels_in_test.extend(true_labels.view(-1).cpu().tolist())
                    # 'confidence' is the avg of expert confidences [N_q]
                    all_confidences.extend(confidence.cpu().numpy())

            if not all_labels_in_test: continue

            if task_type == 'classification':
                valid_indices = [i for i, label in enumerate(all_labels_in_test) if label != -100]
                if not valid_indices: continue
                valid_preds = [all_preds[i] for i in valid_indices]
                valid_labels = [all_labels_in_test[i] for i in valid_indices]
                valid_confidences = [all_confidences[i] for i in valid_indices]
                if not valid_labels: continue
                avg_conf = np.mean(valid_confidences)
                print(
                    f"[{k}-shot] Accuracy: {accuracy_score(valid_labels, valid_preds):.4f} | Avg. Confidence: {avg_conf:.4f}")
            else:
                preds = inverse_transform_time(np.array(all_preds));
                preds[preds < 0] = 0
                labels = inverse_transform_time(np.array(all_labels_in_test))
                avg_conf = np.mean(all_confidences)
                print(
                    f"[{k}-shot] MAE: {mean_absolute_error(labels, preds):.4f} | R-squared: {r2_score(labels, preds):.4f} | Avg. Confidence: {avg_conf:.4f}")
