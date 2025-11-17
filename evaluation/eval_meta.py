# File: evaluation/eval_meta.py
import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from collections import defaultdict

# --- Import from project files ---
from time_transf import inverse_transform_time


def evaluate_model(model, test_tasks, num_shots_list, num_test_episodes=100):
    """
    Standard episodic meta-learning evaluation.
    Samples support/query sets for each episode.
    (Moved from testing.py)
    """
    print("\n🔬 Starting meta-testing on the Transformer-based Meta-Learner...")
    model.eval()
    for task_type, task_data in test_tasks.items():
        print(f"\n--- Evaluating task: {task_type} ---")
        if not task_data:
            print(f"Skipping {task_type}: No test data available.")
            continue
        if task_type == 'classification':
            class_dict = defaultdict(list)
            # Assumes task_data is (prefix, label) or (prefix, label, case_id)
            # 🔻 MODIFIED: Iterated over task_data, not test_tasks 🔻
            for task_item in task_data:
                # 🔺 END MODIFIED 🔺
                seq, label = task_item[0], task_item[1]
                class_dict[label].append((seq, label))
            class_dict = {c: items for c, items in class_dict.items() if len(items) >= max(num_shots_list) + 1}
            available_classes = list(class_dict.keys())
            if len(available_classes) < 2:
                print("Classification test skipped: Need at least 2 classes with sufficient examples.")
                continue
            N_WAYS_TEST = min(len(available_classes), 7)
            print(f"Running classification test as a {N_WAYS_TEST}-way task.")

        for k in num_shots_list:
            # 🔻 MODIFIED: Added all_confidences 🔻
            all_preds, all_labels, all_confidences = [], [], []
            # 🔺 END MODIFIED 🔺

            # Re-build class_dict for sampling (to handle regression case)
            if task_type == 'classification':
                ep_class_dict = defaultdict(list)
                for task_item in task_data:
                    seq, label = task_item[0], task_item[1]
                    ep_class_dict[label].append((seq, label))
                ep_class_dict = {c: items for c, items in ep_class_dict.items() if len(items) >= k + 1}

            for _ in range(num_test_episodes):
                support_set, query_set = [], []
                if task_type == 'classification':
                    eligible_classes = [c for c, items in ep_class_dict.items() if len(items) >= k + 1]
                    if len(eligible_classes) < N_WAYS_TEST: continue
                    episode_classes = random.sample(eligible_classes, N_WAYS_TEST)
                    for cls in episode_classes:
                        samples = random.sample(ep_class_dict[cls], k + 1)
                        support_set.extend(samples[:k]);
                        query_set.append(samples[k])
                else:
                    # Regression: task_data is a list of (prefix, label, case_id)
                    if len(task_data) < k + 1: continue
                    random.shuffle(task_data)
                    # Slicing will work, just ignores the 3rd element (case_id)
                    support_set_raw, query_set_raw = task_data[:k], task_data[k:k + 1]
                    support_set = [(s[0], s[1]) for s in support_set_raw]
                    query_set = [(q[0], q[1]) for q in query_set_raw]

                if not support_set or not query_set: continue
                with torch.no_grad():
                    # 🔻 MODIFIED: Unpack confidence 🔻
                    predictions, true_labels, confidence = model(support_set, query_set, task_type)
                    # 🔺 END MODIFIED 🔺
                if predictions is None or true_labels is None or torch.all(true_labels == -100): continue
                if task_type == 'classification':
                    all_preds.extend(torch.argmax(predictions, dim=1).cpu().numpy())
                    all_labels.extend(true_labels.cpu().numpy())

                    # --- 🔻 FIX 🔻 ---
                    # The `confidence` tensor from MoEModel is already 1D (shape [N_q]).
                    # We no longer need to call torch.max() on it.
                    all_confidences.extend(confidence.cpu().numpy())
                    # --- 🔺 END FIX 🔺 ---

                else:
                    all_preds.extend(predictions.view(-1).cpu().tolist())
                    # 🔻 MODIFIED: Added missing () to .cpu() 🔻
                    all_labels.extend(true_labels.view(-1).cpu().tolist())
                    # 🔺 END MODIFIED 🔺
                    # 🔻 MODIFIED: Store regression confidence 🔻
                    all_confidences.extend(confidence.cpu().numpy())
                    # 🔺 END MODIFIED 🔺
            if not all_labels: continue
            if task_type == 'classification':
                # Filter out invalid -100 labels
                valid_indices = [i for i, label in enumerate(all_labels) if label != -100]
                if not valid_indices: continue
                valid_preds = [all_preds[i] for i in valid_indices]
                valid_labels = [all_labels[i] for i in valid_indices]
                # 🔻 MODIFIED: Filter and report confidence 🔻
                valid_confidences = [all_confidences[i] for i in valid_indices]
                if not valid_labels: continue
                avg_conf = np.mean(valid_confidences)
                print(
                    f"[{k}-shot] Accuracy: {accuracy_score(valid_labels, valid_preds):.4f} | Avg. Confidence: {avg_conf:.4f}")
                # 🔺 END MODIFIED 🔺
            else:
                preds = inverse_transform_time(np.array(all_preds));
                preds[preds < 0] = 0
                labels = inverse_transform_time(np.array(all_labels))
                # 🔻 MODIFIED: Report confidence 🔻
                avg_conf = np.mean(all_confidences)
                print(
                    f"[{k}-shot] MAE: {mean_absolute_error(labels, preds):.4f} | R-squared: {r2_score(labels, preds):.4f} | Avg. Confidence: {avg_conf:.4f}")
                # 🔺 END MODIFIED 🔺
