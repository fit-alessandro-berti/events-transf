# testing.py
import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from collections import defaultdict


def test(model, test_tasks, num_shots_list, num_test_episodes=100):
    """
    Evaluates the model's in-context learning performance on unseen tasks.
    """
    print("\n🔬 Starting meta-testing on unseen process...")
    model.eval()
    results = {}

    for task_type, task_data in test_tasks.items():
        print(f"\n--- Evaluating task: {task_type} ---")
        results[task_type] = {}

        # --- Classification-specific setup ---
        if task_type == 'classification':
            class_dict = defaultdict(list)
            for seq, label in task_data:
                class_dict[label].append((seq, label))

            # Filter classes: must have at least 2 examples for a support/query split.
            class_dict = {c: items for c, items in class_dict.items() if len(items) >= 2}

            available_classes = list(class_dict.keys())

            # Check if evaluation is possible *before* looping
            if len(available_classes) < 2:
                print("Classification test skipped: Need at least 2 classes with sufficient examples.")
                continue

            # FIX: The number of "ways" is determined by the data, not hardcoded.
            N_WAYS_TEST = len(available_classes)
            print(f"Running classification test as a {N_WAYS_TEST}-way task.")
        # ------------------------------------

        for k in num_shots_list:
            all_preds, all_labels = [], []
            episodes_generated = 0

            for _ in range(num_test_episodes):
                support_set, query_set = [], []

                if task_type == 'classification':
                    # Use all available classes for the episode
                    episode_classes = available_classes

                    pool = []
                    for c in episode_classes:
                        pool.extend(class_dict[c])

                    if not pool: continue

                    query_example = random.choice(pool)
                    support_pool = [item for item in pool if item != query_example]

                    if len(support_pool) < k:
                        continue

                    support_set = random.sample(support_pool, k)
                    query_set = [query_example]

                else:  # Regression
                    if len(task_data) < k + 1:
                        continue
                    random.shuffle(task_data)
                    support_set = task_data[:k]
                    query_set = task_data[k:k + 1]

                if not support_set or not query_set:
                    continue

                episodes_generated += 1
                with torch.no_grad():
                    predictions, true_labels = model(support_set, query_set, task_type)

                if predictions is None or true_labels is None: continue

                if task_type == 'classification':
                    pred_idx = torch.argmax(predictions, dim=1).cpu().numpy()
                    all_preds.extend(pred_idx)
                    all_labels.extend(true_labels.cpu().numpy())
                else:
                    all_preds.extend(predictions.view(-1).cpu().tolist())
                    all_labels.extend(true_labels.view(-1).cpu().tolist())

            if episodes_generated == 0:
                print(f"Could not generate valid episodes to test with K={k} shots.")
                continue

            # Compute metrics
            if task_type == 'classification':
                accuracy = accuracy_score(all_labels, all_preds)
                print(f"[{k}-shot] Accuracy: {accuracy:.4f}")
                results[task_type][k] = {'accuracy': accuracy}
            else:
                valid_preds = [p for p, l in zip(all_preds, all_labels) if not np.isnan(p) and not np.isnan(l)]
                valid_labels = [l for p, l in zip(all_preds, all_labels) if not np.isnan(p) and not np.isnan(l)]

                if not valid_labels:
                    print(f"[{k}-shot] MAE: NaN (No valid predictions)")
                    continue

                mae = mean_absolute_error(valid_labels, valid_preds)
                r2 = r2_score(valid_labels, valid_preds)
                print(f"[{k}-shot] MAE: {mae:.4f} | R-squared: {r2:.4f}")
                results[task_type][k] = {'mae': mae, 'r2': r2}

    return results
