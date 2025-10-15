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

        class_dict = None
        if task_type == 'classification':
            class_dict = defaultdict(list)
            for seq, label in task_data:
                class_dict[label].append((seq, label))

            # Filter out classes that don't have enough examples
            # A class needs at least 2 examples to be used in an episode (one for support, one for query).
            class_dict = {c: items for c, items in class_dict.items() if len(items) >= 2}
            if not class_dict:
                print("Not enough data for classification testing (no class has >= 2 examples).")
                continue

        for k in num_shots_list:
            all_preds, all_labels = [], []

            for _ in range(num_test_episodes):
                support_set, query_set = [], []

                if task_type == 'classification':
                    # FIX: Implement a consistent N-way K-shot evaluation protocol.
                    # This ensures the task difficulty is consistent across different `k` values.
                    N_WAYS_TEST = 5  # The number of classes in each test episode.

                    available_classes = list(class_dict.keys())
                    if len(available_classes) < N_WAYS_TEST:
                        continue  # Not enough unique classes in the test set to form a full episode.

                    # 1. Sample N_WAYS classes for the episode.
                    episode_classes = random.sample(available_classes, N_WAYS_TEST)

                    # 2. Create a pool of all available examples from these classes.
                    pool = []
                    for c in episode_classes:
                        pool.extend(class_dict[c])

                    # 3. Sample a query example from the pool.
                    query_example = random.choice(pool)

                    # 4. Create a pool for the support set (all items except the query).
                    support_pool = [item for item in pool if item != query_example]

                    # 5. Sample K support examples from the support pool.
                    if len(support_pool) < k:
                        continue  # Skip if we can't form a full k-shot support set.

                    support_set = random.sample(support_pool, k)
                    query_set = [query_example]

                else:  # Regression
                    if len(task_data) < k + 1:
                        continue
                    # Shuffle data for each episode to ensure variety
                    random.shuffle(task_data)
                    support_set = task_data[:k]
                    query_set = task_data[k:k + 1]

                if not support_set or not query_set:
                    continue

                with torch.no_grad():
                    predictions, true_labels = model(support_set, query_set, task_type)

                if predictions is None or true_labels is None: continue

                if task_type == 'classification':
                    pred_idx = torch.argmax(predictions, dim=1).cpu().numpy()
                    all_preds.extend(pred_idx)
                    all_labels.extend(true_labels.cpu().numpy())
                else:  # regression
                    all_preds.extend(predictions.view(-1).cpu().tolist())
                    all_labels.extend(true_labels.view(-1).cpu().tolist())

            if not all_labels:
                print(f"Could not generate valid episodes to test with K={k} shots.")
                continue

            # Compute metrics
            if task_type == 'classification':
                accuracy = accuracy_score(all_labels, all_preds)
                print(f"[{k}-shot] Accuracy: {accuracy:.4f}")
                results[task_type][k] = {'accuracy': accuracy}
            else:  # regression
                # Filter out potential NaNs
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
