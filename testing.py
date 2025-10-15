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
            # (need at least 2 for a support/query split)
            class_dict = {c: items for c, items in class_dict.items() if len(items) >= 2}
            if not class_dict:
                print("Not enough data for classification testing (no class has >= 2 examples).")
                continue

        for k in num_shots_list:
            all_preds, all_labels = [], []

            for _ in range(num_test_episodes):
                support_set, query_set = [], []

                if task_type == 'classification':
                    # FIX: Implement stable N-way K-shot sampling to prevent performance degradation.
                    # This logic ensures the number of classes ("ways") is controlled, making the
                    # task difficulty consistent as K (shots) increases.
                    N_WAYS = 5  # Number of classes in the episode's "world"

                    available_classes = list(class_dict.keys())
                    if len(available_classes) < 2: continue  # Need at least one class for query and one for support

                    # 1. Sample N_WAYS classes for the episode.
                    num_ep_classes = min(N_WAYS, len(available_classes))
                    episode_classes = random.sample(available_classes, num_ep_classes)

                    # 2. Choose a query class and sample a query/support pair from it.
                    query_class = random.choice(episode_classes)
                    query_example, support_from_query_class = random.sample(class_dict[query_class], 2)

                    query_set = [query_example]
                    support_set = [support_from_query_class]

                    # 3. Fill the rest of the k-shot support set from the chosen episode classes.
                    remaining_shots = k - 1
                    if remaining_shots > 0:
                        pool = []
                        for c in episode_classes:
                            # Add all items from the chosen classes to the pool, excluding already used ones
                            pool.extend(
                                [item for item in class_dict[c] if item not in query_set and item not in support_set])

                        if pool:
                            support_set.extend(random.sample(pool, min(remaining_shots, len(pool))))

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

                if predictions is None: continue

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
                # Filter out potential NaNs that might have slipped through
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
