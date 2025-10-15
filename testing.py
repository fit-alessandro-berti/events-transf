import torch
import random
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

        if task_type == 'classification':
            class_dict = defaultdict(list)
            for seq, label in task_data:
                class_dict[label].append((seq, label))
            # Filter out classes that don't have at least 2 examples (one for support, one for query)
            class_dict = {c: items for c, items in class_dict.items() if len(items) >= 2}
            if not class_dict:
                print("Not enough data for classification testing (no class has >= 2 examples).")
                continue

        for k in num_shots_list:
            all_preds, all_labels = [], []

            for _ in range(num_test_episodes):
                support_set, query_set = [], []

                # Robust episode creation logic for testing
                if task_type == 'classification':
                    # 1. Select a random class that has enough examples
                    query_class = random.choice(list(class_dict.keys()))

                    # 2. Sample a query and at least one support example from that class
                    samples_from_class = random.sample(class_dict[query_class], 2)
                    query_example = samples_from_class[0]
                    support_example_from_query_class = samples_from_class[1]

                    query_set = [query_example]
                    support_set = [support_example_from_query_class]

                    # 3. Fill the rest of the support set (k-1 shots) from all available data
                    pool = []
                    for c, items in class_dict.items():
                        pool.extend(items)
                    # Make sure not to re-add the items already used
                    pool = [item for item in pool if item not in query_set and item not in support_set]

                    remaining_shots = k - 1
                    if remaining_shots > 0 and pool:
                        support_set.extend(random.sample(pool, min(remaining_shots, len(pool))))

                    if len(support_set) < k:
                        continue  # Skip if we couldn't build a full k-shot support set

                else:  # Regression
                    random.shuffle(task_data)
                    if len(task_data) < k + 1:
                        continue
                    support_set = task_data[:k]
                    query_set = task_data[k:k + 1]

                if not support_set or not query_set:
                    continue

                with torch.no_grad():
                    predictions, true_labels = model(support_set, query_set, task_type)

                if task_type == 'classification':
                    # predictions are log-probs over the *episode's* prototype order,
                    # and true_labels are already mapped to that order by the model.
                    pred_idx = torch.argmax(predictions, dim=1).item()
                    all_preds.append(pred_idx)
                    all_labels.append(int(true_labels[0].item()))
                else:  # regression
                    all_preds.extend([float(p) for p in predictions.view(-1).tolist()])
                    all_labels.extend([float(t) for t in true_labels.view(-1).tolist()])

            if not all_labels:
                print(f"Could not generate valid episodes to test with K={k} shots.")
                continue

            # Compute metrics
            if task_type == 'classification':
                accuracy = accuracy_score(all_labels, all_preds)
                print(f"[{k}-shot] Accuracy: {accuracy:.4f}")
                results[task_type][k] = {'accuracy': accuracy}
            else:  # regression
                mae = mean_absolute_error(all_labels, all_preds)
                r2 = r2_score(all_labels, all_preds)
                print(f"[{k}-shot] MAE: {mae:.4f} | R-squared: {r2:.4f}")
                results[task_type][k] = {'mae': mae, 'r2': r2}

    return results
