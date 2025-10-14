# testing.py
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
        print(f"\n--- Evaluating task: {task_type.upper()} ---")
        results[task_type] = {}

        # For classification, group data by class to build valid episodes
        if task_type == 'classification':
            class_dict = defaultdict(list)
            for seq, label in task_data:
                class_dict[label].append((seq, label))

        for k in num_shots_list:
            all_preds, all_labels = [], []

            for _ in range(num_test_episodes):
                # FIX: Build a valid episode to prevent KeyError
                if task_type == 'classification':
                    # 1. Pick a random class for the query
                    query_class = random.choice(list(class_dict.keys()))

                    # 2. Sample one query example from that class
                    if not class_dict[query_class]: continue
                    query_example = random.choice(class_dict[query_class])

                    # 3. Build the support set
                    support_set = []
                    # Ensure the query class is represented
                    support_set.extend(
                        random.sample([item for item in class_dict[query_class] if item != query_example],
                                      min(k - 1, len(class_dict[query_class]) - 1)))

                    # Fill the rest of the support set with other classes
                    other_items = [item for cls, items in class_dict.items() if cls != query_class for item in items]
                    remaining_shots = k - len(support_set)
                    if remaining_shots > 0 and other_items:
                        support_set.extend(random.sample(other_items, min(remaining_shots, len(other_items))))

                    if not support_set: continue
                    query_set = [query_example]

                else:  # Regression logic remains the same
                    random.shuffle(task_data)
                    if len(task_data) < k + 1: continue
                    support_set = task_data[:k]
                    query_set = task_data[k:k + 1]

                with torch.no_grad():
                    predictions, true_labels = model(support_set, query_set, task_type)

                if task_type == 'classification':
                    preds = torch.argmax(predictions, dim=1)
                    # The label needs to be mapped back from the prototype index
                    support_features_encoded = model._process_batch([s[0] for s in support_set])
                    query_features_encoded = model._process_batch([q[0] for q in query_set])
                    _, proto_classes = model.proto_head.forward_classification(
                        support_features_encoded,
                        torch.LongTensor([s[1] for s in support_set]),
                        query_features_encoded
                    )
                    original_pred = proto_classes[preds.item()].item()
                    all_preds.append(original_pred)
                    all_labels.append(query_set[0][1])
                else:  # regression
                    all_preds.extend(predictions.tolist())
                    all_labels.extend(true_labels.tolist())

            if not all_labels:
                print(f"Not enough data to test with K={k} shots.")
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
