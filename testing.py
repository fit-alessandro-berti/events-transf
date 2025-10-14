# testing.py
import torch
import random
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score


def test(model, test_tasks, num_shots_list, num_test_episodes=100):
    """
    Evaluates the model's in-context learning performance on unseen tasks.

    Args:
        model: The trained meta-learner.
        test_tasks (dict): Tasks from an unseen process.
        num_shots_list (list): A list of K values (number of support shots) to test.
        num_test_episodes (int): Number of evaluation episodes to run.
    """
    print("\n🔬 Starting meta-testing on unseen process...")
    model.eval()
    results = {}

    for task_type, task_data in test_tasks.items():
        print(f"\n--- Evaluating task: {task_type.upper()} ---")
        results[task_type] = {}

        for k in num_shots_list:
            all_preds, all_labels = [], []

            for _ in range(num_test_episodes):
                random.shuffle(task_data)
                if len(task_data) < k + 1:
                    continue

                support_set = task_data[:k]
                # Use one sample for the query
                query_set = task_data[k:k + 1]

                with torch.no_grad():
                    predictions, true_labels = model(support_set, query_set, task_type)

                if task_type == 'classification':
                    preds = torch.argmax(predictions, dim=1)
                    # The label needs to be mapped back from the prototype index
                    _, proto_classes = model.proto_head.forward_classification(
                        model._process_batch([s[0] for s in support_set]),
                        torch.LongTensor([s[1] for s in support_set]),
                        model._process_batch([q[0] for q in query_set])
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
