# testing.py
import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from collections import defaultdict
import os
import re

# --- Stand-alone execution imports ---
from config import CONFIG
from data_generator import XESLogLoader, get_task_data
from components.meta_learner import MetaLearner


def evaluate_model(model, test_tasks, num_shots_list, num_test_episodes=100):
    """
    Evaluates a given model's in-context learning performance on unseen tasks.
    This function can be called from main.py or from the stand-alone script.
    """
    print("\n🔬 Starting meta-testing on unseen process...")
    model.eval()
    results = {}

    for task_type, task_data in test_tasks.items():
        print(f"\n--- Evaluating task: {task_type} ---")
        results[task_type] = {}

        if not task_data:
            print(f"Skipping {task_type}: No test data available.")
            continue

        # --- Classification-specific setup ---
        if task_type == 'classification':
            class_dict = defaultdict(list)
            for seq, label in task_data:
                class_dict[label].append((seq, label))
            class_dict = {c: items for c, items in class_dict.items() if len(items) >= max(num_shots_list) + 1}
            available_classes = list(class_dict.keys())
            if len(available_classes) < 2:
                print("Classification test skipped: Need at least 2 classes with sufficient examples.")
                continue
            N_WAYS_TEST = min(len(available_classes), 7)
            print(f"Running classification test as a {N_WAYS_TEST}-way task.")
        # ------------------------------------

        for k in num_shots_list:
            all_preds, all_labels = [], []
            episodes_generated = 0

            for _ in range(num_test_episodes):
                support_set, query_set = [], []

                if task_type == 'classification':
                    eligible_classes = [c for c, items in class_dict.items() if len(items) >= k + 1]
                    if len(eligible_classes) < N_WAYS_TEST:
                        continue

                    episode_classes = random.sample(eligible_classes, N_WAYS_TEST)

                    for cls in episode_classes:
                        samples = random.sample(class_dict[cls], k + 1)
                        support_set.extend(samples[:k])
                        query_set.append(samples[k])

                    random.shuffle(support_set)
                    random.shuffle(query_set)
                else:  # Regression
                    if len(task_data) < k + 1: continue
                    random.shuffle(task_data)
                    support_set = task_data[:k]
                    query_set = task_data[k:k + 1]

                if not support_set or not query_set: continue

                episodes_generated += 1
                with torch.no_grad():
                    predictions, true_labels = model(support_set, query_set, task_type)

                if predictions is None or true_labels is None or torch.all(true_labels == -100):
                    continue

                if task_type == 'classification':
                    all_preds.extend(torch.argmax(predictions, dim=1).cpu().numpy())
                    all_labels.extend(true_labels.cpu().numpy())
                else:
                    all_preds.extend(predictions.view(-1).cpu().tolist())
                    all_labels.extend(true_labels.view(-1).cpu().tolist())

            if episodes_generated == 0:
                print(f"Could not generate valid episodes to test with K={k} shots.")
                continue

            # Compute metrics
            if task_type == 'classification':
                valid_indices = [i for i, label in enumerate(all_labels) if label != -100]
                if not valid_indices:
                    print(f"[{k}-shot] Accuracy: NaN (No valid predictions)")
                    continue
                accuracy = accuracy_score(np.array(all_labels)[valid_indices], np.array(all_preds)[valid_indices])
                print(f"[{k}-shot] Accuracy: {accuracy:.4f}")
                results[task_type][k] = {'accuracy': accuracy}
            else:  # Regression
                valid_preds_transformed = [p for p, l in zip(all_preds, all_labels) if
                                           not np.isnan(p) and not np.isnan(l)]
                valid_labels_transformed = [l for p, l in zip(all_preds, all_labels) if
                                            not np.isnan(p) and not np.isnan(l)]

                if not valid_labels_transformed:
                    print(f"[{k}-shot] MAE: NaN (No valid predictions)")
                    continue

                valid_preds = np.expm1(valid_preds_transformed)
                valid_labels = np.expm1(valid_labels_transformed)
                valid_preds[valid_preds < 0] = 0

                mae = mean_absolute_error(valid_labels, valid_preds)
                r2 = r2_score(valid_labels, valid_preds)
                print(f"[{k}-shot] MAE: {mae:.4f} | R-squared: {r2:.4f}")
                results[task_type][k] = {'mae': mae, 'r2': r2}

    return results


if __name__ == '__main__':
    print("--- Running Testing Script in Stand-Alone Mode ---")

    checkpoint_dir = './checkpoints'
    if not os.path.isdir(checkpoint_dir): exit(f"❌ Error: Checkpoint directory '{checkpoint_dir}' not found.")

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
    if not checkpoints: exit(f"❌ Error: No checkpoints found in '{checkpoint_dir}'.")

    epoch_map = {int(re.search(r'model_epoch_(\d+).pth', f).group(1)): f for f in checkpoints if
                 re.search(r'model_epoch_(\d+).pth', f)}
    latest_checkpoint_path = os.path.join(checkpoint_dir, epoch_map[max(epoch_map.keys())])
    print(f"🔍 Found latest checkpoint: {epoch_map[max(epoch_map.keys())]}")

    print("\n📦 Loading raw logs for testing...")
    all_paths = {**CONFIG['log_paths']['training'], **CONFIG['log_paths']['testing']}
    loader = XXESLogLoader()
    loader.load_logs(all_paths)

    torch.manual_seed(42)
    np.random.seed(42)
    model = MetaLearner(
        cat_vocabs=CONFIG['fixed_vocab_sizes'],
        num_feat_dim=CONFIG['num_numerical_features'],
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        dropout=CONFIG['dropout']
    )
    print(f"💾 Loading weights from {latest_checkpoint_path}...")
    model.load_state_dict(torch.load(latest_checkpoint_path))

    # --- NEW: Apply a single random mapping for the test run ---
    print("🎲 Applying a random mapping to the test data...")
    loader.remap_logs(CONFIG['fixed_vocab_sizes'])

    test_log_name = list(CONFIG['log_paths']['testing'].keys())[0]
    unseen_log = loader.get_log(test_log_name)

    if not unseen_log:
        exit(f"❌ Error: Test log '{test_log_name}' could not be loaded.")

    test_tasks = {
        'classification': get_task_data(unseen_log, 'classification'),
        'regression': get_task_data(unseen_log, 'regression')
    }

    evaluate_model(
        model,
        test_tasks,
        CONFIG['num_shots_test'],
        CONFIG['num_test_episodes']
    )
