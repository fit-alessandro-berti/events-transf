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
from data_generator import ProcessSimulator, get_task_data
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
            class_dict = {c: items for c, items in class_dict.items() if len(items) >= 2}
            available_classes = list(class_dict.keys())
            if len(available_classes) < 2:
                print("Classification test skipped: Need at least 2 classes with sufficient examples.")
                continue
            N_WAYS_TEST = len(available_classes)
            print(f"Running classification test as a {N_WAYS_TEST}-way task.")
        # ------------------------------------

        for k in num_shots_list:
            all_preds, all_labels = [], []
            episodes_generated = 0

            for _ in range(num_test_episodes):
                support_set, query_set = [], []

                if task_type == 'classification':
                    pool = [item for c in available_classes for item in class_dict[c]]
                    if not pool: continue
                    query_example = random.choice(pool)
                    support_pool = [item for item in pool if item != query_example]
                    if len(support_pool) < k: continue
                    support_set = random.sample(support_pool, k)
                    query_set = [query_example]
                else:  # Regression
                    if len(task_data) < k + 1: continue
                    random.shuffle(task_data)
                    support_set = task_data[:k]
                    query_set = task_data[k:k + 1]

                if not support_set or not query_set: continue

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


if __name__ == '__main__':
    print("--- Running Testing Script in Stand-Alone Mode ---")

    # 1. Find the latest checkpoint
    checkpoint_dir = './checkpoints'
    if not os.path.isdir(checkpoint_dir):
        print(f"Error: Checkpoint directory '{checkpoint_dir}' not found. Please train a model first.")
        exit()

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
    if not checkpoints:
        print(f"Error: No checkpoints found in '{checkpoint_dir}'.")
        exit()

    # Extract epoch numbers and find the latest one
    epoch_map = {}
    for chkpt in checkpoints:
        match = re.search(r'model_epoch_(\d+).pth', chkpt)
        if match:
            epoch_map[int(match.group(1))] = chkpt

    latest_epoch = max(epoch_map.keys())
    latest_checkpoint_file = epoch_map[latest_epoch]
    latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint_file)

    print(f"🔍 Found latest checkpoint: {latest_checkpoint_file}")

    # 2. Initialize model architecture from CONFIG
    torch.manual_seed(42)
    np.random.seed(42)

    # We need a simulator instance to get vocabulary sizes for model initialization
    temp_simulator = ProcessSimulator(num_cases=1)
    cat_vocabs = {
        'activity': len(temp_simulator.vocab['activity']),
        'resource': len(temp_simulator.vocab['resource']),
    }

    model = MetaLearner(
        cat_vocabs=cat_vocabs,
        num_feat_dim=CONFIG['num_numerical_features'],
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        dropout=CONFIG['dropout']
    )

    # 3. Load the checkpoint weights
    print(f"💾 Loading weights from {latest_checkpoint_path}...")
    model.load_state_dict(torch.load(latest_checkpoint_path))

    # 4. Generate fresh test data
    print("\n📦 Generating new test data from an unseen process...")
    simulator = ProcessSimulator(num_cases=CONFIG['num_cases_for_testing'])
    unseen_log = simulator.generate_data_for_model('D_unseen')
    test_tasks = {
        'classification': get_task_data(unseen_log, 'classification'),
        'regression': get_task_data(unseen_log, 'regression')
    }

    # 5. Run the evaluation
    evaluate_model(
        model,
        test_tasks,
        CONFIG['num_shots_test'],
        CONFIG['num_test_episodes']
    )
