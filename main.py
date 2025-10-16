# main.py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from collections import Counter

# --- Import from project files ---
from config import CONFIG
from data_generator import XESLogLoader, get_task_data
from components.meta_learner import MetaLearner
from training import train
from testing import evaluate_model


def calculate_baselines(test_tasks):
    """Calculates performance for trivial baseline models."""
    print("\n--- Evaluating Baseline Models ---")

    cls_labels = [label for _, label in test_tasks['classification']]
    if cls_labels:
        majority_class = Counter(cls_labels).most_common(1)[0][0]
        baseline_preds = [majority_class] * len(cls_labels)
        accuracy = accuracy_score(cls_labels, baseline_preds)
        print(f"Classification (Majority Class Baseline) Accuracy: {accuracy:.4f}")

    reg_labels = np.array([label for _, label in test_tasks['regression']])
    if reg_labels.size > 0:
        mean_value = reg_labels.mean()
        baseline_preds = np.full_like(reg_labels, mean_value)
        mae = mean_absolute_error(reg_labels, baseline_preds)
        r2 = r2_score(reg_labels, baseline_preds)
        print(f"Regression (Mean Baseline) MAE: {mae:.4f} | R-squared: {r2:.4f} (by definition)")


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # --- 1. Data Loading and Pre-processing ---
    print("1. Initializing data loader and building vocabulary from training logs...")
    loader = XESLogLoader()
    # Load training logs to build vocabulary and semantic embeddings
    loader.load_and_build_vocab_from_training_logs(CONFIG['log_paths']['training'])

    # Process training logs into traces with embeddings
    training_logs = loader.process_logs(CONFIG['log_paths']['training'])
    # Process test logs using the same vocabulary
    testing_logs = loader.process_logs(CONFIG['log_paths']['testing'])

    print("\n2. Creating training and testing tasks...")
    # Create training tasks from the processed training logs
    training_tasks = {
        'classification': [get_task_data(log, 'classification') for log in training_logs.values()],
        'regression': [get_task_data(log, 'regression') for log in training_logs.values()]
    }

    # Create test tasks from the processed test logs
    test_log_name = list(CONFIG['log_paths']['testing'].keys())[0]
    unseen_log = testing_logs.get(test_log_name)
    if not unseen_log:
        print(f"\n❌ Error: Test log '{test_log_name}' could not be processed. Please check the file path.")
        return

    test_tasks = {
        'classification': get_task_data(unseen_log, 'classification'),
        'regression': get_task_data(unseen_log, 'regression')
    }

    # --- 2. Model Initialization ---
    print("\n3. Initializing model with semantic embedding support...")
    model = MetaLearner(
        embedding_dim=CONFIG['embedding_dim'],
        num_feat_dim=CONFIG['num_numerical_features'],
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        dropout=CONFIG['dropout']
    )
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- 3. Training ---
    print("\n4. Starting training...")
    train(model, training_tasks, CONFIG)

    # --- 4. Testing ---
    print("\n5. Starting testing on unseen log...")
    evaluate_model(model, test_tasks, CONFIG['num_shots_test'], CONFIG['num_test_episodes'])

    # --- 5. Baselines ---
    calculate_baselines(test_tasks)


if __name__ == '__main__':
    main()
