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

    # Classification: Majority Class Baseline
    cls_labels = [label for _, label in test_tasks['classification']]
    if cls_labels:
        majority_class = Counter(cls_labels).most_common(1)[0][0]
        baseline_preds = [majority_class] * len(cls_labels)
        accuracy = accuracy_score(cls_labels, baseline_preds)
        print(f"Classification (Majority Class Baseline) Accuracy: {accuracy:.4f}")

    # Regression: Mean Value Baseline
    reg_labels = np.array([label for _, label in test_tasks['regression']])
    if reg_labels.size > 0:
        mean_value = reg_labels.mean()
        baseline_preds = np.full_like(reg_labels, mean_value)
        mae = mean_absolute_error(reg_labels, baseline_preds)
        r2 = r2_score(reg_labels, baseline_preds)
        print(f"Regression (Mean Baseline) MAE: {mae:.4f} | R-squared: {r2:.4f} (by definition)")


def main():
    # For reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("1. Loading data from XES files...")
    # --- Load all logs to build a unified vocabulary ---
    # The loader needs all paths (training + testing) to create a complete vocabulary
    all_paths = {**CONFIG['log_paths']['training'], **CONFIG['log_paths']['testing']}

    loader = XESLogLoader()
    loader.load_logs(all_paths)

    # Get vocabularies and individual logs from the loader
    cat_vocabs = loader.get_vocabs()

    # Dynamically retrieve training logs based on keys in config
    training_logs = {name: loader.get_log(name) for name in CONFIG['log_paths']['training']}

    if not all(training_logs.values()):
        print("\n❌ Error: One or more training logs could not be loaded. Please check file paths in './logs/'.")
        return

    training_tasks = {
        'classification': [get_task_data(log, 'classification') for log in training_logs.values()],
        'regression': [get_task_data(log, 'regression') for log in training_logs.values()]
    }

    print("\n2. Initializing model...")
    model = MetaLearner(
        cat_vocabs=cat_vocabs,
        num_feat_dim=CONFIG['num_numerical_features'],
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        dropout=CONFIG['dropout']
    )

    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    print("\n3. Starting training...")
    train(model, training_tasks, CONFIG)

    print("\n4. Preparing test data from an unseen process...")
    # Get the name of the test log (assumes one for simplicity)
    test_log_name = list(CONFIG['log_paths']['testing'].keys())[0]
    unseen_log = loader.get_log(test_log_name)

    if not unseen_log:
        print(f"\n❌ Error: Test log '{test_log_name}' could not be loaded. Please check the file path in './logs/'.")
        return

    test_tasks = {
        'classification': get_task_data(unseen_log, 'classification'),
        'regression': get_task_data(unseen_log, 'regression')
    }

    print("\n5. Starting testing...")
    evaluate_model(model, test_tasks, CONFIG['num_shots_test'], CONFIG['num_test_episodes'])

    # Add baseline evaluation for context
    calculate_baselines(test_tasks)


if __name__ == '__main__':
    main()
