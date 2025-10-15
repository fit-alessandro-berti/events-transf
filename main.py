# main.py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from collections import Counter
from data_generator import ProcessSimulator, get_task_data
from components.meta_learner import MetaLearner
from training import train
from testing import test

# --- Configuration ---
CONFIG = {
    # Model Hyperparameters (Increased capacity)
    'd_model': 128,
    'n_heads': 8,
    'n_layers': 3,
    'dropout': 0.1,
    'num_numerical_features': 3,  # cost, time_from_start, time_from_previous

    # Meta-Learning Parameters (More robust episodes)
    'num_shots_range': (3, 10),  # This range is fine, but could be widened to (2, 15)
    'num_queries': 10,
    'num_shots_test': [1, 5, 10],

    # Training Parameters (Significantly more training)
    'lr': 1e-4,
    'epochs': 25,
    'episodes_per_epoch': 1000,

    # Increase test episodes for more stable evaluation
    'num_test_episodes': 1000,
}


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

    print("1. Generating data...")
    simulator = ProcessSimulator(num_cases=500)

    cat_vocabs = {
        'activity': len(simulator.vocab['activity']),
        'resource': len(simulator.vocab['resource']),
    }

    log_a = simulator.generate_data_for_model('A')
    log_b = simulator.generate_data_for_model('B')
    log_c = simulator.generate_data_for_model('C')

    training_tasks = {
        'classification': [
            get_task_data(log_a, 'classification'),
            get_task_data(log_b, 'classification'),
            get_task_data(log_c, 'classification'),
        ],
        'regression': [
            get_task_data(log_a, 'regression'),
            get_task_data(log_b, 'regression'),
            get_task_data(log_c, 'regression'),
        ]
    }

    print("2. Initializing model...")
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
    unseen_log = simulator.generate_data_for_model('D_unseen')
    test_tasks = {
        'classification': get_task_data(unseen_log, 'classification'),
        'regression': get_task_data(unseen_log, 'regression')
    }

    print("\n5. Starting testing...")
    test(model, test_tasks, CONFIG['num_shots_test'], CONFIG['num_test_episodes'])

    # Add baseline evaluation for context
    calculate_baselines(test_tasks)


if __name__ == '__main__':
    main()
