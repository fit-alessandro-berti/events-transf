# main.py
import numpy as np
import torch
import os

# --- Import from project files ---
from config import CONFIG
from data_generator import XESLogLoader, get_task_data
from components.meta_learner import MetaLearner
from training import train


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    artifacts_path = os.path.join(checkpoint_dir, 'training_artifacts.pth')

    # --- 1. Data Preparation ---
    print("--- Phase 1: Preparing Training Data ---")
    loader = XESLogLoader()

    # Fit loader on training logs to create map and store embeddings
    loader.fit(CONFIG['log_paths']['training'])

    # Save these artifacts for the stand-alone test script
    loader.save_training_artifacts(artifacts_path)

    # Transform ONLY the training logs
    training_logs = loader.transform(CONFIG['log_paths']['training'])

    print("\n--- Phase 2: Creating Training Tasks ---")
    training_tasks = {
        'classification': [get_task_data(log, 'classification') for log in training_logs.values()],
        'regression': [get_task_data(log, 'regression') for log in training_logs.values()]
    }

    # --- 3. Model Initialization ---
    print("\n--- Phase 3: Initializing Model ---")
    model = MetaLearner(
        embedding_dim=CONFIG['embedding_dim'],
        num_feat_dim=CONFIG['num_numerical_features'],
        d_model=CONFIG['d_model'], n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'], dropout=CONFIG['dropout']
    )
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- 4. Training ---
    print("\n--- Phase 4: Starting Model Training ---")
    train(model, training_tasks, CONFIG)

    print("\n✅ Training complete. Run 'testing.py' to evaluate.")


if __name__ == '__main__':
    main()
