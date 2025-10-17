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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    strategy = CONFIG['embedding_strategy']
    print(f"--- Running with embedding strategy: '{strategy}' ---")

    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    artifacts_path = os.path.join(checkpoint_dir, 'training_artifacts.pth')

    # --- 1. Data Preparation ---
    print("--- Phase 1: Preparing Training Data ---")
    loader = XESLogLoader(strategy=strategy, sbert_model_name=CONFIG['pretrained_settings']['sbert_model'])
    loader.fit(CONFIG['log_paths']['training'])
    loader.save_training_artifacts(artifacts_path)
    training_logs = loader.transform(CONFIG['log_paths']['training'])

    print("\n--- Phase 2: Creating Training Tasks ---")
    training_tasks = {
        'classification': [get_task_data(log, 'classification') for log in training_logs.values()],
        'regression': [get_task_data(log, 'regression') for log in training_logs.values()]
    }

    # --- 3. Model Initialization ---
    print("\n--- Phase 3: Initializing Model ---")
    if strategy == 'pretrained':
        model_params = {
            'embedding_dim': CONFIG['pretrained_settings']['embedding_dim'],
        }
    else: # learned
        model_params = {
            'vocab_sizes': {
                'activity': len(loader.activity_to_id),
                'resource': len(loader.resource_to_id)
            },
            'embedding_dims': {
                'activity': CONFIG['learned_settings']['activity_embedding_dim'],
                'resource': CONFIG['learned_settings']['resource_embedding_dim']
            }
        }

    model = MetaLearner(
        strategy=strategy,
        num_feat_dim=CONFIG['num_numerical_features'],
        d_model=CONFIG['d_model'], n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'], dropout=CONFIG['dropout'],
        **model_params
    ).to(device)

    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- 4. Training ---
    print("\n--- Phase 4: Starting Model Training ---")
    # Pass the loader object to the train function for vocabulary access
    train(model, training_tasks, loader, CONFIG)

    print("\n✅ Training complete. Run 'testing.py' to evaluate.")


if __name__ == '__main__':
    main()
