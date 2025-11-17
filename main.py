# main.py
import numpy as np
import torch
import os

# --- Import from project files ---
from config import CONFIG
# 🔻🔻🔻 MODIFIED IMPORTS 🔻🔻🔻
# from data_generator import XESLogLoader, get_task_data # No longer used directly
# from components.meta_learner import MetaLearner # No longer used directly
from utils.data_utils import get_task_data
from utils.model_utils import init_loader, create_model
# 🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺
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
    # 🔻🔻🔻 MODIFIED 🔻🔻🔻
    # loader = XESLogLoader(strategy=strategy, sbert_model_name=CONFIG['pretrained_settings']['sbert_model'])
    loader = init_loader(CONFIG)
    # 🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺
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
    # 🔻🔻🔻 MODIFIED 🔻🔻🔻
    # (Replaced all model init logic with the helper function)
    model = create_model(CONFIG, loader, device)
    # 🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺

    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- 4. Training ---
    print("\n--- Phase 4: Starting Model Training ---")
    train(model, training_tasks, loader, CONFIG)

    print("\n✅ Training complete. Run 'testing.py' to evaluate.")


if __name__ == '__main__':
    main()
