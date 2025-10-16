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
    activity_map_path = os.path.join(checkpoint_dir, 'activity_map.pth')

    # --- 1. Data Preparation ---
    print("---  fase 1: Preparazione dei dati di training ---")
    loader = XESLogLoader()

    # Fit loader on training logs to create the activity -> integer label mapping
    loader.fit(CONFIG['log_paths']['training'])

    # Save this mapping so the stand-alone test script can use it
    loader.save_activity_map(activity_map_path)

    # Transform ONLY the training logs to get data with embeddings
    training_logs = loader.transform(CONFIG['log_paths']['training'])

    print("\n--- 2. Creazione dei task di training ---")
    training_tasks = {
        'classification': [get_task_data(log, 'classification') for log in training_logs.values()],
        'regression': [get_task_data(log, 'regression') for log in training_logs.values()]
    }

    # --- 3. Model Initialization ---
    print("\n--- 3. Inizializzazione del modello ---")
    model = MetaLearner(
        embedding_dim=CONFIG['embedding_dim'],
        num_feat_dim=CONFIG['num_numerical_features'],
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        dropout=CONFIG['dropout']
    )
    print(f"Il modello ha {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parametri allenabili.")

    # --- 4. Training ---
    print("\n--- 4. Inizio del training del modello ---")
    train(model, training_tasks, CONFIG)

    print("\n✅ Training completato. Eseguire 'testing.py' per la valutazione.")


if __name__ == '__main__':
    main()
