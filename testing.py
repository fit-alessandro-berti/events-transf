# File: testing.py
import torch
import numpy as np
import os
import re
import warnings

# --- Stand-alone execution imports ---
from config import CONFIG
# 🔻🔻🔻 MODIFIED IMPORTS 🔻🔻🔻
# from data_generator import XESLogLoader, get_task_data # No longer used directly
# from components.meta_learner import MetaLearner # No longer used directly
from time_transf import inverse_transform_time # Still needed? No, moved to eval files.
from utils.data_utils import get_task_data
from utils.model_utils import init_loader, create_model, load_model_weights
from evaluation import evaluate_model, evaluate_retrieval_augmented, evaluate_sklearn_baselines
# 🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺

# 🔻🔻🔻
# All evaluation functions (evaluate_model, _get_all_test_embeddings,
# evaluate_retrieval_augmented, _extract_features_for_sklearn,
# evaluate_sklearn_baselines) have been REMOVED from this file.
# They are now in the `evaluation/` directory.
# 🔺🔺🔺


if __name__ == '__main__':
    strategy = CONFIG['embedding_strategy']
    print(f"--- Running Testing Script in Stand-Alone Mode (strategy: '{strategy}') ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_dir = './checkpoints'
    artifacts_path = os.path.join(checkpoint_dir, 'training_artifacts.pth')

    # --- Load Data ---
    print("\n📦 Loading test data...")
    # 🔻🔻🔻 MODIFIED 🔻🔻🔻
    loader = init_loader(CONFIG)
    # 🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺
    loader.load_training_artifacts(artifacts_path)
    testing_logs = loader.transform(CONFIG['log_paths']['testing'])

    torch.manual_seed(42);
    np.random.seed(42)

    # --- Model Initialization ---
    # 🔻🔻🔻 MODIFIED 🔻🔻🔻
    # (Replaced model init logic with helper)
    model = create_model(CONFIG, loader, device)
    # 🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺

    # --- Load Weights ---
    # 🔻🔻🔻 MODIFIED 🔻🔻🔻
    # (Replaced weight loading logic with helper)
    load_model_weights(model, checkpoint_dir, device)
    # 🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺

    test_log_name = list(CONFIG['log_paths']['testing'].keys())[0]
    unseen_log = testing_logs.get(test_log_name)
    if not unseen_log: exit(f"❌ Error: Test log '{test_log_name}' could not be processed.")

    print("\n🛠️ Creating test tasks...")
    # This call MUST now return (prefix, label, case_id) tuples
    # for the retrieval_augmented mode to work correctly.
    test_tasks = {
        'classification': get_task_data(unseen_log, 'classification'),
        'regression': get_task_data(unseen_log, 'regression')
    }

    # --- Select Evaluation Mode based on Config ---
    test_mode = CONFIG.get('test_mode', 'meta_learning')

    if test_mode == 'retrieval_augmented':
        print("\n--- Running in Retrieval-Augmented Evaluation Mode ---")
        k_list = CONFIG.get('test_retrieval_k', CONFIG['num_shots_test'])
        # Pass the full task list (which includes case_ids)
        evaluate_retrieval_augmented(model, test_tasks, k_list, CONFIG['num_test_episodes'])

    elif test_mode == 'meta_learning':
        print("\n--- Running in Meta-Learning Evaluation Mode ---")
        evaluate_model(model, test_tasks, CONFIG['num_shots_test'], CONFIG['num_test_episodes'])

    else:
        print(f"⚠️ Warning: Unknown test_mode '{test_mode}'. Defaulting to 'meta_learning'.")
        evaluate_model(model, test_tasks, CONFIG['num_shots_test'], CONFIG['num_test_episodes'])

    # Always run baselines for comparison
    evaluate_sklearn_baselines(model, test_tasks, CONFIG['num_shots_test'], CONFIG['num_test_episodes'])
