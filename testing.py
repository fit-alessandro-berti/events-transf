# File: testing.py
import torch
import numpy as np
import os
import re
import warnings
import argparse  # --- (no new import needed) ---

# --- Stand-alone execution imports ---
from config import CONFIG
from time_transf import inverse_transform_time
from utils.data_utils import get_task_data
from utils.model_utils import init_loader, create_model, load_model_weights
from evaluation import evaluate_model, evaluate_retrieval_augmented, evaluate_sklearn_baselines

# (No helper functions here, they are correctly in /evaluation)


if __name__ == '__main__':

    # --- 🔻 MODIFIED: Argument Parsing 🔻 ---
    # Use argparse to override test parameters from the command line
    parser = argparse.ArgumentParser(description="Run the meta-learning model evaluation script.")

    # Pull defaults directly from the imported CONFIG
    default_config = CONFIG

    # --- NEW: Get available test logs from config ---
    available_test_logs = list(default_config['log_paths']['testing'].keys())
    default_test_log = available_test_logs[0] if available_test_logs else None

    parser.add_argument(
        '--test_log_name',  # <-- 🔻 NEW ARGUMENT 🔻
        type=str,
        default=default_test_log,
        choices=available_test_logs,
        help=f"Name of the test log to evaluate. (default: {default_test_log})"
    )
    # --- 🔺 END NEW 🔺 ---

    parser.add_argument(
        '--test_mode',
        type=str,
        default=default_config['test_mode'],
        choices=['meta_learning', 'retrieval_augmented'],
        help=f"Evaluation mode. (default: {default_config['test_mode']})"
    )

    parser.add_argument(
        '--num_test_episodes',
        type=int,
        default=default_config['num_test_episodes'],
        help=f"Number of episodes to run for testing. (default: {default_config['num_test_episodes']})"
    )

    parser.add_argument(
        '--test_retrieval_k',
        type=int,
        nargs='+',  # Accept one or more integer values
        default=default_config['test_retrieval_k'],
        help=f"List of k-values for retrieval-augmented mode. (default: {default_config['test_retrieval_k']})"
    )

    args = parser.parse_args()

    # --- Update CONFIG with parsed arguments ---
    # This will override the imported defaults for the rest of this script
    CONFIG['test_mode'] = args.test_mode
    CONFIG['num_test_episodes'] = args.num_test_episodes
    CONFIG['test_retrieval_k'] = args.test_retrieval_k
    # Note: args.test_log_name is used directly below

    print("--- 🚀 Initializing Test Run with Configuration ---")
    print(f"  - Test Log: {args.test_log_name}")  # <-- 🔻 NEW 🔻
    print(f"  - Test Mode: {CONFIG['test_mode']}")
    print(f"  - Test Episodes: {CONFIG['num_test_episodes']}")
    if CONFIG['test_mode'] == 'retrieval_augmented':
        print(f"  - Retrieval K-values: {CONFIG['test_retrieval_k']}")
    # --- 🔺 END MODIFIED 🔺 ---

    strategy = CONFIG['embedding_strategy']
    print(f"--- Running Testing Script in Stand-Alone Mode (strategy: '{strategy}') ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_dir = './checkpoints'
    artifacts_path = os.path.join(checkpoint_dir, 'training_artifacts.pth')

    # --- 🔻 MODIFIED: Load Data 🔻 ---
    print("\n📦 Loading test data...")
    loader = init_loader(CONFIG)
    loader.load_training_artifacts(artifacts_path)

    # --- New: Select *which* test log to transform ---
    test_log_name = args.test_log_name
    if not test_log_name:
        exit("❌ Error: No test log specified or found in config.")

    log_path = CONFIG['log_paths']['testing'].get(test_log_name)
    if not log_path:
        exit(f"❌ Error: Test log key '{test_log_name}' not found in CONFIG['log_paths']['testing'].")

    # Create a new dict with *only* the log we want to transform
    log_to_transform = {test_log_name: log_path}

    # Now, transform only that specific log
    print(f"Transforming log: '{test_log_name}' from {log_path}")
    testing_logs = loader.transform(log_to_transform)
    # --- 🔺 END MODIFIED 🔺 ---

    torch.manual_seed(42);
    np.random.seed(42)

    # --- Model Initialization ---
    model = create_model(CONFIG, loader, device)

    # --- Load Weights ---
    load_model_weights(model, checkpoint_dir, device)

    # --- 🔻 MODIFIED: Get correct log 🔻 ---
    # test_log_name is already defined from args
    unseen_log = testing_logs.get(test_log_name)
    if not unseen_log:
        exit(f"❌ Error: Test log '{test_log_name}' could not be processed.")
    # --- 🔺 END MODIFIED 🔺 ---

    print("\n🛠️ Creating test tasks...")
    # This call MUST now return (prefix, label, case_id) tuples
    # for the retrieval_augmented mode to work correctly.
    test_tasks = {
        'classification': get_task_data(unseen_log, 'classification'),
        'regression': get_task_data(unseen_log, 'regression')
    }

    # --- Select Evaluation Mode based on Config ---
    # This part now uses the (potentially overridden) CONFIG values
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
