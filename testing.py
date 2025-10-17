# testing.py
import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.exceptions import ConvergenceWarning
from collections import defaultdict
import os
import re
import warnings

# --- Stand-alone execution imports ---
from config import CONFIG
from data_generator import XESLogLoader, get_task_data
from components.meta_learner import MetaLearner
from time_transf import inverse_transform_time

# Suppress convergence warnings from scikit-learn for small K
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def evaluate_model(model, test_tasks, num_shots_list, num_test_episodes=100):
    # This function remains unchanged in its core logic, but let's provide it for completeness
    print("\n🔬 Starting meta-testing on the Transformer-based Meta-Learner...")
    model.eval()
    results = {}
    device = next(model.parameters()).device

    for task_type, task_data in test_tasks.items():
        print(f"\n--- Evaluating task: {task_type} ---")
        if not task_data:
            print(f"Skipping {task_type}: No test data available.")
            continue

        if task_type == 'classification':
            class_dict = defaultdict(list)
            for seq, label in task_data:
                class_dict[label].append((seq, label))
            class_dict = {c: items for c, items in class_dict.items() if len(items) >= max(num_shots_list) + 1}
            available_classes = list(class_dict.keys())
            if len(available_classes) < 2:
                print("Classification test skipped: Need at least 2 classes with sufficient examples.")
                continue
            N_WAYS_TEST = min(len(available_classes), 7)
            print(f"Running classification test as a {N_WAYS_TEST}-way task.")

        for k in num_shots_list:
            all_preds, all_labels = [], []
            episodes_generated = 0

            for _ in range(num_test_episodes):
                support_set, query_set = [], []

                if task_type == 'classification':
                    eligible_classes = [c for c, items in class_dict.items() if len(items) >= k + 1]
                    if len(eligible_classes) < N_WAYS_TEST: continue
                    episode_classes = random.sample(eligible_classes, N_WAYS_TEST)
                    for cls in episode_classes:
                        samples = random.sample(class_dict[cls], k + 1)
                        support_set.extend(samples[:k])
                        query_set.append(samples[k])
                    random.shuffle(support_set)
                    random.shuffle(query_set)
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
                if torch.all(true_labels == -100): continue

                if task_type == 'classification':
                    pred_idx = torch.argmax(predictions, dim=1).cpu().numpy()
                    all_preds.extend(pred_idx)
                    all_labels.extend(true_labels.cpu().numpy())
                else:
                    all_preds.extend(predictions.view(-1).cpu().tolist())
                    all_labels.extend(true_labels.view(-1).cpu().tolist())

            if episodes_generated == 0:
                print(f"[{k}-shot] Skipped: Not enough data to generate episodes.")
                continue

            if task_type == 'classification':
                accuracy = accuracy_score(all_labels, all_preds)
                print(f"[{k}-shot] Accuracy: {accuracy:.4f}")
            else:  # Regression
                valid_preds = inverse_transform_time(np.array(all_preds))
                valid_labels = inverse_transform_time(np.array(all_labels))
                valid_preds[valid_preds < 0] = 0
                mae = mean_absolute_error(valid_labels, valid_preds)
                r2 = r2_score(valid_labels, valid_preds)
                print(f"[{k}-shot] MAE: {mae:.4f} | R-squared: {r2:.4f}")

def _extract_features_for_sklearn(model, trace):
    """
    Uses the trained meta-learner model to encode a single trace prefix
    into a fixed-size feature vector for use with scikit-learn.
    """
    model.eval()
    with torch.no_grad():
        # The model's _process_batch expects a list of sequences
        encoded_vector = model._process_batch([trace])
        # Return the feature vector as a numpy array
        return encoded_vector.squeeze(0).cpu().numpy()

def evaluate_sklearn_baselines(model, test_tasks, num_shots_list, num_test_episodes=100):
    print("\n🧪 Starting evaluation of Scikit-Learn Baselines (using learned representations)...")
    for task_type, task_data in test_tasks.items():
        print(f"\n--- Baseline task: {task_type} ---")
        if not task_data:
            print(f"Skipping {task_type}: No test data available.")
            continue

        if task_type == 'classification':
            class_dict = defaultdict(list)
            for seq, label in task_data:
                class_dict[label].append((seq, label))
            class_dict = {c: items for c, items in class_dict.items() if len(items) >= max(num_shots_list) + 1}
            available_classes = list(class_dict.keys())
            if len(available_classes) < 2: continue
            N_WAYS_TEST = min(len(available_classes), 7)

        for k in num_shots_list:
            all_preds, all_labels = [], []

            for _ in range(num_test_episodes):
                support_set, query_set = [], []
                if task_type == 'classification':
                    eligible_classes = [c for c, items in class_dict.items() if len(items) >= k + 1]
                    if len(eligible_classes) < N_WAYS_TEST: continue
                    episode_classes = random.sample(eligible_classes, N_WAYS_TEST)
                    for cls in episode_classes:
                        samples = random.sample(class_dict[cls], k + 1)
                        support_set.extend(samples[:k])
                        query_set.append(samples[k])
                else:  # Regression
                    if len(task_data) < k + 1: continue
                    random.shuffle(task_data)
                    support_set = task_data[:k]
                    query_set = task_data[k:k + 1]

                if not support_set or not query_set: continue

                # Use the model to extract features
                X_train = np.array([_extract_features_for_sklearn(model, s[0]) for s in support_set])
                y_train = np.array([s[1] for s in support_set])
                X_test = np.array([_extract_features_for_sklearn(model, q[0]) for q in query_set])
                y_test = np.array([q[1] for q in query_set])

                if task_type == 'classification':
                    if len(np.unique(y_train)) < 2: continue
                    sk_model = LogisticRegression(max_iter=100)
                else:
                    sk_model = Ridge()
                try:
                    sk_model.fit(X_train, y_train)
                    preds = sk_model.predict(X_test)
                    all_preds.extend(preds)
                    all_labels.extend(y_test)
                except ValueError:
                    continue

            if not all_labels:
                print(f"[{k}-shot] Skipped: Not enough data to generate episodes.")
                continue

            if task_type == 'classification':
                accuracy = accuracy_score(all_labels, all_preds)
                print(f"[{k}-shot] Logistic Regression Accuracy: {accuracy:.4f}")
            else:
                valid_preds = inverse_transform_time(np.array(all_preds))
                valid_labels = inverse_transform_time(np.array(all_labels))
                valid_preds[valid_preds < 0] = 0
                mae = mean_absolute_error(valid_labels, valid_preds)
                r2 = r2_score(valid_labels, valid_preds)
                print(f"[{k}-shot] Ridge Regression MAE: {mae:.4f} | R-squared: {r2:.4f}")


if __name__ == '__main__':
    print("--- Running Testing Script in Stand-Alone Mode ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_dir = './checkpoints'
    if not os.path.isdir(checkpoint_dir): exit(f"❌ Error: Checkpoint directory '{checkpoint_dir}' not found.")

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
    if not checkpoints: exit(f"❌ Error: No model checkpoints found in '{checkpoint_dir}'.")

    epoch_map = {int(re.search(r'model_epoch_(\d+).pth', f).group(1)): f for f in checkpoints}
    latest_checkpoint_path = os.path.join(checkpoint_dir, epoch_map[max(epoch_map.keys())])
    print(f"🔍 Found latest checkpoint: {os.path.basename(latest_checkpoint_path)}")

    print("\n📦 Loading test data...")
    artifacts_path = os.path.join(checkpoint_dir, 'training_artifacts.pth')

    loader = XESLogLoader()
    # Load the training artifacts (maps, initial embeddings)
    loader.load_training_artifacts(artifacts_path)
    # Transform ONLY the test logs, mapping unseen activities
    testing_logs = loader.transform(CONFIG['log_paths']['testing'])

    torch.manual_seed(42)
    np.random.seed(42)

    # --- Model Initialization ---
    vocab_sizes = {
        'activity': len(loader.activity_to_id),
        'resource': len(loader.resource_to_id)
    }
    embedding_dims = {
        'activity': CONFIG['activity_embedding_dim'],
        'resource': CONFIG['resource_embedding_dim']
    }
    model = MetaLearner(
        vocab_sizes=vocab_sizes,
        embedding_dims=embedding_dims,
        num_feat_dim=CONFIG['num_numerical_features'],
        d_model=CONFIG['d_model'], n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'], dropout=CONFIG['dropout']
    ).to(device)

    # Initialize embeddings in the same way as in training for consistency,
    # even though we are about to load trained weights. This ensures the model
    # architecture is identical.
    model.initialize_embeddings(
        initial_activity_embs=loader.initial_activity_embeddings,
        initial_resource_embs=loader.initial_resource_embeddings,
        similarity_coeff=CONFIG['similarity_coeff']
    )

    print(f"💾 Loading weights from {latest_checkpoint_path}...")
    model.load_state_dict(torch.load(latest_checkpoint_path, map_location=device))

    test_log_name = list(CONFIG['log_paths']['testing'].keys())[0]
    unseen_log = testing_logs.get(test_log_name)
    if not unseen_log: exit(f"❌ Error: Test log '{test_log_name}' could not be processed.")

    print("\n🛠️ Creating test tasks...")
    test_tasks = {
        'classification': get_task_data(unseen_log, 'classification'),
        'regression': get_task_data(unseen_log, 'regression')
    }

    evaluate_model(model, test_tasks, CONFIG['num_shots_test'], CONFIG['num_test_episodes'])
    evaluate_sklearn_baselines(model, test_tasks, CONFIG['num_shots_test'], CONFIG['num_test_episodes'])
