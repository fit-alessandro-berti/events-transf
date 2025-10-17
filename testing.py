# testing.py
import torch
import random
import numpy as np
import pandas as pd
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

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def evaluate_model(model, test_tasks, num_shots_list, num_test_episodes=100):
    # This function's logic remains the same and does not need modification.
    print("\n🔬 Starting meta-testing on the Transformer-based Meta-Learner...")
    model.eval()
    for task_type, task_data in test_tasks.items():
        print(f"\n--- Evaluating task: {task_type} ---")
        if not task_data:
            print(f"Skipping {task_type}: No test data available.")
            continue
        if task_type == 'classification':
            class_dict = defaultdict(list)
            for seq, label in task_data: class_dict[label].append((seq, label))
            class_dict = {c: items for c, items in class_dict.items() if len(items) >= max(num_shots_list) + 1}
            available_classes = list(class_dict.keys())
            if len(available_classes) < 2:
                print("Classification test skipped: Need at least 2 classes with sufficient examples.")
                continue
            N_WAYS_TEST = min(len(available_classes), 7)
            print(f"Running classification test as a {N_WAYS_TEST}-way task.")

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
                        support_set.extend(samples[:k]); query_set.append(samples[k])
                else:
                    if len(task_data) < k + 1: continue
                    random.shuffle(task_data)
                    support_set, query_set = task_data[:k], task_data[k:k + 1]
                if not support_set or not query_set: continue
                with torch.no_grad():
                    predictions, true_labels = model(support_set, query_set, task_type)
                if predictions is None or true_labels is None or torch.all(true_labels == -100): continue
                if task_type == 'classification':
                    all_preds.extend(torch.argmax(predictions, dim=1).cpu().numpy())
                    all_labels.extend(true_labels.cpu().numpy())
                else:
                    all_preds.extend(predictions.view(-1).cpu().tolist())
                    all_labels.extend(true_labels.view(-1).cpu().tolist())
            if not all_labels: continue
            if task_type == 'classification':
                print(f"[{k}-shot] Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
            else:
                preds = inverse_transform_time(np.array(all_preds)); preds[preds < 0] = 0
                labels = inverse_transform_time(np.array(all_labels))
                print(f"[{k}-shot] MAE: {mean_absolute_error(labels, preds):.4f} | R-squared: {r2_score(labels, preds):.4f}")

def _extract_features_for_sklearn(trace, model, strategy):
    """Extracts a feature vector from a trace for use in sklearn models."""
    if strategy == 'learned':
        # Use the trained model to get a high-quality representation
        model.eval()
        with torch.no_grad():
            encoded_vector = model._process_batch([trace])
            return encoded_vector.squeeze(0).cpu().numpy()
    else: # pretrained
        # Use a simple mean of pre-computed embeddings
        event_vectors = []
        for event in trace:
            semantic_vec = event['activity_embedding'] + event['resource_embedding']
            numerical_vec = np.log1p([event['cost'], event['time_from_start'], event['time_from_previous']])
            event_vectors.append(np.concatenate([semantic_vec, numerical_vec]))
        if not event_vectors:
            return np.zeros(CONFIG['pretrained_settings']['embedding_dim'] + CONFIG['num_numerical_features'])
        return np.mean(np.array(event_vectors), axis=0)

def evaluate_sklearn_baselines(model, test_tasks, num_shots_list, num_test_episodes=100):
    strategy = model.strategy
    print(f"\n🧪 Starting evaluation of Scikit-Learn Baselines (feature extraction: '{strategy}')...")
    for task_type, task_data in test_tasks.items():
        print(f"\n--- Baseline task: {task_type} ---")
        if not task_data:
            print(f"Skipping {task_type}: No test data available.")
            continue
        # (The rest of the function logic is the same, just calling the new feature extractor)
        if task_type == 'classification':
            class_dict = defaultdict(list)
            for seq, label in task_data: class_dict[label].append((seq, label))
            class_dict = {c: items for c, items in class_dict.items() if len(items) >= max(num_shots_list) + 1}
            if len(class_dict.keys()) < 2: continue
            N_WAYS_TEST = min(len(class_dict.keys()), 7)

        for k in num_shots_list:
            all_preds, all_labels = [], []
            for _ in range(num_test_episodes):
                support_set, query_set = [], []
                if task_type == 'classification':
                    eligible_classes = [c for c, items in class_dict.items() if len(items) >= k + 1]
                    if len(eligible_classes) < N_WAYS_TEST: continue
                    episode_classes = random.sample(eligible_classes, N_WAYS_TEST)
                    for cls in episode_classes:
                        samples = random.sample(class_dict[cls], k+1)
                        support_set.extend(samples[:k]); query_set.append(samples[k])
                else: # Regression
                    if len(task_data) < k + 1: continue
                    random.shuffle(task_data)
                    support_set, query_set = task_data[:k], task_data[k:k+1]
                if not support_set or not query_set: continue
                # Use the new feature extraction helper
                X_train = np.array([_extract_features_for_sklearn(s[0], model, strategy) for s in support_set])
                y_train = np.array([s[1] for s in support_set])
                X_test = np.array([_extract_features_for_sklearn(q[0], model, strategy) for q in query_set])
                y_test = np.array([q[1] for q in query_set])
                if task_type == 'classification':
                    if len(np.unique(y_train)) < 2: continue
                    sk_model = LogisticRegression(max_iter=100)
                else:
                    sk_model = Ridge()
                try:
                    sk_model.fit(X_train, y_train)
                    all_preds.extend(sk_model.predict(X_test)); all_labels.extend(y_test)
                except ValueError: continue
            if not all_labels: continue
            if task_type == 'classification':
                print(f"[{k}-shot] Logistic Regression Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
            else:
                preds = inverse_transform_time(np.array(all_preds)); preds[preds < 0] = 0
                labels = inverse_transform_time(np.array(all_labels))
                print(f"[{k}-shot] Ridge Regression MAE: {mean_absolute_error(labels, preds):.4f} | R-squared: {r2_score(labels, preds):.4f}")

if __name__ == '__main__':
    strategy = CONFIG['embedding_strategy']
    print(f"--- Running Testing Script in Stand-Alone Mode (strategy: '{strategy}') ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_dir = './checkpoints'
    if not os.path.isdir(checkpoint_dir): exit("❌ Error: Checkpoint directory not found.")
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
    if not checkpoints: exit("❌ Error: No model checkpoints found.")

    latest_checkpoint_path = os.path.join(checkpoint_dir, sorted(checkpoints, key=lambda f: int(re.search(r'(\d+)', f).group(1)))[-1])
    print(f"🔍 Found latest checkpoint: {os.path.basename(latest_checkpoint_path)}")

    print("\n📦 Loading test data...")
    artifacts_path = os.path.join(checkpoint_dir, 'training_artifacts.pth')
    loader = XESLogLoader(strategy=strategy, sbert_model_name=CONFIG['pretrained_settings']['sbert_model'])
    loader.load_training_artifacts(artifacts_path)
    testing_logs = loader.transform(CONFIG['log_paths']['testing'])

    torch.manual_seed(42); np.random.seed(42)

    # --- Model Initialization ---
    if strategy == 'pretrained':
        model_params = {'embedding_dim': CONFIG['pretrained_settings']['embedding_dim']}
    else: # learned
        model_params = {
            'vocab_sizes': {'activity': len(loader.activity_to_id), 'resource': len(loader.resource_to_id)},
            'embedding_dims': {'activity': CONFIG['learned_settings']['activity_embedding_dim'], 'resource': CONFIG['learned_settings']['resource_embedding_dim']}
        }
    model = MetaLearner(
        strategy=strategy, num_feat_dim=CONFIG['num_numerical_features'],
        d_model=CONFIG['d_model'], n_heads=CONFIG['n_heads'], n_layers=CONFIG['n_layers'], dropout=CONFIG['dropout'], **model_params
    ).to(device)

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
