# File: evaluation/eval_baselines.py
import torch
import numpy as np
import random
from collections import defaultdict
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.exceptions import ConvergenceWarning
import warnings

# --- Import from project files ---
from config import CONFIG
from time_transf import inverse_transform_time


def _extract_features_for_sklearn(trace, model, strategy):
    """
    Extracts a feature vector from a trace for use in sklearn models.
    (Moved from testing.py)
    """
    if strategy == 'learned':
        # Use the trained model to get a high-quality representation
        model.eval()
        with torch.no_grad():
            encoded_vector = model._process_batch([trace])
            return encoded_vector.squeeze(0).cpu().numpy()
    else:  # pretrained
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
    """
    Evaluates simple sklearn models on the same episodic tasks.
    (Moved from testing.py)
    """
    strategy = model.strategy
    print(f"\n🧪 Starting evaluation of Scikit-Learn Baselines (feature extraction: '{strategy}')...")

    # Suppress warnings from sklearn models
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for task_type, task_data in test_tasks.items():
        print(f"\n--- Baseline task: {task_type} ---")
        if not task_data:
            print(f"Skipping {task_type}: No test data available.")
            continue

        # Re-build class_dict for sampling (to handle regression case)
        if task_type == 'classification':
            class_dict = defaultdict(list)
            # Assumes task_data is (prefix, label) or (prefix, label, case_id)
            for task_item in task_data:
                seq, label = task_item[0], task_item[1]
                class_dict[label].append((seq, label))
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
                        samples = random.sample(class_dict[cls], k + 1)
                        support_set.extend(samples[:k]);
                        query_set.append(samples[k])
                else:  # Regression
                    # Regression: task_data is a list of (prefix, label, case_id)
                    if len(task_data) < k + 1: continue
                    random.shuffle(task_data)
                    support_set_raw, query_set_raw = task_data[:k], task_data[k:k + 1]
                    support_set = [(s[0], s[1]) for s in support_set_raw]
                    query_set = [(q[0], q[1]) for q in query_set_raw]
                if not support_set or not query_set: continue

                # Use the feature extraction helper
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
                    all_preds.extend(sk_model.predict(X_test));
                    all_labels.extend(y_test)
                except ValueError:
                    continue
            if not all_labels: continue
            if task_type == 'classification':
                print(f"[{k}-shot] Logistic Regression Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
            else:
                preds = inverse_transform_time(np.array(all_preds));
                preds[preds < 0] = 0
                labels = inverse_transform_time(np.array(all_labels))
                print(
                    f"[{k}-shot] Ridge Regression MAE: {mean_absolute_error(labels, preds):.4f} | R-squared: {r2_score(labels, preds):.4f}")
