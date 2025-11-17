# File: testing.py
import torch
import torch.nn.functional as F
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
from tqdm import tqdm

# --- Stand-alone execution imports ---
from config import CONFIG
from data_generator import XESLogLoader, get_task_data
from components.meta_learner import MetaLearner
from time_transf import inverse_transform_time

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def evaluate_model(model, test_tasks, num_shots_list, num_test_episodes=100):
    """
    Standard episodic meta-learning evaluation.
    Samples support/query sets for each episode.
    """
    print("\n🔬 Starting meta-testing on the Transformer-based Meta-Learner...")
    model.eval()
    for task_type, task_data in test_tasks.items():
        print(f"\n--- Evaluating task: {task_type} ---")
        if not task_data:
            print(f"Skipping {task_type}: No test data available.")
            continue
        if task_type == 'classification':
            class_dict = defaultdict(list)
            # Assumes task_data is (prefix, label) or (prefix, label, case_id)
            for task_item in task_data:
                seq, label = task_item[0], task_item[1]
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

            # Re-build class_dict for sampling (to handle regression case)
            if task_type == 'classification':
                ep_class_dict = defaultdict(list)
                for task_item in task_data:
                    seq, label = task_item[0], task_item[1]
                    ep_class_dict[label].append((seq, label))
                ep_class_dict = {c: items for c, items in ep_class_dict.items() if len(items) >= k + 1}

            for _ in range(num_test_episodes):
                support_set, query_set = [], []
                if task_type == 'classification':
                    eligible_classes = [c for c, items in ep_class_dict.items() if len(items) >= k + 1]
                    if len(eligible_classes) < N_WAYS_TEST: continue
                    episode_classes = random.sample(eligible_classes, N_WAYS_TEST)
                    for cls in episode_classes:
                        samples = random.sample(ep_class_dict[cls], k + 1)
                        support_set.extend(samples[:k]);
                        query_set.append(samples[k])
                else:
                    # Regression: task_data is a list of (prefix, label, case_id)
                    if len(task_data) < k + 1: continue
                    random.shuffle(task_data)
                    # Slicing will work, just ignores the 3rd element (case_id)
                    support_set_raw, query_set_raw = task_data[:k], task_data[k:k + 1]
                    support_set = [(s[0], s[1]) for s in support_set_raw]
                    query_set = [(q[0], q[1]) for q in query_set_raw]

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
                # Filter out invalid -100 labels
                valid_indices = [i for i, label in enumerate(all_labels) if label != -100]
                if not valid_indices: continue
                valid_preds = [all_preds[i] for i in valid_indices]
                valid_labels = [all_labels[i] for i in valid_indices]
                if not valid_labels: continue
                print(f"[{k}-shot] Accuracy: {accuracy_score(valid_labels, valid_preds):.4f}")
            else:
                preds = inverse_transform_time(np.array(all_preds));
                preds[preds < 0] = 0
                labels = inverse_transform_time(np.array(all_labels))
                print(
                    f"[{k}-shot] MAE: {mean_absolute_error(labels, preds):.4f} | R-squared: {r2_score(labels, preds):.4f}")


def _get_all_test_embeddings(model, test_tasks_list, batch_size=64):
    """
    Helper function to compute embeddings for all (prefix, label, case_id) tuples.

    *** ASSUMPTION ***: This function assumes test_tasks_list contains tuples of
    (prefix, label, case_id) as generated by a modified get_task_data function.
    """
    all_embeddings = []
    all_labels = []
    all_case_ids = []  # <-- For contamination fix

    device = next(model.parameters()).device
    model.eval()

    try:
        # Check if data has 3 items (prefix, label, case_id)
        _ = test_tasks_list[0][2]
    except (IndexError, TypeError):
        print("\n" + "=" * 50)
        print("❌ ERROR in _get_all_test_embeddings:")
        print("Test data does not contain case_ids.")
        print("Please modify get_task_data in data_generator.py to return:")
        print("(prefix, label, case_id) tuples.")
        print("Aborting retrieval-augmented evaluation.")
        print("=" * 50 + "\n")
        return None, None, None

    with torch.no_grad():
        for i in tqdm(range(0, len(test_tasks_list), batch_size), desc="Pre-computing test embeddings"):
            batch_tasks = test_tasks_list[i:i + batch_size]
            sequences = [t[0] for t in batch_tasks]
            labels = [t[1] for t in batch_tasks]
            case_ids = [t[2] for t in batch_tasks]  # <-- Get case_ids

            if not sequences: continue

            # Use the model's internal processing function
            encoded_batch = model._process_batch(sequences)

            all_embeddings.append(encoded_batch.cpu())
            all_labels.extend(labels)
            all_case_ids.extend(case_ids)  # <-- Store case_ids

    if not all_embeddings:
        return None, None, None

    all_embeddings_tensor = torch.cat(all_embeddings, dim=0).to(device)
    all_labels_tensor = torch.as_tensor(all_labels, device=device)
    all_case_ids_array = np.array(all_case_ids)  # Use numpy for easy string comparison

    return all_embeddings_tensor, all_labels_tensor, all_case_ids_array


def evaluate_retrieval_augmented(model, test_tasks, num_retrieval_k_list, num_test_queries=200):
    """
    Retrieval-Augmented evaluation.
    1. Computes all test embeddings.
    2. For each query, finds k-NN *from other cases* to form the support set.
    """
    print("\n🔬 Starting Retrieval-Augmented Evaluation...")
    model.eval()

    task_embeddings = {}

    # --- 1. Pre-compute all embeddings ---
    for task_type, task_data in test_tasks.items():
        if not task_data:
            print(f"Skipping {task_type}: No test data available.")
            continue

        # Get embeddings, labels, and case_ids
        embeddings, labels, case_ids = _get_all_test_embeddings(model, task_data)

        if embeddings is None:  # Error already printed in helper
            return

        # L2-normalize for efficient cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        task_embeddings[task_type] = (embeddings, labels, case_ids)  # Store all 3
        print(f"  - Pre-computed {embeddings.shape[0]} embeddings for {task_type}.")

    # --- 2. Evaluate using k-NN retrieval ---
    for task_type, (all_embeddings, all_labels, all_case_ids) in task_embeddings.items():
        print(f"\n--- Evaluating task: {task_type} ---")

        num_total_samples = all_embeddings.shape[0]
        if num_total_samples < 2:
            print("Skipping: Not enough samples to evaluate.")
            continue

        num_queries = min(num_test_queries, num_total_samples)
        query_indices = random.sample(range(num_total_samples), num_queries)

        for k in num_retrieval_k_list:
            # Need at least k+1 samples (1 query, k support) *from different cases*
            # This check is complex, so we'll just check if k < N
            if k >= num_total_samples:
                print(f"Skipping [k={k}]: k is larger than total samples.")
                continue

            all_preds, all_true_labels = [], []

            for query_idx in query_indices:
                query_embedding = all_embeddings[query_idx:query_idx + 1]  # [1, D]
                query_label = all_labels[query_idx]
                query_case_id = all_case_ids[query_idx]  # <-- Get query case_id

                # --- Find k-NN Support Set ---
                # Cosine similarity: (1, D) @ (D, N) -> (1, N)
                sims = query_embedding @ all_embeddings.T

                # --- CONTAMINATION FIX ---
                # Find all indices that share the same case ID as the query
                same_case_indices = np.where(all_case_ids == query_case_id)[0]

                # Mask out ALL samples from the same case (including the query itself)
                same_case_indices_tensor = torch.from_numpy(same_case_indices).to(sims.device)
                sims[0, same_case_indices_tensor] = -float('inf')
                # --- END FIX ---

                # Get top k most similar (now guaranteed to be from different cases)
                top_k_indices = torch.topk(sims.squeeze(0), k).indices

                support_embeddings = all_embeddings[top_k_indices]  # [k, D]
                support_labels = all_labels[top_k_indices]  # [k]

                # --- Get Prediction ---
                with torch.no_grad():
                    if task_type == 'classification':
                        logits, proto_classes = model.proto_head.forward_classification(
                            support_embeddings, support_labels, query_embedding
                        )

                        # --- "NO INVALID QUERIES" FIX ---
                        # Get the index of the top logit (e.g., 0, 1, 2...)
                        pred_label_idx = torch.argmax(logits, dim=1).item()

                        # Find the *actual* class label (e.g., "Activity C")
                        # that corresponds to this index
                        predicted_class_label = proto_classes[pred_label_idx].item()

                        all_preds.append(predicted_class_label)
                        all_true_labels.append(query_label.item())
                        # --- END FIX ---

                    else:  # Regression
                        prediction = model.proto_head.forward_regression(
                            support_embeddings, support_labels.float(), query_embedding
                        )
                        all_preds.append(prediction.item())
                        all_true_labels.append(query_label.item())

            if not all_true_labels: continue

            # --- 3. Report Metrics ---
            if task_type == 'classification':
                # No need to filter for -100 anymore
                print(
                    f"[{k}-NN] Retrieval Accuracy: {accuracy_score(all_true_labels, all_preds):.4f} (on {len(all_true_labels)} queries)")
            else:
                preds = inverse_transform_time(np.array(all_preds));
                preds[preds < 0] = 0
                labels = inverse_transform_time(np.array(all_true_labels))
                print(
                    f"[{k}-NN] Retrieval MAE: {mean_absolute_error(labels, preds):.4f} | R-squared: {r2_score(labels, preds):.4f}")


def _extract_features_for_sklearn(trace, model, strategy):
    """Extracts a feature vector from a trace for use in sklearn models."""
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
    strategy = model.strategy
    print(f"\n🧪 Starting evaluation of Scikit-Learn Baselines (feature extraction: '{strategy}')...")
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


if __name__ == '__main__':
    strategy = CONFIG['embedding_strategy']
    print(f"--- Running Testing Script in Stand-Alone Mode (strategy: '{strategy}') ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_dir = './checkpoints'
    if not os.path.isdir(checkpoint_dir): exit("❌ Error: Checkpoint directory not found.")
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
    if not checkpoints: exit("❌ Error: No model checkpoints found.")

    latest_checkpoint_path = os.path.join(checkpoint_dir,
                                          sorted(checkpoints, key=lambda f: int(re.search(r'(\d+)', f).group(1)))[-1])
    print(f"🔍 Found latest checkpoint: {os.path.basename(latest_checkpoint_path)}")

    print("\n📦 Loading test data...")
    artifacts_path = os.path.join(checkpoint_dir, 'training_artifacts.pth')
    loader = XESLogLoader(strategy=strategy, sbert_model_name=CONFIG['pretrained_settings']['sbert_model'])
    loader.load_training_artifacts(artifacts_path)
    testing_logs = loader.transform(CONFIG['log_paths']['testing'])

    torch.manual_seed(42);
    np.random.seed(42)

    # --- Model Initialization ---
    if strategy == 'pretrained':
        model_params = {'embedding_dim': CONFIG['pretrained_settings']['embedding_dim']}
    else:  # learned
        model_params = {
            'char_vocab_size': len(loader.char_to_id),
            'char_embedding_dim': CONFIG['learned_settings']['char_embedding_dim'],
            'char_cnn_output_dim': CONFIG['learned_settings']['char_cnn_output_dim'],
        }
    model = MetaLearner(
        strategy=strategy, num_feat_dim=CONFIG['num_numerical_features'],
        d_model=CONFIG['d_model'], n_heads=CONFIG['n_heads'], n_layers=CONFIG['n_layers'], dropout=CONFIG['dropout'],
        **model_params
    ).to(device)

    # Pass the character vocabulary to the model
    if strategy == 'learned':
        model.set_char_vocab(loader.char_to_id)

    print(f"💾 Loading weights from {latest_checkpoint_path}...")
    model.load_state_dict(torch.load(latest_checkpoint_path, map_location=device))

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
