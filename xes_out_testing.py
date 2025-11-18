# File: xes_out_testing.py
import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import warnings
import pm4py
import copy
from datetime import datetime, timedelta
from tqdm import tqdm

# --- Import from project files ---
from config import CONFIG
from time_transf import transform_time, inverse_transform_time
from utils.model_utils import init_loader, create_model, load_model_weights
from utils.retrieval_utils import find_knn_indices

# Suppress pm4py warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pm4py')


# --- 🔻 FUNCTION MODIFIED 🔻 ---
def get_all_prefix_tasks(log, max_seq_len=10):
    """
    Generates a list of all possible prefix tasks from a log.

    Returns a list of dicts, where each dict contains:
    - 'prefix': The list of event dictionaries.
    - 'cls_label': The ground-truth activity_id of the *next* event.
    - 'reg_label': The ground-truth (transformed) remaining time.
    - 'case_id': The original case ID.
    - 'actual_next_activity_name': The ground-truth *name* of the next activity.
    - 'actual_remaining_time_hr': The ground-truth *untransformed* remaining time.
    """
    print("🛠️ Generating all prefix tasks from log...")
    tasks = []
    if not log: return tasks

    for trace in tqdm(log, desc="  - Processing traces"):
        if len(trace) < 2: continue

        case_id = trace[0]['case_id']

        # Iterate from prefix-length 1 up to prefix-length N-1
        for i in range(1, len(trace)):
            prefix = trace[:i]
            if len(prefix) > max_seq_len:
                prefix = prefix[-max_seq_len:]

            next_event = trace[i]
            last_event_in_prefix = prefix[-1]

            # --- Ground Truth Values ---

            # 1. Classification
            cls_label = next_event.get('activity_id', -100)  # -100 if unseen
            actual_next_activity_name = next_event.get('activity_name', 'Unknown')

            # 2. Regression
            remaining_time_sec = trace[-1]['timestamp'] - last_event_in_prefix['timestamp']
            actual_remaining_time_hr = remaining_time_sec / 3600.0
            if actual_remaining_time_hr < 0:
                actual_remaining_time_hr = 0.0
            reg_label = transform_time(actual_remaining_time_hr)

            # --- Store Task ---
            if cls_label != -100:
                tasks.append({
                    'prefix': prefix,
                    'cls_label': cls_label,
                    'reg_label': reg_label,
                    'case_id': case_id,
                    'actual_next_activity_name': actual_next_activity_name,
                    'actual_remaining_time_hr': actual_remaining_time_hr
                })
    print(f"✅ Generated {len(tasks)} prefix tasks.")
    return tasks


# --- 🔺 END MODIFIED FUNCTION 🔺 ---


def compute_all_embeddings(model, all_prefixes, batch_size=64):
    """
    Computes embeddings for all prefixes in batches.
    """
    print("🧠 Computing embeddings for all prefixes...")
    device = next(model.parameters()).device
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(all_prefixes), batch_size), desc="  - Embedding batches"):
            batch_prefixes = all_prefixes[i:i + batch_size]
            if not batch_prefixes:
                continue

            # For MoEModel, this returns the average embedding
            encoded_batch = model._process_batch(batch_prefixes)
            all_embeddings.append(encoded_batch.cpu())

    if not all_embeddings:
        return torch.empty(0, 0, device=device)

    all_embeddings_tensor = torch.cat(all_embeddings, dim=0).to(device)
    # L2-normalize for efficient cosine similarity
    all_embeddings_norm = F.normalize(all_embeddings_tensor, p=2, dim=1)
    print(f"✅ Computed {all_embeddings_norm.shape[0]} embeddings.")
    return all_embeddings_norm


# --- 🔻 FUNCTION MODIFIED 🔻 ---
def create_xes_trace(prefix_events, new_case_id,
                     pred_rem_time, pred_activity_name,
                     actual_rem_time, actual_next_activity):
    """
    Creates a single pm4py Trace object from a prefix and predictions.
    """
    new_trace = pm4py.objects.log.obj.Trace()
    new_trace.attributes['concept:name'] = new_case_id
    last_event_timestamp = None

    # 1. Add all prefix events
    for event_idx, event_data in enumerate(prefix_events):
        new_event_data = copy.deepcopy(event_data)

        new_event = pm4py.objects.log.obj.Event()
        new_event['concept:name'] = new_event_data['activity_name']
        new_event['org:resource'] = new_event_data.get('resource_name', 'Unknown')
        new_event['time:timestamp'] = datetime.fromtimestamp(new_event_data['timestamp'])

        for key, value in new_event_data.items():
            if key not in ['activity_name', 'resource_name', 'timestamp', 'concept:name', 'org:resource',
                            'time:timestamp',
                            'activity', 'resource', 'activity_embedding', 'resource_embedding']:
                new_event[key] = value

        # 2. Annotate the *last* event with all requested attributes
        if event_idx == len(prefix_events) - 1:
            # --- Renamed per request ---
            new_event['predictedRemainingTime'] = float(pred_rem_time)
            # --- Added per request ---
            new_event['actualRemainingTime'] = float(actual_rem_time)
            new_event['actualNextActivity'] = str(actual_next_activity)

            last_event_timestamp = new_event['time:timestamp']

        new_trace.append(new_event)

    # 3. Add the *new* predicted event
    if last_event_timestamp is not None:
        pred_event = pm4py.objects.log.obj.Event()
        pred_event['concept:name'] = pred_activity_name
        pred_event['org:resource'] = 'PREDICTED'
        pred_event['time:timestamp'] = last_event_timestamp + timedelta(seconds=1)
        pred_event['lifecycle:transition'] = 'complete'
        pred_event['predicted'] = True
        new_trace.append(pred_event)

    return new_trace


# --- 🔺 END MODIFIED FUNCTION 🔺 ---


if __name__ == '__main__':

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run prediction script to generate a new XES log.")

    default_config = CONFIG
    available_test_logs = list(default_config['log_paths']['testing'].keys())
    default_test_log = available_test_logs[0] if available_test_logs else None

    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help="Path to save the output XES log (e.g., 'predictions.xes.gz')."
    )

    parser.add_argument(
        '--test_log_name',
        type=str,
        default=default_test_log,
        choices=available_test_logs,
        help=f"Name of the test log to process. (default: {default_test_log})"
    )

    parser.add_argument(
        '--test_mode',
        type=str,
        default='retrieval_augmented',
        choices=['meta_learning', 'retrieval_augmented'],
        help=f"Evaluation mode. (default: retrieval_augmented)"
    )

    parser.add_argument(
        '--inference_k',
        type=int,
        default=default_config['test_retrieval_k'][0],  # Default to first k-value
        help="The 'k' to use for k-NN support set retrieval."
    )

    args = parser.parse_args()

    if args.test_mode != 'retrieval_augmented':
        print(f"❌ Error: This script requires 'retrieval_augmented' mode to build support sets for inference.")
        print("Please run with --test_mode retrieval_augmented")
        exit(1)

    print("--- 🚀 Initializing Prediction Script ---")
    print(f"  - Input Log: {args.test_log_name}")
    print(f"  - Output File: {args.output_file}")
    print(f"  - Inference Mode: {args.test_mode}")
    print(f"  - Support Set k: {args.inference_k}")

    # --- 1. Load Model & Data ---
    strategy = CONFIG['embedding_strategy']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_dir = './checkpoints'
    artifacts_path = os.path.join(checkpoint_dir, 'training_artifacts.pth')

    print("\n📦 Loading model and data artifacts...")
    loader = init_loader(CONFIG)
    loader.load_training_artifacts(artifacts_path)

    id_to_activity_name = loader.training_activity_names

    model = create_model(CONFIG, loader, device)
    load_model_weights(model, checkpoint_dir, device)
    model.eval()

    proto_head_to_use = model.experts[0].proto_head

    log_path = CONFIG['log_paths']['testing'].get(args.test_log_name)
    log_to_transform = {args.test_log_name: log_path}
    testing_logs = loader.transform(log_to_transform)
    unseen_log = testing_logs.get(args.test_log_name)

    if not unseen_log:
        exit(f"❌ Error: Test log '{args.test_log_name}' could not be processed.")

    # --- 2. Generate Tasks and Embeddings ---
    all_tasks = get_all_prefix_tasks(unseen_log)
    if not all_tasks:
        exit(f"❌ Error: No valid prefixes (length > 1) found in the log.")

    all_prefixes = [t['prefix'] for t in all_tasks]
    all_embeddings_norm = compute_all_embeddings(model, all_prefixes).to(device)

    # Get all labels and case_ids for building support sets
    all_case_ids_array = np.array([t['case_id'] for t in all_tasks])
    all_cls_labels = torch.tensor([t['cls_label'] for t in all_tasks], dtype=torch.long, device=device)
    all_reg_labels = torch.tensor([t['reg_label'] for t in all_tasks], dtype=torch.float32, device=device)

    # --- 3. Main Prediction Loop ---
    print(f"🚀 Starting prediction for {len(all_tasks)} prefixes...")
    output_log = pm4py.objects.log.obj.EventLog()
    k_for_support = args.inference_k

    # --- 🔻 NEW: Lists to store metrics 🔻 ---
    all_time_errors = []
    activity_matches = []
    # --- 🔺 END NEW 🔺 ---

    for i, task in enumerate(tqdm(all_tasks, desc="  - Predicting")):
        query_embedding_norm = all_embeddings_norm[i:i + 1]  # [1, D]
        query_case_id = task['case_id']
        query_prefix = task['prefix']

        # --- 🔻 Get ground truth values from task 🔻 ---
        actual_rem_time = task['actual_remaining_time_hr']
        actual_next_activity = task['actual_next_activity_name']
        # --- 🔺 END 🔺 ---

        # --- Find k-NN Support Set (excluding same-case) ---
        same_case_indices_np = np.where(all_case_ids_array == query_case_id)[0]
        mask_tensor = torch.from_numpy(same_case_indices_np).to(device)

        support_indices = find_knn_indices(
            query_embedding_norm,
            all_embeddings_norm,
            k=k_for_support,
            indices_to_mask=mask_tensor
        )

        if support_indices.numel() == 0:
            # Cannot predict without a support set
            predicted_activity_name = "Error (No Support)"
            predicted_rem_time_hours = 0.0
        else:
            # --- Prediction 1: Next Activity (Classification) ---
            cls_support_embeddings = all_embeddings_norm[support_indices]
            cls_support_labels = all_cls_labels[support_indices]

            with torch.no_grad():
                logits, proto_classes, _ = proto_head_to_use.forward_classification(
                    cls_support_embeddings, cls_support_labels, query_embedding_norm
                )

            pred_label_idx = torch.argmax(logits, dim=1).item()
            predicted_activity_id = proto_classes[pred_label_idx].item()

            if 0 <= predicted_activity_id < len(id_to_activity_name):
                predicted_activity_name = id_to_activity_name[predicted_activity_id]
            else:
                predicted_activity_name = "Unknown (Pred. ID)"

            # --- Prediction 2: Remaining Time (Regression) ---
            reg_support_embeddings = all_embeddings_norm[support_indices]
            reg_support_labels = all_reg_labels[support_indices]

            with torch.no_grad():
                prediction, _ = proto_head_to_use.forward_regression(
                    reg_support_embeddings, reg_support_labels, query_embedding_norm
                )

            predicted_reg_value = prediction[0].item()
            predicted_rem_time_hours = inverse_transform_time(predicted_reg_value)
            if predicted_rem_time_hours < 0:
                predicted_rem_time_hours = 0.0

        # --- 🔻 NEW: Store metrics for this prefix 🔻 ---
        all_time_errors.append(abs(predicted_rem_time_hours - actual_rem_time))
        activity_matches.append(predicted_activity_name == actual_next_activity)
        # --- 🔺 END NEW 🔺 ---

        # --- 4. Create and append the new XES trace ---
        new_case_id = f"{query_case_id}@{len(query_prefix)}"

        # --- 🔻 Pass new values to function 🔻 ---
        new_trace = create_xes_trace(
            query_prefix,
            new_case_id,
            predicted_rem_time_hours,
            predicted_activity_name,
            actual_rem_time,
            actual_next_activity
        )
        # --- 🔺 END 🔺 ---
        output_log.append(new_trace)

    # --- 🔻 NEW: 5. Calculate and add log-level attributes 🔻 ---
    mean_avg_error_time = np.mean(all_time_errors) if all_time_errors else 0.0
    pct_activity_match = np.mean(activity_matches) if activity_matches else 0.0

    output_log.attributes["mean_remaining_time_mae"] = float(mean_avg_error_time)
    output_log.attributes["next_activity_accuracy"] = float(pct_activity_match)

    print("\n📈 Log-level Metrics Calculated:")
    print(f"  - Next Activity Accuracy: {pct_activity_match:.4f}")
    print(f"  - Remaining Time MAE (hours): {mean_avg_error_time:.4f}")
    # --- 🔺 END NEW 🔺 ---

    # --- 6. Save the final XES log ---
    print(f"\n💾 Saving {len(output_log)} predicted traces to {args.output_file}...")
    # --- 🔻 MODIFIED: Use requested export command 🔻 ---
    pm4py.write_xes(output_log, args.output_file, variant_str="line_by_line", compression='gzip' in args.output_file)
    # --- 🔺 END MODIFIED 🔺 ---
    print("✅ Done.")
