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


# --- FUNCTION MODIFIED ---
def get_all_prefix_tasks(log, max_seq_len=10):
    """
    Generates a list of all possible prefix tasks from a log.
    Now also records whether the penultimate event's activity equals the last event's activity.
    """
    print("Generating all prefix tasks from log...")
    tasks = []
    if not log: return tasks

    for trace in tqdm(log, desc="  - Processing traces"):
        if len(trace) < 3: continue  # Need at least 3 events to have a penultimate event

        case_id = trace[0]['case_id']

        for i in range(1, len(trace)):
            prefix = trace[:i]
            if len(prefix) > max_seq_len:
                prefix = prefix[-max_seq_len:]

            next_event = trace[i]
            last_event_in_prefix = prefix[-1]

            # --- Classification ---
            cls_label = next_event.get('activity_id', -100)
            actual_next_activity_name = next_event.get('activity_name', 'Unknown')

            # --- Regression ---
            remaining_time_sec = trace[-1]['timestamp'] - last_event_in_prefix['timestamp']
            actual_remaining_time_hr = remaining_time_sec / 3600.0
            if actual_remaining_time_hr < 0:
                actual_remaining_time_hr = 0.0
            reg_label = transform_time(actual_remaining_time_hr)

            # --- NEW: Is the penultimate activity the same as the last one? ---
            is_repeated = 0.0
            if len(prefix) >= 2:
                penultimate_activity = prefix[-2].get('activity_name', '')
                last_activity = prefix[-1].get('activity_name', '')
                is_repeated = 1.0 if penultimate_activity == last_activity else 0.0

            if cls_label != -100:
                tasks.append({
                    'prefix': prefix,
                    'cls_label': cls_label,
                    'reg_label': reg_label,
                    'case_id': case_id,
                    'actual_next_activity_name': actual_next_activity_name,
                    'actual_remaining_time_hr': actual_remaining_time_hr,
                    'is_last_activity_repeated': is_repeated   # <-- NEW field
                })
    print(f"Generated {len(tasks)} prefix tasks.")
    return tasks


def compute_all_embeddings(model, all_prefixes, batch_size=64):
    print("Computing embeddings for all prefixes...")
    device = next(model.parameters()).device
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(all_prefixes), batch_size), desc="  - Embedding batches"):
            batch_prefixes = all_prefixes[i:i + batch_size]
            if not batch_prefixes:
                continue

            encoded_batch = model._process_batch(batch_prefixes)
            all_embeddings.append(encoded_batch.cpu())

    if not all_embeddings:
        return torch.empty(0, 0, device=device)

    all_embeddings_tensor = torch.cat(all_embeddings, dim=0).to(device)
    all_embeddings_norm = F.normalize(all_embeddings_tensor, p=2, dim=1)
    print(f"Computed {all_embeddings_norm.shape[0]} embeddings.")
    return all_embeddings_norm


def create_xes_trace(prefix_events, new_case_id,
                     pred_rem_time, pred_activity_name,
                     actual_rem_time, actual_next_activity):
    new_trace = pm4py.objects.log.obj.Trace()
    new_trace.attributes['concept:name'] = new_case_id
    last_event_timestamp = None

    for event_idx, event_data in enumerate(prefix_events):
        new_event_data = copy.deepcopy(event_data)
        new_event = pm4py.objects.log.obj.Event()
        new_event['concept:name'] = new_event_data['activity_name']
        new_event['org:resource'] = new_event_data.get('resource_name', 'Unknown')
        new_event['time:timestamp'] = datetime.fromtimestamp(new_event_data['timestamp'])

        for key, value in new_event_data.items():
            if key not in ['activity_name', 'resource_name', 'timestamp', 'concept:name', 'org:resource',
                           'time:timestamp', 'activity', 'resource', 'activity_embedding', 'resource_embedding']:
                new_event[key] = value

        # Annotate the LAST event in the prefix with predictions and ground-truth
        if event_idx == len(prefix_events) - 1:
            new_event['predictedRemainingTime'] = float(pred_rem_time)
            new_event['actualRemainingTime'] = float(actual_rem_time)
            new_event['actualNextActivity'] = str(actual_next_activity)
            last_event_timestamp = new_event['time:timestamp']

        new_trace.append(new_event)

    # Add the predicted next event
    if last_event_timestamp is not None:
        pred_event = pm4py.objects.log.obj.Event()
        pred_event['concept:name'] = pred_activity_name
        pred_event['org:resource'] = 'PREDICTED'
        pred_event['time:timestamp'] = last_event_timestamp + timedelta(seconds=1)
        pred_event['lifecycle:transition'] = 'complete'
        pred_event['predicted'] = True
        new_trace.append(pred_event)

    return new_trace


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run prediction script to generate a new XES log.")
    default_config = CONFIG
    available_test_logs = list(default_config['log_paths']['testing'].keys())
    default_test_log = available_test_logs[0] if available_test_logs else None

    parser.add_argument('--output_file', type=str, required=True,
                        help="Path to save the output XES log (e.g., 'predictions.xes.gz').")
    parser.add_argument('--test_log_name', type=str, default=default_test_log,
                        choices=available_test_logs,
                        help=f"Name of the test log to process. (default: {default_test_log})")
    parser.add_argument('--test_mode', type=str, default='retrieval_augmented',
                        choices=['meta_learning', 'retrieval_augmented'])
    parser.add_argument('--inference_k', type=int,
                        default=default_config['test_retrieval_k'][0],
                        help="The 'k' to use for k-NN support set retrieval.")
    args = parser.parse_args()

    if args.test_mode != 'retrieval_augmented':
        print("This script requires 'retrieval_augmented' mode.")
        exit(1)

    print("--- Initializing Prediction Script ---")
    print(f"  - Input Log: {args.test_log_name}")
    print(f"  - Output File: {args.output_file}")
    print(f"  - Inference Mode: {args.test_mode}")
    print(f"  - Support Set k: {args.inference_k}")

    strategy = CONFIG['embedding_strategy']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_dir = './checkpoints'
    artifacts_path = os.path.join(checkpoint_dir, 'training_artifacts.pth')

    print("\nLoading model and data artifacts...")
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
        exit(f"Test log '{args.test_log_name}' could not be processed.")

    # --- Generate tasks (now includes is_last_activity_repeated) ---
    all_tasks = get_all_prefix_tasks(unseen_log)
    if not all_tasks:
        exit("No valid prefixes found in the log.")

    all_prefixes = [t['prefix'] for t in all_tasks]
    all_embeddings_norm = compute_all_embeddings(model, all_prefixes).to(device)

    all_case_ids_array = np.array([t['case_id'] for t in all_tasks])
    all_cls_labels = torch.tensor([t['cls_label'] for t in all_tasks], dtype=torch.long, device=device)
    all_reg_labels = torch.tensor([t['reg_label'] for t in all_tasks], dtype=torch.float32, device=device)

    # --- NEW: Containers for the two log-level metrics ---
    repeated_activity_ratios = []          # For each prefix: 1.0 if penultimate == last, else 0.0
    remaining_time_errors = []             # |actualRemainingTime - predictedRemainingTime| for each prefix

    print(f"Starting prediction for {len(all_tasks)} prefixes...")
    output_log = pm4py.objects.log.obj.EventLog()
    k_for_support = args.in Dernference_k

    for i, task in enumerate(tqdm(all_tasks, desc="  - Predicting")):
        query_embedding_norm = all_embeddings_norm[i:i + 1]
        query_case_id = task['case_id']
        query_prefix = task['prefix']

        actual_rem_time = task['actual_remaining_time_hr']
        actual_next_activity = task['actual_next_activity_name']

        # --- Record whether last activity is repeated (penultimate == last) ---
        repeated_activity_ratios.append(task['is_last_activity_repeated'])

        same_case_indices_np = np.where(all_case_ids_array == query_case_id)[0]
        mask_tensor = torch.from_numpy(same_case_indices_np).to(device)

        support_indices = find_knn_indices(
            query_embedding_norm,
            all_embeddings_norm,
            k=k_for_support,
            indices_to_mask=mask_tensor
        )

        if support_indices.numel() == 0:
            predicted_activity_name = "Error (No Support)"
            predicted_rem_time_hours = 0.0
        else:
            # Classification
            cls_support_embeddings = all_embeddings_norm[support_indices]
            cls_support_labels = all_cls_labels[support_indices]

            with torch.no_grad():
                logits, proto_classes, _ = proto_head_to_use.forward_classification(
                    cls_support_embeddings, cls_support_labels, query_embedding_norm
                )

            pred_label_idx = torch.argmax(logits, dim=1).item()
            predicted_activity_id = proto_classes[pred_label_idx].item()
            predicted_activity_name = (id_to_activity_name[predicted_activity_id]
                                     if 0 <= predicted_activity_id < len(id_to_activity_name)
                                     else "Unknown (Pred. ID)")

            # Regression
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

        # --- Record MAE for remaining time ---
        remaining_time_errors.append(abs(actual_rem_time - predicted_rem_time_hours))

        new_case_id = f"{query_case_id}@{len(query_prefix)}"
        new_trace = create_xes_trace(
            query_prefix,
            new_case_id,
            predicted_rem_time_hours,
            predicted_activity_name,
            actual_rem_time,
            actual_next_activity
        )
        output_log.append(new_trace)

    # --- Compute and attach the two requested log-level attributes ---
    if repeated_activity_ratios:
        repeated_percentage = np.mean(repeated_activity_ratios)   # proportion (0.0 – 1.0)
        output_log.attributes["repeatedLastActivityPercentage"] = float(repeated_percentage)

    if remaining_time_errors:
        mae_remaining_time = np.mean(remaining_time_errors)
        output_log.attributes["meanAbsoluteErrorRemainingTime"] = float(mae_remaining_time)

    # --- Save the final XES log ---
    print(f"\nSaving {len(output_log)} predicted traces to {args.output_file}...")
    pm4py.write_xes(output_log, args.output_file, variant_str="line_by_line", compression='gzip' in args.output_file)
    print("Done.")
    print(f"   • repeatedLastActivityPercentage = {output_log.attributes.get('repeatedLastActivityPercentage', 'N/A'):.4f}")
    print(f"   • meanAbsoluteErrorRemainingTime = {output_log.attributes.get('meanAbsoluteErrorRemainingTime', 'N/A'):.4f}")
