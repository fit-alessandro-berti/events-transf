# test.py

import argparse
import random
import torch
import numpy as np

from data_generator import (
    EpisodeGenerator,
    collate_batch,
    SPECIAL_TOKENS,
    ACTIVITY_VOCAB_SIZE,
    get_reverse_vocab,
)
from model import IOTransformer
from main import D_MODEL, N_LAYERS, N_HEADS, NUM_NUM_FEATURES, NUM_TIME_FEATURES

# Optional XES generator
try:
    from test_data_generator import XesEpisodeGenerator
    _HAS_XES_GEN = True
except Exception:
    _HAS_XES_GEN = False


# ANSI color codes for printing
class Colors:
    GREEN = '\033[92m'   # Provided context
    RED = '\033[91m'     # Model's prediction
    BLUE = '\033[94m'    # Ground truth
    YELLOW = '\033[93m'  # Headers
    RESET = '\033[0m'    # Reset to default color


def evaluate_model(model, generator, k_shots, task, n_episodes=100):
    """Evaluates the model for a given number of shots and task."""
    model.eval()
    device = next(model.parameters()).device

    total_correct = 0
    total_mae = 0.0
    total_count = 0

    with torch.no_grad():
        for _ in range(n_episodes):
            episode = generator.create_episode(k_shots, task)
            label_idx = len(episode['tokens']) - 1  # query <LABEL> is final token

            batch = collate_batch([episode])  # Batch size = 1
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            activity_logits, time_logits = model(batch)
            query_logits = activity_logits[0, label_idx, :] if task == 'next_activity' else time_logits[0, label_idx, :]

            prediction = torch.argmax(query_logits).item()

            if task == 'next_activity':
                true_label = batch['query_true_tokens'].item() - len(SPECIAL_TOKENS)
                if prediction == true_label:
                    total_correct += 1
            else:
                # Use the generator's bucket->continuous mapping
                predicted_time = generator.bucket_to_continuous(prediction)
                true_time = batch['query_true_continuous'].item()
                total_mae += abs(predicted_time - true_time)

            total_count += 1

    if task == 'next_activity':
        accuracy = total_correct / max(1, total_count)
        return {"accuracy": accuracy}
    else:
        mae = total_mae / max(1, total_count)
        return {"mae": mae}


def show_prediction_example(model, generator, k_shots, task, rev_vocab):
    """Generates one episode, runs inference, and prints a colored, human-readable output."""
    model.eval()
    device = next(model.parameters()).device

    print(f"\n{Colors.YELLOW}--- Example for Task: {task.upper()}, K-shots: {k_shots} ---{Colors.RESET}")

    with torch.no_grad():
        episode = generator.create_episode(k_shots, task)
        batch = collate_batch([episode])

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        activity_logits, time_logits = model(batch)

        # The query <LABEL> is the last token
        label_idx = len(episode['tokens']) - 1
        query_logits = activity_logits[0, label_idx, :] if task == 'next_activity' else time_logits[0, label_idx, :]
        predicted_token_id = torch.argmax(query_logits).item()

        if task == 'next_activity':
            predicted_global_id = predicted_token_id + len(SPECIAL_TOKENS)
        else:
            predicted_global_id = predicted_token_id + len(SPECIAL_TOKENS) + ACTIVITY_VOCAB_SIZE

        true_global_id = batch['query_true_tokens'].item()

        # --- Printing ---
        context_tokens = [rev_vocab.get(tid, f"ID_{tid}") for tid in episode['tokens'][:-1]]
        print(f"{Colors.GREEN}CONTEXT: {' '.join(context_tokens)} <LABEL>{Colors.RESET}")

        predicted_name = rev_vocab.get(predicted_global_id, f"ID_{predicted_global_id}")
        true_name = rev_vocab.get(true_global_id, f"ID_{true_global_id}")

        print(f"{Colors.RED}  -> PREDICTION: {predicted_name}{Colors.RESET}")
        print(f"{Colors.BLUE}  -> TRUTH:      {true_name}{Colors.RESET}")
        if task == 'remaining_time':
            true_time = batch['query_true_continuous'].item()
            pred_bucket = predicted_token_id
            print(f"{Colors.BLUE}     (Continuous Truth: {true_time:.2f}){Colors.RESET}")
            print(f"{Colors.RED}     (Predicted mid-pt: {generator.bucket_to_continuous(pred_bucket):.2f}){Colors.RESET}")


def run_evaluation_suite(model, generator, n_episodes=100):
    """Runs the quantitative evaluation across all tasks and shot configurations."""
    shot_configs = [0, 4, 8]
    tasks = ['next_activity', 'remaining_time']
    results = {}

    for task in tasks:
        print(f"\n--- Evaluating Task: {task.upper()} ---")
        task_results = {}
        for k in shot_configs:
            metrics = evaluate_model(model, generator, k_shots=k, task=task, n_episodes=n_episodes)
            task_results[f"{k}-shot"] = metrics
            metric_str = ", ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
            print(f"  {k}-shot Performance -> {metric_str}")
        results[task] = task_results

    print("\n--- Evaluation Summary ---")
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the ICL Transformer model on synthetic or XES data.")
    parser.add_argument('--data_source', choices=['synthetic', 'xes'], default='synthetic',
                        help="Which data generator to use.")
    parser.add_argument('--xes_path', type=str, default=None, help="Path to the XES file (required if data_source=xes).")
    parser.add_argument('--n_episodes', type=int, default=100, help="Number of evaluation episodes per (task,shot).")
    parser.add_argument('--print_predictions', action='store_true',
                        help="If set, prints a few human-readable prediction examples with colors.")
    # Optional XES attribute keys
    parser.add_argument('--activity_key', type=str, default='concept:name')
    parser.add_argument('--timestamp_key', type=str, default='time:timestamp')
    parser.add_argument('--resource_key', type=str, default='org:resource')
    parser.add_argument('--group_key', type=str, default='org:group')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Build generator
    # -------------------------
    if args.data_source == 'synthetic':
        print("--- Using synthetic (simulated) generator ---")
        test_generator = EpisodeGenerator(
            num_cases=300,
            max_case_len=50,
            num_cat_features=3,
            num_num_features=NUM_NUM_FEATURES,
            n_models=3,
            seed=123  # different seed for test split
        )
    else:
        if not _HAS_XES_GEN:
            raise RuntimeError("XES generator unavailable (pm4py not installed?).")
        if not args.xes_path:
            raise ValueError("Please provide --xes_path when using data_source=xes.")
        print(f"--- Using XES generator on {args.xes_path} ---")
        test_generator = XesEpisodeGenerator(
            xes_path=args.xes_path,
            num_num_features=NUM_NUM_FEATURES,
            activity_key=args.activity_key,
            timestamp_key=args.timestamp_key,
            resource_key=args.resource_key,
            group_key=args.group_key,
            seed=123
        )

    # -------------------------
    # Load model
    # -------------------------
    model = IOTransformer(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        cat_cardinalities=test_generator.cat_cardinalities,
        num_num_features=NUM_NUM_FEATURES,
        num_time_features=NUM_TIME_FEATURES
    ).to(device)
    try:
        model.load_state_dict(torch.load("io_transformer.pth", map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Could not find trained model 'io_transformer.pth'. Please train the model first.")
        exit()

    # -------------------------
    # Evaluate
    # -------------------------
    run_evaluation_suite(model, test_generator, n_episodes=args.n_episodes)

    if args.print_predictions:
        print("\n" + "=" * 25 + " QUALITATIVE EXAMPLES " + "=" * 25)
        rev_vocab = get_reverse_vocab()
        for task in ['next_activity', 'remaining_time']:
            for k in [0, 4, 8]:
                show_prediction_example(model, test_generator, k, task, rev_vocab)
