# test.py

import torch
import numpy as np
import argparse
from data_generator import EpisodeGenerator, collate_batch, SPECIAL_TOKENS, ACTIVITY_VOCAB_SIZE, get_reverse_vocab
from model import IOTransformer
from main import D_MODEL, N_LAYERS, N_HEADS, NUM_CAT_FEATURES, NUM_NUM_FEATURES, NUM_TIME_FEATURES


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

            # After fixes, the final token is the query <LABEL>; compute logits at that position.
            label_idx = len(episode['tokens']) - 1

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
            else:  # remaining_time
                # Convert prediction bucket back to continuous value (bucket midpoint)
                max_rem_time = generator.max_case_len * 3.0
                bucket_width = max_rem_time / 50
                predicted_time = (prediction + 0.5) * bucket_width

                true_time = batch['query_true_continuous'].item()
                total_mae += abs(predicted_time - true_time)

            total_count += 1

    if task == 'next_activity':
        accuracy = total_correct / total_count
        return {"accuracy": accuracy}
    else:
        mae = total_mae / total_count
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

        # Map back to vocab indices
        if task == 'next_activity':
            predicted_global_id = predicted_token_id + len(SPECIAL_TOKENS)
        else:
            predicted_global_id = predicted_token_id + len(SPECIAL_TOKENS) + ACTIVITY_VOCAB_SIZE

        true_global_id = batch['query_true_tokens'].item()

        # --- Printing ---
        # 1. Provided Context
        context_tokens = [rev_vocab.get(tid, f"ID_{tid}") for tid in episode['tokens'][:-1]]
        print(f"{Colors.GREEN}CONTEXT: {' '.join(context_tokens)} <LABEL>{Colors.RESET}")


        # 2. Prediction vs. Truth
        predicted_name = rev_vocab.get(predicted_global_id, f"ID_{predicted_global_id}")
        true_name = rev_vocab.get(true_global_id, f"ID_{true_global_id}")

        print(f"{Colors.RED}  -> PREDICTION: {predicted_name}{Colors.RESET}")
        print(f"{Colors.BLUE}  -> TRUTH:      {true_name}{Colors.RESET}")
        if task == 'remaining_time':
             # Also show continuous value for context
            true_time = batch['query_true_continuous'].item()
            print(f"{Colors.BLUE}     (Continuous Truth: {true_time:.2f}){Colors.RESET}")


def run_evaluation_suite(model, generator):
    """Runs the quantitative evaluation across all tasks and shot configurations."""
    shot_configs = [0, 4, 8]
    tasks = ['next_activity', 'remaining_time']
    results = {}

    for task in tasks:
        print(f"\n--- Evaluating Task: {task.upper()} ---")
        task_results = {}
        for k in shot_configs:
            metrics = evaluate_model(model, generator, k_shots=k, task=task)
            task_results[f"{k}-shot"] = metrics
            metric_str = ", ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
            print(f"  {k}-shot Performance -> {metric_str}")
        results[task] = task_results

    print("\n--- Evaluation Summary ---")
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the ICL Transformer model.")
    parser.add_argument(
        '--print_predictions',
        action='store_true',
        help="If set, prints a few human-readable prediction examples with colors."
    )
    args = parser.parse_args()

    print("--- Initializing Model and Data ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test set generator
    test_generator = EpisodeGenerator(
        num_cases=300, max_case_len=50,
        num_cat_features=NUM_CAT_FEATURES,
        num_num_features=NUM_NUM_FEATURES,
        seed=123 # Use a different seed for test set
    )

    # Load the trained model
    model = IOTransformer(
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
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

    # Run the standard quantitative evaluation
    run_evaluation_suite(model, test_generator)

    # If the flag is set, run the qualitative examples
    if args.print_predictions:
        print("\n" + "="*25 + " QUALITATIVE EXAMPLES " + "="*25)
        rev_vocab = get_reverse_vocab()
        shot_configs = [0, 4, 8]
        tasks = ['next_activity', 'remaining_time']
        for task in tasks:
            for k in shot_configs:
                show_prediction_example(model, test_generator, k, task, rev_vocab)
