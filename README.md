# Events Transformer

Events Transformer is a meta-learning Transformer for process mining event logs (XES). It learns from multiple event logs and adapts to new logs with few examples, targeting two tasks:

- **Next-activity prediction** (classification).
- **Remaining-time prediction** (regression).

It supports two embedding strategies for event attributes:

- **learned**: character-level CNN embeddings.
- **pretrained**: Sentence-Transformers embeddings for activity/resource names.

## Project layout

- `main.py`: training entry point.
- `testing.py`: evaluation entry point.
- `config.py`: default configuration and log paths.
- `data_generator.py`: XES loader and feature/embedding preparation.
- `components/`: model components (Transformer, MoE, meta-learner heads).
- `evaluation/`: evaluation routines (meta-learning and retrieval-augmented).
- `logs/`: sample XES logs and a simulation script.

## Setup

1) Create and activate a Python environment.
2) Install dependencies:

```bash
pip install -r requirements.txt
```

## Data expectations

The loader expects XES logs with these event attributes:

- `concept:name` (activity label)
- `time:timestamp` (event timestamp)
- `org:resource` (resource name; missing values default to `Unknown`)
- `amount` (cost; missing values default to 0.0)

Default training/testing logs are configured in `config.py` under `CONFIG['log_paths']`. Sample logs are already in `logs/`.

## Training

Run training with the defaults:

```bash
python main.py --checkpoint_dir ./checkpoints
```

Common options:

- `--embedding_strategy learned|pretrained`
- `--training_strategy episodic|retrieval|mixed`
- `--resume` (resume from latest checkpoint)
- `--stop_after_epoch N`

The script saves checkpoints and artifacts in `--checkpoint_dir`.

## Evaluation

Evaluate a trained model against a test log key from `config.py`:

```bash
python testing.py --checkpoint_dir ./checkpoints --test_log_name D_unseen
```

You can also pass a direct path to a `.xes` or `.xes.gz` file:

```bash
python testing.py --checkpoint_dir ./checkpoints --test_log_name ./logs/00013_clos2rep.xes.gz
```

To run retrieval-augmented evaluation:

```bash
python testing.py \
  --checkpoint_dir ./checkpoints \
  --test_log_name D_unseen \
  --test_mode retrieval_augmented \
  --test_retrieval_k 1 5 10 20 \
  --test_retrieval_prediction_mode proto_head \
  --test_retrieval_report_confidence_buckets
```

Use `--test_retrieval_prediction_mode foundation_knn` to bypass prototypical heads and predict directly with kNN over foundation-model feature embeddings.
Confidence-bucket reporting uses 5 fixed buckets in `[0,1]` and is applied only when `--test_retrieval_prediction_mode proto_head`.

## Simulating new logs (optional)

Generate synthetic XES logs using pm4py:

```bash
python logs/perform_simulation.py --output logs/simulated_log.xes.gz --num-logs 3
```

Then point `config.py` to the new files or pass them directly to `testing.py`.

## Notes

- GPU is optional; the code uses CUDA if available.
- The pretrained embedding strategy downloads the Sentence-Transformers model specified in `config.py`.
