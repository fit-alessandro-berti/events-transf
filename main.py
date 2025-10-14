
import argparse
import json

from data_generator import generate_traces
from episode_builder import EpisodeBuilder
from io_transformer import IOTransformer, IOTransformerConfig
from train import Trainer, TrainConfig
from eval import eval_icl
from utils import set_seed, get_device


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--train-cases", type=int, default=500, help="Cases per process model")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--eval-episodes", type=int, default=30)
    return p


def main(args):
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    traces, activity2id, resource2id = generate_traces(n_cases_per_model=args.train_cases, seed=args.seed)
    print(f"Generated traces: {len(traces)} ; activities: {len(activity2id)} ; resources: {len(resource2id)}")

    builder = EpisodeBuilder(traces, activity2id, resource2id, max_seq_len=args.max_seq_len, seed=args.seed)

    cfg = IOTransformerConfig(
        num_activities=len(activity2id),
        num_resources=len(resource2id),
        num_dims=2,  # cost_z, pos_frac
        d_model=256,
        n_layers=6,
        n_heads=8,
        max_seq_len=args.max_seq_len,
        dropout=0.1,
        n_classes=len(activity2id),
    )
    model = IOTransformer(cfg).to(device)
    trainer = Trainer(model, device, TrainConfig())

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        for step in range(args.steps_per_epoch):
            if step % 2 == 0:
                task = "cls" if (step // 2) % 2 == 0 else "reg"
                K = [0, 4, 8, 16][(step // 4) % 4]
                batch = builder.build_batch(task=task, K=K, batch_size=args.batch_size, device=device)
                losses = trainer.train_icl_step(batch, task=task)
            else:
                batch = builder.build_causal_batch(batch_size=args.batch_size, device=device)
                losses = trainer.train_causal_step(batch)
            if step % 50 == 0:
                print("  step %04d | " % step + " | ".join([f"{k}:{v:.4f}" for k, v in losses.items()]))

    report = eval_icl(model, builder, device=device, Ks=(0, 4, 16), episodes=args.eval_episodes)
    print("\n=== Evaluation (ICL) ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
