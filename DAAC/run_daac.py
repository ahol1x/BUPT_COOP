from __future__ import annotations

import argparse
from pathlib import Path

from daac.trainer import run_experiments
from daac.utils import write_json


STRATEGIES = [
    "adaptive",
    "prompt_only",
    "tae_only",
    "adapter_each_task",
    "mote_fusion",
    "all_combined",
    "finetune",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAAC: Difficulty-Aware Adaptive Controller prototype.")
    parser.add_argument("--strategy", choices=STRATEGIES, default="adaptive")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=None)
    parser.add_argument("--dataset", default="cifar100", choices=["synthetic", "debug", "cifar100"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1991, 1993, 1995])

    parser.add_argument("--init-classes", type=int, default=0)
    parser.add_argument("--increment", type=int, default=10)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="adamw")
    parser.add_argument("--mask-old-classes-during-train", action="store_true")

    parser.add_argument("--backbone", default="tiny_vit", choices=["tiny_vit"])
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--adapter-bottleneck", type=int, default=16)

    parser.add_argument("--prestudy-batches", type=int, default=2)
    parser.add_argument("--grad-sensitivity-scale", type=float, default=50.0)
    parser.add_argument("--w-novelty", type=float, default=0.35)
    parser.add_argument("--w-entropy", type=float, default=0.20)
    parser.add_argument("--w-grad", type=float, default=0.25)
    parser.add_argument("--w-layer", type=float, default=0.10)
    parser.add_argument("--w-ambiguity", type=float, default=0.10)
    parser.add_argument("--low-threshold", type=float, default=0.35)
    parser.add_argument("--high-threshold", type=float, default=0.65)
    parser.add_argument("--ambiguity-threshold", type=float, default=0.45)
    parser.add_argument("--novelty-threshold", type=float, default=0.60)

    parser.add_argument("--distill-weight", type=float, default=0.2)
    parser.add_argument("--distill-temperature", type=float, default=2.0)

    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--fast-dev-epochs", type=int, default=1)
    args = parser.parse_args()
    if args.fast_dev_run:
        args.dataset = "synthetic" if args.dataset in {"debug", "cifar100"} else args.dataset
        args.epochs = min(args.epochs, args.fast_dev_epochs)
        args.batch_size = min(args.batch_size, 16)
        args.increment = min(args.increment, 2)
        args.max_tasks = args.max_tasks or 3
        args.embed_dim = min(args.embed_dim, 32)
        args.depth = min(args.depth, 2)
        args.num_heads = min(args.num_heads, 4)
        args.adapter_bottleneck = min(args.adapter_bottleneck, 8)
        args.prestudy_batches = min(args.prestudy_batches, 1)
    return args


def main() -> None:
    args = parse_args()
    summaries = run_experiments(args)
    output_root = Path(args.output_dir) / "daac" / args.dataset
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "latest_run_summaries.json", summaries)


if __name__ == "__main__":
    main()
