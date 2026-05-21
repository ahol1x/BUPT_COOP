from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-daac")

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DAAC result summaries.")
    parser.add_argument("--root", default="outputs/daac")
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def read_rows(root: Path, dataset: str) -> list[dict]:
    rows = []
    dataset_root = root / dataset
    for metrics_path in dataset_root.glob("*/*/metrics.csv"):
        strategy = metrics_path.parents[1].name
        seed = metrics_path.parent.name
        with metrics_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["strategy"] = strategy
                row["seed"] = seed
                rows.append(row)
    return rows


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def group_mean(rows: list[dict], value_key: str) -> dict[str, dict[int, float]]:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["strategy"]][int(row["task_id"])].append(as_float(row, value_key))
    return {
        strategy: {
            task_id: sum(values) / len(values)
            for task_id, values in task_values.items()
        }
        for strategy, task_values in grouped.items()
    }


def bar_last(rows: list[dict], value_key: str) -> dict[str, float]:
    by_run = defaultdict(list)
    for row in rows:
        by_run[(row["strategy"], row["seed"])].append(row)
    by_strategy = defaultdict(list)
    for (strategy, _), run_rows in by_run.items():
        run_rows = sorted(run_rows, key=lambda item: int(item["task_id"]))
        by_strategy[strategy].append(as_float(run_rows[-1], value_key))
    return {strategy: sum(values) / len(values) for strategy, values in by_strategy.items()}


def plot_curve(series: dict[str, dict[int, float]], title: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for strategy, values in sorted(series.items()):
        xs = sorted(values)
        ys = [values[x] for x in xs]
        plt.plot(xs, ys, marker="o", label=strategy)
    plt.title(title)
    plt.xlabel("Task")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_bar(values: dict[str, float], title: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    items = sorted(values.items())
    plt.bar([item[0] for item in items], [item[1] for item in items])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_strategy_scatter(rows: list[dict], path: Path) -> None:
    plt.figure(figsize=(8, 5))
    strategies = sorted({row["selected_strategy"] for row in rows})
    strategy_to_y = {strategy: index for index, strategy in enumerate(strategies)}
    xs = [as_float(row, "difficulty_score") for row in rows]
    ys = [strategy_to_y[row["selected_strategy"]] for row in rows]
    plt.scatter(xs, ys, alpha=0.75)
    plt.yticks(range(len(strategies)), strategies)
    plt.xlabel("Difficulty score")
    plt.ylabel("Selected strategy")
    plt.title("Difficulty Score vs Selected Strategy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / args.dataset / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(root, args.dataset)
    if not rows:
        raise SystemExit(f"No metrics.csv files found under {root / args.dataset}")

    plot_curve(group_mean(rows, "task_accuracy"), "Accuracy Curve", "Accuracy", out_dir / "accuracy_curve.png")
    plot_bar(bar_last(rows, "average_incremental_accuracy"), "Average Incremental Accuracy", "Accuracy", out_dir / "average_accuracy.png")
    plot_bar(bar_last(rows, "task_accuracy"), "Final Accuracy", "Accuracy", out_dir / "final_accuracy.png")
    plot_curve(group_mean(rows, "trainable_params"), "Trainable Parameters", "Parameters", out_dir / "trainable_params.png")
    plot_curve(group_mean(rows, "peak_cuda_memory_mb"), "Peak CUDA Memory", "MB", out_dir / "memory.png")
    plot_curve(group_mean(rows, "training_time_sec"), "Training Time", "Seconds", out_dir / "training_time.png")
    plot_strategy_scatter(rows, out_dir / "difficulty_vs_strategy.png")
    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
