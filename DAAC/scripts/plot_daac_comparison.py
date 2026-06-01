from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-daac")

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DAAC comparison results.")
    parser.add_argument("--root", default="outputs/daac_compare")
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--scenario", default="B0-Inc10")
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        warn(f"missing {path}")
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result


def mean_std(values: list[float]) -> tuple[float, float]:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return math.nan, math.nan
    mean = sum(clean) / len(clean)
    if len(clean) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in clean) / (len(clean) - 1)
    return mean, math.sqrt(var)


def by_strategy_summary(rows: list[dict], key: str) -> dict[str, tuple[float, float]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(f(row.get(key)))
    return {strategy: mean_std(values) for strategy, values in grouped.items()}


def grouped_bar(data: dict[str, tuple[float, float]], title: str, ylabel: str, path: Path, log: bool = False) -> None:
    items = sorted(data.items())
    if not items:
        warn(f"skipping {path.name}; no data")
        return
    labels = [item[0] for item in items]
    means = [item[1][0] for item in items]
    stds = [item[1][1] for item in items]
    plt.figure(figsize=(9, 5.4))
    plt.bar(labels, means, yerr=stds, capsize=4)
    if log:
        plt.yscale("log")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_accuracy_curve(rows: list[dict], path: Path) -> None:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["strategy"]][int(float(row["task_id"]))].append(f(row.get("average_incremental_accuracy")))
    plt.figure(figsize=(9, 5.4))
    for strategy, task_values in sorted(grouped.items()):
        tasks = sorted(task_values)
        means, stds = [], []
        for task in tasks:
            mean, std = mean_std(task_values[task])
            means.append(mean)
            stds.append(std)
        plt.plot(tasks, means, marker="o", label=strategy)
        if any(std > 0 for std in stds if not math.isnan(std)):
            lower = [m - s for m, s in zip(means, stds)]
            upper = [m + s for m, s in zip(means, stds)]
            plt.fill_between(tasks, lower, upper, alpha=0.15)
    plt.title("Average Incremental Accuracy Over Tasks")
    plt.xlabel("Task ID")
    plt.ylabel("Average incremental accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_efficiency_params(summary_rows: list[dict], path: Path) -> None:
    strategies = sorted({row["strategy"] for row in summary_rows})
    trainable = by_strategy_summary(summary_rows, "trainable_params")
    total = by_strategy_summary(summary_rows, "total_params")
    if not strategies:
        warn("skipping efficiency_params_bar.png; no summary rows")
        return
    x = list(range(len(strategies)))
    width = 0.38
    train_means = [trainable.get(strategy, (math.nan, 0))[0] for strategy in strategies]
    total_means = [total.get(strategy, (math.nan, 0))[0] for strategy in strategies]
    clean = [value for value in train_means + total_means if not math.isnan(value) and value > 0]
    use_log = bool(clean) and max(clean) / max(min(clean), 1.0) > 100
    plt.figure(figsize=(9.5, 5.4))
    plt.bar([i - width / 2 for i in x], train_means, width=width, label="trainable")
    plt.bar([i + width / 2 for i in x], total_means, width=width, label="total")
    if use_log:
        plt.yscale("log")
    plt.title("Parameter Cost")
    plt.ylabel("Parameters")
    plt.xticks(x, strategies, rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_memory_time(summary_rows: list[dict], path: Path) -> None:
    memory = by_strategy_summary(summary_rows, "peak_cuda_memory_mb")
    time = by_strategy_summary(summary_rows, "total_training_time_sec")
    strategies = sorted({row["strategy"] for row in summary_rows})
    if not strategies:
        warn("skipping memory_time_comparison.png; no summary rows")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for axis, data, title, ylabel in [
        (axes[0], memory, "Peak CUDA Memory", "MB"),
        (axes[1], time, "Total Training Time", "Seconds"),
    ]:
        means = [data.get(strategy, (math.nan, 0))[0] for strategy in strategies]
        stds = [data.get(strategy, (math.nan, 0))[1] for strategy in strategies]
        axis.bar(strategies, means, yerr=stds, capsize=4)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def strategy_points(summary_rows: list[dict], x_key: str, y_key: str) -> dict[str, tuple[float, float]]:
    grouped_x = defaultdict(list)
    grouped_y = defaultdict(list)
    for row in summary_rows:
        grouped_x[row["strategy"]].append(f(row.get(x_key)))
        grouped_y[row["strategy"]].append(f(row.get(y_key)))
    points = {}
    for strategy in sorted(grouped_x):
        points[strategy] = (mean_std(grouped_x[strategy])[0], mean_std(grouped_y[strategy])[0])
    return points


def plot_scatter(points: dict[str, tuple[float, float]], title: str, xlabel: str, ylabel: str, path: Path) -> None:
    if not points:
        warn(f"skipping {path.name}; no data")
        return
    plt.figure(figsize=(8, 5.6))
    for strategy, (x_value, y_value) in points.items():
        if math.isnan(x_value) or math.isnan(y_value):
            continue
        plt.scatter([x_value], [y_value], s=70)
        plt.annotate(strategy, (x_value, y_value), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_adaptive_timeline(rows: list[dict], path: Path) -> None:
    adaptive = [row for row in rows if row["strategy"] == "adaptive"]
    if not adaptive:
        warn("skipping adaptive_strategy_timeline.png; adaptive rows unavailable")
        return
    choices = sorted({row["selected_strategy"] for row in adaptive})
    seeds = sorted({row["seed"] for row in adaptive})
    choice_to_index = {choice: index for index, choice in enumerate(choices)}
    seed_to_y = {seed: index for index, seed in enumerate(seeds)}
    plt.figure(figsize=(9, max(3.8, 0.5 * len(seeds) + 2)))
    for choice in choices:
        xs, ys = [], []
        for row in adaptive:
            if row["selected_strategy"] == choice:
                xs.append(f(row["task_id"]))
                ys.append(seed_to_y[row["seed"]])
        plt.scatter(xs, ys, s=90, label=choice)
    plt.title("Adaptive Strategy Choices Per Task")
    plt.xlabel("Task ID")
    plt.ylabel("Seed")
    plt.yticks(list(seed_to_y.values()), list(seed_to_y.keys()))
    plt.grid(alpha=0.25)
    plt.legend(title="Selected strategy", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_difficulty_components(rows: list[dict], path: Path) -> None:
    adaptive = [row for row in rows if row["strategy"] == "adaptive"]
    if not adaptive:
        warn("skipping difficulty_components_curve.png; adaptive rows unavailable")
        return
    components = [
        "difficulty_score",
        "novelty",
        "entropy",
        "gradient_sensitivity",
        "layer_importance_ratio",
        "expert_ambiguity",
    ]
    task_values = defaultdict(lambda: defaultdict(list))
    for row in adaptive:
        task = int(float(row["task_id"]))
        for component in components:
            task_values[component][task].append(f(row.get(component)))
    plt.figure(figsize=(9, 5.4))
    for component in components:
        tasks = sorted(task_values[component])
        means = [mean_std(task_values[component][task])[0] for task in tasks]
        plt.plot(tasks, means, marker="o", label=component)
    plt.title("Adaptive Difficulty Components")
    plt.xlabel("Task ID")
    plt.ylabel("Score")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_normalized_tradeoff(summary_rows: list[dict], path: Path) -> None:
    metrics = [
        ("accuracy", "final_accuracy", True),
        ("forgetting", "forgetting", False),
        ("params", "trainable_params", False),
        ("memory", "peak_cuda_memory_mb", False),
        ("time", "total_training_time_sec", False),
    ]
    strategies = sorted({row["strategy"] for row in summary_rows})
    raw = {label: by_strategy_summary(summary_rows, key) for label, key, _ in metrics}
    normalized: dict[str, list[float]] = {strategy: [] for strategy in strategies}
    for label, _, higher_better in metrics:
        means = {strategy: raw[label].get(strategy, (math.nan, 0))[0] for strategy in strategies}
        clean = [value for value in means.values() if not math.isnan(value)]
        if not clean:
            warn(f"missing metric for normalized tradeoff: {label}")
            for strategy in strategies:
                normalized[strategy].append(math.nan)
            continue
        lo, hi = min(clean), max(clean)
        for strategy in strategies:
            value = means[strategy]
            if math.isnan(value):
                score = math.nan
            elif hi == lo:
                score = 1.0
            elif higher_better:
                score = (value - lo) / (hi - lo)
            else:
                score = (hi - value) / (hi - lo)
            normalized[strategy].append(score)
    x = list(range(len(strategies)))
    width = 0.16
    plt.figure(figsize=(10, 5.8))
    for metric_index, (label, _, _) in enumerate(metrics):
        offsets = [index + (metric_index - 2) * width for index in x]
        plt.bar(offsets, [normalized[strategy][metric_index] for strategy in strategies], width=width, label=label)
    plt.title("Normalized Efficiency-Performance Tradeoff")
    plt.ylabel("Normalized score (higher is better)")
    plt.xticks(x, strategies, rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    scenario_root = Path(args.root) / args.dataset / args.scenario
    out_dir = Path(args.out_dir) if args.out_dir else scenario_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    result_rows = read_csv(scenario_root / "aggregated_results.csv")
    summary_rows = read_csv(scenario_root / "aggregated_summary.csv")
    if not result_rows or not summary_rows:
        raise SystemExit("No aggregated results found. Run scripts/aggregate_daac_results.py first.")

    plot_accuracy_curve(result_rows, out_dir / "accuracy_curve.png")
    grouped_bar(by_strategy_summary(summary_rows, "final_accuracy"), "Final Accuracy", "Accuracy", out_dir / "final_accuracy_bar.png")
    grouped_bar(by_strategy_summary(summary_rows, "average_incremental_accuracy"), "Average Incremental Accuracy", "Accuracy", out_dir / "avg_incremental_accuracy_bar.png")
    forgetting = by_strategy_summary(summary_rows, "forgetting")
    if all(math.isnan(value[0]) for value in forgetting.values()):
        warn("skipping forgetting_bar.png; forgetting unavailable")
    else:
        grouped_bar(forgetting, "Forgetting Score (Lower Is Better)", "Forgetting", out_dir / "forgetting_bar.png")
    plot_efficiency_params(summary_rows, out_dir / "efficiency_params_bar.png")
    grouped_bar(by_strategy_summary(summary_rows, "number_of_adapters"), "Adapter Count After Final Task", "Adapters", out_dir / "adapter_count_bar.png")
    plot_memory_time(summary_rows, out_dir / "memory_time_comparison.png")
    plot_scatter(
        strategy_points(summary_rows, "trainable_params", "final_accuracy"),
        "Final Accuracy vs Trainable Parameters",
        "Trainable parameters",
        "Final accuracy",
        out_dir / "accuracy_vs_params_scatter.png",
    )
    plot_scatter(
        strategy_points(summary_rows, "total_training_time_sec", "final_accuracy"),
        "Final Accuracy vs Training Time",
        "Total training time (sec)",
        "Final accuracy",
        out_dir / "accuracy_vs_time_scatter.png",
    )
    plot_adaptive_timeline(result_rows, out_dir / "adaptive_strategy_timeline.png")
    plot_difficulty_components(result_rows, out_dir / "difficulty_components_curve.png")
    plot_normalized_tradeoff(summary_rows, out_dir / "normalized_tradeoff_bar.png")
    print(f"Figures written to {out_dir}")


if __name__ == "__main__":
    main()
