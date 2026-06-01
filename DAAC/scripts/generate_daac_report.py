from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a DAAC comparison markdown report.")
    parser.add_argument("--root", default="outputs/daac_compare")
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--scenario", default="B0-Inc10")
    parser.add_argument("--figures-dir", default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
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


def fmt(value: float, digits: int = 2) -> str:
    if math.isnan(value):
        return "NaN"
    return f"{value:.{digits}f}"


def fmt_mean_std(values: list[float], digits: int = 2) -> str:
    mean, std = mean_std(values)
    return f"{fmt(mean, digits)} ± {fmt(std, digits)}"


def group_values(rows: list[dict], key: str) -> dict[str, list[float]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(f(row.get(key)))
    return dict(grouped)


def strategy_mean(rows: list[dict], strategy: str, key: str) -> float:
    return mean_std([f(row.get(key)) for row in rows if row["strategy"] == strategy])[0]


def best_strategy(rows: list[dict], key: str, higher_better: bool) -> tuple[str, float]:
    values = []
    for strategy in sorted({row["strategy"] for row in rows}):
        mean = strategy_mean(rows, strategy, key)
        if not math.isnan(mean):
            values.append((strategy, mean))
    if not values:
        return "N/A", math.nan
    return max(values, key=lambda item: item[1]) if higher_better else min(values, key=lambda item: item[1])


def percent_reduction(new_value: float, old_value: float) -> float:
    if math.isnan(new_value) or math.isnan(old_value) or old_value == 0:
        return math.nan
    return 100.0 * (old_value - new_value) / old_value


def signed_diff(left: float, right: float) -> float:
    if math.isnan(left) or math.isnan(right):
        return math.nan
    return left - right


def main() -> None:
    args = parse_args()
    scenario_root = Path(args.root) / args.dataset / args.scenario
    figures_dir = Path(args.figures_dir) if args.figures_dir else scenario_root / "figures"
    summary_path = scenario_root / "aggregated_summary.csv"
    rows = read_csv(summary_path)
    if not rows:
        raise SystemExit(f"No aggregated summary found at {summary_path}")

    strategies = sorted({row["strategy"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    command_path = scenario_root / "command_used.txt"
    command_used = command_path.read_text(encoding="utf-8").strip() if command_path.exists() else "Not recorded."

    grouped = {key: group_values(rows, key) for key in [
        "final_accuracy",
        "average_incremental_accuracy",
        "forgetting",
        "trainable_params",
        "number_of_adapters",
        "peak_cuda_memory_mb",
        "total_training_time_sec",
        "total_pre_study_time_sec",
    ]}

    lines = [
        "# DAAC Basic Comparison Report",
        "",
        "## Experiment Setup",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Scenario: `{args.scenario}`",
        f"- Strategies: `{', '.join(strategies)}`",
        f"- Seeds: `{', '.join(seeds)}`",
        f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "Command/config recorded by runner:",
        "",
        "```text",
        command_used,
        "```",
        "",
        "## Main Results",
        "",
        "| strategy | final accuracy | avg incremental accuracy | forgetting | trainable params | adapters | peak CUDA memory MB | total training sec | total pre-study sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for strategy in strategies:
        lines.append(
            "| {strategy} | {final} | {avg} | {forgetting} | {params} | {adapters} | {memory} | {train_time} | {pre_time} |".format(
                strategy=strategy,
                final=fmt_mean_std(grouped["final_accuracy"].get(strategy, [])),
                avg=fmt_mean_std(grouped["average_incremental_accuracy"].get(strategy, [])),
                forgetting=fmt_mean_std(grouped["forgetting"].get(strategy, [])),
                params=fmt_mean_std(grouped["trainable_params"].get(strategy, []), digits=0),
                adapters=fmt_mean_std(grouped["number_of_adapters"].get(strategy, []), digits=1),
                memory=fmt_mean_std(grouped["peak_cuda_memory_mb"].get(strategy, [])),
                train_time=fmt_mean_std(grouped["total_training_time_sec"].get(strategy, [])),
                pre_time=fmt_mean_std(grouped["total_pre_study_time_sec"].get(strategy, [])),
            )
        )

    best_final = best_strategy(rows, "final_accuracy", True)
    best_avg = best_strategy(rows, "average_incremental_accuracy", True)
    best_forgetting = best_strategy(rows, "forgetting", False)
    best_params = best_strategy(rows, "trainable_params", False)
    best_memory = best_strategy(rows, "peak_cuda_memory_mb", False)
    best_time = best_strategy(rows, "total_training_time_sec", False)

    lines += [
        "",
        "## Best Strategy By Metric",
        "",
        f"- Best final accuracy: `{best_final[0]}` ({fmt(best_final[1])})",
        f"- Best average incremental accuracy: `{best_avg[0]}` ({fmt(best_avg[1])})",
        f"- Lowest forgetting: `{best_forgetting[0]}` ({fmt(best_forgetting[1])})",
        f"- Lowest trainable params: `{best_params[0]}` ({fmt(best_params[1], 0)})",
        f"- Lowest peak CUDA memory: `{best_memory[0]}` ({fmt(best_memory[1])} MB)",
        f"- Fastest training: `{best_time[0]}` ({fmt(best_time[1])} sec)",
        "",
        "## Efficiency-Performance Interpretation",
        "",
    ]

    adaptive_final = strategy_mean(rows, "adaptive", "final_accuracy")
    adaptive_params = strategy_mean(rows, "adaptive", "trainable_params")
    adaptive_memory = strategy_mean(rows, "adaptive", "peak_cuda_memory_mb")
    adaptive_time = strategy_mean(rows, "adaptive", "total_training_time_sec")

    comparisons = [
        ("adapter_each_task", "final accuracy difference", signed_diff(adaptive_final, strategy_mean(rows, "adapter_each_task", "final_accuracy")), "accuracy points"),
        ("adapter_each_task", "trainable parameter reduction", percent_reduction(adaptive_params, strategy_mean(rows, "adapter_each_task", "trainable_params")), "%"),
        ("all_combined", "memory reduction", percent_reduction(adaptive_memory, strategy_mean(rows, "all_combined", "peak_cuda_memory_mb")), "%"),
        ("all_combined", "training time reduction", percent_reduction(adaptive_time, strategy_mean(rows, "all_combined", "total_training_time_sec")), "%"),
        ("tae_only", "final accuracy difference", signed_diff(adaptive_final, strategy_mean(rows, "tae_only", "final_accuracy")), "accuracy points"),
        ("prompt_only", "final accuracy difference", signed_diff(adaptive_final, strategy_mean(rows, "prompt_only", "final_accuracy")), "accuracy points"),
    ]
    for baseline, label, value, unit in comparisons:
        if baseline not in strategies:
            lines.append(f"- Adaptive vs `{baseline}`: unavailable because `{baseline}` was not run.")
        elif math.isnan(value):
            lines.append(f"- Adaptive vs `{baseline}` {label}: unavailable due to missing or zero-valued metric.")
        else:
            lines.append(f"- Adaptive vs `{baseline}` {label}: {fmt(value)} {unit}.")

    figure_names = [
        "accuracy_curve.png",
        "final_accuracy_bar.png",
        "avg_incremental_accuracy_bar.png",
        "forgetting_bar.png",
        "efficiency_params_bar.png",
        "adapter_count_bar.png",
        "memory_time_comparison.png",
        "accuracy_vs_params_scatter.png",
        "accuracy_vs_time_scatter.png",
        "adaptive_strategy_timeline.png",
        "difficulty_components_curve.png",
        "normalized_tradeoff_bar.png",
    ]
    lines += [
        "",
        "## Figures",
        "",
    ]
    for figure in figure_names:
        path = figures_dir / figure
        if path.exists():
            rel = path.relative_to(scenario_root)
            lines.append(f"- [{figure}]({rel})")
        else:
            lines.append(f"- `{figure}` was not generated.")

    lines += [
        "",
        "## Notes",
        "",
        "- This report is for first-stage feasibility testing, not SOTA claims.",
        "- CPU-only runs have non-comparable timing and zero CUDA memory by design.",
        "- Missing metrics are represented as `NaN` in aggregated CSVs.",
    ]

    report_path = scenario_root / "DAAC_basic_comparison_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
