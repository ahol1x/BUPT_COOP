from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


RESULT_COLUMNS = [
    "dataset",
    "scenario",
    "strategy",
    "seed",
    "task_id",
    "selected_strategy",
    "difficulty_score",
    "novelty",
    "entropy",
    "gradient_sensitivity",
    "layer_importance_ratio",
    "expert_ambiguity",
    "task_accuracy",
    "average_incremental_accuracy",
    "final_accuracy",
    "forgetting",
    "trainable_params",
    "total_params",
    "number_of_adapters",
    "number_of_prompts",
    "training_time_sec",
    "pre_study_time_sec",
    "peak_cuda_memory_mb",
]

SUMMARY_COLUMNS = [
    "dataset",
    "scenario",
    "strategy",
    "seed",
    "tasks",
    "final_accuracy",
    "average_incremental_accuracy",
    "forgetting",
    "trainable_params",
    "total_params",
    "number_of_adapters",
    "number_of_prompts",
    "peak_cuda_memory_mb",
    "total_training_time_sec",
    "total_pre_study_time_sec",
    "metrics_path",
    "summary_path",
]


NUMERIC_FIELDS = {
    "task_id",
    "difficulty_score",
    "novelty",
    "entropy",
    "gradient_sensitivity",
    "layer_importance_ratio",
    "expert_ambiguity",
    "task_accuracy",
    "average_incremental_accuracy",
    "final_accuracy",
    "forgetting",
    "trainable_params",
    "total_params",
    "number_of_adapters",
    "number_of_prompts",
    "training_time_sec",
    "pre_study_time_sec",
    "peak_cuda_memory_mb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate DAAC comparison metrics.")
    parser.add_argument("--root", default="outputs/daac_compare")
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--scenario", default="B0-Inc10")
    return parser.parse_args()


class WarningOnce:
    def __init__(self) -> None:
        self.seen: set[str] = set()

    def warn(self, message: str) -> None:
        if message in self.seen:
            return
        self.seen.add(message)
        print(f"WARNING: {message}", file=sys.stderr)


def read_json(path: Path, warnings: WarningOnce) -> dict:
    if not path.exists():
        warnings.warn(f"missing summary file {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        warnings.warn(f"could not parse {path}: {exc}")
        return {}


def numeric(value, warnings: WarningOnce, field: str, context: Path):
    if value is None or value == "":
        return "NaN"
    try:
        return float(value)
    except (TypeError, ValueError):
        warnings.warn(f"non-numeric value for {field} in {context}; using NaN")
        return "NaN"


def value_from(row: dict, keys: list[str], field: str, context: Path, warnings: WarningOnce):
    for key in keys:
        if key in row:
            return row[key] if row[key] != "" else "NaN"
    warnings.warn(f"missing field {field} in {context}; using NaN")
    return "NaN"


def read_metrics(path: Path, warnings: WarningOnce) -> list[dict]:
    if not path.exists():
        warnings.warn(f"missing metrics file {path}")
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_dirs(scenario_root: Path) -> list[Path]:
    return sorted(path for path in scenario_root.glob("*/seed_*") if path.is_dir())


def extract_seed(run_dir: Path) -> str:
    name = run_dir.name
    return name.split("seed_", 1)[1] if name.startswith("seed_") else name


def aggregate(args: argparse.Namespace) -> tuple[Path, Path]:
    warnings = WarningOnce()
    scenario_root = Path(args.root) / args.dataset / args.scenario
    result_rows: list[dict] = []
    summary_rows: list[dict] = []

    for run_dir in run_dirs(scenario_root):
        strategy = run_dir.parent.name
        seed = extract_seed(run_dir)
        metrics_path = run_dir / "metrics.csv"
        summary_path = run_dir / "summary.json"
        metrics = read_metrics(metrics_path, warnings)
        summary = read_json(summary_path, warnings)

        for metric_row in metrics:
            out = {
                "dataset": args.dataset,
                "scenario": args.scenario,
                "strategy": strategy,
                "seed": seed,
                "task_id": value_from(metric_row, ["task_id"], "task_id", metrics_path, warnings),
                "selected_strategy": metric_row.get("selected_strategy", "NaN"),
                "difficulty_score": value_from(metric_row, ["difficulty_score"], "difficulty_score", metrics_path, warnings),
                "novelty": value_from(metric_row, ["novelty"], "novelty", metrics_path, warnings),
                "entropy": value_from(metric_row, ["entropy"], "entropy", metrics_path, warnings),
                "gradient_sensitivity": value_from(metric_row, ["gradient_sensitivity"], "gradient_sensitivity", metrics_path, warnings),
                "layer_importance_ratio": value_from(metric_row, ["layer_importance_ratio"], "layer_importance_ratio", metrics_path, warnings),
                "expert_ambiguity": value_from(metric_row, ["expert_ambiguity"], "expert_ambiguity", metrics_path, warnings),
                "task_accuracy": value_from(metric_row, ["task_accuracy"], "task_accuracy", metrics_path, warnings),
                "average_incremental_accuracy": value_from(metric_row, ["average_incremental_accuracy"], "average_incremental_accuracy", metrics_path, warnings),
                "final_accuracy": value_from(metric_row, ["final_accuracy"], "final_accuracy", metrics_path, warnings),
                "forgetting": value_from(metric_row, ["forgetting", "forgetting_score"], "forgetting", metrics_path, warnings),
                "trainable_params": value_from(metric_row, ["trainable_params"], "trainable_params", metrics_path, warnings),
                "total_params": value_from(metric_row, ["total_params"], "total_params", metrics_path, warnings),
                "number_of_adapters": value_from(metric_row, ["number_of_adapters"], "number_of_adapters", metrics_path, warnings),
                "number_of_prompts": value_from(metric_row, ["number_of_prompts"], "number_of_prompts", metrics_path, warnings),
                "training_time_sec": value_from(metric_row, ["training_time_sec"], "training_time_sec", metrics_path, warnings),
                "pre_study_time_sec": value_from(metric_row, ["pre_study_time_sec"], "pre_study_time_sec", metrics_path, warnings),
                "peak_cuda_memory_mb": value_from(metric_row, ["peak_cuda_memory_mb"], "peak_cuda_memory_mb", metrics_path, warnings),
            }
            for field in NUMERIC_FIELDS:
                out[field] = numeric(out[field], warnings, field, metrics_path)
            result_rows.append(out)

        last = metrics[-1] if metrics else {}
        total_training = sum(float(row.get("training_time_sec") or 0.0) for row in metrics)
        total_pre = sum(float(row.get("pre_study_time_sec") or 0.0) for row in metrics)
        peak_memory = max([float(row.get("peak_cuda_memory_mb") or 0.0) for row in metrics] or [math.nan])
        summary_row = {
            "dataset": args.dataset,
            "scenario": args.scenario,
            "strategy": strategy,
            "seed": seed,
            "tasks": len(metrics),
            "final_accuracy": summary.get("final_accuracy", last.get("final_accuracy") or last.get("task_accuracy") or "NaN"),
            "average_incremental_accuracy": summary.get("average_incremental_accuracy", last.get("average_incremental_accuracy") or "NaN"),
            "forgetting": summary.get("forgetting_score", last.get("forgetting_score") or last.get("forgetting") or "NaN"),
            "trainable_params": last.get("trainable_params", "NaN"),
            "total_params": last.get("total_params", "NaN"),
            "number_of_adapters": last.get("number_of_adapters", "NaN"),
            "number_of_prompts": last.get("number_of_prompts", "NaN"),
            "peak_cuda_memory_mb": peak_memory,
            "total_training_time_sec": total_training,
            "total_pre_study_time_sec": total_pre,
            "metrics_path": str(metrics_path),
            "summary_path": str(summary_path),
        }
        for field in [
            "tasks",
            "final_accuracy",
            "average_incremental_accuracy",
            "forgetting",
            "trainable_params",
            "total_params",
            "number_of_adapters",
            "number_of_prompts",
            "peak_cuda_memory_mb",
            "total_training_time_sec",
            "total_pre_study_time_sec",
        ]:
            summary_row[field] = numeric(summary_row[field], warnings, field, summary_path)
        summary_rows.append(summary_row)

    if not result_rows:
        warnings.warn(f"no metrics rows found under {scenario_root}")

    scenario_root.mkdir(parents=True, exist_ok=True)
    result_path = scenario_root / "aggregated_results.csv"
    summary_path = scenario_root / "aggregated_summary.csv"
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(result_rows)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)
    return result_path, summary_path


def main() -> None:
    args = parse_args()
    result_path, summary_path = aggregate(args)
    print(f"Aggregated task metrics: {result_path}")
    print(f"Aggregated run summaries: {summary_path}")


if __name__ == "__main__":
    main()
