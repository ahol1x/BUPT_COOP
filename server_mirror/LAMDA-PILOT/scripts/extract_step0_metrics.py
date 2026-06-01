#!/usr/bin/env python
"""Extract Step 0 metrics from PILOT logs.

This is intentionally small and stdlib-only. It does not alter training; it
summarizes the metrics already printed by trainer.py plus wrapper-run metadata.
"""

import argparse
import ast
import csv
import json
import re
from pathlib import Path


def read_text(path):
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(errors="replace")


def last_number(pattern, text, cast=float):
    values = re.findall(pattern, text)
    if not values:
        return None
    return cast(values[-1])


def all_numbers(pattern, text, cast=float):
    return [cast(value) for value in re.findall(pattern, text)]


def literal_matches(pattern, text):
    values = []
    for match in re.findall(pattern, text):
        try:
            values.append(ast.literal_eval(match))
        except (SyntaxError, ValueError):
            continue
    return values


def parse_gpu_peak_mib(path):
    p = Path(path) if path else None
    if not p or not p.exists():
        return None

    peak = None
    with p.open(newline="", errors="replace") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            raw_value = row[2].strip().replace("MiB", "").strip()
            try:
                value = float(raw_value)
            except ValueError:
                continue
            peak = value if peak is None else max(peak, value)
    return peak


def parse_wall_clock_seconds(path):
    p = Path(path) if path else None
    if not p or not p.exists():
        return None
    text = p.read_text(errors="replace").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def build_summary(args):
    training_text = read_text(args.wrapper_log) or read_text(args.repo_log)

    total_params_by_task = all_numbers(r"All params:\s*([0-9]+)", training_text, int)
    trainable_params_by_task = all_numbers(
        r"Trainable params:\s*([0-9]+)", training_text, int
    )
    cnn_top1_curves = literal_matches(r"CNN top1 curve:\s*(\[[^\n]+\])", training_text)
    cnn_top5_curves = literal_matches(r"CNN top5 curve:\s*(\[[^\n]+\])", training_text)
    cnn_grouped = literal_matches(r"CNN:\s*(\{[^\n]+\})", training_text)

    top1_curve = cnn_top1_curves[-1] if cnn_top1_curves else []
    top5_curve = cnn_top5_curves[-1] if cnn_top5_curves else []
    final_grouped = cnn_grouped[-1] if cnn_grouped else {}

    average_incremental_accuracy = last_number(
        r"Average Accuracy \(CNN\):\s*([0-9.+\-eE]+)", training_text
    )
    if average_incremental_accuracy is None and top1_curve:
        average_incremental_accuracy = sum(top1_curve) / len(top1_curve)

    final_accuracy = top1_curve[-1] if top1_curve else final_grouped.get("total")

    summary = {
        "run_name": args.run_name,
        "method": "finetune",
        "dataset_requested": "CIFAR100",
        "dataset_config_key": "cifar224",
        "split": "B0-Inc10",
        "seed": 1993,
        "config": args.config,
        "source_logs": {
            "wrapper_log": args.wrapper_log,
            "repo_log": args.repo_log,
            "gpu_memory_samples": args.gpu_memory_samples,
            "wall_clock_seconds": args.wall_clock_seconds,
        },
        "params": {
            "total_params_by_task": total_params_by_task,
            "trainable_params_by_task": trainable_params_by_task,
            "final_total_params": total_params_by_task[-1]
            if total_params_by_task
            else None,
            "final_trainable_params": trainable_params_by_task[-1]
            if trainable_params_by_task
            else None,
        },
        "cnn": {
            "top1_curve": top1_curve,
            "top5_curve": top5_curve,
            "average_incremental_accuracy": average_incremental_accuracy,
            "final_accuracy": final_accuracy,
            "forgetting": last_number(
                r"Forgetting \(CNN\):\s*([0-9.+\-eE]+)", training_text
            ),
            "grouped_accuracy_by_eval": cnn_grouped,
            "final_grouped_accuracy": final_grouped,
        },
        "runtime": {
            "wall_clock_seconds": parse_wall_clock_seconds(args.wall_clock_seconds),
            "peak_gpu_memory_mib": parse_gpu_peak_mib(args.gpu_memory_samples),
        },
        "notes": [
            "Metrics are parsed from the current wrapper log when available.",
            "The repo's config key 'cifar224' downloads CIFAR100 and applies 224x224 transforms for ViT.",
        ],
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Extract Step 0 Finetune metrics.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--wrapper-log", required=True)
    parser.add_argument("--repo-log", required=True)
    parser.add_argument("--gpu-memory-samples", required=True)
    parser.add_argument("--wall-clock-seconds", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = build_summary(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("Wrote metrics summary to {}".format(output))


if __name__ == "__main__":
    main()
