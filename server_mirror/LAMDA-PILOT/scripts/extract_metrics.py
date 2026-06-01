#!/usr/bin/env python
"""Extract PILOT metrics from repo and wrapper logs.

The repo prints numpy scalar reprs inside lists and dicts, for example
``np.float64(98.9)``. This extractor normalizes those values before parsing
the logged Python literals.
"""

import argparse
import ast
import csv
import json
import re
from pathlib import Path


NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def read_text(path):
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(errors="replace")


def load_config(path):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def clean_int(value):
    return int(str(value).replace(",", ""))


def normalize_numpy_scalars(text):
    scalar = re.compile(
        r"np\.(?:float16|float32|float64|float_|float|int8|int16|int32|int64|int_|int|uint8|uint16|uint32|uint64)\(("
        + NUMBER_RE
        + r")\)"
    )
    previous = None
    while previous != text:
        previous = text
        text = scalar.sub(r"\1", text)
    return text


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def literal_matches(pattern, text):
    values = []
    for match in re.findall(pattern, text):
        try:
            values.append(to_jsonable(ast.literal_eval(match)))
        except (SyntaxError, ValueError):
            continue
    return values


def last_number(pattern, text, cast=float):
    values = re.findall(pattern, text)
    if not values:
        return None
    return cast(values[-1])


def all_numbers(pattern, text, cast=float):
    return [cast(value) for value in re.findall(pattern, text)]


def parse_gpu_peak_mib(path, text):
    p = Path(path) if path else None
    peak = None
    if p and p.exists():
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
    if peak is not None:
        return peak
    return last_number(r"Peak GPU memory MiB:\s*(" + NUMBER_RE + r")", text)


def parse_wall_clock_seconds(path, text):
    p = Path(path) if path else None
    if p and p.exists():
        raw = p.read_text(errors="replace").strip()
        if raw:
            try:
                return float(raw)
            except ValueError:
                pass
    return last_number(r"Wall[- ]clock seconds:\s*(" + NUMBER_RE + r")", text)


def infer_split(config):
    init_cls = config.get("init_cls")
    increment = config.get("increment")
    if init_cls is None or increment is None:
        return None
    if init_cls == increment:
        return "B0-Inc{}".format(increment)
    return "B{}-Inc{}".format(init_cls, increment)


def first_seed(config):
    seed = config.get("seed")
    if isinstance(seed, list) and seed:
        return seed[0]
    return seed


def training_text(wrapper_log, repo_log):
    wrapper_text = read_text(wrapper_log)
    repo_text = read_text(repo_log)
    if wrapper_text.strip():
        text = wrapper_text
        if repo_text and "CNN top1 curve:" not in wrapper_text:
            text = text + "\n" + repo_text
    else:
        text = repo_text
    return normalize_numpy_scalars(text)


def build_summary(args):
    config = load_config(args.config)
    text = training_text(args.wrapper_log, args.repo_log)

    total_params_by_task = all_numbers(r"All params:\s*([0-9,]+)", text, clean_int)
    trainable_params_by_task = all_numbers(
        r"Trainable params:\s*([0-9,]+)", text, clean_int
    )
    if not total_params_by_task:
        total_params_by_task = all_numbers(
            r"([0-9,]+)\s+model total parameters\.", text, clean_int
        )
    if not trainable_params_by_task:
        trainable_params_by_task = all_numbers(
            r"([0-9,]+)\s+model training parameters\.", text, clean_int
        )

    cnn_top1_curves = literal_matches(r"CNN top1 curve:\s*(\[[^\n\r]+\])", text)
    cnn_top5_curves = literal_matches(r"CNN top5 curve:\s*(\[[^\n\r]+\])", text)
    nme_top1_curves = literal_matches(r"NME top1 curve:\s*(\[[^\n\r]+\])", text)
    nme_top5_curves = literal_matches(r"NME top5 curve:\s*(\[[^\n\r]+\])", text)
    cnn_grouped = literal_matches(r"CNN:\s*(\{[^\n\r]+\})", text)
    nme_grouped = literal_matches(r"NME:\s*(\{[^\n\r]+\})", text)

    top1_curve = cnn_top1_curves[-1] if cnn_top1_curves else []
    top5_curve = cnn_top5_curves[-1] if cnn_top5_curves else []
    final_grouped = cnn_grouped[-1] if cnn_grouped else {}

    average_incremental_accuracy = last_number(
        r"Average Accuracy \(CNN\):\s*(" + NUMBER_RE + r")", text
    )
    if average_incremental_accuracy is None and top1_curve:
        average_incremental_accuracy = sum(top1_curve) / len(top1_curve)

    final_accuracy = top1_curve[-1] if top1_curve else final_grouped.get("total")
    forgetting = last_number(r"Forgetting \(CNN\):\s*(" + NUMBER_RE + r")", text)

    params = {
        "total_params_by_task": total_params_by_task,
        "trainable_params_by_task": trainable_params_by_task,
        "final_total_params": total_params_by_task[-1]
        if total_params_by_task
        else None,
        "final_trainable_params": trainable_params_by_task[-1]
        if trainable_params_by_task
        else None,
    }
    runtime = {
        "wall_clock_seconds": parse_wall_clock_seconds(args.wall_clock_seconds, text),
        "peak_gpu_memory_mib": parse_gpu_peak_mib(args.gpu_memory_samples, text),
    }

    summary = {
        "run_name": args.run_name,
        "method": args.method or config.get("model_name"),
        "dataset_requested": args.dataset_requested,
        "dataset_config_key": args.dataset_config_key or config.get("dataset"),
        "split": args.split or infer_split(config),
        "seed": args.seed if args.seed is not None else first_seed(config),
        "config": args.config,
        "source_logs": {
            "wrapper_log": args.wrapper_log,
            "repo_log": args.repo_log,
            "gpu_memory_samples": args.gpu_memory_samples,
            "wall_clock_seconds": args.wall_clock_seconds,
        },
        "top1_curve": top1_curve,
        "top5_curve": top5_curve,
        "final_accuracy": final_accuracy,
        "average_incremental_accuracy": average_incremental_accuracy,
        "forgetting": forgetting,
        "params": params,
        "runtime": runtime,
        "cnn": {
            "top1_curve": top1_curve,
            "top5_curve": top5_curve,
            "average_incremental_accuracy": average_incremental_accuracy,
            "final_accuracy": final_accuracy,
            "forgetting": forgetting,
            "grouped_accuracy_by_eval": cnn_grouped,
            "final_grouped_accuracy": final_grouped,
        },
        "nme": {
            "top1_curve": nme_top1_curves[-1] if nme_top1_curves else [],
            "top5_curve": nme_top5_curves[-1] if nme_top5_curves else [],
            "grouped_accuracy_by_eval": nme_grouped,
            "final_grouped_accuracy": nme_grouped[-1] if nme_grouped else {},
        },
        "notes": [
            "Metrics are parsed from wrapper logs when available, otherwise from repo logs.",
            "Numpy scalar reprs such as np.float64(98.9) are normalized before literal parsing.",
        ],
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Extract PILOT metrics from logs.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--wrapper-log", required=True)
    parser.add_argument("--repo-log", required=True)
    parser.add_argument("--gpu-memory-samples", required=True)
    parser.add_argument("--wall-clock-seconds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method")
    parser.add_argument("--dataset-requested", default="CIFAR100")
    parser.add_argument("--dataset-config-key")
    parser.add_argument("--split")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    summary = build_summary(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("Wrote metrics summary to {}".format(output))


if __name__ == "__main__":
    main()
