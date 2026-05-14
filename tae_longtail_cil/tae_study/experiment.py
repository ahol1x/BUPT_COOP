from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import statistics
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tae_study.data import CIFARCache, LongTailCase, make_cifar10_cache, make_loader
from tae_study.models import SmallCILNet


TOTAL_CLASSES = 10
CLASSES_PER_TASK = 2
METHOD_BASELINE = "traditional_full_update"
METHOD_TAE = "tae_ced_top_p"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run a 10-case long-tailed CIL comparison between baseline and TaE/CEd."
    )
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "final_results")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--studies", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--classes-per-task", type=int, default=CLASSES_PER_TASK)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--head-train-per-class", type=int, default=5000)
    parser.add_argument("--tail-ratio", type=float, default=0.1)
    parser.add_argument("--min-train-per-class", type=int, default=50)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--tae-budget", type=float, default=0.15)
    parser.add_argument("--mask-batches", type=int, default=20)
    parser.add_argument("--centroid-min-weight", type=float, default=0.1)
    parser.add_argument("--centroid-max-weight", type=float, default=0.1)
    parser.add_argument("--class-weight-beta", type=float, default=0.95)
    parser.add_argument("--highlight-cases", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--test-tasks", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> None:
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    cache = make_cifar10_cache(args.data_dir, args.download_data)
    result_path = args.output_dir / "per_case_results.csv"
    rows = read_result_rows(result_path) if args.resume and result_path.exists() else []
    completed = {(row["case"], row["method"]) for row in rows}

    config_path = args.output_dir / "run_config.json"
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config_path.write_text(json.dumps(serializable_args, indent=2), encoding="utf-8")

    for case_index in range(1, args.studies + 1):
        case = make_longtail_case(case_index, args)
        for method in (METHOD_BASELINE, METHOD_TAE):
            key = (case.name, method)
            if key in completed:
                continue
            print(f"running {case.name} method={method} seed={case.seed}")
            row = run_method(case, method, args, device, cache)
            rows.append(row)
            write_result_rows(rows, result_path, args.test_tasks)
            completed.add(key)

    write_result_rows(rows, result_path, args.test_tasks)
    highlighted = write_highlight_rows(rows, args.output_dir, args.highlight_cases, args.test_tasks)
    summary = summarize(rows, args.test_tasks, args.highlight_cases)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_html_report(rows, highlighted, summary, args.output_dir / "comparison.html", args.test_tasks)
    write_method_notes(args.output_dir / "paper_method_notes.md")
    print(f"wrote results to {args.output_dir}")


def validate_args(args: argparse.Namespace) -> None:
    if TOTAL_CLASSES % args.classes_per_task != 0:
        raise ValueError("TOTAL_CLASSES must be divisible by --classes-per-task.")
    if args.tae_budget <= 0 or args.tae_budget > 1:
        raise ValueError("--tae-budget must be in (0, 1].")
    if args.tail_ratio <= 0 or args.tail_ratio > 1:
        raise ValueError("--tail-ratio must be in (0, 1].")
    max_task = TOTAL_CLASSES // args.classes_per_task
    invalid_tasks = [task for task in args.test_tasks if task < 1 or task > max_task]
    if invalid_tasks:
        raise ValueError(f"--test-tasks contains invalid task ids: {invalid_tasks}")


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_longtail_case(case_index: int, args: argparse.Namespace) -> LongTailCase:
    seed = args.base_seed + case_index - 1
    rng = random.Random(seed)
    order = list(range(TOTAL_CLASSES))
    rng.shuffle(order)
    train_counts: dict[int, int] = {}
    for rank, label in enumerate(order):
        exponent = rank / max(TOTAL_CLASSES - 1, 1)
        count = round(args.head_train_per_class * (args.tail_ratio ** exponent))
        train_counts[label] = max(args.min_train_per_class, count)
    return LongTailCase(
        name=f"case{case_index:02d}",
        seed=seed,
        class_order=order,
        train_counts=train_counts,
        test_count_per_class=args.test_per_class,
    )


def task_classes(case: LongTailCase, args: argparse.Namespace) -> list[list[int]]:
    step = args.classes_per_task
    return [case.class_order[i : i + step] for i in range(0, TOTAL_CLASSES, step)]


def run_method(
    case: LongTailCase,
    method: str,
    args: argparse.Namespace,
    device: torch.device,
    cache: CIFARCache,
) -> dict[str, str]:
    set_seed(case.seed)
    model = SmallCILNet(num_classes=TOTAL_CLASSES, feature_dim=128).to(device)
    centroids = torch.zeros(TOTAL_CLASSES, model.feature_dim, device=device, requires_grad=True)
    seen_labels: list[int] = []
    curve: list[float] = []
    per_task_accuracy: dict[int, float] = {}

    for task_index, current_labels in enumerate(task_classes(case, args), start=1):
        seen_labels.extend(current_labels)
        train_loader = make_loader(
            cache.train,
            current_labels,
            case.train_counts,
            seed=case.seed + task_index,
            batch_size=args.batch_size,
            shuffle=True,
        )
        eval_loader = make_loader(
            cache.test,
            seen_labels,
            {label: case.test_count_per_class for label in range(TOTAL_CLASSES)},
            seed=case.seed + 1000 + task_index,
            batch_size=args.batch_size,
            shuffle=False,
        )

        masks = None
        if method == METHOD_TAE:
            init_new_centroids(model, centroids, train_loader, current_labels, device)
            if task_index > 1:
                masks = select_tae_mask(
                    model=model,
                    loader=train_loader,
                    seen_labels=seen_labels,
                    device=device,
                    budget=args.tae_budget,
                    batches=args.mask_batches,
                )

        train_task(
            model=model,
            centroids=centroids,
            loader=train_loader,
            seen_labels=seen_labels,
            current_labels=current_labels,
            case=case,
            method=method,
            device=device,
            args=args,
            masks=masks,
        )
        accuracy = evaluate(model, eval_loader, seen_labels, device)
        curve.append(accuracy)
        per_task_accuracy[task_index] = accuracy

    row: dict[str, str] = {
        "case": case.name,
        "study_seed": str(case.seed),
        "method": method,
        "class_order": " ".join(map(str, case.class_order)),
        "train_counts": " ".join(f"{label}:{case.train_counts[label]}" for label in sorted(case.train_counts)),
        "curve": " ".join(f"{value:.4f}" for value in curve),
        "final_accuracy": f"{curve[-1]:.6f}",
        "average_accuracy": f"{statistics.mean(curve):.6f}",
    }
    for task in args.test_tasks:
        row[f"task{task}_accuracy"] = f"{per_task_accuracy[task]:.6f}"
    return row


def remap_labels(labels: torch.Tensor, seen_labels: list[int]) -> torch.Tensor:
    mapped = torch.empty_like(labels)
    for local_index, global_label in enumerate(seen_labels):
        mapped[labels == global_label] = local_index
    return mapped


def class_weights(case: LongTailCase, seen_labels: list[int], beta: float, device: torch.device) -> torch.Tensor:
    weights = torch.ones(TOTAL_CLASSES, device=device)
    for label in seen_labels:
        n = max(case.train_counts[label], 1)
        effective = (1.0 - beta**n) / max(1.0 - beta, 1e-8)
        weights[label] = 1.0 / max(effective, 1e-8)
    weights = weights / weights[seen_labels].mean()
    return weights


def select_tae_mask(
    model: nn.Module,
    loader: DataLoader,
    seen_labels: list[int],
    device: torch.device,
    budget: float,
    batches: int,
) -> dict[str, torch.Tensor]:
    model.train()
    grad_scores: dict[str, torch.Tensor] = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    seen = torch.tensor(seen_labels, dtype=torch.long, device=device)
    used_batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        model.zero_grad(set_to_none=True)
        logits = model(images).index_select(1, seen)
        local_labels = remap_labels(labels, seen_labels).to(device)
        loss = F.cross_entropy(logits, local_labels)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_scores[name].add_(param.grad.detach().abs())
        used_batches += 1
        if used_batches >= batches:
            break

    flat_scores = torch.cat([score.flatten() for score in grad_scores.values()])
    k = max(1, int(math.ceil(flat_scores.numel() * budget)))
    threshold = torch.topk(flat_scores, k, sorted=False).values.min()
    return {name: (score >= threshold).float() for name, score in grad_scores.items()}


@torch.no_grad()
def init_new_centroids(
    model: SmallCILNet,
    centroids: torch.Tensor,
    loader: DataLoader,
    new_labels: list[int],
    device: torch.device,
) -> None:
    model.eval()
    sums = {label: torch.zeros(centroids.shape[1], device=device) for label in new_labels}
    counts = {label: 0 for label in new_labels}
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        _, features = model(images, return_features=True)
        for label in new_labels:
            mask = labels == label
            if mask.any():
                sums[label] += features[mask].sum(dim=0)
                counts[label] += int(mask.sum().item())
    for label in new_labels:
        if counts[label] > 0:
            centroids.data[label] = sums[label] / counts[label]


def centroid_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor,
    seen_labels: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    norm_features = F.normalize(features, dim=1)
    norm_centroids = F.normalize(centroids, dim=1)
    l_min = -F.cosine_similarity(norm_features, norm_centroids[labels], dim=1).mean()
    seen = torch.tensor(seen_labels, dtype=torch.long, device=features.device)
    seen_centroids = norm_centroids.index_select(0, seen)
    if len(seen_labels) <= 1:
        return l_min, torch.zeros((), device=features.device)
    similarity = seen_centroids @ seen_centroids.T
    off_diag = ~torch.eye(len(seen_labels), dtype=torch.bool, device=features.device)
    l_max = similarity[off_diag].mean()
    return l_min, l_max


def apply_model_mask(model: nn.Module, masks: dict[str, torch.Tensor] | None) -> None:
    if masks is None:
        return
    for name, param in model.named_parameters():
        if param.grad is not None and name in masks:
            param.grad.mul_(masks[name])


def apply_centroid_mask(centroids: torch.Tensor, current_labels: list[int]) -> None:
    if centroids.grad is None:
        return
    mask = torch.zeros_like(centroids.grad)
    mask[current_labels] = 1.0
    centroids.grad.mul_(mask)


def train_task(
    model: SmallCILNet,
    centroids: torch.Tensor,
    loader: DataLoader,
    seen_labels: list[int],
    current_labels: list[int],
    case: LongTailCase,
    method: str,
    device: torch.device,
    args: argparse.Namespace,
    masks: dict[str, torch.Tensor] | None,
) -> None:
    params = list(model.parameters())
    if method == METHOD_TAE:
        params.append(centroids)
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    seen = torch.tensor(seen_labels, dtype=torch.long, device=device)
    weights = None
    if method == METHOD_TAE:
        weights = class_weights(case, seen_labels, args.class_weight_beta, device).index_select(0, seen)

    for _ in range(args.epochs):
        model.train()
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, features = model(images, return_features=True)
            seen_logits = logits.index_select(1, seen)
            local_labels = remap_labels(labels, seen_labels).to(device)
            loss = F.cross_entropy(seen_logits, local_labels, weight=weights)
            if method == METHOD_TAE:
                l_min, l_max = centroid_loss(features, labels, centroids, seen_labels)
                loss = (
                    loss
                    + args.centroid_min_weight * l_min
                    + args.centroid_max_weight * l_max
                )
            loss.backward()
            if method == METHOD_TAE:
                apply_model_mask(model, masks)
                apply_centroid_mask(centroids, current_labels)
            optimizer.step()


@torch.no_grad()
def evaluate(model: SmallCILNet, loader: DataLoader, seen_labels: list[int], device: torch.device) -> float:
    model.eval()
    seen = torch.tensor(seen_labels, dtype=torch.long, device=device)
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images).index_select(1, seen)
        predictions = seen[logits.argmax(dim=1)]
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())
    return 100.0 * correct / max(total, 1)


def result_fieldnames(test_tasks: list[int]) -> list[str]:
    fields = [
        "case",
        "study_seed",
        "method",
        "class_order",
        "train_counts",
        "curve",
    ]
    fields.extend(f"task{task}_accuracy" for task in test_tasks)
    fields.extend(["final_accuracy", "average_accuracy"])
    return fields


def write_result_rows(rows: list[dict[str, str]], path: Path, test_tasks: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = result_fieldnames(test_tasks)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_result_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_highlight_rows(
    rows: list[dict[str, str]],
    output_dir: Path,
    highlight_cases: list[int],
    test_tasks: list[int],
) -> list[dict[str, str]]:
    names = {f"case{case_index:02d}" for case_index in highlight_cases}
    highlighted = [row for row in rows if row["case"] in names]
    write_result_rows(highlighted, output_dir / "highlight_case_results.csv", test_tasks)
    return highlighted


def summarize(rows: list[dict[str, str]], test_tasks: list[int], highlight_cases: list[int]) -> dict:
    summary: dict[str, object] = {
        "methods": {},
        "differences_tae_minus_traditional": {},
        "highlight_cases": [f"case{case_index:02d}" for case_index in highlight_cases],
    }
    metrics = [f"task{task}_accuracy" for task in test_tasks] + ["final_accuracy", "average_accuracy"]
    by_method = {method: [row for row in rows if row["method"] == method] for method in (METHOD_BASELINE, METHOD_TAE)}
    for method, method_rows in by_method.items():
        summary["methods"][method] = {
            metric: describe([float(row[metric]) for row in method_rows])
            for metric in metrics
        }

    for metric in metrics:
        baseline = {row["case"]: float(row[metric]) for row in by_method[METHOD_BASELINE]}
        tae = {row["case"]: float(row[metric]) for row in by_method[METHOD_TAE]}
        diffs = [tae[name] - baseline[name] for name in sorted(baseline) if name in tae]
        summary["differences_tae_minus_traditional"][metric] = describe(diffs)
    return summary


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": std,
        "min": min(values),
        "max": max(values),
    }


def parse_curve(value: str) -> list[float]:
    return [float(item) for item in value.split()]


def mean_curve(rows: list[dict[str, str]], method: str) -> list[float]:
    curves = [parse_curve(row["curve"]) for row in rows if row["method"] == method]
    return [statistics.mean(points) for points in zip(*curves)]


def svg_line_chart(title: str, series: dict[str, list[float]]) -> str:
    width, height = 780, 360
    left, right, top, bottom = 58, 28, 24, 48
    colors = {
        METHOD_BASELINE: "#2563eb",
        METHOD_TAE: "#dc2626",
        "traditional": "#2563eb",
        "tae": "#dc2626",
    }
    values = [value for points in series.values() for value in points]
    ymin = max(0.0, math.floor(min(values) / 10.0) * 10.0)
    ymax = min(100.0, math.ceil(max(values) / 10.0) * 10.0)
    if ymax <= ymin:
        ymax = ymin + 10.0

    def x_pos(index: int, count: int) -> float:
        if count == 1:
            return left
        return left + index * ((width - left - right) / (count - 1))

    def y_pos(value: float) -> float:
        return top + (ymax - value) * ((height - top - bottom) / (ymax - ymin))

    y_ticks = []
    for tick_index in range(6):
        value = ymin + tick_index * (ymax - ymin) / 5
        y = y_pos(value)
        y_ticks.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e5e7eb" />'
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{value:.0f}</text>'
        )

    lines = []
    circles = []
    for label, points in series.items():
        color = colors.get(label, "#111827")
        encoded = " ".join(
            f"{x_pos(index, len(points)):.2f},{y_pos(value):.2f}"
            for index, value in enumerate(points)
        )
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{encoded}" />')
        for index, value in enumerate(points):
            circles.append(
                f'<circle cx="{x_pos(index, len(points)):.2f}" cy="{y_pos(value):.2f}" r="4" fill="{color}">'
                f"<title>{html.escape(label)} T{index + 1}: {value:.2f}%</title></circle>"
            )

    x_ticks = []
    task_count = len(next(iter(series.values())))
    for index in range(task_count):
        x = x_pos(index, task_count)
        x_ticks.append(
            f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{height - bottom}" y2="{height - bottom + 5}" stroke="#6b7280" />'
            f'<text x="{x:.2f}" y="{height - 20}" text-anchor="middle">Task {index + 1}</text>'
        )

    return f"""
    <section>
      <h2>{html.escape(title)}</h2>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
        <rect width="{width}" height="{height}" fill="white" />
        {''.join(y_ticks)}
        <line x1="{left}" x2="{width - right}" y1="{height - bottom}" y2="{height - bottom}" stroke="#111827" />
        <line x1="{left}" x2="{left}" y1="{top}" y2="{height - bottom}" stroke="#111827" />
        {''.join(x_ticks)}
        {''.join(lines)}
        {''.join(circles)}
        <text x="{width / 2}" y="{height - 2}" text-anchor="middle">Incremental task</text>
        <text x="16" y="{height / 2}" text-anchor="middle" transform="rotate(-90 16 {height / 2})">Accuracy (%)</text>
      </svg>
    </section>
    """


def write_html_report(
    rows: list[dict[str, str]],
    highlighted: list[dict[str, str]],
    summary: dict,
    path: Path,
    test_tasks: list[int],
) -> None:
    mean_series = {
        METHOD_BASELINE: mean_curve(rows, METHOD_BASELINE),
        METHOD_TAE: mean_curve(rows, METHOD_TAE),
    }
    charts = [svg_line_chart("Mean accuracy across all long-tail cases", mean_series)]
    for case_name in sorted({row["case"] for row in highlighted}):
        case_rows = [row for row in highlighted if row["case"] == case_name]
        series = {row["method"]: parse_curve(row["curve"]) for row in case_rows}
        if len(series) == 2:
            charts.append(svg_line_chart(f"{case_name} detailed curve", series))

    metric_rows = []
    metrics = [f"task{task}_accuracy" for task in test_tasks] + ["final_accuracy", "average_accuracy"]
    for metric in metrics:
        baseline = summary["methods"][METHOD_BASELINE][metric]["mean"]
        tae = summary["methods"][METHOD_TAE][metric]["mean"]
        diff = summary["differences_tae_minus_traditional"][metric]["mean"]
        metric_rows.append(
            "<tr>"
            f"<td>{html.escape(metric)}</td>"
            f"<td>{baseline:.2f}</td>"
            f"<td>{tae:.2f}</td>"
            f"<td>{diff:+.2f}</td>"
            "</tr>"
        )

    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TaE Long-tail CIL Comparison</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #111827; }}
    h1 {{ font-size: 24px; margin-bottom: 8px; }}
    h2 {{ font-size: 18px; margin-top: 28px; }}
    p {{ max-width: 900px; line-height: 1.45; color: #374151; }}
    svg {{ width: min(100%, 920px); height: auto; border: 1px solid #e5e7eb; }}
    table {{ border-collapse: collapse; margin-top: 24px; min-width: 760px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f3f4f6; }}
    .legend {{ display: flex; gap: 18px; margin: 16px 0; }}
    .swatch {{ display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: -1px; }}
  </style>
</head>
<body>
  <h1>TaE Long-tail CIL Comparison</h1>
  <p>This report compares a traditional full-update baseline with a TaE/CEd model across shuffled long-tailed CIFAR-10 class-incremental studies. The highlighted case curves show the requested case 1 and case 5 details when those cases are present.</p>
  <div class="legend">
    <span><span class="swatch" style="background:#2563eb"></span>{html.escape(METHOD_BASELINE)}</span>
    <span><span class="swatch" style="background:#dc2626"></span>{html.escape(METHOD_TAE)}</span>
  </div>
  {''.join(charts)}
  <h2>Mean Summary</h2>
  <table>
    <thead><tr><th>Metric</th><th>Traditional mean</th><th>TaE/CEd mean</th><th>Difference</th></tr></thead>
    <tbody>{''.join(metric_rows)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def write_method_notes(path: Path) -> None:
    body = """# What This Study Implements From Paper 0471

Paper 0471 addresses long-tailed class-incremental learning, where each new task can contain head classes with many samples and tail classes with very few samples. Traditional full-update finetuning often overfits the new head classes and overwrites older class representations.

This folder implements three ideas from the paper:

1. Task-aware parameter selection: before each new task after task 1, the runner accumulates gradients on the incoming task and selects the top-p percent most sensitive parameters.
2. Frozen majority update: during that task, only the selected parameters are updated, while the rest of the model is protected from drift.
3. Centroid-enhanced representation: each class has a learnable centroid. The model pulls features toward their own class centroid and pushes different class centroids apart. A class reweighting term also reduces head-class dominance.

Expected benefit over the traditional baseline:

- Less catastrophic forgetting because most older parameters are not changed on every new task.
- Better tail-class feature separation because the centroid loss gives sparse tail classes a stronger geometric target.
- Lower adaptation cost than fully expanding a whole backbone for every task because only a selected parameter subset is trainable per task.

The code here is a compact, server-friendly study runner inspired by the paper and the LAMDA-PILOT baseline folder. It does not modify the downloaded LAMDA-PILOT submodule.
"""
    path.write_text(body, encoding="utf-8")

