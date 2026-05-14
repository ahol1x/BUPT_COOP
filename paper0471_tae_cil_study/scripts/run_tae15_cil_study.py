from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CLASSES_PER_TASK = 2
TOTAL_CLASSES = 10


@dataclass(frozen=True)
class CaseConfig:
    name: str
    train_counts: dict[int, int]
    test_count_per_class: int
    shuffled_order: bool


@dataclass(frozen=True)
class CIFARCache:
    train: "IndexedCIFAR"
    test: "IndexedCIFAR"


class SmallCILNet(nn.Module):
    def __init__(self, num_classes: int = TOTAL_CLASSES, feature_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        z = self.projector(self.features(x))
        logits = self.classifier(z)
        if return_features:
            return logits, z
        return logits


class IndexedCIFAR:
    def __init__(self, dataset):
        self.dataset = dataset
        self.targets = np.asarray(dataset.targets)
        self.by_class = {
            label: np.where(self.targets == label)[0].tolist()
            for label in range(TOTAL_CLASSES)
        }

    def subset(self, labels: list[int], per_class_counts: dict[int, int], rng: np.random.Generator) -> Subset:
        indices: list[int] = []
        for label in labels:
            class_indices = np.array(self.by_class[label])
            count = min(per_class_counts[label], len(class_indices))
            chosen = rng.choice(class_indices, size=count, replace=False)
            indices.extend(chosen.tolist())
        rng.shuffle(indices)
        return Subset(self.dataset, indices)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_order(seed: int, shuffled: bool) -> list[int]:
    order = list(range(TOTAL_CLASSES))
    if shuffled:
        rng = random.Random(seed)
        rng.shuffle(order)
    return order


def task_classes(order: list[int]) -> list[list[int]]:
    return [order[i : i + CLASSES_PER_TASK] for i in range(0, TOTAL_CLASSES, CLASSES_PER_TASK)]


def make_case_configs(args: argparse.Namespace) -> list[CaseConfig]:
    balanced_counts = {label: args.case1_train_per_class for label in range(TOTAL_CLASSES)}
    long_tail_counts = {}
    for rank, label in enumerate(range(TOTAL_CLASSES)):
        exponent = rank / (TOTAL_CLASSES - 1)
        count = round(args.case2_head_train_per_class * (args.case2_tail_ratio ** exponent))
        long_tail_counts[label] = max(args.case2_min_train_per_class, count)
    return [
        CaseConfig("case1_balanced", balanced_counts, args.test_per_class, False),
        CaseConfig("case2_shuffled_long_tail", long_tail_counts, args.test_per_class, True),
    ]


def make_cifar_cache(root: Path) -> CIFARCache:
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    return CIFARCache(
        train=IndexedCIFAR(datasets.CIFAR10(root=str(root), train=True, download=True, transform=transform_train)),
        test=IndexedCIFAR(datasets.CIFAR10(root=str(root), train=False, download=True, transform=transform_test)),
    )


def make_loaders(cache: CIFARCache, case: CaseConfig, labels: list[int], seed: int, batch_size: int):
    rng = np.random.default_rng(seed)
    train_subset = cache.train.subset(labels, case.train_counts, rng)
    test_subset = cache.test.subset(labels, {label: case.test_count_per_class for label in range(TOTAL_CLASSES)}, rng)
    return (
        DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def class_weights(case: CaseConfig, seen_labels: list[int], device: torch.device) -> torch.Tensor:
    weights = torch.ones(TOTAL_CLASSES, device=device)
    beta = 0.95
    for label in seen_labels:
        n = case.train_counts[label]
        effective = (1.0 - beta**n) / (1.0 - beta)
        weights[label] = 1.0 / max(effective, 1e-6)
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
    used = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        model.zero_grad(set_to_none=True)
        logits = model(images)
        masked_logits = logits.index_select(1, seen)
        local_labels = remap_labels(labels, seen_labels).to(device)
        loss = F.cross_entropy(masked_logits, local_labels)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_scores[name].add_(param.grad.detach().abs())
        used += 1
        if used >= batches:
            break

    flat = torch.cat([score.flatten() for score in grad_scores.values()])
    k = max(1, int(math.ceil(flat.numel() * budget)))
    threshold = torch.topk(flat, k, sorted=False).values.min()
    return {name: (score >= threshold).float() for name, score in grad_scores.items()}


def remap_labels(labels: torch.Tensor, seen_labels: list[int]) -> torch.Tensor:
    out = torch.empty_like(labels)
    for local, global_label in enumerate(seen_labels):
        out[labels == global_label] = local
    return out


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
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            _, feats = model(images, return_features=True)
            for label in new_labels:
                mask = labels == label
                if mask.any():
                    sums[label] += feats[mask].sum(dim=0)
                    counts[label] += int(mask.sum().item())
    for label in new_labels:
        if counts[label] > 0:
            centroids.data[label] = sums[label] / counts[label]


def centroid_loss(features: torch.Tensor, labels: torch.Tensor, centroids: torch.Tensor, seen_labels: list[int]) -> torch.Tensor:
    norm_features = F.normalize(features, dim=1)
    norm_centroids = F.normalize(centroids, dim=1)
    l_min = -F.cosine_similarity(norm_features, norm_centroids[labels], dim=1).mean()
    seen = torch.tensor(seen_labels, dtype=torch.long, device=features.device)
    seen_centroids = norm_centroids.index_select(0, seen)
    if len(seen_labels) <= 1:
        return l_min
    sim = seen_centroids @ seen_centroids.T
    off_diag = ~torch.eye(len(seen_labels), dtype=torch.bool, device=features.device)
    l_max = sim[off_diag].mean()
    return l_min + l_max


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
    case: CaseConfig,
    method: str,
    device: torch.device,
    args: argparse.Namespace,
    masks: dict[str, torch.Tensor] | None,
) -> None:
    params = list(model.parameters())
    if method == "tae15":
        params.append(centroids)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    seen = torch.tensor(seen_labels, dtype=torch.long, device=device)
    weights = class_weights(case, seen_labels, device) if method == "tae15" else None
    for _ in range(args.epochs):
        model.train()
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, feats = model(images, return_features=True)
            masked_logits = logits.index_select(1, seen)
            local_labels = remap_labels(labels, seen_labels).to(device)
            local_weights = weights.index_select(0, seen) if weights is not None else None
            loss = F.cross_entropy(masked_logits, local_labels, weight=local_weights)
            if method == "tae15":
                loss = loss + args.centroid_weight * centroid_loss(feats, labels, centroids, seen_labels)
            loss.backward()
            if method == "tae15":
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
        preds = seen[logits.argmax(dim=1)]
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
    return 100.0 * correct / max(total, 1)


def run_one(case: CaseConfig, method: str, seed: int, args: argparse.Namespace, device: torch.device, cache: CIFARCache) -> dict:
    set_seed(seed)
    model = SmallCILNet().to(device)
    centroids = torch.zeros(TOTAL_CLASSES, 128, device=device, requires_grad=True)
    order = class_order(seed, case.shuffled_order)
    tasks = task_classes(order)
    seen_labels: list[int] = []
    curve: list[float] = []
    for task_index, current_labels in enumerate(tasks):
        seen_labels.extend(current_labels)
        train_loader, eval_loader = make_loaders(cache, case, current_labels, seed + task_index, args.batch_size)
        _, seen_eval_loader = make_loaders(cache, case, seen_labels, seed + 1000 + task_index, args.batch_size)
        masks = None
        if method == "tae15":
            init_new_centroids(model, centroids, train_loader, current_labels, device)
            if task_index > 0:
                masks = select_tae_mask(
                    model,
                    train_loader,
                    seen_labels,
                    device,
                    args.tae_budget,
                    args.mask_batches,
                )
        train_task(model, centroids, train_loader, seen_labels, current_labels, case, method, device, args, masks)
        curve.append(evaluate(model, seen_eval_loader, seen_labels, device))
    return {
        "case": case.name,
        "method": method,
        "seed": seed,
        "class_order": " ".join(map(str, order)),
        "curve": " ".join(f"{x:.4f}" for x in curve),
        "final_accuracy": curve[-1],
        "average_incremental_accuracy": statistics.mean(curve),
    }


def summarize(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    def safe_stdev(values: list[float]) -> float:
        return statistics.stdev(values) if len(values) > 1 else 0.0

    for case in sorted({row["case"] for row in rows}):
        summary[case] = {}
        case_rows = [row for row in rows if row["case"] == case]
        for metric in ["final_accuracy", "average_incremental_accuracy"]:
            baseline = [row[metric] for row in case_rows if row["method"] == "traditional_full_update"]
            tae = [row[metric] for row in case_rows if row["method"] == "tae15"]
            paired = sorted(
                (
                    (
                        next(row for row in case_rows if row["method"] == "traditional_full_update" and row["seed"] == seed)[metric],
                        next(row for row in case_rows if row["method"] == "tae15" and row["seed"] == seed)[metric],
                    )
                    for seed in sorted({row["seed"] for row in case_rows})
                ),
                key=lambda x: x[0],
            )
            diffs = [b[1] - b[0] for b in paired]
            if len(paired) > 1:
                ttest = stats.ttest_rel([x[1] for x in paired], [x[0] for x in paired])
                t_statistic = float(ttest.statistic)
                p_value = float(ttest.pvalue)
            else:
                t_statistic = float("nan")
                p_value = float("nan")
            diff_sem = stats.sem(diffs) if len(diffs) > 1 else float("nan")
            if len(diffs) > 1 and diff_sem > 0:
                ci_low, ci_high = stats.t.interval(
                    confidence=0.95,
                    df=len(diffs) - 1,
                    loc=statistics.mean(diffs),
                    scale=diff_sem,
                )
            else:
                ci_low = ci_high = statistics.mean(diffs)
            summary[case][metric] = {
                "traditional_mean": statistics.mean(baseline),
                "traditional_std": safe_stdev(baseline),
                "tae15_mean": statistics.mean(tae),
                "tae15_std": safe_stdev(tae),
                "mean_difference_tae_minus_traditional": statistics.mean(diffs),
                "paired_t_statistic": t_statistic,
                "paired_p_value": p_value,
                "significant_at_0.05": bool(p_value < 0.05),
                "difference_95ci_low": float(ci_low),
                "difference_95ci_high": float(ci_high),
                "n": len(diffs),
            }
    return summary


def parse_curve(value: str) -> list[float]:
    return [float(item) for item in value.split()]


def mean_curve(rows: list[dict], case: str, method: str) -> list[float]:
    curves = [parse_curve(row["curve"]) for row in rows if row["case"] == case and row["method"] == method]
    return [statistics.mean(points) for points in zip(*curves)]


def polyline(points: list[tuple[float, float]], color: str) -> str:
    encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{encoded}" />'


def make_chart(case: str, traditional: list[float], tae: list[float]) -> str:
    width, height = 760, 360
    left, right, top, bottom = 58, 24, 24, 48
    values = traditional + tae
    ymin = max(0.0, math.floor(min(values) / 10.0) * 10.0)
    ymax = min(100.0, math.ceil(max(values) / 10.0) * 10.0)
    if ymax <= ymin:
        ymax = ymin + 10.0

    def x_pos(index: int) -> float:
        return left + index * ((width - left - right) / (len(traditional) - 1))

    def y_pos(value: float) -> float:
        return top + (ymax - value) * ((height - top - bottom) / (ymax - ymin))

    trad_points = [(x_pos(i), y_pos(v)) for i, v in enumerate(traditional)]
    tae_points = [(x_pos(i), y_pos(v)) for i, v in enumerate(tae)]
    y_ticks = []
    for i in range(6):
        value = ymin + i * (ymax - ymin) / 5
        y = y_pos(value)
        y_ticks.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e5e7eb" />'
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{value:.0f}</text>'
        )
    x_ticks = []
    for i in range(len(traditional)):
        x = x_pos(i)
        x_ticks.append(
            f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{height - bottom}" y2="{height - bottom + 5}" stroke="#6b7280" />'
            f'<text x="{x:.2f}" y="{height - 20}" text-anchor="middle">T{i + 1}</text>'
        )
    points = []
    for series, color in [(traditional, "#2563eb"), (tae, "#dc2626")]:
        for i, value in enumerate(series):
            points.append(
                f'<circle cx="{x_pos(i):.2f}" cy="{y_pos(value):.2f}" r="4" fill="{color}">'
                f"<title>Task {i + 1}: {value:.2f}%</title></circle>"
            )
    return f"""
    <section>
      <h2>{html.escape(case)}</h2>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(case)} accuracy curves">
        <rect width="{width}" height="{height}" fill="white" />
        {''.join(y_ticks)}
        <line x1="{left}" x2="{width - right}" y1="{height - bottom}" y2="{height - bottom}" stroke="#111827" />
        <line x1="{left}" x2="{left}" y1="{top}" y2="{height - bottom}" stroke="#111827" />
        {''.join(x_ticks)}
        {polyline(trad_points, "#2563eb")}
        {polyline(tae_points, "#dc2626")}
        {''.join(points)}
        <text x="{width / 2}" y="{height - 2}" text-anchor="middle">Incremental task</text>
        <text x="16" y="{height / 2}" text-anchor="middle" transform="rotate(-90 16 {height / 2})">Accuracy (%)</text>
      </svg>
    </section>
    """


def write_visualization(rows: list[dict], summary: dict, path: Path) -> None:
    cases = sorted({row["case"] for row in rows})
    charts = []
    rows_html = []
    for case in cases:
        traditional = mean_curve(rows, case, "traditional_full_update")
        tae = mean_curve(rows, case, "tae15")
        charts.append(make_chart(case, traditional, tae))
        for metric, values in summary[case].items():
            rows_html.append(
                "<tr>"
                f"<td>{html.escape(case)}</td>"
                f"<td>{html.escape(metric)}</td>"
                f"<td>{values['traditional_mean']:.2f}</td>"
                f"<td>{values['tae15_mean']:.2f}</td>"
                f"<td>{values['mean_difference_tae_minus_traditional']:.2f}</td>"
                f"<td>{values['paired_p_value']:.4g}</td>"
                f"<td>{'yes' if values['significant_at_0.05'] else 'no'}</td>"
                "</tr>"
            )
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Paper 0471 TaE 15% CIL Study</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #111827; }}
    h1 {{ font-size: 24px; margin-bottom: 8px; }}
    h2 {{ font-size: 18px; margin-top: 28px; }}
    p {{ max-width: 820px; line-height: 1.45; color: #374151; }}
    svg {{ width: min(100%, 900px); height: auto; border: 1px solid #e5e7eb; }}
    table {{ border-collapse: collapse; margin-top: 24px; min-width: 820px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ background: #f3f4f6; }}
    .legend {{ display: flex; gap: 18px; margin: 16px 0; }}
    .swatch {{ display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: -1px; }}
  </style>
</head>
<body>
  <h1>Paper 0471 TaE 15% CIL Study</h1>
  <p>Mean accuracy curves across seeded studies. The traditional model is full-update finetuning with cross entropy. The paper-0471 model uses top-15% task-aware parameter updates, centroid-enhanced loss, and effective-number reweighting.</p>
  <div class="legend">
    <span><span class="swatch" style="background:#2563eb"></span>Traditional full update</span>
    <span><span class="swatch" style="background:#dc2626"></span>TaE 15%</span>
  </div>
  {''.join(charts)}
  <h2>Statistical Summary</h2>
  <table>
    <thead>
      <tr><th>Case</th><th>Metric</th><th>Traditional mean</th><th>TaE15 mean</th><th>Diff</th><th>Paired p</th><th>p &lt; 0.05</th></tr>
    </thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--studies", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--tae-budget", type=float, default=0.15)
    parser.add_argument("--mask-batches", type=int, default=16)
    parser.add_argument("--centroid-weight", type=float, default=0.1)
    parser.add_argument("--case1-train-per-class", type=int, default=1000)
    parser.add_argument("--case2-head-train-per-class", type=int, default=1000)
    parser.add_argument("--case2-tail-ratio", type=float, default=0.1)
    parser.add_argument("--case2-min-train-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"device={device}")
    cache = make_cifar_cache(args.data_dir)
    rows: list[dict] = []
    csv_path = args.results_dir / "tae15_cil_results.csv"
    if args.resume and csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["seed"] = int(row["seed"])
                row["final_accuracy"] = float(row["final_accuracy"])
                row["average_incremental_accuracy"] = float(row["average_incremental_accuracy"])
                rows.append(row)

    completed = {(row["case"], row["method"], row["seed"]) for row in rows}
    seeds = list(range(args.studies))
    for case in make_case_configs(args):
        for method in ["traditional_full_update", "tae15"]:
            for seed in seeds:
                key = (case.name, method, seed)
                if key in completed:
                    continue
                print(f"running case={case.name} method={method} seed={seed}")
                row = run_one(case, method, seed, args, device, cache)
                rows.append(row)
                write_csv(rows, csv_path)

    summary = summarize(rows)
    summary_path = args.results_dir / "tae15_cil_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_visualization(rows, summary, args.results_dir / "tae15_cil_visualization.html")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
