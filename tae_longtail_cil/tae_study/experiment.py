from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from tae_study.data import CIFARCache, LongTailCase, make_cifar10_cache, make_loader
from tae_study.models import SmallCILNet


TOTAL_CLASSES = 10
CLASSES_PER_TASK = 2
CIFAR_IMAGE_BYTES = 32 * 32 * 3
METHOD_BASELINE = "traditional_full_update"
METHOD_TAE = "tae_ced_top_p"
METHOD_TAE_REVIEW = "tae_review_replay"
METHODS = (METHOD_BASELINE, METHOD_TAE, METHOD_TAE_REVIEW)
TAE_METHODS = {METHOD_TAE, METHOD_TAE_REVIEW}
METHOD_COLORS = {
    METHOD_BASELINE: "#2563eb",
    METHOD_TAE: "#dc2626",
    METHOD_TAE_REVIEW: "#059669",
}


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
    parser.add_argument("--class-weight-beta", type=float, default=0.9999)
    parser.add_argument("--max-class-weight", type=float, default=6.0)
    parser.add_argument("--prototype-replay-weight", type=float, default=0.25)
    parser.add_argument("--logit-adjustment-tau", type=float, default=0.5)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Methods to run. Default runs traditional, TaE, and TaE with review replay.",
    )
    parser.add_argument("--review-exemplars-per-class", type=int, default=100)
    parser.add_argument("--review-epochs", type=int, default=3)
    parser.add_argument("--review-lr-scale", type=float, default=0.25)
    parser.add_argument("--highlight-cases", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--test-tasks", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print live case, task, epoch, and evaluation details.")
    parser.add_argument(
        "--log-every-batches",
        type=int,
        default=0,
        help="With --verbose, also print running training stats every N batches.",
    )
    parser.set_defaults(tae_balanced_sampling=True)
    parser.add_argument(
        "--tae-balanced-sampling",
        dest="tae_balanced_sampling",
        action="store_true",
        help="Use class-balanced sampling for TaE current-task batches.",
    )
    parser.add_argument(
        "--no-tae-balanced-sampling",
        dest="tae_balanced_sampling",
        action="store_false",
        help="Disable class-balanced sampling for TaE.",
    )
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> None:
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    live_log(
        args,
        "starting study "
        f"device={device} studies={args.studies} epochs={args.epochs} "
        f"batch_size={args.batch_size} tae_balanced_sampling={args.tae_balanced_sampling} "
        f"class_weight_beta={args.class_weight_beta} max_class_weight={args.max_class_weight} "
        f"prototype_replay_weight={args.prototype_replay_weight} "
        f"logit_adjustment_tau={args.logit_adjustment_tau} "
        f"review_exemplars_per_class={args.review_exemplars_per_class} "
        f"review_epochs={args.review_epochs} methods={args.methods} output_dir={args.output_dir}",
    )
    live_log(args, f"loading CIFAR-10 data from {args.data_dir} download={args.download_data}")
    cache = make_cifar10_cache(args.data_dir, args.download_data)
    result_path = args.output_dir / "per_case_results.csv"
    rows = read_result_rows(result_path) if args.resume and result_path.exists() else []
    completed = {(row["case"], row["method"]) for row in rows}
    if completed:
        live_log(args, f"resume enabled: found {len(completed)} completed case/method runs")

    config_path = args.output_dir / "run_config.json"
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config_path.write_text(json.dumps(serializable_args, indent=2), encoding="utf-8")

    for case_index in range(1, args.studies + 1):
        case = make_longtail_case(case_index, args)
        for method in args.methods:
            key = (case.name, method)
            if key in completed:
                live_log(args, f"skipping completed {case.name} method={method}")
                continue
            print(f"running {case.name} method={method} seed={case.seed}", flush=True)
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
    print(f"wrote results to {args.output_dir}", flush=True)


def live_log(args: argparse.Namespace, message: str) -> None:
    if getattr(args, "verbose", False):
        print(message, flush=True)


def format_counts(labels: list[int], counts: dict[int, int]) -> str:
    return ", ".join(f"{label}:{counts[label]}" for label in labels)


def mask_coverage(masks: dict[str, torch.Tensor] | None) -> tuple[int, int, float]:
    if masks is None:
        return 0, 0, 0.0
    active = sum(int(mask.sum().item()) for mask in masks.values())
    total = sum(mask.numel() for mask in masks.values())
    percent = 100.0 * active / max(total, 1)
    return active, total, percent


def is_tae_method(method: str) -> bool:
    return method in TAE_METHODS


def uses_review_memory(method: str) -> bool:
    return method == METHOD_TAE_REVIEW


@dataclass
class RuntimeStats:
    training_seconds: float = 0.0
    review_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    review_events: int = 0


class ReviewMemory:
    def __init__(self, exemplars_per_class: int) -> None:
        self.exemplars_per_class = exemplars_per_class
        self.indices_by_class: dict[int, list[int]] = {}

    def update(
        self,
        dataset,
        labels: list[int],
        per_class_counts: dict[int, int],
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        for label in labels:
            class_indices = np.asarray(dataset.by_class[label])
            available_count = min(per_class_counts.get(label, len(class_indices)), len(class_indices))
            if available_count <= 0 or self.exemplars_per_class <= 0:
                self.indices_by_class[label] = []
                continue
            available = rng.choice(class_indices, size=available_count, replace=False)
            keep_count = min(self.exemplars_per_class, available_count)
            chosen = rng.choice(available, size=keep_count, replace=False)
            self.indices_by_class[label] = chosen.tolist()

    def labels(self) -> list[int]:
        return sorted(label for label, indices in self.indices_by_class.items() if indices)

    def indices(self, labels: list[int]) -> list[int]:
        selected: list[int] = []
        for label in labels:
            selected.extend(self.indices_by_class.get(label, []))
        return selected

    def exemplar_count(self) -> int:
        return sum(len(indices) for indices in self.indices_by_class.values())

    def replay_memory_bytes(self) -> int:
        return self.exemplar_count() * CIFAR_IMAGE_BYTES

    def make_loader(
        self,
        dataset,
        labels: list[int],
        batch_size: int,
        seed: int,
        class_balanced: bool = True,
    ) -> DataLoader | None:
        indices = self.indices(labels)
        if not indices:
            return None
        subset = Subset(dataset.dataset, indices)
        generator = torch.Generator()
        generator.manual_seed(seed)
        if not class_balanced:
            return DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                generator=generator,
                pin_memory=torch.cuda.is_available(),
            )
        subset_targets = dataset.targets[np.asarray(indices)]
        class_counts = {
            label: max(int((subset_targets == label).sum()), 1)
            for label in labels
        }
        weights = np.asarray(
            [1.0 / class_counts[int(label)] for label in subset_targets],
            dtype=np.float64,
        )
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(indices),
            replacement=True,
            generator=generator,
        )
        return DataLoader(
            subset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            generator=generator,
            pin_memory=torch.cuda.is_available(),
        )


def validate_args(args: argparse.Namespace) -> None:
    if TOTAL_CLASSES % args.classes_per_task != 0:
        raise ValueError("TOTAL_CLASSES must be divisible by --classes-per-task.")
    if args.tae_budget <= 0 or args.tae_budget > 1:
        raise ValueError("--tae-budget must be in (0, 1].")
    if args.tail_ratio <= 0 or args.tail_ratio > 1:
        raise ValueError("--tail-ratio must be in (0, 1].")
    if args.class_weight_beta <= 0 or args.class_weight_beta >= 1:
        raise ValueError("--class-weight-beta must be in (0, 1).")
    if args.max_class_weight < 1:
        raise ValueError("--max-class-weight must be at least 1.")
    if args.prototype_replay_weight < 0:
        raise ValueError("--prototype-replay-weight must be non-negative.")
    if args.logit_adjustment_tau < 0:
        raise ValueError("--logit-adjustment-tau must be non-negative.")
    if args.review_exemplars_per_class < 0:
        raise ValueError("--review-exemplars-per-class must be non-negative.")
    if args.review_epochs < 0:
        raise ValueError("--review-epochs must be non-negative.")
    if args.review_lr_scale <= 0:
        raise ValueError("--review-lr-scale must be positive.")
    if not args.methods:
        raise ValueError("--methods must include at least one method.")
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
    method_start = time.perf_counter()
    stats = RuntimeStats()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = SmallCILNet(num_classes=TOTAL_CLASSES, feature_dim=128).to(device)
    centroids = torch.zeros(TOTAL_CLASSES, model.feature_dim, device=device, requires_grad=True)
    review_memory = ReviewMemory(args.review_exemplars_per_class) if uses_review_memory(method) else None
    seen_labels: list[int] = []
    curve: list[float] = []
    per_task_accuracy: dict[int, float] = {}
    all_tasks = task_classes(case, args)

    live_log(args, f"[{case.name}][{method}] class_order={case.class_order}")
    live_log(args, f"[{case.name}][{method}] train_counts={format_counts(case.class_order, case.train_counts)}")

    for task_index, current_labels in enumerate(all_tasks, start=1):
        seen_labels.extend(current_labels)
        train_loader = make_loader(
            cache.train,
            current_labels,
            case.train_counts,
            seed=case.seed + task_index,
            batch_size=args.batch_size,
            shuffle=True,
            class_balanced=is_tae_method(method) and args.tae_balanced_sampling,
        )
        eval_loader = make_loader(
            cache.test,
            seen_labels,
            {label: case.test_count_per_class for label in range(TOTAL_CLASSES)},
            seed=case.seed + 1000 + task_index,
            batch_size=args.batch_size,
            shuffle=False,
        )
        centroid_loader = train_loader
        if is_tae_method(method) and args.tae_balanced_sampling:
            centroid_loader = make_loader(
                cache.train,
                current_labels,
                case.train_counts,
                seed=case.seed + task_index,
                batch_size=args.batch_size,
                shuffle=False,
            )
        live_log(
            args,
            f"[{case.name}][{method}] task {task_index}/{len(all_tasks)} "
            f"new_labels={current_labels} seen_labels={seen_labels} "
            f"train_examples={len(train_loader.dataset)} eval_examples={len(eval_loader.dataset)} "
            f"train_batches={len(train_loader)} balanced_sampling={is_tae_method(method) and args.tae_balanced_sampling}",
        )

        masks = None
        if is_tae_method(method):
            live_log(args, f"[{case.name}][{method}] task {task_index}: initializing class centroids")
            init_new_centroids(model, centroids, centroid_loader, current_labels, device)
            if task_index > 1:
                live_log(
                    args,
                    f"[{case.name}][{method}] task {task_index}: selecting TaE mask "
                    f"budget={args.tae_budget:.2f} batches={args.mask_batches}",
                )
                masks = select_tae_mask(
                    model=model,
                    loader=train_loader,
                    seen_labels=seen_labels,
                    case=case,
                    args=args,
                    device=device,
                    budget=args.tae_budget,
                    batches=args.mask_batches,
                )
                active, total, percent = mask_coverage(masks)
                live_log(
                    args,
                    f"[{case.name}][{method}] task {task_index}: mask active_params={active}/{total} "
                    f"({percent:.2f}%)",
                )

        train_start = time.perf_counter()
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
            task_index=task_index,
        )
        stats.training_seconds += time.perf_counter() - train_start
        if is_tae_method(method):
            live_log(args, f"[{case.name}][{method}] task {task_index}: refreshing centroids after training")
            init_new_centroids(model, centroids, centroid_loader, current_labels, device)
        if review_memory is not None:
            review_start = time.perf_counter()
            for class_offset, new_label in enumerate(current_labels):
                review_memory.update(
                    cache.train,
                    [new_label],
                    case.train_counts,
                    seed=case.seed + 5000 + task_index * 100 + class_offset,
                )
                memory_labels = [label for label in seen_labels if label in review_memory.labels()]
                review_loader = review_memory.make_loader(
                    cache.train,
                    memory_labels,
                    args.batch_size,
                    seed=case.seed + 6000 + task_index * 100 + class_offset,
                    class_balanced=True,
                )
                if review_loader is not None and args.review_epochs > 0:
                    stats.review_events += 1
                    live_log(
                        args,
                        f"[{case.name}][{method}] task {task_index}: review after class {new_label} "
                        f"with {review_memory.exemplar_count()} exemplars for {args.review_epochs} epochs",
                    )
                    review_task(
                        model=model,
                        centroids=centroids,
                        loader=review_loader,
                        seen_labels=memory_labels,
                        case=case,
                        device=device,
                        args=args,
                        task_index=task_index,
                        method=method,
                    )
                    centroid_review_loader = review_memory.make_loader(
                        cache.train,
                        memory_labels,
                        args.batch_size,
                        seed=case.seed + 7000 + task_index * 100 + class_offset,
                        class_balanced=False,
                    )
                    if centroid_review_loader is not None:
                        init_new_centroids(model, centroids, centroid_review_loader, memory_labels, device)
            stats.review_seconds += time.perf_counter() - review_start
        logit_adjustment = None
        if is_tae_method(method) and args.logit_adjustment_tau > 0:
            logit_adjustment = make_logit_adjustment(case, seen_labels, args.logit_adjustment_tau, device)
        eval_start = time.perf_counter()
        accuracy = evaluate(model, eval_loader, seen_labels, device, logit_adjustment)
        stats.evaluation_seconds += time.perf_counter() - eval_start
        curve.append(accuracy)
        per_task_accuracy[task_index] = accuracy
        live_log(
            args,
            f"[{case.name}][{method}] task {task_index}: eval_accuracy={accuracy:.2f}% "
            f"curve={' '.join(f'{value:.2f}' for value in curve)}",
        )

    replay_exemplars = review_memory.exemplar_count() if review_memory is not None else 0
    replay_memory_bytes = review_memory.replay_memory_bytes() if review_memory is not None else 0
    centroid_memory_bytes = centroids.numel() * centroids.element_size() if is_tae_method(method) else 0
    extra_memory_bytes = replay_memory_bytes + centroid_memory_bytes
    peak_cuda_memory_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    total_elapsed_seconds = time.perf_counter() - method_start
    row: dict[str, str] = {
        "case": case.name,
        "study_seed": str(case.seed),
        "method": method,
        "class_order": " ".join(map(str, case.class_order)),
        "train_counts": " ".join(f"{label}:{case.train_counts[label]}" for label in sorted(case.train_counts)),
        "curve": " ".join(f"{value:.4f}" for value in curve),
        "final_accuracy": f"{curve[-1]:.6f}",
        "average_accuracy": f"{statistics.mean(curve):.6f}",
        "total_elapsed_seconds": f"{total_elapsed_seconds:.3f}",
        "training_seconds": f"{stats.training_seconds:.3f}",
        "review_seconds": f"{stats.review_seconds:.3f}",
        "evaluation_seconds": f"{stats.evaluation_seconds:.3f}",
        "review_events": str(stats.review_events),
        "replay_exemplars": str(replay_exemplars),
        "replay_memory_bytes": str(replay_memory_bytes),
        "replay_memory_mb": f"{replay_memory_bytes / (1024 ** 2):.6f}",
        "centroid_memory_bytes": str(centroid_memory_bytes),
        "extra_memory_bytes": str(extra_memory_bytes),
        "extra_memory_mb": f"{extra_memory_bytes / (1024 ** 2):.6f}",
        "peak_cuda_memory_bytes": str(peak_cuda_memory_bytes),
        "peak_cuda_memory_mb": f"{peak_cuda_memory_bytes / (1024 ** 2):.6f}",
    }
    for task in args.test_tasks:
        row[f"task{task}_accuracy"] = f"{per_task_accuracy[task]:.6f}"
    live_log(
        args,
        f"[{case.name}][{method}] complete final_accuracy={row['final_accuracy']} "
        f"average_accuracy={row['average_accuracy']}",
    )
    return row


def remap_labels(labels: torch.Tensor, seen_labels: list[int]) -> torch.Tensor:
    mapped = torch.empty_like(labels)
    for local_index, global_label in enumerate(seen_labels):
        mapped[labels == global_label] = local_index
    return mapped


def class_counts(case: LongTailCase, seen_labels: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [max(case.train_counts[label], 1) for label in seen_labels],
        dtype=torch.float32,
        device=device,
    )


def class_weights(
    case: LongTailCase,
    seen_labels: list[int],
    beta: float,
    max_weight: float,
    device: torch.device,
) -> torch.Tensor:
    weights = torch.ones(TOTAL_CLASSES, device=device)
    for label in seen_labels:
        n = max(case.train_counts[label], 1)
        effective = (1.0 - beta**n) / max(1.0 - beta, 1e-8)
        weights[label] = 1.0 / max(effective, 1e-8)
    weights = weights / weights[seen_labels].mean()
    weights = torch.clamp(weights, max=max_weight)
    weights = weights / weights[seen_labels].mean()
    return weights


def balanced_softmax_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    counts: torch.Tensor,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    adjusted_logits = logits + counts.clamp_min(1.0).log().unsqueeze(0)
    per_sample = F.cross_entropy(adjusted_logits, labels, reduction="none")
    if weights is None:
        return per_sample.mean()
    sample_weights = weights.index_select(0, labels)
    return (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)


def make_logit_adjustment(
    case: LongTailCase,
    seen_labels: list[int],
    tau: float,
    device: torch.device,
) -> torch.Tensor:
    counts = class_counts(case, seen_labels, device)
    priors = counts / counts.mean().clamp_min(1e-8)
    return -tau * priors.clamp_min(1e-8).log()


def prototype_replay_loss(
    model: SmallCILNet,
    centroids: torch.Tensor,
    seen_labels: list[int],
    device: torch.device,
) -> torch.Tensor:
    seen = torch.tensor(seen_labels, dtype=torch.long, device=device)
    proto_features = centroids.index_select(0, seen)
    proto_logits = model.classifier(proto_features).index_select(1, seen)
    proto_targets = torch.arange(len(seen_labels), dtype=torch.long, device=device)
    return F.cross_entropy(proto_logits, proto_targets)


def select_tae_mask(
    model: nn.Module,
    loader: DataLoader,
    seen_labels: list[int],
    case: LongTailCase,
    args: argparse.Namespace,
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
    counts = class_counts(case, seen_labels, device)
    weights = class_weights(
        case,
        seen_labels,
        args.class_weight_beta,
        args.max_class_weight,
        device,
    ).index_select(0, seen)
    used_batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        model.zero_grad(set_to_none=True)
        logits = model(images).index_select(1, seen)
        local_labels = remap_labels(labels, seen_labels).to(device)
        loss = balanced_softmax_loss(logits, local_labels, counts, weights)
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
    label_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    norm_features = F.normalize(features, dim=1)
    norm_centroids = F.normalize(centroids, dim=1)
    per_sample_l_min = -F.cosine_similarity(norm_features, norm_centroids[labels], dim=1)
    if label_weights is None:
        l_min = per_sample_l_min.mean()
    else:
        class_losses = []
        class_loss_weights = []
        for label in labels.unique().tolist():
            mask = labels == label
            class_losses.append(per_sample_l_min[mask].mean())
            class_loss_weights.append(label_weights[label])
        stacked_losses = torch.stack(class_losses)
        stacked_weights = torch.stack(class_loss_weights)
        l_min = (stacked_losses * stacked_weights).sum() / stacked_weights.sum().clamp_min(1e-8)
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
        if name.startswith("classifier."):
            continue
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
    task_index: int,
) -> None:
    params = list(model.parameters())
    if is_tae_method(method):
        params.append(centroids)
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    seen = torch.tensor(seen_labels, dtype=torch.long, device=device)
    seen_counts = class_counts(case, seen_labels, device)
    global_label_weights = None
    local_label_weights = None
    if is_tae_method(method):
        global_label_weights = class_weights(
            case,
            seen_labels,
            args.class_weight_beta,
            args.max_class_weight,
            device,
        )
        local_label_weights = global_label_weights.index_select(0, seen)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, features = model(images, return_features=True)
            seen_logits = logits.index_select(1, seen)
            local_labels = remap_labels(labels, seen_labels).to(device)
            if is_tae_method(method):
                loss = balanced_softmax_loss(
                    seen_logits,
                    local_labels,
                    seen_counts,
                    local_label_weights,
                )
            else:
                loss = F.cross_entropy(seen_logits, local_labels)
            if is_tae_method(method):
                l_min, l_max = centroid_loss(
                    features,
                    labels,
                    centroids,
                    seen_labels,
                    global_label_weights,
                )
                loss = (
                    loss
                    + args.centroid_min_weight * l_min
                    + args.centroid_max_weight * l_max
                )
                if task_index > 1 and args.prototype_replay_weight > 0:
                    loss = loss + args.prototype_replay_weight * prototype_replay_loss(
                        model,
                        centroids,
                        seen_labels,
                        device,
                    )
            loss.backward()
            if is_tae_method(method):
                apply_model_mask(model, masks)
                apply_centroid_mask(centroids, current_labels)
            optimizer.step()
            batch_total = int(labels.numel())
            epoch_loss += float(loss.detach().item()) * batch_total
            predictions = seen[seen_logits.detach().argmax(dim=1)]
            epoch_correct += int((predictions == labels).sum().item())
            epoch_total += batch_total
            if (
                getattr(args, "verbose", False)
                and args.log_every_batches > 0
                and batch_index % args.log_every_batches == 0
            ):
                running_loss = epoch_loss / max(epoch_total, 1)
                running_acc = 100.0 * epoch_correct / max(epoch_total, 1)
                print(
                    f"[{case.name}][{method}] task {task_index} epoch {epoch}/{args.epochs} "
                    f"batch {batch_index}/{len(loader)} loss={running_loss:.4f} "
                    f"train_acc={running_acc:.2f}%",
                    flush=True,
                )
        if getattr(args, "verbose", False):
            mean_loss = epoch_loss / max(epoch_total, 1)
            train_acc = 100.0 * epoch_correct / max(epoch_total, 1)
            elapsed = time.perf_counter() - epoch_start
            print(
                f"[{case.name}][{method}] task {task_index} epoch {epoch}/{args.epochs} done "
                f"loss={mean_loss:.4f} train_acc={train_acc:.2f}% time={elapsed:.1f}s",
                flush=True,
            )


def review_task(
    model: SmallCILNet,
    centroids: torch.Tensor,
    loader: DataLoader,
    seen_labels: list[int],
    case: LongTailCase,
    device: torch.device,
    args: argparse.Namespace,
    task_index: int,
    method: str,
) -> None:
    params = list(model.parameters()) + [centroids]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr * args.review_lr_scale,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    seen = torch.tensor(seen_labels, dtype=torch.long, device=device)
    seen_counts = class_counts(case, seen_labels, device)
    global_label_weights = class_weights(
        case,
        seen_labels,
        args.class_weight_beta,
        args.max_class_weight,
        device,
    )
    local_label_weights = global_label_weights.index_select(0, seen)

    for epoch in range(1, args.review_epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, features = model(images, return_features=True)
            seen_logits = logits.index_select(1, seen)
            local_labels = remap_labels(labels, seen_labels).to(device)
            loss = balanced_softmax_loss(
                seen_logits,
                local_labels,
                seen_counts,
                local_label_weights,
            )
            l_min, l_max = centroid_loss(
                features,
                labels,
                centroids,
                seen_labels,
                global_label_weights,
            )
            loss = loss + args.centroid_min_weight * l_min + args.centroid_max_weight * l_max
            if task_index > 1 and args.prototype_replay_weight > 0:
                loss = loss + args.prototype_replay_weight * prototype_replay_loss(
                    model,
                    centroids,
                    seen_labels,
                    device,
                )
            loss.backward()
            apply_centroid_mask(centroids, seen_labels)
            optimizer.step()

            batch_total = int(labels.numel())
            epoch_loss += float(loss.detach().item()) * batch_total
            predictions = seen[seen_logits.detach().argmax(dim=1)]
            epoch_correct += int((predictions == labels).sum().item())
            epoch_total += batch_total
            if (
                getattr(args, "verbose", False)
                and args.log_every_batches > 0
                and batch_index % args.log_every_batches == 0
            ):
                running_loss = epoch_loss / max(epoch_total, 1)
                running_acc = 100.0 * epoch_correct / max(epoch_total, 1)
                print(
                    f"[{case.name}][{method}] task {task_index} review epoch {epoch}/{args.review_epochs} "
                    f"batch {batch_index}/{len(loader)} loss={running_loss:.4f} "
                    f"review_acc={running_acc:.2f}%",
                    flush=True,
                )
        if getattr(args, "verbose", False):
            mean_loss = epoch_loss / max(epoch_total, 1)
            review_acc = 100.0 * epoch_correct / max(epoch_total, 1)
            elapsed = time.perf_counter() - epoch_start
            print(
                f"[{case.name}][{method}] task {task_index} review epoch {epoch}/{args.review_epochs} done "
                f"loss={mean_loss:.4f} review_acc={review_acc:.2f}% time={elapsed:.1f}s",
                flush=True,
            )


@torch.no_grad()
def evaluate(
    model: SmallCILNet,
    loader: DataLoader,
    seen_labels: list[int],
    device: torch.device,
    logit_adjustment: torch.Tensor | None = None,
) -> float:
    model.eval()
    seen = torch.tensor(seen_labels, dtype=torch.long, device=device)
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images).index_select(1, seen)
        if logit_adjustment is not None:
            logits = logits + logit_adjustment.unsqueeze(0)
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
    fields.extend(
        [
            "final_accuracy",
            "average_accuracy",
            "total_elapsed_seconds",
            "training_seconds",
            "review_seconds",
            "evaluation_seconds",
            "review_events",
            "replay_exemplars",
            "replay_memory_bytes",
            "replay_memory_mb",
            "centroid_memory_bytes",
            "extra_memory_bytes",
            "extra_memory_mb",
            "peak_cuda_memory_bytes",
            "peak_cuda_memory_mb",
        ]
    )
    return fields


def write_result_rows(rows: list[dict[str, str]], path: Path, test_tasks: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = result_fieldnames(test_tasks)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
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
    methods = method_order(rows)
    summary: dict[str, object] = {
        "methods": {},
        "differences_vs_traditional": {},
        "highlight_cases": [f"case{case_index:02d}" for case_index in highlight_cases],
    }
    metrics = summary_metrics(test_tasks)
    by_method = {method: [row for row in rows if row["method"] == method] for method in methods}
    for method, method_rows in by_method.items():
        summary["methods"][method] = {
            metric: describe(numeric_values(method_rows, metric))
            for metric in metrics
        }

    baseline_by_case = {row["case"]: row for row in by_method.get(METHOD_BASELINE, [])}
    for method in methods:
        if method == METHOD_BASELINE:
            continue
        summary["differences_vs_traditional"][method] = {}
        for metric in metrics:
            diffs = []
            for row in by_method[method]:
                baseline_row = baseline_by_case.get(row["case"])
                if baseline_row is None:
                    continue
                value = parse_float(row.get(metric))
                baseline_value = parse_float(baseline_row.get(metric))
                if value is not None and baseline_value is not None:
                    diffs.append(value - baseline_value)
            summary["differences_vs_traditional"][method][metric] = describe(diffs)

    if METHOD_TAE in summary["differences_vs_traditional"]:
        summary["differences_tae_minus_traditional"] = summary["differences_vs_traditional"][METHOD_TAE]
    return summary


def summary_metrics(test_tasks: list[int]) -> list[str]:
    return [
        *[f"task{task}_accuracy" for task in test_tasks],
        "final_accuracy",
        "average_accuracy",
        "total_elapsed_seconds",
        "training_seconds",
        "review_seconds",
        "evaluation_seconds",
        "review_events",
        "replay_exemplars",
        "replay_memory_mb",
        "extra_memory_mb",
        "peak_cuda_memory_mb",
    ]


def method_order(rows: list[dict[str, str]]) -> list[str]:
    present = {row["method"] for row in rows}
    ordered = [method for method in METHODS if method in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def numeric_values(rows: list[dict[str, str]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = parse_float(row.get(metric))
        if value is not None:
            values.append(value)
    return values


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
    if not curves:
        return []
    return [statistics.mean(points) for points in zip(*curves)]


def svg_line_chart(title: str, series: dict[str, list[float]]) -> str:
    width, height = 780, 360
    left, right, top, bottom = 58, 28, 24, 48
    values = [value for points in series.values() for value in points]
    if not values:
        return ""
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
        color = METHOD_COLORS.get(label, "#111827")
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
    methods = method_order(rows)
    mean_series = {
        method: mean_curve(rows, method)
        for method in methods
        if mean_curve(rows, method)
    }
    charts = [svg_line_chart("Mean accuracy across all long-tail cases", mean_series)]
    for case_name in sorted({row["case"] for row in highlighted}):
        case_rows = [row for row in highlighted if row["case"] == case_name]
        series = {row["method"]: parse_curve(row["curve"]) for row in case_rows}
        if series:
            charts.append(svg_line_chart(f"{case_name} detailed curve", series))

    metric_rows = []
    metrics = summary_metrics(test_tasks)
    for metric in metrics:
        for method in methods:
            mean = summary["methods"][method][metric]["mean"]
            diff = 0.0
            if method != METHOD_BASELINE:
                diff = summary["differences_vs_traditional"][method][metric]["mean"]
            metric_rows.append(
                "<tr>"
                f"<td>{html.escape(metric)}</td>"
                f"<td>{html.escape(method)}</td>"
                f"<td>{mean:.3f}</td>"
                f"<td>{diff:+.3f}</td>"
                "</tr>"
            )
    legend = "".join(
        f'<span><span class="swatch" style="background:{METHOD_COLORS.get(method, "#111827")}"></span>'
        f"{html.escape(method)}</span>"
        for method in methods
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
  <p>This report compares a traditional full-update baseline, TaE/CEd, and optionally TaE with review replay across shuffled long-tailed CIFAR-10 class-incremental studies. Memory columns report persistent extra method memory, with replay images estimated as raw 32x32 RGB CIFAR exemplars.</p>
  <div class="legend">
    {legend}
  </div>
  {''.join(charts)}
  <h2>Mean Summary</h2>
  <table>
    <thead><tr><th>Metric</th><th>Method</th><th>Mean</th><th>Difference vs traditional</th></tr></thead>
    <tbody>{''.join(metric_rows)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def write_method_notes(path: Path) -> None:
    body = """# What This Study Implements From Paper 0471

Paper 0471 addresses long-tailed class-incremental learning, where each new task can contain head classes with many samples and tail classes with very few samples. Traditional full-update finetuning often overfits the new head classes and overwrites older class representations.

This folder implements the following ideas from the paper:

1. Task-aware parameter selection: before each new task after task 1, the runner accumulates gradients on the incoming task and selects the top-p percent most sensitive parameters.
2. Frozen majority update: during that task, only the selected feature parameters are updated, while the rest of the feature extractor is protected from drift. The classifier remains trainable so tail correction and prototype replay can directly update class boundaries.
3. Long-tail-aware task batches: TaE uses class-balanced sampling inside each current task so tail classes influence mask selection and training instead of being drowned out by head classes.
4. Long-tail loss correction: TaE uses balanced-softmax cross-entropy plus bounded effective-number class weights, which makes the current tail classes more expensive to misclassify while avoiding unstable unbounded weights.
5. Centroid-enhanced representation: each class has a learnable centroid. The model pulls features toward their own class centroid and pushes different class centroids apart using a class-balanced centroid loss.
6. Centroid prototype replay: on later tasks, old class centroids are used as lightweight feature prototypes so the classifier keeps a direct old-class signal without replaying old images.
7. Evaluation-time logit adjustment: TaE applies a small prior correction at evaluation to reduce head-class bias under balanced test classes.
8. Review replay: the `tae_review_replay` method stores a small class-balanced exemplar memory. After each newly introduced class is studied, it adds that class to memory and reviews all remembered seen classes before evaluation.

Expected benefit over the traditional baseline:

- Less catastrophic forgetting because most older parameters are not changed on every new task.
- Better tail-class feature separation because the centroid loss gives sparse tail classes a stronger geometric target.
- Stronger old-class retention in the review variant because every newly learned task is followed by a balanced review of old and new classes.
- Lower adaptation cost than fully expanding a whole backbone for every task because only a selected parameter subset is trainable per task.

The result CSV records `total_elapsed_seconds`, `training_seconds`, `review_seconds`, `review_events`, `replay_exemplars`, `replay_memory_mb`, `extra_memory_mb`, and `peak_cuda_memory_mb`. Replay memory is estimated as raw CIFAR RGB images: exemplar_count x 32 x 32 x 3 bytes. Extra memory also includes the class centroid tensor used by TaE methods.

The code here is a compact, server-friendly study runner inspired by the paper and the LAMDA-PILOT baseline folder. It does not modify the downloaded LAMDA-PILOT submodule.
"""
    path.write_text(body, encoding="utf-8")
