from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from daac.controller import (
    ControllerWeights,
    DifficultyEstimator,
    DifficultySignals,
    SelectorConfig,
    StrategyDecision,
    StrategySelector,
    TopPGradientMasker,
    fixed_strategy_decision,
)
from daac.data import IncrementalDataModule
from daac.model import DAACModel
from daac.utils import (
    CsvLogger,
    count_parameters,
    ensure_dir,
    normalized_entropy,
    now_seconds,
    peak_cuda_memory_mb,
    pick_device,
    set_seed,
    write_json,
)


METRIC_FIELDS = [
    "task_id",
    "selected_strategy",
    "difficulty_score",
    "novelty",
    "entropy",
    "gradient_sensitivity",
    "layer_importance_ratio",
    "expert_ambiguity",
    "top_p",
    "trainable_params",
    "total_params",
    "number_of_adapters",
    "number_of_prompts",
    "training_time_sec",
    "pre_study_time_sec",
    "peak_cuda_memory_mb",
    "task_accuracy",
    "average_incremental_accuracy",
    "final_accuracy",
    "forgetting_score",
]


class DAACExperiment:
    def __init__(self, args, strategy: str, seed: int) -> None:
        self.args = args
        self.strategy = strategy
        self.seed = seed
        self.device = pick_device(args.device)
        self.output_dir = (
            Path(args.output_dir)
            / "daac"
            / args.dataset
            / strategy
            / str(seed)
        )
        ensure_dir(self.output_dir)
        metrics_path = self.output_dir / "metrics.csv"
        if metrics_path.exists():
            metrics_path.unlink()
        self.csv_logger = CsvLogger(self.output_dir / "metrics.csv", METRIC_FIELDS)
        weights = ControllerWeights(
            novelty=args.w_novelty,
            entropy=args.w_entropy,
            gradient=args.w_grad,
            layer=args.w_layer,
            ambiguity=args.w_ambiguity,
        )
        selector_config = SelectorConfig(
            low_threshold=args.low_threshold,
            high_threshold=args.high_threshold,
            ambiguity_threshold=args.ambiguity_threshold,
            novelty_threshold=args.novelty_threshold,
        )
        self.estimator = DifficultyEstimator(
            weights=weights,
            max_batches=args.prestudy_batches,
            grad_scale=args.grad_sensitivity_scale,
        )
        self.selector = StrategySelector(selector_config)
        self.class_prototypes: dict[int, torch.Tensor] = {}
        self.expert_task_classes: dict[int, set[int]] = {0: set()}
        self.rows: list[dict] = []
        self.per_task_history: dict[int, list[float]] = {}

    def run(self) -> dict:
        set_seed(self.seed)
        data = IncrementalDataModule(
            dataset=self.args.dataset,
            data_dir=Path(self.args.data_dir),
            seed=self.seed,
            init_classes=self.args.init_classes,
            increment=self.args.increment,
            batch_size=self.args.batch_size,
            fast_dev_run=self.args.fast_dev_run,
            download=self.args.download,
            num_workers=self.args.num_workers,
            max_tasks=self.args.max_tasks,
        )
        model = DAACModel(
            max_classes=data.total_classes,
            image_size=self.args.image_size,
            patch_size=self.args.patch_size,
            embed_dim=self.args.embed_dim,
            depth=self.args.depth,
            num_heads=self.args.num_heads,
            adapter_bottleneck=self.args.adapter_bottleneck,
        ).to(self.device)
        write_json(
            self.output_dir / "run_config.json",
            {
                "args": vars(self.args),
                "strategy": self.strategy,
                "seed": self.seed,
                "resolved_dataset": data.dataset,
                "task_specs": [asdict(spec) for spec in data.task_specs],
            },
        )

        for task_index, task_spec in enumerate(data.task_specs):
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            total_classes = task_spec.end_class
            train_loader = data.train_loader_for_task(task_index, shuffle=True)
            pre_start = now_seconds()
            signals = self.estimator.estimate(
                model=model,
                loader=train_loader,
                task_index=task_index,
                total_classes=total_classes,
                prototypes=self.class_prototypes,
                expert_task_classes=self.expert_task_classes,
                device=self.device,
            )
            pre_seconds = now_seconds() - pre_start
            decision = self._decision(task_index, signals)
            self._maybe_expand_modules(model, task_index, decision)
            self._configure_prompts(model, decision, signals)
            masker = self._configure_trainable_and_mask(model, train_loader, total_classes, decision)
            trainable_params = masker.selected if masker is not None else count_parameters(model, trainable_only=True)
            train_start = now_seconds()
            self._train_task(model, train_loader, total_classes, task_spec.start_class, decision, masker)
            train_seconds = now_seconds() - train_start
            self._update_prototypes(model, data, task_index)
            task_accuracy, per_task_acc = self._evaluate(model, data, task_index, decision)
            self._record_task_history(per_task_acc)
            avg_acc = float(np.mean([row["task_accuracy"] for row in self.rows] + [task_accuracy]))
            forgetting = self._forgetting()
            row = self._row(
                task_id=task_spec.task_id,
                decision=decision,
                signals=signals,
                trainable_params=trainable_params,
                total_params=count_parameters(model),
                adapter_count=model.adapter_count(),
                prompt_count=model.prompt_pool.number_of_prompts(),
                train_seconds=train_seconds,
                pre_seconds=pre_seconds,
                memory_mb=peak_cuda_memory_mb(self.device),
                task_accuracy=task_accuracy,
                avg_acc=avg_acc,
                final_accuracy=task_accuracy if task_index == data.nb_tasks - 1 else "",
                forgetting=forgetting,
            )
            self.csv_logger.append(row)
            self.rows.append(row)
            model.prompt_pool.commit_current()

        summary = self._summary()
        write_json(self.output_dir / "summary.json", summary)
        return summary

    def _decision(self, task_index: int, signals: DifficultySignals) -> StrategyDecision:
        if self.strategy == "adaptive":
            return self.selector.select(task_index, signals)
        return fixed_strategy_decision(self.strategy, task_index, signals)

    def _maybe_expand_modules(self, model: DAACModel, task_index: int, decision: StrategyDecision) -> None:
        if task_index == 0:
            model.current_adapter_id = 0
            self.expert_task_classes.setdefault(0, set())
            return
        if decision.strategy in {"new_adapter", "all_combined"}:
            adapter_id = model.add_adapter(clone_last=False)
            self.expert_task_classes.setdefault(adapter_id, set())

    def _configure_prompts(self, model: DAACModel, decision: StrategyDecision, signals: DifficultySignals) -> None:
        if decision.strategy in {"base_train", "prompt_or_light_update", "tae_top_p", "all_combined", "weighted_expert_fusion"}:
            layers = signals.important_layers or list(range(model.depth))
            model.prompt_pool.start_task(layers)
        else:
            model.prompt_pool.start_task([])

    def _configure_trainable_and_mask(
        self,
        model: DAACModel,
        train_loader,
        total_classes: int,
        decision: StrategyDecision,
    ) -> TopPGradientMasker | None:
        model.freeze_all()
        masker = None
        if decision.strategy == "finetune":
            model.enable_all()
            return None

        model.enable_classifier()
        if decision.strategy in {"base_train", "new_adapter"}:
            model.enable_current_adapter()
        elif decision.strategy == "prompt_or_light_update":
            model.enable_prompts()
        elif decision.strategy == "weighted_expert_fusion":
            model.enable_prompts()
        elif decision.strategy == "tae_top_p":
            model.enable_current_adapter()
            model.enable_prompts()
            masker = TopPGradientMasker.build(
                model,
                train_loader,
                total_classes=total_classes,
                top_p=decision.top_p,
                device=self.device,
                max_batches=self.args.prestudy_batches,
            )
        elif decision.strategy == "all_combined":
            model.enable_current_adapter()
            model.enable_prompts()
            masker = TopPGradientMasker.build(
                model,
                train_loader,
                total_classes=total_classes,
                top_p=decision.top_p,
                device=self.device,
                max_batches=self.args.prestudy_batches,
            )
        else:
            raise ValueError(f"Unsupported selected strategy {decision.strategy}")
        return masker

    def _optimizer(self, model: DAACModel):
        params = [param for param in model.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters selected for this task.")
        if self.args.optimizer == "sgd":
            return optim.SGD(params, lr=self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay)
        if self.args.optimizer == "adam":
            return optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        return optim.AdamW(params, lr=self.args.lr, weight_decay=self.args.weight_decay)

    def _train_task(
        self,
        model: DAACModel,
        train_loader,
        total_classes: int,
        start_class: int,
        decision: StrategyDecision,
        masker: TopPGradientMasker | None,
    ) -> None:
        model.train()
        optimizer = self._optimizer(model)
        epochs = self.args.epochs
        if self.args.fast_dev_run:
            epochs = min(epochs, self.args.fast_dev_epochs)
        for _ in range(epochs):
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                output = model(inputs, total_classes=total_classes)
                logits = output["logits"]
                if self.args.mask_old_classes_during_train and start_class > 0:
                    logits = logits.clone()
                    logits[:, :start_class] = -1e4
                loss = F.cross_entropy(logits, targets)
                if decision.strategy in {"weighted_expert_fusion", "all_combined"}:
                    loss = loss + self._distillation_loss(model, inputs, output["logits"], start_class)
                optimizer.zero_grad()
                loss.backward()
                if masker is not None:
                    masker.apply(model)
                optimizer.step()

    def _distillation_loss(
        self,
        model: DAACModel,
        inputs: torch.Tensor,
        current_logits: torch.Tensor,
        known_classes: int,
    ) -> torch.Tensor:
        if known_classes <= 1 or model.adapter_count() <= 1 or self.args.distill_weight <= 0:
            return torch.zeros((), device=inputs.device)
        old_logits = []
        with torch.no_grad():
            for adapter_id in range(model.adapter_count()):
                if adapter_id == model.current_adapter_id:
                    continue
                old = model(inputs, adapter_id=adapter_id, total_classes=known_classes)["logits"]
                old_logits.append(old)
        if not old_logits:
            return torch.zeros((), device=inputs.device)
        teacher = torch.stack(old_logits, dim=0).mean(dim=0)
        temp = self.args.distill_temperature
        student = current_logits[:, :known_classes]
        kd = F.kl_div(
            F.log_softmax(student / temp, dim=1),
            F.softmax(teacher / temp, dim=1),
            reduction="batchmean",
        ) * (temp * temp)
        return self.args.distill_weight * kd

    @torch.no_grad()
    def _update_prototypes(self, model: DAACModel, data: IncrementalDataModule, task_index: int) -> None:
        model.eval()
        loader = data.train_loader_for_task(task_index, shuffle=False)
        features_by_class: dict[int, list[torch.Tensor]] = {}
        for inputs, targets in loader:
            output = model(inputs.to(self.device), total_classes=data.seen_classes_after(task_index))
            features = output["features"].detach().cpu()
            for feature, label in zip(features, targets):
                features_by_class.setdefault(int(label.item()), []).append(feature)
        for class_id, values in features_by_class.items():
            prototype = torch.stack(values, dim=0).mean(dim=0)
            prototype = F.normalize(prototype, dim=0)
            self.class_prototypes[class_id] = prototype
            self.expert_task_classes.setdefault(model.current_adapter_id, set()).add(class_id)

    @torch.no_grad()
    def _evaluate(
        self,
        model: DAACModel,
        data: IncrementalDataModule,
        task_index: int,
        decision: StrategyDecision,
    ) -> tuple[float, dict[int, float]]:
        model.eval()
        total_classes = data.seen_classes_after(task_index)
        loader = data.test_loader_seen(task_index)
        correct = 0
        total = 0
        task_correct: dict[int, int] = {}
        task_total: dict[int, int] = {}
        use_fusion = decision.use_fusion or self.strategy == "mote_fusion"
        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            if use_fusion and model.adapter_count() > 1 and self.class_prototypes:
                logits, _ = model.fusion_logits(inputs, self.class_prototypes, self.expert_task_classes, total_classes)
            else:
                logits = model(inputs, total_classes=total_classes)["logits"]
            preds = logits.argmax(dim=1).cpu()
            targets_cpu = targets.cpu()
            correct += int((preds == targets_cpu).sum().item())
            total += int(targets_cpu.numel())
            for spec_index, spec in enumerate(data.task_specs[: task_index + 1]):
                mask = (targets_cpu >= spec.start_class) & (targets_cpu < spec.end_class)
                if mask.any():
                    task_correct[spec_index] = task_correct.get(spec_index, 0) + int((preds[mask] == targets_cpu[mask]).sum().item())
                    task_total[spec_index] = task_total.get(spec_index, 0) + int(mask.sum().item())
        acc = 100.0 * correct / max(total, 1)
        per_task = {
            spec_index: 100.0 * task_correct.get(spec_index, 0) / max(task_total.get(spec_index, 0), 1)
            for spec_index in range(task_index + 1)
        }
        return round(acc, 4), per_task

    def _record_task_history(self, per_task_acc: dict[int, float]) -> None:
        for task_index, acc in per_task_acc.items():
            self.per_task_history.setdefault(task_index, []).append(acc)

    def _forgetting(self) -> float:
        if len(self.per_task_history) <= 1:
            return 0.0
        values = []
        max_task = max(self.per_task_history)
        for task_index, history in self.per_task_history.items():
            if task_index == max_task or not history:
                continue
            values.append(max(history) - history[-1])
        return round(float(np.mean(values)), 4) if values else 0.0

    def _row(
        self,
        task_id: int,
        decision: StrategyDecision,
        signals: DifficultySignals,
        trainable_params: int,
        total_params: int,
        adapter_count: int,
        prompt_count: int,
        train_seconds: float,
        pre_seconds: float,
        memory_mb: float,
        task_accuracy: float,
        avg_acc: float,
        final_accuracy,
        forgetting: float,
    ) -> dict:
        return {
            "task_id": task_id,
            "selected_strategy": decision.strategy,
            "difficulty_score": round(signals.difficulty_score, 6),
            "novelty": round(signals.novelty, 6),
            "entropy": round(signals.entropy, 6),
            "gradient_sensitivity": round(signals.gradient_sensitivity, 6),
            "layer_importance_ratio": round(signals.layer_importance_ratio, 6),
            "expert_ambiguity": round(signals.expert_ambiguity, 6),
            "top_p": decision.top_p,
            "trainable_params": int(trainable_params),
            "total_params": int(total_params),
            "number_of_adapters": int(adapter_count),
            "number_of_prompts": int(prompt_count),
            "training_time_sec": round(train_seconds, 4),
            "pre_study_time_sec": round(pre_seconds, 4),
            "peak_cuda_memory_mb": round(memory_mb, 4),
            "task_accuracy": task_accuracy,
            "average_incremental_accuracy": round(avg_acc, 4),
            "final_accuracy": final_accuracy,
            "forgetting_score": forgetting,
        }

    def _summary(self) -> dict:
        final_acc = self.rows[-1]["task_accuracy"] if self.rows else 0.0
        avg_acc = float(np.mean([row["task_accuracy"] for row in self.rows])) if self.rows else 0.0
        summary = {
            "strategy": self.strategy,
            "seed": self.seed,
            "dataset": self.args.dataset,
            "final_accuracy": round(final_acc, 4),
            "average_incremental_accuracy": round(avg_acc, 4),
            "forgetting_score": self._forgetting(),
            "tasks": len(self.rows),
            "metrics_csv": str(self.output_dir / "metrics.csv"),
            "rows": self.rows,
        }
        return summary


def run_experiments(args) -> list[dict]:
    summaries = []
    strategies = args.strategies or [args.strategy]
    seeds = args.seeds
    for strategy in strategies:
        for seed in seeds:
            experiment = DAACExperiment(args, strategy=strategy, seed=seed)
            summary = experiment.run()
            summaries.append(summary)
            print(
                f"finished strategy={strategy} seed={seed} "
                f"final={summary['final_accuracy']:.2f} avg={summary['average_incremental_accuracy']:.2f} "
                f"logs={summary['metrics_csv']}",
                flush=True,
            )
    return summaries
