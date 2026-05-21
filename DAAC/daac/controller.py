from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.nn import functional as F

from daac.model import DAACModel
from daac.utils import cosine_logits, normalized_entropy


@dataclass
class DifficultySignals:
    novelty: float = 1.0
    entropy: float = 0.0
    gradient_sensitivity: float = 0.0
    layer_importance_ratio: float = 0.0
    expert_ambiguity: float = 0.0
    difficulty_score: float = 0.0
    important_layers: list[int] | None = None


@dataclass
class ControllerWeights:
    novelty: float = 0.35
    entropy: float = 0.20
    gradient: float = 0.25
    layer: float = 0.10
    ambiguity: float = 0.10


@dataclass
class SelectorConfig:
    low_threshold: float = 0.35
    high_threshold: float = 0.65
    ambiguity_threshold: float = 0.45
    novelty_threshold: float = 0.60


@dataclass
class StrategyDecision:
    strategy: str
    top_p: float = 0.0
    use_fusion: bool = False


class DifficultyEstimator:
    def __init__(
        self,
        weights: ControllerWeights,
        max_batches: int = 2,
        grad_scale: float = 50.0,
    ) -> None:
        self.weights = weights
        self.max_batches = max_batches
        self.grad_scale = grad_scale

    def estimate(
        self,
        model: DAACModel,
        loader,
        task_index: int,
        total_classes: int,
        prototypes: dict[int, torch.Tensor],
        expert_task_classes: dict[int, set[int]],
        device: torch.device,
    ) -> DifficultySignals:
        was_training = model.training
        model.to(device)
        features, logits, layer_outputs, labels = self._collect_forward(
            model, loader, total_classes, device
        )
        novelty = self._novelty(features, prototypes, task_index, device)
        entropy = self._entropy(logits)
        layer_ratio, important_layers = self._layer_importance(layer_outputs)
        grad_sensitivity = self._gradient_sensitivity(model, loader, total_classes, device)
        expert_ambiguity = self._expert_ambiguity(model, loader, prototypes, expert_task_classes, total_classes, device)
        score = (
            self.weights.novelty * novelty
            + self.weights.entropy * entropy
            + self.weights.gradient * grad_sensitivity
            + self.weights.layer * layer_ratio
            + self.weights.ambiguity * expert_ambiguity
        )
        model.train(was_training)
        return DifficultySignals(
            novelty=float(novelty),
            entropy=float(entropy),
            gradient_sensitivity=float(grad_sensitivity),
            layer_importance_ratio=float(layer_ratio),
            expert_ambiguity=float(expert_ambiguity),
            difficulty_score=float(max(0.0, min(1.0, score))),
            important_layers=important_layers,
        )

    @torch.no_grad()
    def _collect_forward(self, model: DAACModel, loader, total_classes: int, device: torch.device):
        model.eval()
        all_features, all_logits, all_labels = [], [], []
        layer_accum: list[list[torch.Tensor]] = [[] for _ in range(model.depth)]
        for batch_id, (inputs, targets) in enumerate(loader):
            if batch_id >= self.max_batches:
                break
            output = model(inputs.to(device), total_classes=total_classes, return_layers=True)
            all_features.append(output["features"].detach().cpu())
            all_logits.append(output["logits"].detach().cpu())
            all_labels.append(targets.detach().cpu())
            for layer_id, layer_output in enumerate(output["layer_outputs"]):
                layer_accum[layer_id].append(layer_output.detach().cpu())
        features = torch.cat(all_features, dim=0) if all_features else torch.empty(0, model.embed_dim)
        logits = torch.cat(all_logits, dim=0) if all_logits else torch.empty(0, total_classes)
        labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
        layer_outputs = [
            torch.cat(layer_values, dim=0) if layer_values else torch.empty(0, model.embed_dim)
            for layer_values in layer_accum
        ]
        return features, logits, layer_outputs, labels

    def _novelty(
        self,
        features: torch.Tensor,
        prototypes: dict[int, torch.Tensor],
        task_index: int,
        device: torch.device,
    ) -> float:
        if task_index == 0 or not prototypes or features.numel() == 0:
            return 1.0
        current_mean = F.normalize(features.mean(dim=0, keepdim=True), dim=-1)
        proto_tensor = torch.stack([proto.detach().cpu() for proto in prototypes.values()], dim=0)
        proto_tensor = F.normalize(proto_tensor, dim=-1)
        max_sim = (current_mean @ proto_tensor.t()).max().item()
        novelty = 1.0 - max(-1.0, min(1.0, max_sim))
        return max(0.0, min(1.0, novelty))

    def _entropy(self, logits: torch.Tensor) -> float:
        if logits.numel() == 0 or logits.shape[1] <= 1:
            return 0.0
        probs = torch.softmax(logits, dim=1)
        return float(normalized_entropy(probs).mean().item())

    def _layer_importance(self, layer_outputs: list[torch.Tensor]) -> tuple[float, list[int]]:
        if not layer_outputs:
            return 0.0, []
        scores = []
        previous = None
        for output in layer_outputs:
            if output.numel() == 0:
                scores.append(0.0)
                continue
            if previous is None or previous.shape != output.shape:
                score = output.norm(dim=1).mean().item()
            else:
                score = (output - previous).norm(dim=1).mean().item()
            scores.append(float(score))
            previous = output
        mean_score = sum(scores) / max(len(scores), 1)
        important = [index for index, score in enumerate(scores) if score >= mean_score and score > 0]
        return len(important) / max(len(scores), 1), important

    def _gradient_sensitivity(self, model: DAACModel, loader, total_classes: int, device: torch.device) -> float:
        candidate_params = model.candidate_named_parameters()
        previous_flags = {name: param.requires_grad for name, param in candidate_params}
        for _, param in candidate_params:
            param.requires_grad_(True)
            param.grad = None
        model.train()
        total_sq = 0.0
        total_elems = 0
        batches = 0
        for batch_id, (inputs, targets) in enumerate(loader):
            if batch_id >= self.max_batches:
                break
            logits = model(inputs.to(device), total_classes=total_classes)["logits"]
            loss = F.cross_entropy(logits, targets.to(device))
            model.zero_grad(set_to_none=True)
            loss.backward()
            for _, param in candidate_params:
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                total_sq += float((grad * grad).sum().item())
                total_elems += grad.numel()
            batches += 1
        model.zero_grad(set_to_none=True)
        for name, param in candidate_params:
            param.requires_grad_(previous_flags[name])
        if total_elems == 0 or batches == 0:
            return 0.0
        rms = math.sqrt(total_sq / total_elems)
        return float(1.0 - math.exp(-self.grad_scale * rms))

    @torch.no_grad()
    def _expert_ambiguity(
        self,
        model: DAACModel,
        loader,
        prototypes: dict[int, torch.Tensor],
        expert_task_classes: dict[int, set[int]],
        total_classes: int,
        device: torch.device,
    ) -> float:
        if model.adapter_count() <= 1 or not prototypes:
            return 0.0
        entropies = []
        reliable_ratios = []
        for batch_id, (inputs, _) in enumerate(loader):
            if batch_id >= self.max_batches:
                break
            _, weights = model.fusion_logits(inputs.to(device), prototypes, expert_task_classes, total_classes)
            probs = weights.transpose(0, 1)
            entropies.append(normalized_entropy(probs).mean().item())
            reliable_ratios.append((weights > (1.0 / max(model.adapter_count(), 1))).float().sum(dim=0).mean().item() / model.adapter_count())
        if not entropies:
            return 0.0
        entropy_score = sum(entropies) / len(entropies)
        ratio_score = sum(reliable_ratios) / len(reliable_ratios)
        return float(max(0.0, min(1.0, 0.7 * entropy_score + 0.3 * ratio_score)))


class StrategySelector:
    def __init__(self, config: SelectorConfig) -> None:
        self.config = config

    def select(self, task_index: int, signals: DifficultySignals) -> StrategyDecision:
        if task_index == 0:
            return StrategyDecision(strategy="base_train")
        if (
            signals.difficulty_score >= self.config.high_threshold
            and signals.novelty >= self.config.novelty_threshold
        ):
            return StrategyDecision(strategy="new_adapter")
        if signals.expert_ambiguity >= self.config.ambiguity_threshold:
            return StrategyDecision(strategy="weighted_expert_fusion", use_fusion=True)
        if signals.difficulty_score >= self.config.low_threshold:
            return StrategyDecision(strategy="tae_top_p", top_p=self.top_p_for(signals.difficulty_score))
        return StrategyDecision(strategy="prompt_or_light_update")

    @staticmethod
    def top_p_for(difficulty_score: float) -> float:
        if difficulty_score < 0.45:
            return 0.05
        if difficulty_score < 0.55:
            return 0.10
        return 0.20


def fixed_strategy_decision(strategy: str, task_index: int, signals: DifficultySignals) -> StrategyDecision:
    if task_index == 0 and strategy not in {"finetune"}:
        return StrategyDecision(strategy="base_train")
    if strategy == "adaptive":
        raise ValueError("adaptive must use StrategySelector")
    if strategy == "prompt_only":
        return StrategyDecision(strategy="prompt_or_light_update")
    if strategy == "tae_only":
        return StrategyDecision(strategy="tae_top_p", top_p=StrategySelector.top_p_for(signals.difficulty_score))
    if strategy == "adapter_each_task":
        return StrategyDecision(strategy="new_adapter")
    if strategy == "mote_fusion":
        return StrategyDecision(strategy="new_adapter", use_fusion=True)
    if strategy == "all_combined":
        return StrategyDecision(strategy="all_combined", top_p=StrategySelector.top_p_for(signals.difficulty_score), use_fusion=True)
    if strategy == "finetune":
        return StrategyDecision(strategy="finetune")
    raise ValueError(f"Unknown strategy '{strategy}'")


class TopPGradientMasker:
    def __init__(self, masks: dict[str, torch.Tensor], selected: int, total: int) -> None:
        self.masks = masks
        self.selected = selected
        self.total = total

    @classmethod
    def build(
        cls,
        model: DAACModel,
        loader,
        total_classes: int,
        top_p: float,
        device: torch.device,
        max_batches: int = 2,
        include_backbone: bool = False,
    ) -> "TopPGradientMasker":
        named_params = model.candidate_named_parameters(include_backbone=include_backbone)
        previous_flags = {name: param.requires_grad for name, param in named_params}
        for _, param in named_params:
            param.requires_grad_(True)
            param.grad = None

        model.train()
        grads: dict[str, torch.Tensor] = {}
        for batch_id, (inputs, targets) in enumerate(loader):
            if batch_id >= max_batches:
                break
            logits = model(inputs.to(device), total_classes=total_classes)["logits"]
            loss = F.cross_entropy(logits, targets.to(device))
            model.zero_grad(set_to_none=True)
            loss.backward()
            for name, param in named_params:
                if param.grad is None:
                    continue
                current = param.grad.detach().abs().cpu()
                grads[name] = grads.get(name, torch.zeros_like(current)) + current

        flat = torch.cat([grad.flatten() for grad in grads.values()]) if grads else torch.tensor([])
        total = int(flat.numel())
        if total == 0:
            masks = {name: torch.ones_like(param.detach(), dtype=param.dtype) for name, param in named_params}
            selected = sum(mask.numel() for mask in masks.values())
        else:
            selected = max(1, int(total * max(0.0, min(1.0, top_p))))
            threshold = torch.topk(flat, k=selected, largest=True).values.min()
            masks = {
                name: (grads[name] >= threshold).to(dtype=param.dtype)
                for name, param in named_params
                if name in grads
            }

        model.zero_grad(set_to_none=True)
        for name, param in named_params:
            param.requires_grad_(previous_flags[name])
        return cls(masks=masks, selected=selected, total=total)

    def apply(self, model: DAACModel) -> None:
        named = dict(model.named_parameters())
        for name, mask in self.masks.items():
            param = named.get(name)
            if param is None or param.grad is None:
                continue
            param.grad.mul_(mask.to(param.grad.device))
