from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from daac.utils import cosine_logits


class Adapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int, scale: float = 0.1) -> None:
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.scale = scale
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.relu(self.down(x))) * self.scale


class AdapterStack(nn.Module):
    def __init__(self, num_layers: int, dim: int, bottleneck: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Adapter(dim, bottleneck) for _ in range(num_layers)])

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> Adapter:
        return self.layers[index]


class TinyTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor, adapter: Adapter | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + attn_out
        mlp_out = self.mlp(self.norm2(x))
        if adapter is not None:
            mlp_out = mlp_out + adapter(x)
        return x + mlp_out


class PromptPool(nn.Module):
    def __init__(self, num_layers: int, dim: int, prompt_scale: float = 0.02) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.prompt_scale = prompt_scale
        self.current_prompts = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim), requires_grad=False) for _ in range(num_layers)]
        )
        self.global_prompts = nn.ParameterList()
        self.global_layers: list[int] = []
        self.active_layers: set[int] = set()

    def start_task(self, layers: Iterable[int] | None) -> None:
        source = range(self.num_layers) if layers is None else layers
        active = sorted(set(int(layer) for layer in source))
        self.active_layers = set(layer for layer in active if 0 <= layer < self.num_layers)
        for layer in range(self.num_layers):
            prompt = torch.empty(self.dim)
            if layer in self.active_layers:
                nn.init.normal_(prompt, mean=0.0, std=self.prompt_scale)
                self.current_prompts[layer] = nn.Parameter(prompt, requires_grad=True)
            else:
                self.current_prompts[layer] = nn.Parameter(torch.zeros(self.dim), requires_grad=False)

    def commit_current(self) -> None:
        for layer in sorted(self.active_layers):
            prompt = self.current_prompts[layer].detach().clone()
            if prompt.abs().sum().item() == 0:
                continue
            frozen = nn.Parameter(prompt, requires_grad=False)
            self.global_prompts.append(frozen)
            self.global_layers.append(layer)
        for layer in range(self.num_layers):
            self.current_prompts[layer].requires_grad_(False)
        self.active_layers = set()

    def prompt_for_layer(self, layer: int, device: torch.device) -> torch.Tensor | None:
        prompts = []
        for stored_layer, prompt in zip(self.global_layers, self.global_prompts):
            if stored_layer == layer:
                prompts.append(prompt.to(device))
        if layer in self.active_layers:
            prompts.append(self.current_prompts[layer].to(device))
        if not prompts:
            return None
        return torch.stack(prompts, dim=0).mean(dim=0)

    def number_of_prompts(self) -> int:
        return len(self.global_prompts) + len(self.active_layers)


@dataclass
class ForwardOutput:
    features: torch.Tensor
    logits: torch.Tensor
    layer_outputs: list[torch.Tensor]


class DAACModel(nn.Module):
    def __init__(
        self,
        max_classes: int,
        image_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 64,
        depth: int = 3,
        num_heads: int = 4,
        adapter_bottleneck: int = 16,
    ) -> None:
        super().__init__()
        self.max_classes = max_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.adapter_bottleneck = adapter_bottleneck
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([TinyTransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.prompt_pool = PromptPool(depth, embed_dim)
        self.adapters = nn.ModuleList()
        self.classifier = nn.Linear(embed_dim, max_classes)
        self.current_adapter_id = 0
        self.add_adapter()
        self._init_backbone()
        self.freeze_backbone()

    def _init_backbone(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def freeze_backbone(self) -> None:
        for name, param in self.named_parameters():
            if name.startswith("patch_embed") or name.startswith("blocks") or name in {"cls_token", "pos_embed"} or name.startswith("norm"):
                param.requires_grad = False

    def add_adapter(self, clone_last: bool = False) -> int:
        if clone_last and len(self.adapters) > 0:
            new_adapter = copy.deepcopy(self.adapters[-1])
        else:
            new_adapter = AdapterStack(self.depth, self.embed_dim, self.adapter_bottleneck)
        self.adapters.append(new_adapter)
        self.current_adapter_id = len(self.adapters) - 1
        return self.current_adapter_id

    def adapter_count(self) -> int:
        return len(self.adapters)

    def freeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def enable_classifier(self) -> None:
        for param in self.classifier.parameters():
            param.requires_grad = True

    def enable_current_adapter(self) -> None:
        for param in self.adapters[self.current_adapter_id].parameters():
            param.requires_grad = True

    def enable_adapter(self, adapter_id: int) -> None:
        for param in self.adapters[adapter_id].parameters():
            param.requires_grad = True

    def enable_prompts(self) -> None:
        for prompt in self.prompt_pool.current_prompts:
            if prompt.requires_grad:
                prompt.requires_grad_(True)

    def enable_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def candidate_named_parameters(self, include_backbone: bool = False) -> list[tuple[str, nn.Parameter]]:
        names: list[tuple[str, nn.Parameter]] = []
        current_prefix = f"adapters.{self.current_adapter_id}."
        for name, param in self.named_parameters():
            if name.startswith("classifier") or name.startswith(current_prefix) or name.startswith("prompt_pool.current_prompts"):
                names.append((name, param))
            elif include_backbone and not name.startswith("prompt_pool.global_prompts"):
                names.append((name, param))
        return names

    def prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        return tokens + self.pos_embed[:, : tokens.shape[1], :]

    def forward_features(
        self,
        x: torch.Tensor,
        adapter_id: int | None = None,
        return_layers: bool = False,
        prompt_enabled: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        adapter_id = self.current_adapter_id if adapter_id is None else adapter_id
        tokens = self.prepare_tokens(x)
        layer_outputs: list[torch.Tensor] = []
        for layer_id, block in enumerate(self.blocks):
            prompt = self.prompt_pool.prompt_for_layer(layer_id, tokens.device) if prompt_enabled else None
            if prompt is not None:
                tokens = tokens.clone()
                tokens[:, 0, :] = tokens[:, 0, :] + prompt
            adapter = self.adapters[adapter_id][layer_id] if adapter_id is not None and adapter_id < len(self.adapters) else None
            tokens = block(tokens, adapter)
            if return_layers:
                layer_outputs.append(tokens[:, 0, :])
        features = self.norm(tokens)[:, 0, :]
        return features, layer_outputs

    def forward(
        self,
        x: torch.Tensor,
        adapter_id: int | None = None,
        total_classes: int | None = None,
        return_layers: bool = False,
        prompt_enabled: bool = True,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        features, layer_outputs = self.forward_features(
            x,
            adapter_id=adapter_id,
            return_layers=return_layers,
            prompt_enabled=prompt_enabled,
        )
        logits = self.classifier(features)
        if total_classes is not None:
            logits = logits[:, :total_classes]
        return {"features": features, "logits": logits, "layer_outputs": layer_outputs}

    @torch.no_grad()
    def expert_features(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for adapter_id in range(len(self.adapters)):
            features, _ = self.forward_features(x, adapter_id=adapter_id, prompt_enabled=True)
            outputs.append(features)
        return torch.stack(outputs, dim=0)

    @torch.no_grad()
    def fusion_logits(
        self,
        x: torch.Tensor,
        prototypes: dict[int, torch.Tensor],
        expert_task_classes: dict[int, set[int]],
        total_classes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not prototypes:
            logits = self.forward(x, total_classes=total_classes)["logits"]
            weights = torch.ones(1, x.shape[0], device=x.device)
            return logits, weights

        class_ids = sorted(class_id for class_id in prototypes if class_id < total_classes)
        proto_tensor = torch.stack([prototypes[class_id].to(x.device) for class_id in class_ids], dim=0)
        expert_features = self.expert_features(x)
        expert_scores = []
        expert_logits = []
        for expert_id in range(expert_features.shape[0]):
            logits = cosine_logits(expert_features[expert_id], proto_tensor)
            expert_logits.append(logits)
            top_values, top_indices = torch.topk(logits, k=min(2, logits.shape[1]), dim=1)
            pred_classes = torch.tensor([class_ids[int(idx)] for idx in top_indices[:, 0]], device=x.device)
            scope = expert_task_classes.get(expert_id, set())
            reliable = torch.tensor([int(class_id.item()) in scope for class_id in pred_classes], device=x.device)
            top1 = top_values[:, 0]
            if top_values.shape[1] > 1:
                scs = (top_values[:, 0] - top_values[:, 1]) / top_values[:, 0].abs().clamp_min(1e-6)
            else:
                scs = torch.ones_like(top1)
            score = top1 + top1.abs() * scs
            score = torch.where(reliable, score, torch.full_like(score, -1e4))
            expert_scores.append(score)
        score_tensor = torch.stack(expert_scores, dim=0)
        no_reliable = (score_tensor > -1e3).sum(dim=0) == 0
        if no_reliable.any():
            raw_conf = torch.stack([logits.max(dim=1).values for logits in expert_logits], dim=0)
            score_tensor[:, no_reliable] = raw_conf[:, no_reliable]
        weights = torch.softmax(score_tensor, dim=0)
        fused = (weights.unsqueeze(-1) * expert_features).sum(dim=0)
        logits = cosine_logits(fused, proto_tensor)
        dense_logits = torch.full((x.shape[0], total_classes), -1e4, device=x.device)
        for offset, class_id in enumerate(class_ids):
            dense_logits[:, class_id] = logits[:, offset]
        return dense_logits, weights
