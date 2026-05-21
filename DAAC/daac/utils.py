from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_seconds() -> float:
    return time.perf_counter()


def normalize_rows_for_json(value):
    if is_dataclass(value):
        return normalize_rows_for_json(asdict(value))
    if isinstance(value, dict):
        return {k: normalize_rows_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_rows_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [normalize_rows_for_json(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(normalize_rows_for_json(payload), indent=2), encoding="utf-8")


class CsvLogger:
    def __init__(self, path: Path, fieldnames: Iterable[str]) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        ensure_dir(path.parent)
        self._wrote_header = path.exists() and path.stat().st_size > 0

    def append(self, row: dict) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if not self._wrote_header:
                writer.writeheader()
                self._wrote_header = True
            clean = {name: row.get(name, "") for name in self.fieldnames}
            writer.writerow(clean)


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    if den == 0:
        return default
    return num / den


def normalized_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if probs.shape[-1] <= 1:
        return torch.zeros(probs.shape[:-1], device=probs.device)
    entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=-1)
    return entropy / math.log(probs.shape[-1])


def cosine_logits(features: torch.Tensor, prototypes: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
    features = torch.nn.functional.normalize(features, dim=-1)
    prototypes = torch.nn.functional.normalize(prototypes, dim=-1)
    return scale * features @ prototypes.t()


def peak_cuda_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
