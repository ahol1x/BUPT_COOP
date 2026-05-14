from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class LongTailCase:
    name: str
    seed: int
    class_order: list[int]
    train_counts: dict[int, int]
    test_count_per_class: int


@dataclass(frozen=True)
class CIFARCache:
    train: "IndexedCIFAR"
    test: "IndexedCIFAR"


class IndexedCIFAR:
    def __init__(self, dataset, num_classes: int) -> None:
        self.dataset = dataset
        self.num_classes = num_classes
        self.targets = np.asarray(dataset.targets)
        self.by_class = {
            label: np.where(self.targets == label)[0].tolist()
            for label in range(num_classes)
        }

    def subset(
        self,
        labels: list[int],
        per_class_counts: dict[int, int],
        rng: np.random.Generator,
    ) -> Subset:
        indices: list[int] = []
        for label in labels:
            class_indices = np.asarray(self.by_class[label])
            requested = per_class_counts.get(label, len(class_indices))
            count = min(requested, len(class_indices))
            chosen = rng.choice(class_indices, size=count, replace=False)
            indices.extend(chosen.tolist())
        rng.shuffle(indices)
        return Subset(self.dataset, indices)


def make_cifar10_cache(data_dir: Path, download: bool) -> CIFARCache:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    return CIFARCache(
        train=IndexedCIFAR(
            datasets.CIFAR10(
                root=str(data_dir),
                train=True,
                download=download,
                transform=train_transform,
            ),
            num_classes=10,
        ),
        test=IndexedCIFAR(
            datasets.CIFAR10(
                root=str(data_dir),
                train=False,
                download=download,
                transform=test_transform,
            ),
            num_classes=10,
        ),
    )


def make_loader(
    dataset: IndexedCIFAR,
    labels: list[int],
    per_class_counts: dict[int, int],
    seed: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    rng = np.random.default_rng(seed)
    subset = dataset.subset(labels, per_class_counts, rng)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )

