from __future__ import annotations

import pickle
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class TensorCILDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.images = images.float()
        self.labels = labels.long()

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


@dataclass
class TaskSpec:
    task_id: int
    start_class: int
    end_class: int

    @property
    def classes(self) -> list[int]:
        return list(range(self.start_class, self.end_class))


class IncrementalDataModule:
    def __init__(
        self,
        dataset: str,
        data_dir: Path,
        seed: int,
        init_classes: int,
        increment: int,
        batch_size: int,
        fast_dev_run: bool = False,
        download: bool = False,
        num_workers: int = 0,
        max_tasks: int | None = None,
    ) -> None:
        self.dataset = dataset.lower()
        self.data_dir = data_dir
        self.seed = seed
        self.init_classes = init_classes
        self.increment = increment
        self.batch_size = batch_size
        self.fast_dev_run = fast_dev_run
        self.download = download
        self.num_workers = num_workers

        if self.fast_dev_run or self.dataset in {"synthetic", "debug"}:
            train_images, train_labels, test_images, test_labels = make_synthetic_cil(seed=seed)
            self.dataset = "synthetic"
        elif self.dataset == "cifar100":
            train_images, train_labels, test_images, test_labels = load_cifar100(data_dir, download)
        else:
            raise ValueError(f"Unknown dataset '{dataset}'. Use synthetic or cifar100.")

        if self.fast_dev_run:
            self.total_classes = min(6, int(train_labels.max().item()) + 1)
            keep_train = train_labels < self.total_classes
            keep_test = test_labels < self.total_classes
            train_images, train_labels = train_images[keep_train], train_labels[keep_train]
            test_images, test_labels = test_images[keep_test], test_labels[keep_test]
            self.increment = min(2, self.increment)
            self.init_classes = 0
        else:
            self.total_classes = int(train_labels.max().item()) + 1

        self.train_dataset = TensorCILDataset(train_images, train_labels)
        self.test_dataset = TensorCILDataset(test_images, test_labels)
        self.task_specs = self._build_task_specs(max_tasks=max_tasks)

    def _build_task_specs(self, max_tasks: int | None = None) -> list[TaskSpec]:
        increments: list[int] = []
        if self.init_classes > 0:
            increments.append(self.init_classes)
        remaining = self.total_classes - sum(increments)
        while remaining > 0:
            step = min(self.increment, remaining)
            increments.append(step)
            remaining -= step
        if max_tasks is not None:
            increments = increments[:max_tasks]
        specs = []
        cursor = 0
        for index, size in enumerate(increments):
            specs.append(TaskSpec(task_id=index + 1, start_class=cursor, end_class=cursor + size))
            cursor += size
        self.total_classes = cursor
        return specs

    @property
    def nb_tasks(self) -> int:
        return len(self.task_specs)

    def seen_classes_after(self, task_index: int) -> int:
        return self.task_specs[task_index].end_class

    def task_size(self, task_index: int) -> int:
        spec = self.task_specs[task_index]
        return spec.end_class - spec.start_class

    def loader_for_classes(
        self,
        classes: Sequence[int],
        train: bool,
        shuffle: bool,
        batch_size: int | None = None,
    ) -> DataLoader:
        dataset = self.train_dataset if train else self.test_dataset
        labels = dataset.labels
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for class_id in classes:
            mask |= labels == int(class_id)
        indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        generator = torch.Generator()
        generator.manual_seed(self.seed + len(indices) + (13 if train else 29))
        return DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            generator=generator,
        )

    def train_loader_for_task(self, task_index: int, shuffle: bool = True) -> DataLoader:
        return self.loader_for_classes(self.task_specs[task_index].classes, train=True, shuffle=shuffle)

    def test_loader_seen(self, task_index: int) -> DataLoader:
        seen = range(0, self.seen_classes_after(task_index))
        return self.loader_for_classes(list(seen), train=False, shuffle=False)

    def test_loader_for_task(self, task_index: int) -> DataLoader:
        return self.loader_for_classes(self.task_specs[task_index].classes, train=False, shuffle=False)


def make_synthetic_cil(
    seed: int,
    total_classes: int = 10,
    train_per_class: int = 36,
    test_per_class: int = 18,
    image_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    class_patterns = torch.randn(total_classes, 3, image_size, image_size, generator=generator)
    grid = torch.linspace(-1.0, 1.0, image_size)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    base = torch.stack([xx, yy, xx * yy], dim=0)
    class_patterns = 0.55 * class_patterns + 0.45 * base.unsqueeze(0)
    class_patterns = torch.tanh(class_patterns)

    def sample(split_count: int, noise: float) -> tuple[torch.Tensor, torch.Tensor]:
        images, labels = [], []
        for class_id in range(total_classes):
            proto = class_patterns[class_id]
            for sample_id in range(split_count):
                jitter = torch.randn(proto.shape, generator=generator) * noise
                shift = ((sample_id % 5) - 2) * 0.015
                images.append((proto + jitter + shift).clamp(-2.0, 2.0))
                labels.append(class_id)
        return torch.stack(images), torch.tensor(labels, dtype=torch.long)

    train_images, train_labels = sample(train_per_class, noise=0.22)
    test_images, test_labels = sample(test_per_class, noise=0.25)
    return train_images, train_labels, test_images, test_labels


def load_cifar100(data_dir: Path, download: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    root = data_dir / "cifar-100-python"
    if not root.exists() and download:
        download_cifar100_archive(data_dir)
    if root.exists():
        return load_cifar100_pickles(root)

    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError(
            "CIFAR100 requires either local data/cifar-100-python files or torchvision. "
            "Run fast debug with --fast_dev_run, or install torchvision / provide local CIFAR100 data."
        ) from exc

    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.CIFAR100(str(data_dir), train=True, download=download, transform=transform)
    test = datasets.CIFAR100(str(data_dir), train=False, download=download, transform=transform)
    train_images = torch.stack([train[i][0] for i in range(len(train))])
    train_labels = torch.tensor(train.targets, dtype=torch.long)
    test_images = torch.stack([test[i][0] for i in range(len(test))])
    test_labels = torch.tensor(test.targets, dtype=torch.long)
    return normalize_images(train_images), train_labels, normalize_images(test_images), test_labels


def download_cifar100_archive(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "cifar-100-python.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", archive)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(data_dir)


def load_cifar100_pickles(root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def read(name: str):
        with (root / name).open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        data = payload["data"].reshape(-1, 3, 32, 32)
        labels = np.asarray(payload["fine_labels"], dtype=np.int64)
        return torch.from_numpy(data).float() / 255.0, torch.from_numpy(labels).long()

    train_images, train_labels = read("train")
    test_images, test_labels = read("test")
    return normalize_images(train_images), train_labels, normalize_images(test_images), test_labels


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.5071, 0.4867, 0.4408], dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761], dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std
