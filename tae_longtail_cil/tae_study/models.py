from __future__ import annotations

import torch
from torch import nn


class SmallCILNet(nn.Module):
    """Compact image classifier used for both baseline and TaE studies.

    The architecture is intentionally small so a 10-case comparison can be run
    on a single server without modifying the downloaded LAMDA-PILOT submodule.
    """

    def __init__(self, num_classes: int = 10, feature_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
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
        features = self.projector(self.features(x))
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits

