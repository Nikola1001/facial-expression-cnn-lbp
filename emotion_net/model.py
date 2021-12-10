"""Compact four-convolution classifier with a separate LBP branch."""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18

from .features import BINS, GRID


class EmotionNet(nn.Module):
    def __init__(self, classes: int = 7):
        super().__init__()
        channels = (1, 32, 64, 128, 128)
        layers = []
        for index in range(4):
            layers.extend((nn.Conv2d(channels[index], channels[index + 1], 3, padding=1, bias=False),
                           nn.BatchNorm2d(channels[index + 1]), nn.ReLU(inplace=True)))
            if index in (1, 3):
                layers.append(nn.MaxPool2d(2))
        layers.extend((nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten()))
        self.cnn = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3 + GRID * GRID * BINS, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, classes),
        )

    def forward(self, image: torch.Tensor, lbp: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat((self.cnn(image), lbp), dim=1))


class ResNet18LBP(nn.Module):
    """ImageNet-initialized CNN with a trainable projection of fixed LBP features."""

    def __init__(self, pretrained_path=None, use_lbp: bool = True):
        super().__init__()
        self.backbone = resnet18(weights=None)
        if pretrained_path is not None:
            from hashlib import sha256
            from pathlib import Path

            path = Path(pretrained_path)
            if not path.is_file():
                raise FileNotFoundError(f"Pretrained weights missing: {path}")
            digest = sha256(path.read_bytes()).hexdigest()
            if not digest.startswith("f37072fd"):
                raise ValueError("Unexpected ResNet18 weights checksum")
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state)
        self.backbone.fc = nn.Identity()
        self.use_lbp = use_lbp
        self.lbp_projection = nn.Sequential(
            nn.LayerNorm(GRID * GRID * BINS),
            nn.Linear(GRID * GRID * BINS, 128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 7),
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor, lbp: torch.Tensor) -> torch.Tensor:
        # Dataset supplies grayscale pixels in [-1, 1].
        x = F.interpolate((image + 1) * 0.5, size=(112, 112), mode="bilinear", align_corners=False)
        x = x.expand(-1, 3, -1, -1)
        x = (x - self.mean) / self.std
        image_features = self.backbone(x)
        if not self.use_lbp:
            lbp = torch.zeros_like(lbp)
        texture_features = self.lbp_projection(lbp)
        return self.classifier(torch.cat((image_features, texture_features), dim=1))


class LBPExpert(nn.Module):
    """Small classifier trained only on spatial LBP histograms."""

    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(GRID * GRID * BINS),
            nn.Linear(GRID * GRID * BINS, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 7),
        )

    def forward(self, image: torch.Tensor, lbp: torch.Tensor) -> torch.Tensor:
        return self.classifier(lbp)
