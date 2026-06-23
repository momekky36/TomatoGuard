"""
model_scratch.py
-----------------
A small, fully-from-scratch CNN used as a CPU-friendly baseline when
pretrained ImageNet weights can't be downloaded (e.g. restricted-network
environments). It's far smaller than EfficientNet-B0, so every layer can
actually be trained in a reasonable time on CPU.

For the *recommended* approach (transfer learning, much higher accuracy
with less data/compute), see model.py + the README's Colab instructions.
"""
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, in_size: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # ->64
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # ->32
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # ->16
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_scratch_model(num_classes: int):
    return SimpleCNN(num_classes)
