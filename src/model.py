"""
model.py
--------
Transfer-learning classifier: a pretrained EfficientNet-B0 backbone (trained
on ImageNet) with its classification head replaced for our N disease
classes. Only the head + last block are fine-tuned by default, which trains
fast and works well on a few thousand images.
"""
import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True):
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        # Unfreeze the last conv block so the model can still adapt
        # low-level features slightly to leaf textures.
        for param in model.features[-1].parameters():
            param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
