"""
Hair Type Classifier Model — EfficientNet-B0 with transfer learning.
"""

import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes: int = 4, pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    """
    Build an EfficientNet-B0 model with a custom classification head.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 4: Straight, Wavy, Curly, Kinky).
    pretrained : bool
        Whether to load ImageNet-pretrained weights.
    freeze_backbone : bool
        If True, freeze all backbone (feature extractor) layers so only the
        classifier head is trained initially. Call ``unfreeze_backbone()``
        later for full fine-tuning.

    Returns
    -------
    nn.Module
        The configured model.
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Replace the classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )

    if freeze_backbone:
        _freeze_backbone(model)

    return model


def _freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> dict:
    """Return counts of total vs trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = create_model(num_classes=4, pretrained=True, freeze_backbone=True)
    info = count_parameters(model)
    print(f"Model ready — {info['trainable']:,} trainable / {info['total']:,} total params")

    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}  (expected [1, 4])")
