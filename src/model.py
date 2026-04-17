"""
src/model.py
Build MobileNetV3-Large with transfer learning.
Selective backbone unfreezing + custom classifier head.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Large_Weights


def build_model(num_classes: int, cfg: dict) -> nn.Module:
    """
    Build MobileNetV3-Large with ImageNet-pretrained weights.

    Strategy:
      - Load IMAGENET1K_V1 weights
      - Freeze backbone except last 3 feature blocks (features[-3:])
      - Replace classifier[-1] Linear(1280 → num_classes)

    Args:
        num_classes: number of output classes
        cfg:         loaded config dict

    Returns:
        nn.Module ready for training
    """
    weights = (MobileNet_V3_Large_Weights.IMAGENET1K_V1
               if cfg["training"]["pretrained"] else None)

    model = models.mobilenet_v3_large(weights=weights)

    if cfg["training"]["freeze_backbone"]:
        # Freeze all feature extraction layers
        for param in model.features.parameters():
            param.requires_grad = False
        # Unfreeze last 3 blocks for fine-tuning
        for block in model.features[-3:]:
            for param in block.parameters():
                param.requires_grad = True

    # Replace final linear layer
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


def count_parameters(model: nn.Module) -> dict:
    """Return total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    return {"total": total, "trainable": trainable}


def print_model_summary(model: nn.Module, num_classes: int) -> None:
    """Print architecture summary."""
    params = count_parameters(model)
    print(f"\n{'='*45}")
    print(f"Model : MobileNetV3-Large")
    print(f"Output classes  : {num_classes}")
    print(f"Total params    : {params['total']:,}")
    print(f"Trainable params: {params['trainable']:,}  "
          f"({100*params['trainable']/params['total']:.1f}%)")
    print(f"{'='*45}\n")


def load_checkpoint(model: nn.Module, checkpoint_path: str,
                    device: torch.device) -> dict:
    """
    Load a saved checkpoint into model (in-place).

    Returns:
        checkpoint dict (contains epoch, best_val_acc, etc.)
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}, "
          f"Best val acc: {ckpt.get('best_val_acc', '?'):.4f}")
    return ckpt
