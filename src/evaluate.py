"""
src/evaluate.py
Final test evaluation:
  - Confusion matrix
  - Classification report (precision, recall, F1)
  - Per-class accuracy
  - ONNX export
"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")   # headless safe for Colab
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run full inference pass over loader.

    Returns:
        (all_preds, all_labels) as numpy arrays
    """
    model.eval()
    all_preds  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with autocast():
            outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str | None = None,
) -> None:
    """Print sklearn classification report and optionally save as .txt."""
    # Shorten class names for display
    short_names = [c.replace("grape_", "") for c in class_names]
    report = classification_report(
        y_true, y_pred, target_names=short_names, digits=4
    )
    print("\n" + "=" * 70)
    print("Classification Report")
    print("=" * 70)
    print(report)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(report)
        print(f"Report saved: {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str | None = None,
    normalize: bool = True,
) -> None:
    """
    Plot and optionally save confusion matrix.

    Args:
        normalize: if True, show row-normalized (recall) percentages
    """
    short_names = [c.replace("grape_", "") for c in class_names]
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt     = ".2f"
        title   = "Confusion Matrix (Row-Normalized)"
    else:
        cm_plot = cm
        fmt     = "d"
        title   = "Confusion Matrix (Counts)"

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm_plot, annot=True, fmt=fmt,
        xticklabels=short_names, yticklabels=short_names,
        cmap="Blues", linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved: {save_path}")
    plt.show()
    plt.close(fig)


def plot_training_curves(
    history: dict,
    save_path: str | None = None,
) -> None:
    """Plot loss and accuracy curves side-by-side."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train", linewidth=1.8)
    ax1.plot(epochs, history["val_loss"],   label="Val",   linewidth=1.8)
    ax1.set_title("Loss", fontsize=13)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train", linewidth=1.8)
    ax2.plot(epochs, history["val_acc"],   label="Val",   linewidth=1.8)
    ax2.set_title("Accuracy", fontsize=13)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("MobileNetV3-Large — Grape Disease Detection",
                 fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved: {save_path}")
    plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────
# ONNX Export
# ──────────────────────────────────────────────

def export_onnx(
    model: nn.Module,
    save_path: str,
    image_size: int = 224,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Export PyTorch model to ONNX with dynamic batch size.

    Args:
        model:      trained model (already loaded)
        save_path:  output .onnx file path
        image_size: model input spatial size
    """
    import onnx

    model.eval().to(device)
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    # Validate
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported and verified: {save_path}")
