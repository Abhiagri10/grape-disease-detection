"""
src/train.py
Training loop with:
  - Mixed-precision (torch.cuda.amp)
  - Per-epoch val evaluation
  - Early stopping
  - Best-model checkpointing to Google Drive
"""
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────
# One epoch
# ──────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Run one training epoch.

    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        # Gradient clipping — stabilises training with AMP
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        correct      += predicted.eq(labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Run validation / test pass (no gradient).

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            outputs = model(images)
            loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        correct      += predicted.eq(labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, correct / total


# ──────────────────────────────────────────────
# Full training run
# ──────────────────────────────────────────────

def train(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    cfg: dict,
    device: torch.device,
) -> Dict[str, List]:
    """
    Full training + validation loop.

    Returns:
        history dict with lists: train_loss, train_acc, val_loss, val_acc
    """
    epochs    = cfg["training"]["epochs"]
    patience  = cfg["training"]["early_stopping_patience"]
    model_dir = Path(cfg["outputs"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / cfg["outputs"]["best_model_name"]

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg["training"]["label_smoothing"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["scheduler"]["T_max"]
    )

    scaler = GradScaler()

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    best_val_acc   = 0.0
    patience_count = 0

    print(f"\n{'Ep':>4}  {'Tr Loss':>8}  {'Tr Acc':>7}  "
          f"{'Va Loss':>8}  {'Va Acc':>7}  {'LR':>10}  {'Time':>6}")
    print("─" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device
        )
        va_loss, va_acc = validate(
            model, loaders["val"], criterion, device
        )
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f"{epoch:>4}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  "
              f"{va_loss:>8.4f}  {va_acc:>7.4f}  {lr_now:>10.2e}  "
              f"{elapsed:>5.1f}s")

        # Save best
        if va_acc > best_val_acc:
            best_val_acc   = va_acc
            patience_count = 0
            torch.save(
                {
                    "epoch":          epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc":   best_val_acc,
                    "config":         cfg,
                },
                save_path,
            )
            print(f"       ✓ Best model saved  (val_acc={best_val_acc:.4f})")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\nEarly stop at epoch {epoch} — "
                      f"no improvement for {patience} epochs.")
                break

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    print(f"Model saved: {save_path}")
    return history
