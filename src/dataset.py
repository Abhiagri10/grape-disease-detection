"""
src/dataset.py
Dataset class, stratified splitting, and DataLoader factory.
All logic is reproducible via fixed seed.
"""
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────

def get_transforms(cfg: dict, split: str) -> transforms.Compose:
    """
    Returns torchvision transforms for train / val / test.

    Args:
        cfg:   loaded config dict
        split: 'train' | 'val' | 'test'
    """
    mean = cfg["preprocessing"]["imagenet_mean"]
    std  = cfg["preprocessing"]["imagenet_std"]
    size = cfg["preprocessing"]["image_size"]
    aug  = cfg["augmentation"]

    normalize = transforms.Normalize(mean=mean, std=std)

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=aug["horizontal_flip_p"]),
            transforms.RandomVerticalFlip(p=aug["vertical_flip_p"]),
            transforms.RandomRotation(degrees=aug["rotation_degrees"]),
            transforms.ColorJitter(
                brightness=aug["color_jitter"]["brightness"],
                contrast=aug["color_jitter"]["contrast"],
                saturation=aug["color_jitter"]["saturation"],
                hue=aug["color_jitter"]["hue"],
            ),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=aug["random_erasing_p"]),
        ])
    else:
        # val and test — deterministic
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            normalize,
        ])


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class GrapeDiseaseDataset(Dataset):
    """
    Image classification dataset for grape disease classes.

    Args:
        samples:    list of (image_path, class_index) tuples
        class_names: ordered list of class name strings
        transform:  torchvision transform pipeline
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        class_names: List[str],
        transform: transforms.Compose,
    ):
        self.samples     = samples
        self.class_names = class_names
        self.transform   = transform
        self._skipped    = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            # Return a black tensor if file is corrupted
            img = Image.new("RGB", (224, 224), color=0)
            self._skipped += 1
        img = self.transform(img)
        return img, label


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def collect_samples(data_dir: str, class_names: List[str],
                    exts: List[str]) -> List[Tuple[str, int]]:
    """
    Walk data_dir and collect (filepath, class_idx) pairs.
    Only folders whose names are in class_names are included.

    Returns:
        List of (absolute_path, label_index) tuples
    """
    exts_set = {e.lower() for e in exts}
    samples  = []
    for idx, cls in enumerate(class_names):
        cls_dir = Path(data_dir) / cls
        if not cls_dir.exists():
            raise FileNotFoundError(
                f"Class folder not found: {cls_dir}\n"
                f"Check dataset extraction and config class names."
            )
        for fp in sorted(cls_dir.iterdir()):
            if fp.suffix.lower() in exts_set:
                samples.append((str(fp), idx))
    return samples


def split_samples(
    samples: List[Tuple[str, int]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List, List, List]:
    """
    Stratified 3-way split.

    Returns:
        (train_samples, val_samples, test_samples)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    labels = [s[1] for s in samples]

    # First split: train vs (val + test)
    temp_ratio = val_ratio + test_ratio
    train, temp, _, temp_labels = train_test_split(
        samples, labels,
        test_size=temp_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test (equal halves of temp)
    val_frac = val_ratio / temp_ratio
    val, test = train_test_split(
        temp,
        test_size=(1.0 - val_frac),
        stratify=temp_labels,
        random_state=seed,
    )
    return train, val, test


def print_split_summary(
    train: List, val: List, test: List,
    class_names: List[str],
) -> None:
    """Print per-class and total counts for each split."""
    total = len(train) + len(val) + len(test)
    print(f"\n{'='*55}")
    print(f"{'Split Summary':^55}")
    print(f"{'='*55}")
    print(f"{'Class':<45} {'Train':>5} {'Val':>5} {'Test':>5}")
    print(f"{'-'*55}")
    for idx, cls in enumerate(class_names):
        n_tr = sum(1 for s in train if s[1] == idx)
        n_va = sum(1 for s in val   if s[1] == idx)
        n_te = sum(1 for s in test  if s[1] == idx)
        print(f"{cls:<45} {n_tr:>5} {n_va:>5} {n_te:>5}")
    print(f"{'-'*55}")
    print(f"{'TOTAL':<45} {len(train):>5} {len(val):>5} {len(test):>5}")
    print(f"  Combined: {total}  |  "
          f"Train {len(train)/total*100:.1f}%  "
          f"Val {len(val)/total*100:.1f}%  "
          f"Test {len(test)/total*100:.1f}%")
    print(f"{'='*55}\n")


# ──────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────

def build_dataloaders(
    train: List, val: List, test: List,
    class_names: List[str],
    cfg: dict,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders.

    Returns:
        dict with keys 'train', 'val', 'test'
    """
    bs          = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]

    datasets = {
        "train": GrapeDiseaseDataset(train, class_names,
                                     get_transforms(cfg, "train")),
        "val":   GrapeDiseaseDataset(val,   class_names,
                                     get_transforms(cfg, "val")),
        "test":  GrapeDiseaseDataset(test,  class_names,
                                     get_transforms(cfg, "test")),
    }

    loaders = {
        "train": DataLoader(datasets["train"], batch_size=bs,
                            shuffle=True,  num_workers=num_workers,
                            pin_memory=True, drop_last=True),
        "val":   DataLoader(datasets["val"],   batch_size=bs,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True),
        "test":  DataLoader(datasets["test"],  batch_size=bs,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True),
    }
    return loaders
