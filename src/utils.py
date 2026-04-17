"""
src/utils.py
Shared utilities:
  - Reproducibility seeding
  - Device detection
  - Class weight computation (for imbalanced datasets)
  - Zip selective extraction (grape classes only)
"""
import os
import random
import zipfile
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic ops — may slow down training slightly
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────

def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU : {props.name}")
        print(f"  VRAM: {props.total_memory / 1e9:.1f} GB")
    return device


# ──────────────────────────────────────────────
# Dataset extraction
# ──────────────────────────────────────────────

def extract_grape_only(
    zip_path: str,
    extract_dir: str,
    species_prefix: str = "grape_",
    nested_parent: str | None = None,
    skip_if_exists: bool = True,
) -> None:
    """
    Selectively extract only grape_* folders from the dataset zip.

    Handles both flat and nested zip structures:
      - Flat  : grape_anthracnose/img.jpg
      - Nested: Ranveer_Realistic_dataset_2000_1126/grape_anthracnose/img.jpg

    In the nested case, the parent prefix is stripped so extracted files
    land directly under extract_dir/grape_anthracnose/.

    Args:
        zip_path:       full path to the .zip file
        extract_dir:    destination root directory
        species_prefix: folder name prefix to keep (default: "grape_")
        nested_parent:  top-level folder inside zip, if any (auto-detected
                        if None; set explicitly via config nested_parent)
        skip_if_exists: skip if extract_dir already contains grape folders
    """
    extract_path = Path(extract_dir)

    # Skip if grape class folders already present
    if skip_if_exists and extract_path.exists():
        existing_grape = [
            p for p in extract_path.iterdir()
            if p.is_dir() and p.name.startswith(species_prefix)
        ]
        if existing_grape:
            print(f"Extraction skipped — {len(existing_grape)} grape folders "
                  f"already in {extract_dir}")
            return

    extract_path.mkdir(parents=True, exist_ok=True)

    print(f"Opening zip: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_members = zf.namelist()

        # ── Auto-detect or confirm nested parent ──────────────────
        if nested_parent is None:
            # Detect: is there a single top-level directory that wraps everything?
            top_dirs = {m.split("/")[0] for m in all_members if "/" in m}
            flat_grape = [m for m in all_members
                          if m.startswith(species_prefix)]
            if flat_grape:
                nested_parent = ""   # flat structure — no stripping needed
            elif len(top_dirs) == 1:
                nested_parent = top_dirs.pop()
                print(f"  Auto-detected nested parent: '{nested_parent}'")
            else:
                nested_parent = ""

        # ── Filter grape members ───────────────────────────────────
        prefix_with_parent = (
            f"{nested_parent}/{species_prefix}" if nested_parent else species_prefix
        )
        grape_members = [m for m in all_members
                         if prefix_with_parent in m and not m.endswith("/")]

        print(f"  Total zip entries : {len(all_members)}")
        print(f"  Grape file entries: {len(grape_members)}")
        print(f"  Nested parent     : '{nested_parent or '(none — flat)'}'")

        if not grape_members:
            # Diagnostic dump to help the user fix config
            sample = all_members[:10]
            raise ValueError(
                f"No grape entries found with prefix '{prefix_with_parent}'.\n"
                f"First 10 zip entries:\n" + "\n".join(f"  {e}" for e in sample) +
                f"\n\nFix: set nested_parent in configs/config.yaml to match "
                f"the top-level folder name shown above."
            )

        # ── Extract with parent-prefix stripping ──────────────────
        for i, member in enumerate(grape_members, 1):
            # Strip nested parent so file lands at extract_dir/grape_X/file.jpg
            if nested_parent and member.startswith(nested_parent + "/"):
                rel_path = member[len(nested_parent) + 1:]  # strip "Parent/"
            else:
                rel_path = member

            dest = extract_path / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())

            if i % 300 == 0:
                print(f"  Extracted {i}/{len(grape_members)}...")

    # Final verification
    extracted_dirs = [p.name for p in extract_path.iterdir()
                      if p.is_dir() and p.name.startswith(species_prefix)]
    print(f"Extraction complete → {extract_dir}")
    print(f"  Grape class folders extracted: {len(extracted_dirs)}")


# ──────────────────────────────────────────────
# Class balance
# ──────────────────────────────────────────────

def compute_class_weights(
    samples: List[Tuple[str, int]],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.
    Useful when class imbalance exists.

    Returns:
        FloatTensor of shape (num_classes,) on device
    """
    counts = Counter(s[1] for s in samples)
    total  = len(samples)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls_idx in range(num_classes):
        n = counts.get(cls_idx, 1)
        weights[cls_idx] = total / (num_classes * n)
    weights = weights / weights.sum() * num_classes  # normalize
    print(f"Class weights: {weights.tolist()}")
    return weights.to(device)


# ──────────────────────────────────────────────
# Print helpers
# ──────────────────────────────────────────────

def print_class_distribution(
    samples: List[Tuple[str, int]],
    class_names: List[str],
) -> None:
    """Print per-class image counts and percentage."""
    counts = Counter(s[1] for s in samples)
    total  = len(samples)
    print(f"\n{'='*50}")
    print(f"{'Class Distribution':^50}")
    print(f"{'='*50}")
    print(f"{'Class':<45} {'N':>5}  {'%':>5}")
    print(f"{'-'*50}")
    for idx, name in enumerate(class_names):
        n = counts.get(idx, 0)
        print(f"{name:<45} {n:>5}  {100*n/total:>4.1f}%")
    print(f"{'-'*50}")
    print(f"{'TOTAL':<45} {total:>5}  100.0%")
    print(f"{'='*50}\n")
