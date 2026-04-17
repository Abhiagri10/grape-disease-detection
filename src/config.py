"""
src/config.py
Load and validate YAML configuration.
"""
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    """Load config.yaml and return as nested dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    """Fail fast on missing critical keys."""
    required = [
        ("dataset", "classes"),
        ("dataset", "split"),
        ("training", "batch_size"),
        ("training", "epochs"),
        ("optimizer", "lr"),
    ]
    for section, key in required:
        assert key in cfg.get(section, {}), \
            f"Config missing: [{section}][{key}]"

    splits = cfg["dataset"]["split"]
    total = sum(splits.values())
    assert abs(total - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {total:.4f}"

    assert len(cfg["dataset"]["classes"]) > 0, \
        "No classes defined in config"


if __name__ == "__main__":
    cfg = load_config()
    print(f"Config loaded: {len(cfg['dataset']['classes'])} classes, "
          f"batch_size={cfg['training']['batch_size']}, "
          f"epochs={cfg['training']['epochs']}")
