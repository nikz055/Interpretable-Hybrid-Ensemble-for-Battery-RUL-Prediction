from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
