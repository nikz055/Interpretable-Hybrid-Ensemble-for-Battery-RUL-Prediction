"""Assemble per-cell feature matrices for a feature set and horizon T."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .batteryml_ref import SET_A, severson_features
from .physical import set_b, set_c
from .window import Window

FEATURE_SETS: dict[str, Callable] = {
    "A": lambda w: {k: severson_features(w)[k] for k in SET_A},
    "B": set_b,
    "C": set_c,
}


def build_features(caches: dict, cells: list[str], feature_set: str, T: int,
                   keep: np.ndarray | None = None) -> pd.DataFrame:
    fn = FEATURE_SETS[feature_set]
    rows = [fn(Window(caches[c], T, keep)) for c in cells]
    df = pd.DataFrame(rows, index=pd.Index(cells, name="cell_id")).astype(float)
    return df.replace([np.inf, -np.inf], np.nan)


def labels_for(caches: dict, cells: list[str]) -> np.ndarray:
    return np.array([caches[c]["label"] for c in cells], dtype=float)
