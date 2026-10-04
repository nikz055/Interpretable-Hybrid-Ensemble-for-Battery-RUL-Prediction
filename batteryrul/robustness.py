"""Controlled corruptions of the held-out *test* inputs (Experiment 5).

Models are always trained on clean training cells; only test inputs are
corrupted, which mimics deploying a model on lower-quality field data.

* missing   - a random fraction of test feature values is set to NaN and
              filled by the train-fitted median imputer inside the pipeline
* noise     - Gaussian measurement noise added to the raw per-cycle capacity
              signals (Qd, Qc, Qdlin) and to discharge-voltage summaries, then
              features are recomputed (signal-level, not feature-level, noise)
* sparse    - only every k-th cycle (1, 1+k, ...) is available; features are
              recomputed from the retained cycles
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features.build import build_features


def mask_missing(X: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    Xm = X.copy()
    mask = rng.random(X.shape) < rate
    Xm = Xm.mask(mask)
    return Xm


def noisy_cache(cache: dict, cap_sigma: float, volt_sigma: float, rng: np.random.Generator) -> dict:
    c = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in cache.items()}
    n = len(c["qd"])
    c["qd"] = c["qd"] + rng.normal(0, cap_sigma, n)
    c["qc"] = c["qc"] + rng.normal(0, cap_sigma, n)
    c["qdlin"] = (c["qdlin"] + rng.normal(0, cap_sigma, c["qdlin"].shape)).astype(np.float32)
    c["v_dis_avg"] = c["v_dis_avg"] + rng.normal(0, volt_sigma, n)
    c["e_dis"] = c["e_dis"] + c["qd"] * rng.normal(0, volt_sigma, n)
    return c


def noisy_features(caches: dict, cells, feature_set: str, T: int, cap_sigma: float,
                   volt_sigma: float, rng: np.random.Generator) -> pd.DataFrame:
    noisy = {c: noisy_cache(caches[c], cap_sigma, volt_sigma, rng) for c in cells}
    return build_features(noisy, cells, feature_set, T)


def sparse_features(caches: dict, cells, feature_set: str, T: int, every: int) -> pd.DataFrame:
    keep = np.zeros(T, dtype=bool)
    keep[::every] = True
    return build_features(caches, cells, feature_set, T, keep=keep)
