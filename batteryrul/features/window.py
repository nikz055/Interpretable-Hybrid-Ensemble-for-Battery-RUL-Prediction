"""The early-life observation window: the only object feature functions see.

``Window(cache, T)`` slices every cached signal to the first T cycles
(0-based indices 0..T-1, i.e. cycle numbers 1..T) *before* anything else
happens, so no feature can read cycle T+1 or later, the final capacity, or the
label. ``keep`` optionally drops cycles inside the window (sparse-sampling
robustness test); lookups then fall back to the nearest retained cycle at or
before the requested one.
"""
from __future__ import annotations

import numpy as np

from ..cache import SCALAR_SIGNALS

FORBIDDEN_KEYS = ("label", "legacy_label", "qd_full", "n_cycles_recorded")


class Window:
    def __init__(self, cache: dict, T: int, keep: np.ndarray | None = None):
        n_avail = len(cache["qd"])
        if n_avail < T:
            raise ValueError(f"{cache.get('cell_id')}: only {n_avail} cached cycles < T={T}")
        idx = np.arange(T)
        if keep is not None:
            idx = idx[np.asarray(keep, dtype=bool)[:T]]
        if len(idx) < 2:
            raise ValueError("Window needs at least two retained cycles")
        self.T = T
        self.idx = idx                                   # 0-based cycle index
        self.cell_id = str(cache.get("cell_id", ""))
        self.nominal_capacity_ah = float(cache["nominal_capacity_ah"])
        self._sig = {k: np.asarray(cache[k], dtype=float)[:T][idx] for k in SCALAR_SIGNALS}
        self.qdlin = np.asarray(cache["qdlin"], dtype=float)[:T][idx]

    def __getitem__(self, name: str) -> np.ndarray:
        if name in FORBIDDEN_KEYS:
            raise KeyError(f"'{name}' contains future information and is not a feature input")
        return self._sig[name]

    def pos(self, k: int) -> int:
        """Position of cycle index k, or of the nearest retained index <= k."""
        k = int(np.clip(k, 0, self.T - 1))
        p = np.searchsorted(self.idx, k, side="right") - 1
        return int(max(p, 0))

    def at(self, name: str, k: int) -> float:
        return float(self[name][self.pos(k)])

    def qdlin_at(self, k: int) -> np.ndarray:
        return self.qdlin[self.pos(k)]

    def between(self, a: int, b: int) -> np.ndarray:
        """Positions of retained cycles with a <= index < b."""
        return np.where((self.idx >= a) & (self.idx < b))[0]
