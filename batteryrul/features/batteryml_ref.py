"""Feature Set A: re-implementation of BatteryML's MATR feature extractors.

Ported from batteryml/feature/severson.py and voltage_capacity_matrix.py with
the MATR-1 config values (critical_cycles = [2, 9, 99], diff_base = 8,
cycle 98 for QdLinear, use_precalculated_qdlin = True). BatteryML quirks are
kept on purpose so the reproduced baselines are comparable:

* "Early discharge capacity" indexes the *sliced* capacity list
  Qd[c0:c2][c0], i.e. cycle index 4, not cycle index 2.
* "Average early charge time" sums dt where I < 0, which in MATR is the
  discharge step (time is in minutes).
* "Integral of temperature over time" is log(mean cycle-average temperature).
* NaN / inf feature values are replaced by 0.

For other horizons T the late critical cycle becomes T-1 (and the reference
cycle min(9, T//2)), so the features never use cycles beyond T.
"""
from __future__ import annotations

import numpy as np

from .window import Window

EPS = 1e-8


def median_smooth(x: np.ndarray, window_size: int = 10) -> np.ndarray:
    """BatteryML ``smooth``: running median over [i-10, i+10] (returns medians)."""
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():          # rare path: keep exact NaN propagation of np.median
        n = len(x)
        return np.array([np.median(x[max(0, i - window_size):min(n, i + window_size + 1)])
                         for i in range(n)])
    # Vectorised: NaN padding makes truncated edge windows identical to the loop
    padded = np.pad(x, window_size, constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * window_size + 1)
    return np.nanmedian(windows, axis=1)


def critical_cycles(T: int) -> tuple[int, int, int]:
    if T == 100:
        return 2, 9, 99
    return 2, min(9, T // 2), T - 1


def delta_q_curve(win: Window, early: int, late: int) -> np.ndarray:
    """Smoothed Qd(V)_late - Qd(V)_early with NaNs removed (Severson DeltaQ)."""
    diff = median_smooth(win.qdlin_at(late) - win.qdlin_at(early))
    return diff[~np.isnan(diff)]


def _linfit(x, y):
    m = np.isfinite(y)
    if m.sum() < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x[m], y[m], 1)
    return slope, intercept


def severson_features(win: Window) -> dict:
    c0, c1, c2 = critical_cycles(win.T)
    dq = delta_q_curve(win, c1, c2)
    f = {}
    if len(dq) <= 1:
        dq = np.zeros(2)
    std, var = dq.std(ddof=1), dq.var(ddof=1)          # torch defaults (unbiased)
    centred = dq - dq.mean()
    f["dq_min_log"] = np.log10(abs(dq.min()) + EPS)
    f["dq_var_log"] = np.log10(var + EPS)
    f["dq_skew_log"] = np.log10(abs((centred ** 3).mean()) / (std ** 3 + EPS) + EPS)
    f["dq_kurt_log"] = np.log10((centred ** 4).mean() / (var ** 2 + EPS) + EPS)

    pos = win.between(c0, c2)
    qd = win["qd"][pos]
    x = (win.idx[pos] - c0).astype(float)
    f["qd_early"] = qd[min(c0, len(qd) - 1)]
    f["qd_max_minus_early"] = np.nanmax(qd) - f["qd_early"]
    f["qd_slope"], f["qd_intercept"] = _linfit(x, qd)

    early_t = win["t_bml_chg"][win.between(0, 4)]
    early_t = early_t[np.isfinite(early_t)]
    f["t_early_chg_log"] = np.log((early_t.mean() if len(early_t) else 0.0) + EPS)

    inc = win.between(c0, c2 + 1)
    temp = win["temp_mean"][inc]
    temp = temp[np.isfinite(temp)]
    f["temp_avg_log"] = np.log((temp.mean() if len(temp) else 0.0) + EPS)
    f["ir_min"] = np.min(win["ir"][inc])
    f["ir_change"] = win.at("ir", c2) - win.at("ir", c0)
    return {k: (0.0 if not np.isfinite(v) else float(v)) for k, v in f.items()}


# BatteryML model feature lists (names mapped to the keys above)
VARIANCE_MODEL = ["dq_var_log"]
DISCHARGE_MODEL = ["dq_min_log", "dq_var_log", "dq_skew_log", "dq_kurt_log",
                   "qd_early", "qd_max_minus_early"]
FULL_MODEL = ["dq_min_log", "dq_var_log", "qd_slope", "qd_intercept", "qd_early",
              "t_early_chg_log", "temp_avg_log", "ir_min", "ir_change"]
SET_A = ["dq_min_log", "dq_var_log", "dq_skew_log", "dq_kurt_log", "qd_early",
         "qd_max_minus_early", "qd_slope", "qd_intercept", "t_early_chg_log",
         "temp_avg_log", "ir_min", "ir_change"]


def qdlinear_feature(win: Window, diff_base: int = 8, cycle: int = 98) -> np.ndarray:
    """BatteryML VoltageCapacityMatrixFeatureExtractor (cycles_to_keep=[cycle])."""
    base = median_smooth(win.qdlin_at(diff_base))
    cur = median_smooth(win.qdlin_at(cycle))
    diff = median_smooth(cur - base)
    diff[~np.isfinite(diff)] = 0.0
    return diff
