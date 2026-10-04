"""Regression metrics in cycles, life-group summaries and paired statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def regression_metrics(y_true, y_pred, y_ref_mean: float | None = None) -> dict:
    """All metrics are in cycles unless stated.

    Normalised errors divide by the mean *true* test cycle life (unless a
    reference mean is given). MAPE is safe here because labels are > 100
    cycles by construction (BatteryML min_rul_limit).
    """
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    e = p - y
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mae = float(np.mean(np.abs(e)))
    ref = float(np.mean(y)) if y_ref_mean is None else y_ref_mean
    ss_res, ss_tot = np.sum(e ** 2), np.sum((y - y.mean()) ** 2)
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "MedAE": float(np.median(np.abs(e))),
        "Bias": float(np.mean(e)),                 # mean signed error (pred - true)
        "nRMSE": rmse / ref,
        "nMAE": mae / ref,
        "MAPE": float(np.mean(np.abs(e) / y)),
        "MaxAE": float(np.max(np.abs(e))),
    }


def life_groups(y, edges) -> np.ndarray:
    """Assign short / medium / long using tertile edges fitted on training labels."""
    y = np.asarray(y, float)
    return np.where(y <= edges[0], "short", np.where(y <= edges[1], "medium", "long"))


def tertile_edges(y_train) -> tuple[float, float]:
    q = np.quantile(np.asarray(y_train, float), [1 / 3, 2 / 3])
    return float(q[0]), float(q[1])


def bootstrap_ci(y_true, preds_by_seed: np.ndarray, metric: str = "RMSE",
                 n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Cell-level bootstrap CI of a metric averaged over seeds.

    ``preds_by_seed`` has shape (n_seeds, n_cells). Resampling cells captures
    test-set sampling uncertainty; averaging over seeds removes seed noise.
    """
    y = np.asarray(y_true, float)
    P = np.atleast_2d(preds_by_seed)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        vals[b] = np.mean([regression_metrics(y[i], p[i])[metric] for p in P])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_comparison(y_true, preds_a: np.ndarray, preds_b: np.ndarray,
                      n_boot: int = 2000, seed: int = 0) -> dict:
    """Compare model A against model B on the same test cells.

    Returns the seed-averaged RMSE/MAE difference (A - B; negative favours A),
    a paired cell-bootstrap 95 % CI of the RMSE difference, and a Wilcoxon
    signed-rank test on seed-averaged per-cell absolute errors.
    """
    y = np.asarray(y_true, float)
    A, B = np.atleast_2d(preds_a), np.atleast_2d(preds_b)
    k = min(len(A), len(B))
    A, B = A[:k], B[:k]

    def rmse(P, idx):
        return np.mean([np.sqrt(np.mean((p[idx] - y[idx]) ** 2)) for p in P])

    full = np.arange(len(y))
    d_rmse = rmse(A, full) - rmse(B, full)
    d_mae = np.mean(np.abs(A - y)) - np.mean(np.abs(B - y))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        boots[b] = rmse(A, i) - rmse(B, i)
    ae_a, ae_b = np.abs(A - y).mean(0), np.abs(B - y).mean(0)
    try:
        w = stats.wilcoxon(ae_a, ae_b)
        p_w = float(w.pvalue)
    except ValueError:
        p_w = np.nan
    return {
        "dRMSE": float(d_rmse),
        "dRMSE_lo": float(np.quantile(boots, 0.025)),
        "dRMSE_hi": float(np.quantile(boots, 0.975)),
        "P(dRMSE<0)": float(np.mean(boots < 0)),
        "dMAE": float(d_mae),
        "wilcoxon_p": p_w,
        "cells_A_better": int(np.sum(ae_a < ae_b)),
        "n_cells": int(len(y)),
    }


def summarize_seeds(rows: list[dict]) -> dict:
    """Mean and std over seeds for every numeric key."""
    df = pd.DataFrame(rows)
    out = {}
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.number):
            out[f"{c}_mean"] = float(df[c].mean())
            out[f"{c}_std"] = float(df[c].std(ddof=1)) if len(df) > 1 else 0.0
    return out
