"""Ensemble combiners fitted on out-of-fold (OOF) predictions of training cells.

Constrained linear stack (the proposed model):

    L_hat = sum_i w_i * L_hat_i + b,   w_i >= 0,   sum_i w_i = 1

Weights are fitted on OOF predictions only, so the test cells never influence
them. With the simplex constraint the stack is a convex combination of the
base predictions plus a global bias correction, which keeps it interpretable
and prevents the meta-learner from over-fitting the ~40 training cells.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.base import clone


def oof_predict(estimator, X: np.ndarray, y: np.ndarray, splits) -> np.ndarray:
    """Out-of-fold predictions; each training cell is predicted by a model that never saw it."""
    oof = np.full(len(y), np.nan)
    for trn, val in splits:
        m = clone(estimator).fit(X[trn], y[trn])
        oof[val] = m.predict(X[val])
    if np.isnan(oof).any():
        raise RuntimeError("OOF folds do not cover every training cell")
    return oof


class ConstrainedStacker:
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept

    def fit(self, P: np.ndarray, y: np.ndarray) -> "ConstrainedStacker":
        P, y = np.asarray(P, float), np.asarray(y, float)
        k = P.shape[1]

        def bias(w):
            return float(np.mean(y - P @ w)) if self.fit_intercept else 0.0

        def loss(w):
            r = y - P @ w - bias(w)
            return float(np.mean(r ** 2))

        res = minimize(loss, np.full(k, 1.0 / k), method="SLSQP",
                       bounds=[(0.0, 1.0)] * k,
                       constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                       options={"ftol": 1e-12, "maxiter": 500})
        w = np.clip(res.x, 0, None)
        self.weights_ = w / w.sum()
        self.intercept_ = bias(self.weights_)
        return self

    def predict(self, P: np.ndarray) -> np.ndarray:
        return np.asarray(P, float) @ self.weights_ + self.intercept_


def inverse_mse_weights(P_oof: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Validation-weighted average: w_i proportional to 1 / OOF-MSE_i."""
    mse = np.mean((np.asarray(P_oof) - np.asarray(y)[:, None]) ** 2, axis=0)
    w = 1.0 / mse
    return w / w.sum()


def crossfit_stack_oof(P_oof: np.ndarray, y: np.ndarray, splits) -> np.ndarray:
    """Honest CV estimate of the stack: weights for fold k are fitted on the
    OOF predictions of the other folds only."""
    out = np.full(len(y), np.nan)
    for trn, val in splits:
        out[val] = ConstrainedStacker().fit(P_oof[trn], y[trn]).predict(P_oof[val])
    return out
