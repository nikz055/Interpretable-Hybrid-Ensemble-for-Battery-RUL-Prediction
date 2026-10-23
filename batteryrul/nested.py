"""Repeated nested battery-level resampling with conformal uncertainty (v2).

For one horizon T and one outer split (outer-train / outer-test cells):

1. Inner grouped 5-fold CV on the outer-train cells only scores a small, fixed
   set of candidate pipelines (feature set x model x hyper-parameter):
     * life regression  -> lowest RMSE of ln(cycle life)
     * early-failure classification -> lowest log-loss (proper scoring rule)
2. The chosen pipeline is refitted on all outer-train cells and applied to the
   outer-test cells.
3. Uncertainty is calibrated on the inner out-of-fold predictions of the chosen
   pipeline (cross-conformal):
     * 90 % life intervals: exp(ln y_hat +- q), q = conformal quantile of |ln y - ln y_hat|
     * class-conditional (Mondrian) conformal pass/fail sets: a cell gets
       {fail}, {pass} or {fail, pass} (= "not sure yet, keep testing")
4. Alarm thresholds for several target recalls (operating points that trade
   missed failures against false alarms) are set on the same inner OOF
   probabilities.

The outer-test cells influence nothing above. The v1 approach (PLSR on Set C,
alarm = predicted life below a cutoff) is run through the identical procedure
as the baseline, so the comparison is like-for-like.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .models import Winsorizer, wrap

PLS_K = (2, 3, 4, 6, 8)


def _pls(k):
    return lambda: wrap(PLSRegression(n_components=k, scale=False))


def _ridge(a):
    return lambda: wrap(Ridge(alpha=a))


def _gp():
    kernel = ConstantKernel(1.0) * RBF(length_scale=10.0) + WhiteKernel(0.1)
    return wrap(GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, random_state=0))


REG_CANDIDATES = (
    [(f"plsr_k{k}_{fs}", fs, _pls(k)) for fs in ("C", "D") for k in PLS_K]
    + [(f"ridge_a{a}_{fs}", fs, _ridge(a)) for fs in ("C", "D") for a in (1, 10, 100)]
    + [("gp_C", "C", _gp)]
)
BASELINE_REG = [c for c in REG_CANDIDATES if c[0].startswith("plsr_") and c[1] == "C"]   # = v1 best model


def _logit(c):
    def make():
        return Pipeline([("winsor", Winsorizer()), ("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()), ("clf", LogisticRegression(C=c, max_iter=5000))])
    return make


CLF_CANDIDATES = [(f"logit_C{c}_{fs}", fs, _logit(c)) for fs in ("A", "C", "D") for c in (0.01, 0.03, 0.1, 0.3, 1.0)]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile; inf when there are too few scores."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n or n == 0 else float(s[k - 1])


def recall_cutoff(score: np.ndarray, positive: np.ndarray, target: float) -> float:
    """Smallest cutoff c such that flagging score <= c reaches target recall."""
    for c in np.sort(np.unique(score)):
        if (score[positive] <= c).mean() >= target:
            return float(c)
    return float(np.max(score))


def _inner_oof_reg(make, X, y, splits):
    p = np.full(len(y), np.nan)
    for trn, val in splits:
        p[val] = np.ravel(make().fit(X[trn], y[trn]).predict(X[val]))
    return np.clip(p, 1.0, None)


def _inner_oof_clf(make, X, f, splits):
    p = np.full(len(f), np.nan)
    for trn, val in splits:
        if f[trn].all() or (~f[trn]).all():
            p[val] = f[trn].mean()
        else:
            p[val] = make().fit(X[trn], f[trn]).predict_proba(X[val])[:, 1]
    return np.clip(p, 1e-6, 1 - 1e-6)


def _select_reg(cands, Xs, y, splits):
    best = None
    for name, fs, make in cands:
        oof = _inner_oof_reg(make, Xs[fs], y, splits)
        score = float(np.sqrt(np.mean((np.log(oof) - np.log(y)) ** 2)))
        if best is None or score < best[1]:
            best = (name, score, fs, make, oof)
    return best


def run_outer_fold(T: int, Xs: dict, Q: np.ndarray | None, y: np.ndarray, cells: list[str],
                   trn: np.ndarray, tst: np.ndarray, inner_splits, cfg: dict, tag: dict) -> tuple[list, dict]:
    """Fit/select/calibrate on cells[trn] only; predict cells[tst]."""
    thr, alphas, ia, target = (cfg["failure_threshold_cycles"], cfg["alphas"], cfg["interval_alpha"],
                               cfg["target_recall"])
    Xtr = {k: v[trn] for k, v in Xs.items()}
    Xte = {k: v[tst] for k, v in Xs.items()}
    ytr, fail_tr = y[trn], y[trn] <= thr

    # --- life regression: new (all candidates) and baseline (v1 = PLSR on Set C) ---
    out = {}
    for role, cands in (("new", REG_CANDIDATES), ("base", BASELINE_REG)):
        name, score, fs, make, oof = _select_reg(cands, Xtr, ytr, inner_splits)
        model = make().fit(Xtr[fs], ytr)
        pred = np.clip(np.ravel(model.predict(Xte[fs])), 1.0, None)
        q = conformal_quantile(np.abs(np.log(oof) - np.log(ytr)), ia)
        cut = recall_cutoff(oof, fail_tr, target)               # v1-style alarm on predicted life
        out[role] = dict(name=name, score=score, pred=pred, lo=pred * np.exp(-q), hi=pred * np.exp(q),
                         alarm=pred <= cut, cut=cut,
                         alarms={rr: pred <= recall_cutoff(oof, fail_tr, rr) for rr in cfg["alarm_recalls"]})
    dummy = np.full(len(tst), np.exp(np.mean(np.log(ytr))))

    bml = None
    if Q is not None:                                            # BatteryML PLSR on QdLinear (k = 9, fixed)
        m = wrap(PLSRegression(n_components=9, scale=False)).fit(Q[trn], ytr)
        bml = np.clip(np.ravel(m.predict(Q[tst])), 1.0, None)

    # --- early-failure classifier ---
    best = None
    for name, fs, make in CLF_CANDIDATES:
        oof = _inner_oof_clf(make, Xtr[fs], fail_tr, inner_splits)
        ll = float(-np.mean(fail_tr * np.log(oof) + (~fail_tr) * np.log(1 - oof)))
        if best is None or ll < best[1]:
            best = (name, ll, fs, make, oof)
    cname, cll, cfs, cmake, coof = best
    clf = cmake().fit(Xtr[cfs], fail_tr)
    p = clf.predict_proba(Xte[cfs])[:, 1]
    alarm_cut = recall_cutoff(-coof, fail_tr, target)
    alarm_cuts = {rr: recall_cutoff(-coof, fail_tr, rr) for rr in cfg["alarm_recalls"]}
    rows = []
    sets = {}
    for a in alphas:
        q_fail = conformal_quantile(1 - coof[fail_tr], a)        # nonconformity of the true class
        q_pass = conformal_quantile(coof[~fail_tr], a)
        sets[a] = (1 - p <= q_fail, p <= q_pass)
    for i, j in enumerate(tst):
        r = {**tag, "T": T, "cell_id": cells[j], "y_true": y[j], "early_failure": bool(y[j] <= thr),
             "reg_choice": out["new"]["name"], "pred_life": out["new"]["pred"][i],
             "lo90": out["new"]["lo"][i], "hi90": out["new"]["hi"][i],
             "base_choice": out["base"]["name"], "base_pred_life": out["base"]["pred"][i],
             "base_lo90": out["base"]["lo"][i], "base_hi90": out["base"]["hi"][i],
             "base_alarm": bool(out["base"]["alarm"][i]), "dummy_pred_life": dummy[i],
             "clf_choice": cname, "p_fail": p[i], "alarm": bool(-p[i] <= alarm_cut)}
        for rr in cfg["alarm_recalls"]:
            r[f"alarm_r{rr:g}"] = bool(-p[i] <= alarm_cuts[rr])
            r[f"base_alarm_r{rr:g}"] = bool(out["base"]["alarms"][rr][i])
        if bml is not None:
            r["batteryml_plsr_qdlinear"] = bml[i]
        for a in alphas:
            r[f"fail_in_set_a{a:g}"] = bool(sets[a][0][i])
            r[f"pass_in_set_a{a:g}"] = bool(sets[a][1][i])
        rows.append(r)
    sel = {**tag, "T": T, "reg_choice": out["new"]["name"], "reg_inner_log_rmse": out["new"]["score"],
           "base_choice": out["base"]["name"], "base_inner_log_rmse": out["base"]["score"],
           "clf_choice": cname, "clf_inner_logloss": cll}
    return rows, sel
