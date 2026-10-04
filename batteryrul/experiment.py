"""One experimental configuration = (feature set, horizon T).

Protocol for a configuration (identical for every feature set / horizon):

1. Tune ridge, PLSR, RF, XGBoost and MLP by repeated grouped K-fold CV on the
   training cells (test cells are not loaded into this function's tuning path).
2. For each seed: fit each tuned base model on all training cells -> test
   predictions; compute OOF predictions on the fixed fold manifest.
3. Fit ensemble combiners on OOF predictions only, apply them to the frozen
   base-model test predictions.
4. Pre-registered selections, all made on training-CV numbers:
   * best individual model = lowest mean OOF RMSE among ridge/plsr/rf/xgb/mlp
   * final ensemble = stack_rf_xgb_mlp unless stack_rf_xgb has a lower
     cross-fitted CV RMSE (then MLP is dropped)
"""
from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .ensemble import (ConstrainedStacker, crossfit_stack_oof, inverse_mse_weights,
                       oof_predict)
from .metrics import regression_metrics
from .models import make_estimator, repeated_group_folds, tune
from .splits import assert_disjoint, fold_indices

INDIVIDUAL = ["ridge", "plsr", "rf", "xgb", "mlp"]
TUNED = ["ridge", "plsr", "rf", "xgb", "mlp"]


def stack_name(models):
    return "stack_" + "_".join(models)


def run_configuration(tag: str, X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame,
                      y_te: np.ndarray, folds: pd.DataFrame, cfg: dict, out_dir: Path,
                      save_models: bool = False, log=print) -> dict:
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_cells, test_cells = list(X_tr.index), list(X_te.index)
    assert_disjoint(train=train_cells, test=test_cells)
    assert list(X_tr.columns) == list(X_te.columns)
    Xtr, Xte = X_tr.to_numpy(float), X_te.to_numpy(float)
    wq = tuple(cfg["preprocessing"]["winsor_quantiles"])
    seeds = cfg["seeds"]
    base_models = cfg["stacking"]["base_models"]

    # ---- 1. tuning on training cells only ---------------------------------
    labels = pd.Series(y_tr, index=train_cells)
    tcfg = cfg["tuning"]
    tune_splits = repeated_group_folds(train_cells, labels, cfg["split"]["n_folds"],
                                       tcfg["repeats"], tcfg["seed"])
    best, tuning_rows = {}, []
    for name in TUNED:
        params, cv_rmse, res = tune(name, Xtr, y_tr, tune_splits, tcfg["n_iter"], tcfg["seed"], wq)
        best[name] = params
        tuning_rows.append({"model": name, "tuning_cv_rmse": cv_rmse, "params": json.dumps(params)})
        log(f"  [{tag}] tuned {name:5s} CV RMSE {cv_rmse:7.1f}  {params}")
    pd.DataFrame(tuning_rows).to_csv(out_dir / "tuning.csv", index=False)
    with open(out_dir / "best_params.json", "w") as f:
        json.dump(best, f, indent=2)

    # ---- 2./3. seeds: base models, OOF, ensembles ------------------------
    oof_splits = [(trn, val) for _, trn, val in fold_indices(train_cells, folds)]
    pred_rows, weight_rows = [], []

    def record(model, seed, part, cells, y, p):
        pred_rows.append(pd.DataFrame({"model": model, "seed": seed, "partition": part,
                                       "cell_id": cells, "y_true": y, "y_pred": p}))

    for seed in seeds:
        test_p, oof_p = {}, {}
        for name in ["dummy"] + INDIVIDUAL:
            est = make_estimator(name, best.get(name), seed, wq)
            fitted = est.fit(Xtr, y_tr)
            test_p[name] = fitted.predict(Xte)
            oof_p[name] = oof_predict(est, Xtr, y_tr, oof_splits)
            if save_models and seed == seeds[0]:
                (out_dir / "models").mkdir(exist_ok=True)
                joblib.dump(fitted, out_dir / "models" / f"{name}_seed{seed}.joblib")

        ens_test, ens_oof = {}, {}
        for subset in [base_models, ["rf", "xgb"]]:
            key = "mean_" + "_".join(subset)
            ens_test[key] = np.mean([test_p[m] for m in subset], axis=0)
            ens_oof[key] = np.mean([oof_p[m] for m in subset], axis=0)
        P_oof = np.column_stack([oof_p[m] for m in base_models])
        P_te = np.column_stack([test_p[m] for m in base_models])
        w = inverse_mse_weights(P_oof, y_tr)
        key = "valweighted_" + "_".join(base_models)
        ens_test[key], ens_oof[key] = P_te @ w, P_oof @ w
        weight_rows.append({"seed": seed, "combiner": key, **dict(zip(base_models, w)), "intercept": 0.0})

        subsets = [list(c) for r in (2, 3) for c in combinations(base_models, r)]
        for subset in subsets:
            Po = np.column_stack([oof_p[m] for m in subset])
            Pt = np.column_stack([test_p[m] for m in subset])
            st = ConstrainedStacker().fit(Po, y_tr)
            key = stack_name(subset)
            ens_test[key] = st.predict(Pt)
            ens_oof[key] = crossfit_stack_oof(Po, y_tr, oof_splits)   # honest CV estimate
            weight_rows.append({"seed": seed, "combiner": key, **dict(zip(subset, st.weights_)),
                                "intercept": st.intercept_})
            if save_models and seed == seeds[0]:
                joblib.dump(st, out_dir / "models" / f"{key}_seed{seed}.joblib")

        for name, p in {**test_p, **ens_test}.items():
            record(name, seed, "test", test_cells, y_te, p)
        for name, p in {**oof_p, **ens_oof}.items():
            record(name, seed, "oof", train_cells, y_tr, p)

    preds = pd.concat(pred_rows, ignore_index=True)
    preds.to_csv(out_dir / "predictions.csv", index=False)
    weights = pd.DataFrame(weight_rows)
    weights.to_csv(out_dir / "ensemble_weights.csv", index=False)

    # ---- metrics -----------------------------------------------------------
    rows = []
    for (model, seed, part), g in preds.groupby(["model", "seed", "partition"]):
        rows.append({"model": model, "seed": seed, "partition": part,
                     **regression_metrics(g.y_true, g.y_pred)})
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out_dir / "metrics_per_seed.csv", index=False)
    summary = per_seed.groupby(["partition", "model"]).agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.drop(columns=["seed_mean", "seed_std"]).reset_index()
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)

    oof = summary[summary.partition == "oof"].set_index("model")["RMSE_mean"]
    best_individual = oof.loc[INDIVIDUAL].idxmin()
    full = stack_name(base_models)
    reduced = stack_name(["rf", "xgb"])
    final = reduced if (reduced in oof and oof[reduced] < oof[full]) else full
    selection = {"tag": tag, "best_individual_by_cv": best_individual,
                 "final_ensemble_by_cv": final,
                 "cv_rmse": {k: float(v) for k, v in oof.items()},
                 "runtime_s": round(time.time() - t0, 1)}
    with open(out_dir / "selection.json", "w") as f:
        json.dump(selection, f, indent=2)
    test = summary[summary.partition == "test"].set_index("model")
    log(f"  [{tag}] best individual (CV) = {best_individual}, final ensemble (CV) = {final}; "
        f"test RMSE {best_individual} {test.loc[best_individual, 'RMSE_mean']:.1f}, "
        f"{final} {test.loc[final, 'RMSE_mean']:.1f}  ({selection['runtime_s']:.0f}s)")
    return selection
