"""Audit: can the legacy notebook's MATR-1 XGBoost RMSE (103.5) be reproduced?

Re-implements notebooks/legacy/xai-battery-degradation.ipynb (cells 8-16)
line by line on the signal cache, then changes one thing at a time:

  L0  exact legacy pipeline (label = number of recorded cycles)
  L1  L0 but with BatteryML's 80 %-SOH label
  L2  training-only 5-fold CV estimate of the L0/L1 configuration
  L3  test RMSE of 36 neighbouring XGBoost configurations (diagnostic only:
      shows how much a single untuned configuration can move on 42 cells;
      never used to select anything)

Outputs: results/tables/legacy_audit.csv, results/tables/legacy_hparam_sensitivity.csv
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

from batteryrul.cache import load_all  # noqa: E402
from batteryrul.config import load_config, repo_path  # noqa: E402
from batteryrul.features.batteryml_ref import median_smooth  # noqa: E402
from batteryrul.splits import fold_indices, read_manifests  # noqa: E402

EPS = 1e-8
COLUMNS = ["DeltaQ_Variance", "DeltaQ_Minimum", "Early_Discharge_Capacity", "Max_Min_Capacity_Diff",
           "Capacity_Fade_Slope", "Capacity_Fade_Intercept", "Avg_Charge_Time",
           "Temperature_Integral", "Min_Internal_Resistance", "IR_Change"]


def legacy_features(c: dict, cc=(1, 9, 99)) -> dict:
    """notebook cell 9 ``extract_all_features`` (use_precalculated=True)."""
    early, late = c["qdlin"][cc[1]].astype(float), c["qdlin"][cc[2]].astype(float)
    dq = median_smooth(np.nan_to_num(late - early, nan=0.0))
    f = {"DeltaQ_Variance": np.log10(np.var(dq) + EPS),
         "DeltaQ_Minimum": np.log10(np.abs(np.min(dq)) + EPS)}
    caps = c["qd"][cc[0]:cc[2]]
    f["Early_Discharge_Capacity"] = caps[0]
    f["Max_Min_Capacity_Diff"] = caps.max() - caps[0]
    f["Capacity_Fade_Slope"], f["Capacity_Fade_Intercept"] = np.polyfit(np.arange(len(caps)), caps, 1)
    f["Avg_Charge_Time"] = np.log(np.mean(c["t_bml_chg"][:4]) + EPS)
    f["Temperature_Integral"] = np.log(np.mean(c["temp_mean"][cc[0]:cc[2] + 1]) + EPS)
    f["Min_Internal_Resistance"] = np.min(c["ir"][cc[0]:cc[2] + 1])
    f["IR_Change"] = c["ir"][cc[2]] - c["ir"][cc[0]]
    return f


def fit_predict(Xtr, ytr, Xte, **params):
    """notebook cells 15-16: z-score features, log+z-score label, XGBoost."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    ly = np.log(ytr + 1e-8)
    lm, ls = ly.mean(), ly.std() + 1e-8
    p = dict(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0)
    p.update(params)
    model = xgb.XGBRegressor(**p).fit((Xtr - mu) / sd, (ly - lm) / ls)
    return np.exp(model.predict((Xte - mu) / sd) * ls + lm)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    train, test, folds = read_manifests(repo_path(cfg["split"]["manifest_dir"]))
    caches = load_all(repo_path(cfg["data"]["cache_dir"]), train + test)
    X = {c: legacy_features(caches[c]) for c in train + test}
    Xtr = pd.DataFrame([X[c] for c in train])[COLUMNS].to_numpy()
    Xte = pd.DataFrame([X[c] for c in test])[COLUMNS].to_numpy()

    print("b1c1 legacy features (compare with notebook cell 13 output):")
    print(pd.Series(X["b1c1"]).round(6).to_string())

    rows = []
    for tag, key in [("L0 legacy label (n recorded cycles)", "legacy_label"),
                     ("L1 BatteryML label (80% SOH)", "label")]:
        ytr = np.array([caches[c][key] for c in train])
        yte_own = np.array([caches[c][key] for c in test])
        yte_bml = np.array([caches[c]["label"] for c in test])
        p = fit_predict(Xtr, ytr, Xte)
        oof = np.zeros(len(train))
        for _, a, b in fold_indices(train, folds):
            oof[b] = fit_predict(Xtr[a], ytr[a], Xtr[b])
        rows.append({"variant": tag,
                     "test_RMSE_vs_own_label": rmse(yte_own, p),
                     "test_RMSE_vs_batteryml_label": rmse(yte_bml, p),
                     "train_CV_RMSE": rmse(ytr, oof)})
    audit = pd.DataFrame(rows)
    print(audit.round(1).to_string(index=False))

    ytr = np.array([caches[c]["label"] for c in train])
    yte = np.array([caches[c]["label"] for c in test])
    grid = list(itertools.product([2, 3, 4, 6], [50, 100, 300], [0.05, 0.1, 0.3]))
    sens = pd.DataFrame([{"max_depth": d, "n_estimators": n, "learning_rate": lr,
                          "test_RMSE": rmse(yte, fit_predict(Xtr, ytr, Xte, max_depth=d, n_estimators=n, learning_rate=lr))}
                         for d, n, lr in grid])
    q = sens.test_RMSE.describe()
    print(f"36 neighbouring configs (BatteryML label): min {q['min']:.1f}, median {q['50%']:.1f}, max {q['max']:.1f}")

    out = repo_path(cfg["output"]["results_dir"]) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out / "legacy_audit.csv", index=False)
    sens.to_csv(out / "legacy_hparam_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
