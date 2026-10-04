"""Step 2: reproduce BatteryML's sklearn MATR-1 baselines with our re-implementation.

Each baseline mirrors configs/baselines/sklearn/<model>/matr_1.yaml in
microsoft/BatteryML: same split, same features, ZScore feature transform,
LogScale->ZScore label transform, default model hyper-parameters. Agreement with
BatteryML's published table validates our data definitions before any new
model is trained.

Outputs:
    results/tables/batteryml_reproduction.csv
    results/predictions/batteryml_baselines.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cross_decomposition import PLSRegression  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import RBF, DotProduct  # noqa: E402
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from batteryrul.cache import load_all  # noqa: E402
from batteryrul.config import load_config, repo_path  # noqa: E402
from batteryrul.features.batteryml_ref import (DISCHARGE_MODEL, FULL_MODEL,  # noqa: E402
                                               VARIANCE_MODEL, qdlinear_feature,
                                               severson_features)
from batteryrul.features.window import Window  # noqa: E402
from batteryrul.metrics import regression_metrics  # noqa: E402
from batteryrul.splits import read_manifests  # noqa: E402

# BatteryML README, RUL benchmark table, MATR1 column (RMSE in cycles)
PUBLISHED = {
    "Dummy": "398", "Variance model": "136", "Discharge model": "329", "Full model": "167",
    "Ridge (QdLinear)": "116", "PCR (QdLinear)": "90", "PLSR (QdLinear)": "104",
    "Gaussian process (QdLinear)": "154", "XGBoost (QdLinear)": "334",
    "Random forest (QdLinear)": "168±9", "MLP (QdLinear)": "149±3",
}


class ZScore:
    """BatteryML ZScoreDataTransformation (unbiased std, clamped at 1e-8)."""

    def fit(self, x):
        self.mean = x.mean(0, keepdims=True)
        self.std = np.clip(x.std(0, ddof=1, keepdims=True), 1e-8, None)
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


def run(model, Xtr, ytr, Xte):
    fz = ZScore().fit(Xtr)
    ly = np.log(ytr)[:, None]
    lz = ZScore().fit(ly)
    model.fit(fz.transform(Xtr), lz.transform(ly).ravel())
    pred = np.asarray(model.predict(fz.transform(Xte))).reshape(-1, 1)
    return np.exp(lz.inverse(pred)).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    train, test, _ = read_manifests(repo_path(cfg["split"]["manifest_dir"]))
    caches = load_all(repo_path(cfg["data"]["cache_dir"]), train + test)
    y_tr = np.array([caches[c]["label"] for c in train])
    y_te = np.array([caches[c]["label"] for c in test])

    def expert(cells, cols):
        rows = [severson_features(Window(caches[c], 100)) for c in cells]
        return pd.DataFrame(rows)[cols].to_numpy(float)

    def qdlin(cells):
        return np.stack([qdlinear_feature(Window(caches[c], 100)) for c in cells])

    Q_tr, Q_te = qdlin(train), qdlin(test)
    specs = [
        ("Dummy", lambda s: DummyRegressor(), expert(train, VARIANCE_MODEL), expert(test, VARIANCE_MODEL), False),
        ("Variance model", lambda s: LinearRegression(), expert(train, VARIANCE_MODEL), expert(test, VARIANCE_MODEL), False),
        ("Discharge model", lambda s: LinearRegression(), expert(train, DISCHARGE_MODEL), expert(test, DISCHARGE_MODEL), False),
        ("Full model", lambda s: LinearRegression(), expert(train, FULL_MODEL), expert(test, FULL_MODEL), False),
        ("Ridge (QdLinear)", lambda s: Ridge(), Q_tr, Q_te, False),
        ("PCR (QdLinear)", lambda s: make_pipeline(PCA(n_components=12), LinearRegression()), Q_tr, Q_te, False),
        ("PLSR (QdLinear)", lambda s: PLSRegression(n_components=9, scale=False), Q_tr, Q_te, False),
        ("Gaussian process (QdLinear)", lambda s: GaussianProcessRegressor(DotProduct() + RBF()), Q_tr, Q_te, False),
        ("XGBoost (QdLinear)", lambda s: XGBRegressor(random_state=s), Q_tr, Q_te, False),
        ("Random forest (QdLinear)", lambda s: RandomForestRegressor(random_state=s, n_jobs=-1), Q_tr, Q_te, True),
    ]

    rows, preds = [], {"cell_id": test, "y_true": y_te}
    for name, make, Xtr, Xte, seeded in specs:
        seeds = cfg["seeds"] if seeded else [0]
        per_seed = []
        for s in seeds:
            p = run(make(s), Xtr, y_tr, Xte)
            per_seed.append(regression_metrics(y_te, p))
            if s == seeds[0]:
                preds[name] = p
        df = pd.DataFrame(per_seed)
        row = {"model": name, "features": "QdLinear" if "QdLinear" in name else "expert",
               "RMSE": df.RMSE.mean(), "RMSE_std": df.RMSE.std(ddof=1) if len(df) > 1 else 0.0,
               "MAE": df.MAE.mean(), "MAPE": df.MAPE.mean(), "n_seeds": len(seeds),
               "batteryml_published_RMSE": PUBLISHED[name]}
        rows.append(row)
        print(f"{name:30s} RMSE {row['RMSE']:7.1f} (+-{row['RMSE_std']:.1f})  published {PUBLISHED[name]}")

    rows.append({"model": "MLP (QdLinear)", "features": "QdLinear", "RMSE": np.nan,
                 "batteryml_published_RMSE": PUBLISHED["MLP (QdLinear)"],
                 "note": "not re-run (BatteryML torch MLP, 5000 epochs); see our MLP in exp1"})
    out = repo_path(cfg["output"]["results_dir"])
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "tables" / "batteryml_reproduction.csv", index=False)
    pd.DataFrame(preds).to_csv(out / "predictions" / "batteryml_baselines.csv", index=False)


if __name__ == "__main__":
    main()
