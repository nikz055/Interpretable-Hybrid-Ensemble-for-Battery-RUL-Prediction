"""Step 10: confirm SHAP attributions by retraining without / with only each
mechanism group (training-cell grouped CV; the test set is reported but not
used for any conclusion about importance).

Configurations per horizon T and model (PLSR, RF, XGB):
  full                     all Set C features
  drop:<group>             all features except the group
  only:<group>             only the group's features
  degradation_only         all features except charging-protocol proxies
  protocol_metadata_only   C1, switch SOC, C2 of the charging protocol (no cycling data)

Hyper-parameters are those tuned on the full feature set for that horizon
(not re-tuned per configuration; PLSR components are capped at #features).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from batteryrul.config import load_config, repo_path  # noqa: E402
from batteryrul.ensemble import oof_predict  # noqa: E402
from batteryrul.features.groups import GROUP_LABEL, MECHANISM_GROUPS, PROTOCOL_GROUP, groups_for  # noqa: E402
from batteryrul.metrics import regression_metrics  # noqa: E402
from batteryrul.models import make_estimator  # noqa: E402
from batteryrul.splits import fold_indices, read_manifests  # noqa: E402

warnings.filterwarnings("ignore")
MODELS = ["plsr", "rf", "xgb"]


def evaluate(model, params, seed, Xtr, ytr, Xte, yte, splits, wq):
    p = dict(params)
    if model == "plsr":
        p["n_components"] = min(p["n_components"], Xtr.shape[1])
    est = make_estimator(model, p, seed, wq)
    oof = oof_predict(est, Xtr, ytr, splits)
    pred = est.fit(Xtr, ytr).predict(Xte)
    cv, te = regression_metrics(ytr, oof), regression_metrics(yte, pred)
    return {"cv_rmse": cv["RMSE"], "cv_mae": cv["MAE"], "test_rmse": te["RMSE"], "test_mae": te["MAE"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ecfg = cfg["experiments"]["early_warning"]
    res = repo_path(cfg["output"]["results_dir"])
    out, figs = res / "runs" / "early_warning", res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    train, test, folds = read_manifests(repo_path(cfg["split"]["manifest_dir"]))
    splits = [(trn, val) for _, trn, val in fold_indices(train, folds)]
    wq = cfg["preprocessing"]["winsor_quantiles"]
    fs = ecfg["feature_set"]
    proto = pd.read_csv(res / "tables" / "protocols.csv").set_index("cell_id")[["c1", "soc_switch", "c2"]]

    jobs = []
    for T in ecfg["horizons"]:
        df = pd.read_csv(res / "features" / f"features_{fs}_T{T}.csv", index_col="cell_id")
        cols = [c for c in df.columns if c not in ("partition", "cycle_life")]
        ytr, yte = df.loc[train, "cycle_life"].to_numpy(), df.loc[test, "cycle_life"].to_numpy()
        groups = groups_for(cols)
        best = json.loads((res / "runs" / f"T{T}_{fs}" / "best_params.json").read_text())
        configs = {"full": cols, "degradation_only": [c for j, c in enumerate(cols) if j not in groups[PROTOCOL_GROUP]]}
        for g, idx in groups.items():
            configs[f"drop:{g}"] = [c for j, c in enumerate(cols) if j not in idx]
            configs[f"only:{g}"] = [cols[j] for j in idx]
        feats = {name: (df.loc[train, c].to_numpy(float), df.loc[test, c].to_numpy(float)) for name, c in configs.items()}
        feats["protocol_metadata_only"] = (proto.loc[train].to_numpy(float), proto.loc[test].to_numpy(float))
        for name, (Xtr, Xte) in feats.items():
            for m in MODELS:
                for s in ([ecfg["ablation_seeds"][0]] if m == "plsr" else ecfg["ablation_seeds"]):
                    jobs.append(((T, name, m, s, Xtr.shape[1]), (m, best[m], s, Xtr, ytr, Xte, yte, splits, wq)))

    print(f"{len(jobs)} fits x {len(splits) + 1} ...")
    results = Parallel(n_jobs=max(1, (os.cpu_count() or 2) - 1))(delayed(evaluate)(*a) for _, a in jobs)
    raw = pd.DataFrame([{"T": k[0], "config": k[1], "model": k[2], "seed": k[3], "n_features": k[4], **r}
                        for (k, _), r in zip(jobs, results)])
    raw.to_csv(out / "group_ablation_raw.csv", index=False)

    s = raw.groupby(["T", "config", "model"])[["cv_rmse", "cv_mae", "test_rmse", "test_mae", "n_features"]].mean().reset_index()
    full = s[s.config == "full"].set_index(["T", "model"])
    for k in ("cv_rmse", "cv_mae", "test_rmse", "test_mae"):
        s[f"d_{k}"] = s[k] - [full.loc[(t, m), k] for t, m in zip(s["T"], s.model)]
    s.to_csv(out / "group_ablation_summary.csv", index=False)
    # consensus over models (mean of per-model deltas)
    cons = s.groupby(["T", "config"])[["cv_rmse", "cv_mae", "d_cv_rmse", "d_cv_mae", "test_rmse", "d_test_rmse"]].mean().reset_index()
    cons.to_csv(out / "group_ablation_consensus.csv", index=False)

    order = list(MECHANISM_GROUPS)
    Ts = ecfg["horizons"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    w = 0.8 / len(Ts)
    for i, T in enumerate(Ts):
        c = cons[cons["T"] == T].set_index("config")
        x = np.arange(len(order)) + (i - (len(Ts) - 1) / 2) * w
        axes[0].bar(x, [c.loc[f"drop:{g}", "d_cv_mae"] if f"drop:{g}" in c.index else np.nan for g in order], w, label=f"T={T}")
        axes[1].bar(x, [c.loc[f"only:{g}", "cv_mae"] if f"only:{g}" in c.index else np.nan for g in order], w, label=f"T={T}")
    for ax in axes:
        ax.set_xticks(np.arange(len(order)), [GROUP_LABEL[g].replace(" (", "\n(") for g in order], fontsize=7, rotation=20)
        ax.legend()
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("Δ CV MAE when the group is removed (cycles)")
    axes[0].set_title("Drop-group: loss of accuracy without the group")
    axes[1].set_ylabel("CV MAE using only the group (cycles)")
    axes[1].set_title("Only-group: accuracy from the group alone")
    for T in Ts:
        axes[1].axhline(cons[(cons["T"] == T) & (cons.config == "full")].cv_mae.iloc[0], ls=":", lw=0.8, color="grey")
    fig.tight_layout()
    fig.savefig(figs / "early_group_ablation.png", dpi=150)
    plt.close(fig)

    print(cons[cons.config.isin(["full", "degradation_only", "protocol_metadata_only"]) | cons.config.str.startswith("only:")]
          .round(1).to_string(index=False))


if __name__ == "__main__":
    main()
