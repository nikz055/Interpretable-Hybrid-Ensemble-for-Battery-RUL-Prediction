"""Step 5: Experiment 5, robustness to corrupted test inputs (main configuration).

Re-fits the tuned base models of the main configuration on clean training cells
(same seeds, so the models are identical to Experiment 1), re-uses the stored
OOF-fitted ensemble weights, and scores clean vs. corrupted test inputs.

Outputs: results/runs/robustness/{robustness_raw.csv, robustness_summary.csv}
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from batteryrul.cache import load_all  # noqa: E402
from batteryrul.config import load_config, repo_path  # noqa: E402
from batteryrul.metrics import regression_metrics  # noqa: E402
from batteryrul.models import make_estimator  # noqa: E402
from batteryrul.robustness import mask_missing, noisy_features, sparse_features  # noqa: E402
from batteryrul.splits import read_manifests  # noqa: E402

warnings.filterwarnings("ignore")
MODELS = ["ridge", "plsr", "rf", "xgb", "mlp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rcfg = cfg["experiments"]["robustness"]
    res = repo_path(cfg["output"]["results_dir"])
    main_cfg = json.loads((res / "runs" / "main.json").read_text())
    run_dir = res / "runs" / main_cfg["tag"]
    fs, T = main_cfg["feature_set"], main_cfg["horizon"]
    best = json.loads((run_dir / "best_params.json").read_text())
    weights = pd.read_csv(run_dir / "ensemble_weights.csv")
    combiners = [main_cfg["final_ensemble"], "mean_" + "_".join(cfg["stacking"]["base_models"])]

    train, test, _ = read_manifests(repo_path(cfg["split"]["manifest_dir"]))
    df = pd.read_csv(res / "features" / f"features_{fs}_T{T}.csv", index_col="cell_id")
    cols = [c for c in df.columns if c not in ("partition", "cycle_life")]
    Xtr, ytr = df.loc[train, cols], df.loc[train, "cycle_life"].to_numpy()
    Xte, yte = df.loc[test, cols], df.loc[test, "cycle_life"].to_numpy()
    caches = load_all(repo_path(cfg["data"]["cache_dir"]), test)

    # corrupted test inputs: list of (condition, repeat, X)
    rng = np.random.default_rng(12345)
    conditions = [("clean", 0, Xte)]
    for rate in rcfg["missing_rates"]:
        for r in range(rcfg["repeats"]):
            conditions.append((f"missing_{int(rate * 100)}pct", r, mask_missing(Xte, rate, rng)))
    for r in range(rcfg["repeats"]):
        Xn = noisy_features(caches, test, fs, T, rcfg["capacity_noise_ah"], rcfg["voltage_noise_v"], rng)
        conditions.append(("gaussian_noise", r, Xn[cols]))
    conditions.append((f"sparse_every{rcfg['sparse_every']}", 0,
                       sparse_features(caches, test, fs, T, rcfg["sparse_every"])[cols]))
    print(f"{len(conditions)} test-input conditions, set {fs}, T={T}")

    rows = []
    for seed in cfg["seeds"]:
        fitted = {m: make_estimator(m, best[m], seed, cfg["preprocessing"]["winsor_quantiles"]).fit(Xtr.to_numpy(), ytr)
                  for m in MODELS}
        wseed = weights[weights.seed == seed].set_index("combiner")
        for cond, rep, X in conditions:
            P = {m: fitted[m].predict(X.to_numpy(float)) for m in MODELS}
            for comb in combiners:
                if comb.startswith("mean_"):
                    members = comb[len("mean_"):].split("_")
                    P[comb] = np.mean([P[m] for m in members], axis=0)
                else:
                    members = comb[len("stack_"):].split("_")
                    w = wseed.loc[comb]
                    P[comb] = sum(w[m] * P[m] for m in members) + w["intercept"]
            for m, p in P.items():
                met = regression_metrics(yte, p)
                rows.append({"seed": seed, "condition": cond, "repeat": rep, "model": m,
                             "RMSE": met["RMSE"], "MAE": met["MAE"]})
        print(f"  seed {seed} done")

    raw = pd.DataFrame(rows)
    clean = raw[raw.condition == "clean"].set_index(["seed", "model"])[["RMSE", "MAE"]]
    raw = raw.join(clean, on=["seed", "model"], rsuffix="_clean")
    raw["dRMSE"] = raw["RMSE"] - raw["RMSE_clean"]
    raw["dMAE"] = raw["MAE"] - raw["MAE_clean"]
    out = res / "runs" / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "robustness_raw.csv", index=False)
    summ = raw.groupby(["condition", "model"])[["RMSE", "MAE", "dRMSE", "dMAE"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(out / "robustness_summary.csv", index=False)
    print(summ.pivot(index="model", columns="condition", values="dRMSE_mean").round(1).to_string())


if __name__ == "__main__":
    main()
