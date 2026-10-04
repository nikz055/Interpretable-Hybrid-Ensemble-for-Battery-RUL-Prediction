"""Step 4: Experiments 1-4 (main benchmark, feature ablation, ensemble ablation, horizon ablation).

Reads the committed feature CSVs (results/features), so this step can be
re-run without the raw data.

    python scripts/04_run_experiments.py            # all configurations
    python scripts/04_run_experiments.py --skip-existing

Writes results/runs/T{T}_{set}/... and results/runs/main.json, which records
the main feature set chosen by training-CV RMSE of the final ensemble.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from batteryrul.config import load_config, repo_path  # noqa: E402
from batteryrul.experiment import run_configuration  # noqa: E402
from batteryrul.splits import assert_disjoint, read_manifests  # noqa: E402


def load_features(results_dir: Path, fs: str, T: int, train, test):
    df = pd.read_csv(results_dir / "features" / f"features_{fs}_T{T}.csv", index_col="cell_id")
    X = df.drop(columns=["partition", "cycle_life"])
    tr, te = X.loc[train], X.loc[test]
    assert_disjoint(train=tr.index, test=te.index)
    assert (df.loc[train, "partition"] == "train").all() and (df.loc[test, "partition"] == "test").all()
    return tr, df.loc[train, "cycle_life"].to_numpy(), te, df.loc[test, "cycle_life"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = repo_path(cfg["output"]["results_dir"])
    runs = res / "runs"
    train, test, folds = read_manifests(repo_path(cfg["split"]["manifest_dir"]))

    def run(fs, T, save_models):
        tag = f"T{T}_{fs}"
        out = runs / tag
        if args.skip_existing and (out / "selection.json").exists():
            print(f"skip {tag}")
            return json.loads((out / "selection.json").read_text())
        Xtr, ytr, Xte, yte = load_features(res, fs, T, train, test)
        print(f"== {tag}: {Xtr.shape[1]} features, {len(Xtr)} train / {len(Xte)} test cells")
        return run_configuration(tag, Xtr, ytr, Xte, yte, folds, cfg, out, save_models=save_models)

    T_main = cfg["experiments"]["main"]["horizon"]
    sel = {fs: run(fs, T_main, save_models=True) for fs in cfg["experiments"]["main"]["feature_sets"]}

    # main feature set: lowest training-CV RMSE of each set's final ensemble
    cv = {fs: s["cv_rmse"][s["final_ensemble_by_cv"]] for fs, s in sel.items()}
    main_fs = min(cv, key=cv.get)
    main = {"horizon": T_main, "feature_set": main_fs, "tag": f"T{T_main}_{main_fs}",
            "final_ensemble": sel[main_fs]["final_ensemble_by_cv"],
            "best_individual": sel[main_fs]["best_individual_by_cv"],
            "selection_rule": "lowest grouped-CV RMSE (training cells) of the final ensemble",
            "cv_rmse_final_ensemble_by_set": cv}
    (runs / "main.json").write_text(json.dumps(main, indent=2))
    print(f"Main configuration: {main}")

    for T in cfg["experiments"]["horizon_ablation"]["horizons"]:
        if T != T_main:
            run(main_fs, T, save_models=False)


if __name__ == "__main__":
    main()
