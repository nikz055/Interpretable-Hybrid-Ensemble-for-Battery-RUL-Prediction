"""Step 3: build feature matrices for every feature set and horizon.

Outputs:
    results/features/features_{A,B,C}_T{20,50,100}.csv   (cell_id, partition, cycle_life, features...)
    results/tables/feature_dictionary.csv, docs/FEATURE_DICTIONARY.md
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batteryrul.cache import load_all  # noqa: E402
from batteryrul.config import load_config, repo_path  # noqa: E402
from batteryrul.features.build import FEATURE_SETS, build_features, labels_for  # noqa: E402
from batteryrul.features.dictionary import feature_dictionary, to_markdown  # noqa: E402
from batteryrul.splits import read_manifests  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    train, test, _ = read_manifests(repo_path(cfg["split"]["manifest_dir"]))
    cells = train + test
    caches = load_all(repo_path(cfg["data"]["cache_dir"]), cells)
    out = repo_path(cfg["output"]["results_dir"])
    (out / "features").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    horizons = sorted(set(cfg["experiments"]["horizon_ablation"]["horizons"]) | {cfg["experiments"]["main"]["horizon"]})
    for T in horizons:
        for fs in FEATURE_SETS:
            X = build_features(caches, cells, fs, T)
            X.insert(0, "cycle_life", labels_for(caches, cells))
            X.insert(0, "partition", ["train" if c in train else "test" for c in cells])
            X.to_csv(out / "features" / f"features_{fs}_T{T}.csv")
            print(f"T={T:3d} set {fs}: {X.shape[1] - 2} features, {int(X.isna().sum().sum())} NaN")

    d = feature_dictionary()
    d.to_csv(out / "tables" / "feature_dictionary.csv", index=False)
    (repo_path("docs") / "FEATURE_DICTIONARY.md").write_text(to_markdown(d), encoding="utf-8")


if __name__ == "__main__":
    main()
