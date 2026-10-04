"""Step 1: write battery-level split manifests and a label table.

Outputs (committed to git so every run uses identical partitions):
    splits/train_cells.csv   41 BatteryML MATR-1 training cells + cycle life
    splits/test_cells.csv    42 BatteryML MATR-1 test cells (locked)
    splits/cv_folds.csv      grouped 5-fold assignment of the training cells
    results/tables/labels.csv  BatteryML label vs. legacy notebook label
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from batteryrul.cache import load_all  # noqa: E402
from batteryrul.config import load_config, repo_path  # noqa: E402
from batteryrul.splits import (MATR1_TEST, MATR1_TRAIN, assert_disjoint,  # noqa: E402
                               make_cv_folds, write_manifests)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    caches = load_all(repo_path(cfg["data"]["cache_dir"]), MATR1_TRAIN + MATR1_TEST)
    labels = pd.Series({c: caches[c]["label"] for c in caches}, name="cycle_life")
    legacy = pd.Series({c: caches[c]["legacy_label"] for c in caches})

    dropped = labels[labels.isna()].index.tolist()
    if dropped:
        print(f"BatteryML label excludes {dropped} (label <= 100 cycles)")
    train = [c for c in MATR1_TRAIN if c not in dropped]
    test = [c for c in MATR1_TEST if c not in dropped]
    assert_disjoint(train=train, test=test)

    folds = make_cv_folds(train, labels, cfg["split"]["n_folds"], cfg["split"]["fold_seed"])
    write_manifests(repo_path(cfg["split"]["manifest_dir"]), train, test, folds, labels)

    table = pd.DataFrame({
        "cell_id": labels.index,
        "partition": ["train" if c in train else "test" for c in labels.index],
        "batteryml_label": labels.values,
        "legacy_label_n_cycles": legacy.values,
        "n_cycles_recorded": [caches[c]["n_cycles_recorded"] for c in labels.index],
    })
    table["legacy_minus_batteryml"] = table["legacy_label_n_cycles"] - table["batteryml_label"]
    out = repo_path(cfg["output"]["results_dir"]) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "labels.csv", index=False)

    print(f"train={len(train)} test={len(test)} folds={folds['fold'].value_counts().sort_index().tolist()}")
    for part in ("train", "test"):
        y = table.loc[table.partition == part, "batteryml_label"]
        print(f"  {part}: cycle life {y.min():.0f}-{y.max():.0f}, mean {y.mean():.0f}, median {y.median():.0f}")
    d = table["legacy_minus_batteryml"]
    print(f"legacy label - BatteryML label: mean {d.mean():.1f}, max {d.max():.0f}, "
          f"cells differing: {(d != 0).sum()}/{len(d)}; corr {np.corrcoef(table.batteryml_label, table.legacy_label_n_cycles)[0, 1]:.4f}")


if __name__ == "__main__":
    main()
