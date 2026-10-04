"""Battery-level partitions for MATR-1 and programmatic leakage checks.

Protocol (Option B of the study design):
  * Locked test set  = BatteryML ``MATRPrimaryTestTrainTestSplitter`` test cells
    (42 cells; BatteryML drops the outlier b2c1). Never used for tuning,
    feature selection, ensemble weights or SHAP background.
  * Training set     = BatteryML train cells (41 cells).
  * Validation       = grouped K-fold over training cells. Each cell is its own
    group (one feature vector per cell), so a fold never shares a cell with
    another fold. The fold manifest is written once and reused everywhere.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Copied verbatim from batteryml/train_test_split/MATR_split.py
MATR1_TRAIN = [
    'b1c1', 'b1c3', 'b1c5', 'b1c7', 'b1c11', 'b1c15',
    'b1c17', 'b1c19', 'b1c21', 'b1c24', 'b1c26', 'b1c28',
    'b1c30', 'b1c32', 'b1c34', 'b1c36', 'b1c38', 'b1c40',
    'b1c42', 'b1c44', 'b2c0', 'b2c2', 'b2c4', 'b2c6',
    'b2c11', 'b2c13', 'b2c17', 'b2c19', 'b2c21', 'b2c23',
    'b2c25', 'b2c27', 'b2c29', 'b2c31', 'b2c33', 'b2c35',
    'b2c37', 'b2c39', 'b2c41', 'b2c43', 'b2c45',
]
_MATR1_TEST_WITH_OUTLIER = [
    'b1c0', 'b1c2', 'b1c4', 'b1c6', 'b1c9', 'b1c14',
    'b1c16', 'b1c18', 'b1c20', 'b1c23', 'b1c25', 'b1c27',
    'b1c29', 'b1c31', 'b1c33', 'b1c35', 'b1c37', 'b1c39',
    'b1c41', 'b1c43', 'b1c45', 'b2c1', 'b2c3', 'b2c5',
    'b2c10', 'b2c12', 'b2c14', 'b2c18', 'b2c20', 'b2c22',
    'b2c24', 'b2c26', 'b2c28', 'b2c30', 'b2c32', 'b2c34',
    'b2c36', 'b2c38', 'b2c40', 'b2c42', 'b2c44', 'b2c46',
    'b2c47',
]
MATR1_TEST = [c for c in _MATR1_TEST_WITH_OUTLIER if c != 'b2c1']
MATR1_CELLS = MATR1_TRAIN + MATR1_TEST


class LeakageError(AssertionError):
    pass


def assert_disjoint(**partitions) -> None:
    """Raise if any cell id appears in more than one named partition."""
    names = list(partitions)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(partitions[a]) & set(partitions[b])
            if overlap:
                raise LeakageError(f"Cells shared by {a} and {b}: {sorted(overlap)}")


def make_cv_folds(train_cells, labels: pd.Series, n_folds: int, seed: int) -> pd.DataFrame:
    """Grouped, label-stratified K-fold assignment over training cells.

    Cells are sorted by cycle life and dealt into folds in shuffled blocks so
    every fold spans short, medium and long-lived cells.
    """
    rng = np.random.default_rng(seed)
    order = labels.loc[train_cells].sort_values().index.to_list()
    fold = {}
    for start in range(0, len(order), n_folds):
        block = order[start:start + n_folds]
        ids = rng.permutation(n_folds)[:len(block)]
        for cell, k in zip(block, ids):
            fold[cell] = int(k)
    df = pd.DataFrame({"cell_id": list(fold), "fold": list(fold.values())})
    return df.sort_values(["fold", "cell_id"]).reset_index(drop=True)


def fold_indices(cells: list[str], folds: pd.DataFrame):
    """Yield (fold, train_idx, val_idx) positional indices into ``cells``."""
    fmap = folds.set_index("cell_id")["fold"]
    assign = np.array([fmap[c] for c in cells])
    for k in sorted(np.unique(assign)):
        val = np.where(assign == k)[0]
        trn = np.where(assign != k)[0]
        assert_disjoint(fold_train=[cells[i] for i in trn], fold_val=[cells[i] for i in val])
        yield int(k), trn, val


def write_manifests(out_dir: Path, train, test, folds: pd.DataFrame, labels: pd.Series) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_disjoint(train=train, test=test)
    pd.DataFrame({"cell_id": train, "cycle_life": labels.loc[train].values}).to_csv(
        out_dir / "train_cells.csv", index=False)
    pd.DataFrame({"cell_id": test, "cycle_life": labels.loc[test].values}).to_csv(
        out_dir / "test_cells.csv", index=False)
    folds.to_csv(out_dir / "cv_folds.csv", index=False)


def read_manifests(split_dir: Path):
    split_dir = Path(split_dir)
    train = pd.read_csv(split_dir / "train_cells.csv")["cell_id"].tolist()
    test = pd.read_csv(split_dir / "test_cells.csv")["cell_id"].tolist()
    folds = pd.read_csv(split_dir / "cv_folds.csv")
    assert_disjoint(train=train, test=test)
    if set(folds["cell_id"]) != set(train):
        raise LeakageError("CV folds must cover exactly the training cells")
    return train, test, folds
