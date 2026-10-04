import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batteryrul.labels import batteryml_rul_label  # noqa: E402
from batteryrul.splits import (MATR1_TEST, MATR1_TRAIN, LeakageError,  # noqa: E402
                               assert_disjoint, fold_indices, make_cv_folds,
                               read_manifests)


def test_batteryml_split_sizes_and_disjoint():
    assert len(MATR1_TRAIN) == 41
    assert len(MATR1_TEST) == 42
    assert "b2c1" not in MATR1_TEST
    assert_disjoint(train=MATR1_TRAIN, test=MATR1_TEST)


def test_assert_disjoint_detects_overlap():
    with pytest.raises(LeakageError):
        assert_disjoint(train=["b1c1", "b1c2"], test=["b1c2"])


def test_label_matches_batteryml_convention():
    # first sub-threshold capacity at 0-based index 150 -> label 152
    qd = [1.08] * 150 + [0.879] + [0.85] * 10
    assert batteryml_rul_label(qd, 1.1) == 152.0
    # never reaches EOL -> padded: len + 2
    assert batteryml_rul_label([1.0] * 300, 1.1) == 302.0
    # too short -> excluded
    assert math.isnan(batteryml_rul_label([1.0] * 50 + [0.8], 1.1))


def test_cv_folds_are_grouped_and_cover_train():
    labels = pd.Series({c: 300 + 10 * i for i, c in enumerate(MATR1_TRAIN)})
    folds = make_cv_folds(MATR1_TRAIN, labels, 5, 0)
    assert set(folds.cell_id) == set(MATR1_TRAIN)
    assert folds.cell_id.is_unique
    seen = []
    for _, trn, val in fold_indices(MATR1_TRAIN, folds):
        assert not set(trn) & set(val)
        seen.extend(val)
    assert sorted(seen) == list(range(len(MATR1_TRAIN)))


@pytest.mark.skipif(not (ROOT / "splits" / "train_cells.csv").exists(), reason="manifests not generated")
def test_committed_manifests_are_leakage_free():
    train, test, folds = read_manifests(ROOT / "splits")
    assert_disjoint(train=train, test=test)
    assert set(train) <= set(MATR1_TRAIN) and set(test) <= set(MATR1_TEST)
