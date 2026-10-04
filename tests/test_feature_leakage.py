"""Features must depend only on cycles 1..T: corrupting later cycles or the label
must leave every feature unchanged."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batteryrul.cache import QDLIN_POINTS, SCALAR_SIGNALS  # noqa: E402
from batteryrul.features.batteryml_ref import qdlinear_feature, severson_features  # noqa: E402
from batteryrul.features.physical import set_b, set_c  # noqa: E402
from batteryrul.features.window import Window  # noqa: E402


def synthetic_cache(n=100, seed=0):
    rng = np.random.default_rng(seed)
    cache = {k: 1.0 + 0.01 * rng.standard_normal(n) for k in SCALAR_SIGNALS}
    cache["qd"] = np.linspace(1.08, 1.05, n) + 1e-4 * rng.standard_normal(n)
    v = np.linspace(0, 1, QDLIN_POINTS)
    cache["qdlin"] = np.stack([v * (1.07 - 1e-4 * i) for i in range(n)]) + 1e-5 * rng.standard_normal((n, QDLIN_POINTS))
    cache.update(cell_id="syn", nominal_capacity_ah=1.1, label=800.0, legacy_label=820.0,
                 qd_full=np.linspace(1.08, 0.85, 900), n_cycles_recorded=900)
    return cache


def corrupt_after(cache, T):
    bad = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in cache.items()}
    for k in SCALAR_SIGNALS:
        bad[k][T:] = 1e6
    bad["qdlin"][T:] = -5.0
    bad["label"], bad["legacy_label"] = 1.0, 1.0
    bad["qd_full"] = bad["qd_full"] * 0
    return bad


FEATURE_FUNCS = [severson_features, set_b, set_c]


@pytest.mark.parametrize("T", [20, 50, 100])
def test_features_ignore_future_cycles(T):
    cache = synthetic_cache(n=100)
    for fn in FEATURE_FUNCS:
        a = fn(Window(cache, T))
        b = fn(Window(corrupt_after(cache, T), T))
        assert a.keys() == b.keys()
        for k in a:
            assert a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k])), (fn.__name__, k)


def test_sparse_window_uses_only_retained_cycles():
    cache = synthetic_cache(n=100)
    keep = np.zeros(100, bool)
    keep[::5] = True
    bad = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in cache.items()}
    for k in SCALAR_SIGNALS:
        bad[k][~keep] = np.nan
    bad["qdlin"][~keep] = np.nan
    for fn in FEATURE_FUNCS:
        a, b = fn(Window(cache, 100, keep)), fn(Window(bad, 100, keep))
        for k in a:
            assert a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k])), (fn.__name__, k)


def test_qdlinear_ignores_future_cycles():
    cache = synthetic_cache(n=100)
    np.testing.assert_array_equal(qdlinear_feature(Window(cache, 99)),
                                  qdlinear_feature(Window(corrupt_after(cache, 99), 99)))


def test_window_refuses_label_access():
    win = Window(synthetic_cache(), 100)
    for key in ("label", "qd_full", "n_cycles_recorded"):
        with pytest.raises(KeyError):
            win[key]


def test_every_feature_has_exactly_one_mechanism_group():
    from batteryrul.features.dictionary import feature_dictionary
    from batteryrul.features.groups import MECHANISM_GROUPS
    members = [f for feats in MECHANISM_GROUPS.values() for f in feats]
    assert len(members) == len(set(members)), "a feature is listed in two groups"
    missing = set(feature_dictionary().feature) - set(members)
    assert not missing, missing
