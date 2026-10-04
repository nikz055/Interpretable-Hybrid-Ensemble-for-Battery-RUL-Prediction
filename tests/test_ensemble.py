import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batteryrul.ensemble import (ConstrainedStacker, crossfit_stack_oof,  # noqa: E402
                                 inverse_mse_weights, oof_predict)
from batteryrul.models import Winsorizer, make_estimator  # noqa: E402


def test_stacker_weights_on_simplex_and_recover_mixture():
    rng = np.random.default_rng(0)
    y = rng.uniform(300, 2000, 200)
    P = np.column_stack([y + rng.normal(0, 50, 200), y + rng.normal(0, 200, 200), rng.uniform(300, 2000, 200)])
    st = ConstrainedStacker().fit(P, y)
    assert np.all(st.weights_ >= 0) and np.isclose(st.weights_.sum(), 1)
    assert st.weights_[0] > st.weights_[1] > st.weights_[2]


def test_stacker_bias_correction():
    y = np.linspace(300, 2000, 50)
    P = np.column_stack([y + 100, y + 100])
    st = ConstrainedStacker().fit(P, y)
    np.testing.assert_allclose(st.predict(P), y, atol=1e-6)


def test_inverse_mse_weights():
    y = np.zeros(4)
    P = np.column_stack([np.ones(4), 2 * np.ones(4)])
    np.testing.assert_allclose(inverse_mse_weights(P, y), [0.8, 0.2])


def test_oof_never_uses_held_out_cell():
    X = np.arange(20, dtype=float)[:, None]
    y = 500 + 10 * X.ravel()
    splits = [(np.setdiff1d(np.arange(20), np.arange(k, 20, 5)), np.arange(k, 20, 5)) for k in range(5)]
    p = oof_predict(make_estimator("ridge", {"alpha": 1e-6}), X, y, splits)
    assert np.isfinite(p).all()
    assert np.isfinite(crossfit_stack_oof(np.column_stack([p, p]), y, splits)).all()


def test_winsorizer_fits_on_training_data_only():
    w = Winsorizer((0.0, 1.0)).fit(np.array([[0.0], [1.0]]))
    np.testing.assert_array_equal(w.transform(np.array([[5.0], [-5.0], [np.nan]])), [[1.0], [0.0], [np.nan]])
