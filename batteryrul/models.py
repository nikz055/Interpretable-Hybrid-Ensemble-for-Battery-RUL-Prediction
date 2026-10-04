"""Model zoo, leakage-safe preprocessing and grouped-CV hyper-parameter tuning.

Every estimator is wrapped as

    TransformedTargetRegressor(
        regressor = Pipeline([winsorise -> median-impute -> standardise -> model]),
        transformer = log -> z-score)            # BatteryML label transform

so that clipping bounds, imputation medians, scaling statistics and the label
transform are re-fitted on the training cells of every CV fold and of the final
fit; predictions come back in cycles.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBRegressor

from .splits import fold_indices, make_cv_folds

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each feature to quantiles learned on the training data (NaN-aware)."""

    def __init__(self, quantiles=(0.01, 0.99)):
        self.quantiles = quantiles

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.lo_ = np.nanquantile(X, self.quantiles[0], axis=0)
        self.hi_ = np.nanquantile(X, self.quantiles[1], axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lo_, self.hi_)


def target_transformer():
    return Pipeline([
        ("log", FunctionTransformer(np.log, np.exp, check_inverse=False)),
        ("z", StandardScaler()),
    ])


def wrap(model, winsor_q=(0.01, 0.99)) -> TransformedTargetRegressor:
    reg = Pipeline([
        ("winsor", Winsorizer(tuple(winsor_q))),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])
    return TransformedTargetRegressor(regressor=reg, transformer=target_transformer())


# base estimator, search space (keys are the inner model's parameters), search type
MODEL_SPACE = {
    "dummy": (DummyRegressor(), {}, "none"),
    "ridge": (Ridge(), {"alpha": list(np.logspace(-3, 3, 13))}, "grid"),
    "plsr": (PLSRegression(scale=False), {"n_components": [1, 2, 3, 4, 5, 6, 8]}, "grid"),
    "rf": (RandomForestRegressor(n_jobs=1), {
        "n_estimators": [200, 500],
        "max_depth": [None, 3, 5, 8],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": [0.33, 0.5, 1.0, "sqrt"],
    }, "random"),
    "xgb": (XGBRegressor(n_jobs=1, verbosity=0), {
        "max_depth": [2, 3, 4, 6],
        "learning_rate": [0.02, 0.05, 0.1, 0.2],
        "n_estimators": [100, 200, 400, 800],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.8, 1.0],
        "reg_lambda": [0.1, 1.0, 10.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "min_child_weight": [1, 3],
    }, "random"),
    "mlp": (MLPRegressor(early_stopping=True, validation_fraction=0.15, n_iter_no_change=50,
                         max_iter=3000, solver="adam"), {
        "hidden_layer_sizes": [(8,), (16,), (32,), (16, 8)],
        "alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0],
        "learning_rate_init": [1e-3, 3e-3, 1e-2],
    }, "random"),
}
STOCHASTIC = {"rf", "xgb", "mlp"}


def make_estimator(name: str, params: dict | None = None, seed: int | None = None,
                   winsor_q=(0.01, 0.99)) -> TransformedTargetRegressor:
    base, _, _ = MODEL_SPACE[name]
    model = clone(base)
    if params:
        model.set_params(**params)
    if seed is not None and name in STOCHASTIC:
        model.set_params(random_state=seed)
    return wrap(model, winsor_q)


def repeated_group_folds(cells, labels: pd.Series, n_folds: int, repeats: int, seed: int):
    """List of (train_idx, val_idx) for repeated grouped K-fold over training cells."""
    splits = []
    for r in range(repeats):
        folds = make_cv_folds(cells, labels, n_folds, seed + r)
        splits.extend((trn, val) for _, trn, val in fold_indices(cells, folds))
    return splits


def tune(name: str, X: np.ndarray, y: np.ndarray, cv_splits, n_iter: int, seed: int,
         winsor_q=(0.01, 0.99), n_jobs: int = -1) -> tuple[dict, float, pd.DataFrame]:
    """Select hyper-parameters by grouped CV on training cells only (RMSE in cycles)."""
    base, space, kind = MODEL_SPACE[name]
    est = make_estimator(name, seed=seed, winsor_q=winsor_q)
    if kind == "none":
        from sklearn.model_selection import cross_val_score
        s = cross_val_score(est, X, y, cv=cv_splits, scoring="neg_root_mean_squared_error")
        return {}, float(-s.mean()), pd.DataFrame()
    grid = {f"regressor__model__{k}": v for k, v in space.items()}
    n_total = len(ParameterGrid(grid))
    if kind == "grid" or n_total <= n_iter:
        search = GridSearchCV(est, grid, cv=cv_splits, scoring="neg_root_mean_squared_error",
                              n_jobs=n_jobs, error_score=np.nan)
    else:
        search = RandomizedSearchCV(est, grid, n_iter=n_iter, cv=cv_splits, random_state=seed,
                                    scoring="neg_root_mean_squared_error", n_jobs=n_jobs,
                                    error_score=np.nan)
    search.fit(X, y)
    best = {k.replace("regressor__model__", ""): _py(v) for k, v in search.best_params_.items()}
    res = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    return best, float(-search.best_score_), res


def _py(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v
