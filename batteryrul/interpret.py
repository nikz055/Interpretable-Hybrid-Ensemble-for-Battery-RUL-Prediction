"""SHAP helpers. All explanations are computed on held-out cells only; the
KernelSHAP background is a k-means summary of *training* cells.

Units: SHAP values are expressed in ln(cycle life). A value of +0.1 means the
feature moved this cell's prediction up by a factor of exp(0.1) = 1.105 relative
to the explainer's base value. Tree models are explained exactly with
TreeSHAP; the MLP and the stacked ensemble with model-agnostic KernelSHAP.
"""
from __future__ import annotations

import numpy as np
import shap

# Expected sign of each feature's effect on cycle life from degradation reasoning
# (+: larger value -> longer life). "?" = no clear physical prior.
EXPECTED_SIGN = {
    "dq_var_log": "-", "dq_min_log": "-", "dq_kurt_log": "?", "dq_skew_log": "?",
    "dq_rms": "-", "dq_mean": "+", "dv_q50": "+", "dv_dis_avg": "+",
    "qd_retention": "+", "qd_slope": "+", "qd_slope_late": "+", "qd_slope_last10": "+",
    "qd_curvature": "+", "qd_max_minus_ref": "?", "qd_max_minus_early": "?",
    "qd_ref": "?", "qd_early": "?", "qd_mean": "?", "qc_ref": "?", "qd_std": "-", "qd_intercept": "?",
    "ce_mean": "+", "ce_std": "-", "ce_slope": "?",
    "v_dis_avg_mean": "+", "v_dis_avg_std": "-", "e_dis_mean": "+",
    "v_q10_last": "+", "v_q50_last": "+", "v_q90_last": "+",
    "i_chg_mean": "-", "i_chg_std": "?", "i_max": "-",
    "t_chg_mean": "+", "t_dis_mean": "+", "t_cv_mean": "?", "t_early_chg_log": "?",
    "temp_mean": "-", "temp_max": "-", "temp_range": "-", "temp_std": "?", "temp_avg_log": "-",
    "temp_slope": "-", "ir_ref": "-", "ir_min": "-", "ir_change": "-", "ir_slope": "-",
    "dqdv_peak_ratio": "+", "dqdv_peak_shift": "?", "t_dis_slope": "+", "t_chg_slope": "?",
    "t_cv_slope": "?", "qd_resid_std_last10": "-",
}


def _parts(ttr):
    """Split a fitted TransformedTargetRegressor into (preprocess, model, ln-scale, ln-shift)."""
    pipe = ttr.regressor_
    pre = pipe[:-1]
    model = pipe[-1]
    z = ttr.transformer_.named_steps["z"]
    return pre, model, float(z.scale_[0]), float(z.mean_[0])


def tree_shap_log(ttr, X_raw: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact TreeSHAP for RF / XGB, rescaled from z-scored ln(life) to ln(life)."""
    pre, model, scale, shift = _parts(ttr)
    Xt = pre.transform(X_raw)
    if hasattr(model, "get_booster"):
        # XGBoost's built-in exact TreeSHAP (shap<0.47 cannot parse xgboost>=3 models)
        import xgboost as xgb
        contrib = model.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)
        sv, base = contrib[:, :-1], float(contrib[0, -1])
    else:
        expl = shap.TreeExplainer(model)
        sv = np.asarray(expl.shap_values(Xt))
        base = float(np.ravel(expl.expected_value)[0])
    return sv * scale, base * scale + shift


def kernel_shap_log(predict_cycles, X_bg_raw: np.ndarray, X_raw: np.ndarray, n_background: int,
                    nsamples: int, seed: int = 0) -> tuple[np.ndarray, float]:
    """KernelSHAP of ln(predicted cycle life) for any model taking raw features."""
    np.random.seed(seed)                       # KernelSHAP samples coalitions with np.random
    bg = shap.kmeans(X_bg_raw, min(n_background, len(X_bg_raw)))

    def f(X):
        return np.log(np.clip(predict_cycles(X), 1.0, None))

    expl = shap.KernelExplainer(f, bg)
    sv = np.asarray(expl.shap_values(X_raw, nsamples=nsamples, silent=True, l1_reg=False))
    return sv, float(expl.expected_value)


def sign_consistency(feature: str, x: np.ndarray, sv: np.ndarray) -> dict:
    """Direction of the learned association = sign of Spearman(feature value, SHAP)."""
    from scipy.stats import spearmanr
    ok = np.isfinite(x)
    rho = spearmanr(x[ok], sv[ok]).correlation if ok.sum() > 3 and np.std(x[ok]) > 0 else np.nan
    learned = "+" if rho > 0.3 else "-" if rho < -0.3 else "~"
    expected = EXPECTED_SIGN.get(feature, "?")
    if expected == "?" or learned == "~":
        verdict = "not assessed" if expected == "?" else "weak/non-monotone"
    else:
        verdict = "consistent" if learned == expected else "INCONSISTENT"
    return {"spearman_value_vs_shap": float(rho), "learned_sign": learned,
            "expected_sign": expected, "plausibility": verdict}
