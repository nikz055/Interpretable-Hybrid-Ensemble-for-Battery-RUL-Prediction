"""Mechanism groups for early-life features (Feature Sets A and C).

Grouping serves two purposes: (1) correlated features that measure the same
phenomenon (e.g. ΔQ(V) variance and mean-voltage drift) are evaluated jointly,
which gives stable importance estimates; (2) *degradation symptoms* are kept
apart from *charging-protocol proxies*. The charging protocol is chosen before
cycling starts, so it can predict life but it is not a failure symptom.

Group-level SHAP uses additivity: phi_G(x) = sum_{j in G} phi_j(x).
"""
from __future__ import annotations

MECHANISM_GROUPS: dict[str, list[str]] = {
    # loss of capacity and its trajectory (discharge time at fixed 4C is a capacity proxy)
    "capacity_fade": ["qd_ref", "qc_ref", "qd_mean", "qd_std", "qd_retention", "qd_max_minus_ref",
                      "qd_slope", "qd_slope_late", "qd_curvature", "qd_slope_last10",
                      "qd_resid_std_last10", "t_dis_mean", "t_dis_slope",
                      "qd_early", "qd_max_minus_early", "qd_intercept"],
    # change in the shape of the discharge Q(V) curve (Severson ΔQ(V), dQ/dV)
    "discharge_curve_shape": ["dq_var_log", "dq_min_log", "dq_mean", "dq_rms", "dq_skew_log",
                              "dq_kurt_log", "dqdv_peak_shift", "dqdv_peak_ratio"],
    # discharge voltage level and its drift (polarisation / overpotential growth)
    "voltage_polarisation": ["v_dis_avg_mean", "v_dis_avg_std", "e_dis_mean", "v_q10_last",
                             "v_q50_last", "v_q90_last", "dv_q50", "dv_dis_avg"],
    "internal_resistance": ["ir_ref", "ir_min", "ir_change", "ir_slope"],
    "thermal": ["temp_mean", "temp_max", "temp_range", "temp_std", "temp_slope", "temp_avg_log"],
    "coulombic_efficiency": ["ce_mean", "ce_std", "ce_slope"],
    # dominated by the (pre-chosen) fast-charging protocol, not by degradation
    "charging_protocol_proxy": ["i_chg_mean", "i_chg_std", "i_max", "t_chg_mean", "t_cv_mean",
                                "t_chg_slope", "t_cv_slope", "t_early_chg_log"],
}

PROTOCOL_GROUP = "charging_protocol_proxy"
DEGRADATION_GROUPS = [g for g in MECHANISM_GROUPS if g != PROTOCOL_GROUP]

GROUP_LABEL = {
    "capacity_fade": "Capacity fade",
    "discharge_curve_shape": "Discharge-curve shape (ΔQ(V), dQ/dV)",
    "voltage_polarisation": "Voltage / polarisation",
    "internal_resistance": "Internal resistance",
    "thermal": "Thermal",
    "coulombic_efficiency": "Coulombic efficiency",
    "charging_protocol_proxy": "Charging-protocol proxy",
}


def feature_group(feature: str) -> str:
    for g, feats in MECHANISM_GROUPS.items():
        if feature in feats:
            return g
    raise KeyError(f"feature '{feature}' has no mechanism group")


def groups_for(columns) -> dict[str, list[int]]:
    """Column indices per group, restricted to groups present in ``columns``."""
    out: dict[str, list[int]] = {}
    for j, c in enumerate(columns):
        out.setdefault(feature_group(c), []).append(j)
    return out
