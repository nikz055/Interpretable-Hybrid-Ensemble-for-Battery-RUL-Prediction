"""Feature dictionary: formula, unit, source signal, window, interpretation, leakage.

Notation: Qd_k / Qc_k = max discharge / charge capacity of cycle k (Ah);
Q_k(V) = Qdlin of cycle k (Ah on the 3.6->2.0 V grid); k is the 0-based cycle
index (k = 1 is cycle 2); T = horizon; W = cycles 1..T; trajectories are
cleaned (range check + Hampel + in-window interpolation) for Sets B/C.
"""
from __future__ import annotations

import pandas as pd

_NO_LEAK = "Uses cycles 1..T only (Window-enforced, unit-tested)"

# name, set, formula, unit, source, window, interpretation
_ROWS = [
    # ---------------- Set A: BatteryML / Severson reference ----------------
    ("dq_min_log", "A", "log10|min_V ΔQ(V)|, ΔQ = smooth(Q_{T-1}(V) - Q_9(V))", "log10 Ah", "Qdlin", "cycles 10, T",
     "Largest local capacity loss along the discharge curve"),
    ("dq_var_log", "A", "log10 Var_V[ΔQ(V)] (unbiased)", "log10 Ah²", "Qdlin", "cycles 10, T",
     "Non-uniformity of discharge-curve change; Severson's single best predictor"),
    ("dq_skew_log", "A", "log10 |skewness of ΔQ(V)|", "log10 (-)", "Qdlin", "cycles 10, T", "Asymmetry of curve change"),
    ("dq_kurt_log", "A", "log10 kurtosis of ΔQ(V)", "log10 (-)", "Qdlin", "cycles 10, T", "Peakedness of curve change"),
    ("qd_early", "A", "Qd at index 4 (BatteryML indexes Qd[2:99][2])", "Ah", "Qd", "cycle 5", "Initial usable capacity"),
    ("qd_max_minus_early", "A", "max(Qd_2..98) - qd_early", "Ah", "Qd", "cycles 3-99",
     "Early capacity rise (LFP break-in). Sensitive to logging spikes (b1c18)"),
    ("qd_slope", "A", "OLS slope of Qd_k vs k, k = 2..98", "Ah/cycle", "Qd", "cycles 3-99", "Early fade rate"),
    ("qd_intercept", "A", "OLS intercept of the same fit", "Ah", "Qd", "cycles 3-99", "Fitted initial capacity"),
    ("t_early_chg_log", "A", "log mean_{k=0..3} Σ dt[I<0] (BatteryML 'Average early charge time')", "log min", "I, t",
     "cycles 1-4", "Actually the discharge(+CV-hold) duration, mislabelled in BatteryML; protocol-dependent"),
    ("temp_avg_log", "A", "log mean_{k=2..T-1} mean_t T", "log °C", "T", "cycles 3-T",
     "Average cell temperature (BatteryML 'Integral of temperature')"),
    ("ir_min", "A", "min_{k=2..T-1} IR_k", "Ω", "IR", "cycles 3-T", "Lowest internal resistance (0 for batch-1 artefact cells)"),
    ("ir_change", "A", "IR_{T-1} - IR_2", "Ω", "IR", "cycles 3, T", "Early resistance growth"),
    # ---------------- Set B: core physical ----------------
    ("qd_ref", "B", "Qd_1 (cleaned)", "Ah", "Qd", "cycle 2", "Initial discharge capacity"),
    ("qc_ref", "B", "Qc_1 (cleaned)", "Ah", "Qc", "cycle 2", "Initial charge capacity"),
    ("qd_mean", "B", "mean_W Qd", "Ah", "Qd", "W", "Average early capacity"),
    ("qd_std", "B", "std_W Qd", "Ah", "Qd", "W", "Capacity variability over early life"),
    ("qd_retention", "B", "median(Qd_{T-5..T-1}) / median(Qd_{1..3})", "-", "Qd", "cycles 2-4, T-4..T",
     "Capacity retained at cycle T relative to cycle 2"),
    ("qd_max_minus_ref", "B", "max_{k≥1} Qd_k - Qd_1", "Ah", "Qd", "W", "Early capacity rise, spike-robust"),
    ("qd_slope", "B", "OLS slope of Qd vs k over k = 1..T-1, per 100 cycles", "Ah/100 cyc", "Qd", "W", "Average fade rate"),
    ("qd_slope_late", "B", "OLS slope over k = T/2..T-1", "Ah/100 cyc", "Qd", "second half of W", "Recent fade trend"),
    ("ce_mean", "B", "mean_W Qd_k/Qc_k", "-", "Qd, Qc", "W", "Coulombic efficiency (side-reaction losses)"),
    ("ce_std", "B", "std_W Qd_k/Qc_k", "-", "Qd, Qc", "W", "Early-cycle instability"),
    ("ce_slope", "B", "OLS slope of CE vs k", "1/100 cyc", "Qd, Qc", "W", "CE trend"),
    ("v_dis_avg_mean", "B", "mean_W (∫V dQ / ΔQ) over discharge points (I < -0.1 A)", "V", "V, Qd, I", "W",
     "Mean discharge voltage, falls with polarisation / resistance"),
    ("v_dis_avg_std", "B", "std_W of the same", "V", "V, Qd, I", "W", "Voltage-curve variability across cycles"),
    ("e_dis_mean", "B", "mean_W ∫V dQ (discharge)", "Wh", "V, Qd, I", "W", "Area under the discharge voltage curve (energy)"),
    ("v_q10_last", "B", "V where Q_{T-1}(V) = 0.1 Q_{T-1}(2.0 V)", "V", "Qdlin", "cycle T", "Voltage quantile, early discharge"),
    ("v_q50_last", "B", "V at 50 % of delivered capacity", "V", "Qdlin", "cycle T", "Mid-discharge voltage (plateau)"),
    ("v_q90_last", "B", "V at 90 % of delivered capacity", "V", "Qdlin", "cycle T", "End-of-discharge voltage"),
    ("i_chg_mean", "B", "mean_W time-weighted mean charge current (I > 0.05 A)", "A", "I, t", "W",
     "Charging aggressiveness (fast-charge protocol)"),
    ("i_chg_std", "B", "mean_W time-weighted std of charge current", "A", "I, t", "W", "Multi-step protocol current variability"),
    ("i_max", "B", "max_W |I|", "A", "I", "W", "Peak current (highest C-rate step)"),
    ("t_chg_mean", "B", "mean_W Σ dt[I > 0.05 A]", "min", "I, t", "W", "Charge duration"),
    ("t_dis_mean", "B", "mean_W Σ dt[I < -0.1 A]", "min", "I, t", "W", "Discharge duration (≈ capacity / 4C current)"),
    ("t_cv_mean", "B", "mean_W Σ dt[I > 0.05 A and V ≥ 3.595 V]", "min", "I, V, t", "W", "Constant-voltage duration"),
    ("temp_mean", "B", "mean_W mean_t T", "°C", "T", "W", "Average temperature"),
    ("temp_max", "B", "max_W max_t T (Hampel)", "°C", "T", "W", "Peak temperature"),
    ("temp_range", "B", "mean_W (max_t T - min_t T)", "°C", "T", "W", "Within-cycle self-heating"),
    ("temp_std", "B", "std_W mean_t T", "°C", "T", "W", "Temperature variability across cycles"),
    ("ir_ref", "B", "IR_1 (IR ≤ 0 treated as missing)", "Ω", "IR", "cycle 2", "Initial internal resistance"),
    ("ir_min", "B", "min_W IR (IR ≤ 0 missing)", "Ω", "IR", "W", "Lowest internal resistance"),
    ("ir_change", "B", "median(IR_{T-5..T-1}) - median(IR_{1..3})", "Ω", "IR", "W", "Early resistance growth"),
    # ---------------- Set C: trajectory additions ----------------
    ("qd_curvature", "C", "quadratic coefficient of Qd vs k/100 over k = 1..T-1", "Ah/(100 cyc)²", "Qd", "W",
     "Capacity-fade acceleration"),
    ("qd_slope_last10", "C", "OLS slope over the last 10 cycles", "Ah/100 cyc", "Qd", "cycles T-9..T", "Rolling (recent) fade rate"),
    ("qd_resid_std_last10", "C", "std of residuals of that fit", "Ah", "Qd", "cycles T-9..T", "Rolling capacity noise"),
    ("dq_var_log", "C", "as Set A (cycle 10 vs T)", "log10 Ah²", "Qdlin", "cycles 10, T", "Curve-shape change"),
    ("dq_min_log", "C", "as Set A", "log10 Ah", "Qdlin", "cycles 10, T", "Curve-shape change"),
    ("dq_mean", "C", "mean_V ΔQ(V)", "Ah", "Qdlin", "cycles 10, T", "Average curve shift"),
    ("dq_rms", "C", "sqrt(mean_V ΔQ(V)²)", "Ah", "Qdlin", "cycles 10, T", "Curve-shape distance (L2)"),
    ("dv_q50", "C", "v_q50(T) - v_q50(10)", "V", "Qdlin", "cycles 10, T", "Change in mid-discharge voltage"),
    ("dv_dis_avg", "C", "level(v_dis_avg, end) - level(v_dis_avg, ref)", "V", "V, Qd, I", "W", "Polarisation growth"),
    ("dqdv_peak_shift", "C", "V of max -dQ/dV at cycle T minus at cycle 10", "V", "Qdlin", "cycles 10, T",
     "Peak shift; at 4C peaks are polarisation-broadened, interpret relatively"),
    ("dqdv_peak_ratio", "C", "height of max -dQ/dV at T / at 10", "-", "Qdlin", "cycles 10, T", "Peak attenuation (loss of active material)"),
    ("t_chg_slope", "C", "OLS slope of charge duration", "min/100 cyc", "I, t", "W", "Charge duration trend"),
    ("t_dis_slope", "C", "OLS slope of discharge duration", "min/100 cyc", "I, t", "W", "Discharge duration trend (∝ fade)"),
    ("t_cv_slope", "C", "OLS slope of CV duration", "min/100 cyc", "I, V, t", "W", "CV duration trend (∝ resistance)"),
    ("ir_slope", "C", "OLS slope of cleaned IR", "Ω/100 cyc", "IR", "W", "Resistance growth rate"),
    ("temp_slope", "C", "OLS slope of cycle-mean temperature", "°C/100 cyc", "T", "W", "Thermal trend"),
]


def feature_dictionary() -> pd.DataFrame:
    df = pd.DataFrame(_ROWS, columns=["feature", "introduced_in_set", "formula", "unit", "source_signal",
                                      "observation_window", "physical_interpretation"])
    df["leakage_assessment"] = _NO_LEAK
    return df


def to_markdown(df: pd.DataFrame) -> str:
    lines = ["# Feature dictionary", "",
             "Set C = Set B ∪ the rows marked C. Windows are stated for T = 100; "
             "for other horizons replace 100 by T (Set A/C reference cycle becomes min(10, T/2)). "
             "Every feature uses only cycles 1..T, enforced by `batteryrul.features.window.Window` "
             "and checked by `tests/test_feature_leakage.py`.", "",
             "| Feature | Set | Formula | Unit | Source | Window | Physical interpretation |",
             "|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| `{r.feature}` | {r.introduced_in_set} | {r.formula} | {r.unit} | {r.source_signal} | "
                     f"{r.observation_window} | {r.physical_interpretation} |")
    return "\n".join(lines) + "\n"
