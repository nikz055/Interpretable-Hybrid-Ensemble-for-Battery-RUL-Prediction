"""Feature Set B (core physical) and Set C (B + trajectory / curve-shape).

All features are computed from a ``Window`` (cycles 1..T only). Per-cycle
trajectories are cleaned *inside the window* before aggregation:

* capacities outside [0.5, 1.3] x nominal are physically impossible for these
  1.1 Ah LFP cells and set to NaN (e.g. b1c18 shows a 2.88 Ah logging spike);
* internal resistance <= 0 is a logging artefact (batch 1, one cycle per cell)
  and set to NaN;
* a Hampel filter (+-5 cycles, 3 scaled MADs) removes remaining isolated spikes;
* NaNs are filled by linear interpolation over cycle index *within the window*
  (no backward fill from later cycles beyond T is possible, since they are not in the window).

Cycle indices are 0-based: index 1 = cycle 2 is the reference cycle (cycle 1
is a formation-like outlier in MATR), index T-1 = cycle T is the last cycle.
"""
from __future__ import annotations

import numpy as np

from ..cache import QDLIN_VOLTAGE
from .batteryml_ref import critical_cycles, delta_q_curve, median_smooth
from .window import Window

EPS = 1e-8
REF = 1          # cycle 2
LEVEL_SPAN = 5   # cycles averaged (median) to estimate a level at the window end


def hampel(x: np.ndarray, half_window: int = 5, n_sigmas: float = 3.0) -> np.ndarray:
    x = np.asarray(x, float).copy()
    n = len(x)
    for i in range(n):
        seg = x[max(0, i - half_window):min(n, i + half_window + 1)]
        seg = seg[np.isfinite(seg)]
        if len(seg) < 3 or not np.isfinite(x[i]):
            continue
        med = np.median(seg)
        mad = 1.4826 * np.median(np.abs(seg - med))
        if mad > 0 and abs(x[i] - med) > n_sigmas * mad:
            x[i] = np.nan
    return x


def _interp_nan(idx: np.ndarray, x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return x
    if ok.all():
        return x
    return np.interp(idx, idx[ok], x[ok])


def cleaned(win: Window, name: str) -> np.ndarray:
    x = win[name].astype(float).copy()
    if name in ("qd", "qc"):
        nom = win.nominal_capacity_ah
        x[(x < 0.5 * nom) | (x > 1.3 * nom)] = np.nan
    if name == "ir":
        x[x <= 0] = np.nan
    x = hampel(x)
    return _interp_nan(win.idx.astype(float), x)


def _slope(x_idx, y, quadratic=False):
    m = np.isfinite(y)
    if m.sum() < (3 if quadratic else 2):
        return np.nan
    deg = 2 if quadratic else 1
    coef = np.polyfit(x_idx[m] / 100.0, y[m], deg)      # per 100 cycles
    return float(coef[0])


def _level(win, x, a, b):
    """Median of a trajectory over retained cycles with a <= index < b."""
    pos = win.between(a, b)
    if len(pos) == 0:
        pos = np.array([win.pos(a)])
    return float(np.nanmedian(x[pos]))


def voltage_at_fraction(qdlin: np.ndarray, frac: float) -> float:
    """Discharge voltage at which a fraction ``frac`` of that cycle's capacity is delivered."""
    q = median_smooth(qdlin)
    ok = np.isfinite(q)
    if ok.sum() < 10:
        return np.nan
    q, v = np.maximum.accumulate(q[ok]), QDLIN_VOLTAGE[ok]
    return float(np.interp(frac * q[-1], q, v))


def dqdv_peak(qdlin: np.ndarray) -> tuple[float, float]:
    """(voltage, height) of the main -dQ/dV peak of one discharge curve.

    Qdlin is already interpolated on a uniform voltage grid, so the derivative
    is well defined; at 4C the peak is polarisation-broadened, so only its
    *relative* change across early cycles is used.
    """
    q = median_smooth(qdlin)
    if not np.isfinite(q).all():
        return np.nan, np.nan
    dqdv = -np.gradient(q, QDLIN_VOLTAGE)
    k = 25
    dqdv = np.convolve(dqdv, np.ones(k) / k, mode="same")
    inner = slice(50, len(dqdv) - 50)                  # ignore grid edges
    i = int(np.argmax(dqdv[inner])) + 50
    return float(QDLIN_VOLTAGE[i]), float(dqdv[i])


def core_features(win: Window) -> dict:
    T, x = win.T, win.idx.astype(float)
    qd, qc, ir = cleaned(win, "qd"), cleaned(win, "qc"), cleaned(win, "ir")
    ce = qd / qc
    f = {}
    # capacity
    f["qd_ref"] = float(qd[win.pos(REF)])
    f["qc_ref"] = float(qc[win.pos(REF)])
    f["qd_mean"] = float(np.nanmean(qd))
    f["qd_std"] = float(np.nanstd(qd))
    ref_level = _level(win, qd, REF, REF + 3)
    end_level = _level(win, qd, T - LEVEL_SPAN, T)
    f["qd_retention"] = end_level / ref_level
    f["qd_max_minus_ref"] = float(np.nanmax(qd[win.between(REF, T)]) - qd[win.pos(REF)])
    f["qd_slope"] = _slope(x[win.between(REF, T)], qd[win.between(REF, T)])
    late = win.between(T // 2, T)
    f["qd_slope_late"] = _slope(x[late], qd[late])
    # coulombic efficiency
    f["ce_mean"] = float(np.nanmean(ce))
    f["ce_std"] = float(np.nanstd(ce))
    f["ce_slope"] = _slope(x, ce)
    # voltage
    vavg = hampel(win["v_dis_avg"])
    f["v_dis_avg_mean"] = float(np.nanmean(vavg))
    f["v_dis_avg_std"] = float(np.nanstd(vavg))
    f["e_dis_mean"] = float(np.nanmean(hampel(win["e_dis"])))
    last_q = win.qdlin_at(T - 1)
    for frac in (0.1, 0.5, 0.9):
        f[f"v_q{int(frac * 100)}_last"] = voltage_at_fraction(last_q, frac)
    # current
    f["i_chg_mean"] = float(np.nanmean(win["i_chg_mean"]))
    f["i_chg_std"] = float(np.nanmean(win["i_chg_std"]))
    f["i_max"] = float(np.nanmax(win["i_max"]))
    # time (minutes)
    f["t_chg_mean"] = float(np.nanmean(hampel(win["t_chg"])))
    f["t_dis_mean"] = float(np.nanmean(hampel(win["t_dis"])))
    f["t_cv_mean"] = float(np.nanmean(hampel(win["t_cv"])))
    # temperature
    tm = win["temp_mean"]
    f["temp_mean"] = float(np.nanmean(tm))
    f["temp_max"] = float(np.nanmax(hampel(win["temp_max"])))
    f["temp_range"] = float(np.nanmean(win["temp_max"] - win["temp_min"]))
    f["temp_std"] = float(np.nanstd(tm))
    # internal resistance
    f["ir_ref"] = float(ir[win.pos(REF)])
    f["ir_min"] = float(np.nanmin(ir))
    f["ir_change"] = _level(win, ir, T - LEVEL_SPAN, T) - _level(win, ir, REF, REF + 3)
    return f


def trajectory_features(win: Window) -> dict:
    T, x = win.T, win.idx.astype(float)
    qd = cleaned(win, "qd")
    f = {}
    seg = win.between(REF, T)
    f["qd_curvature"] = _slope(x[seg], qd[seg], quadratic=True)
    last10 = win.between(T - 10, T)
    f["qd_slope_last10"] = _slope(x[last10], qd[last10])
    f["qd_resid_std_last10"] = float(np.nanstd(qd[last10] - np.polyval(np.polyfit(x[last10], qd[last10], 1), x[last10]))) \
        if len(last10) >= 3 else np.nan
    # discharge-curve shape change between the reference and the last cycle
    _, c1, c2 = critical_cycles(T)
    dq = delta_q_curve(win, c1, c2)
    if len(dq) > 1:
        f["dq_var_log"] = float(np.log10(dq.var(ddof=1) + EPS))
        f["dq_min_log"] = float(np.log10(abs(dq.min()) + EPS))
        f["dq_mean"] = float(dq.mean())
        f["dq_rms"] = float(np.sqrt(np.mean(dq ** 2)))
    else:
        f.update(dq_var_log=np.nan, dq_min_log=np.nan, dq_mean=np.nan, dq_rms=np.nan)
    q_ref, q_last = win.qdlin_at(c1), win.qdlin_at(c2)
    f["dv_q50"] = voltage_at_fraction(q_last, 0.5) - voltage_at_fraction(q_ref, 0.5)
    vavg = hampel(win["v_dis_avg"])
    f["dv_dis_avg"] = _level(win, vavg, T - LEVEL_SPAN, T) - _level(win, vavg, REF, REF + 3)
    pv_ref, ph_ref = dqdv_peak(q_ref)
    pv_last, ph_last = dqdv_peak(q_last)
    f["dqdv_peak_shift"] = pv_last - pv_ref
    f["dqdv_peak_ratio"] = ph_last / ph_ref if ph_ref else np.nan
    # duration / resistance / temperature trends (per 100 cycles)
    for name in ("t_chg", "t_dis", "t_cv"):
        f[f"{name}_slope"] = _slope(x, hampel(win[name]))
    f["ir_slope"] = _slope(x, cleaned(win, "ir"))
    f["temp_slope"] = _slope(x, win["temp_mean"])
    return f


def set_b(win: Window) -> dict:
    return core_features(win)


def set_c(win: Window) -> dict:
    return {**core_features(win), **trajectory_features(win)}
