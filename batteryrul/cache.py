"""Per-cycle signal cache.

Reading the raw MATR pickles is the only expensive step (~5 GB of Python lists
for MATR-1). We read each cell once and keep, for the first ``T_MAX`` cycles
only, a compact set of per-cycle signals from which every feature set and every
horizon T <= T_MAX is computed.

Leakage boundary: the only quantity taken from cycles beyond ``T_MAX`` is the
capacity trajectory ``qd_full``, which is stored *solely* to compute the label
and is never passed to a feature function (see ``features.window``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import load_cell
from .labels import batteryml_rul_label, legacy_notebook_label

T_MAX = 100
QDLIN_POINTS = 1000
# Severson et al. interpolate Qdlin on linspace(3.6, 2.0, 1000)
QDLIN_VOLTAGE = np.linspace(3.6, 2.0, QDLIN_POINTS)

I_EPS = 0.05          # A, |I| below this is treated as rest
DIS_EPS = 0.1         # A, same threshold BatteryML uses to select discharge points
CV_VOLTAGE = 3.595    # V, charge points above this are in the 3.6 V CV step

SCALAR_SIGNALS = [
    "qd", "qc", "ir",
    "t_bml_chg", "t_chg", "t_dis", "t_cv", "t_rest",
    "e_dis", "e_chg", "v_dis_avg",
    "i_chg_mean", "i_chg_std", "i_max",
    "temp_mean", "temp_max", "temp_min",
]


def _arr(cycle: dict, key: str) -> np.ndarray:
    v = cycle.get(key)
    return np.asarray(v if v is not None else [], dtype=float)


def cycle_signals(cycle: dict) -> dict:
    """Scalar summaries of one cycle. All values come from this cycle only."""
    I = _arr(cycle, "current_in_A")
    V = _arr(cycle, "voltage_in_V")
    t = _arr(cycle, "time_in_s")            # minutes, see io.py
    T = _arr(cycle, "temperature_in_C")
    Qd = _arr(cycle, "discharge_capacity_in_Ah")
    Qc = _arr(cycle, "charge_capacity_in_Ah")
    out = dict.fromkeys(SCALAR_SIGNALS, np.nan)
    if len(I) < 3 or len(t) != len(I):
        return out

    dt = np.diff(t, prepend=t[0])
    dt[0] = 0.0
    chg, dis = I > I_EPS, I < -DIS_EPS
    out["qd"] = np.nanmax(Qd) if len(Qd) else np.nan
    out["qc"] = np.nanmax(Qc) if len(Qc) else np.nan
    ir = cycle.get("internal_resistance_in_ohm")
    out["ir"] = float(ir) if ir is not None else np.nan
    # BatteryML's "Average early charge time" sums dt where I < 0, which in
    # MATR is the *discharge* step. Kept verbatim for the reference features.
    out["t_bml_chg"] = dt[I < 0].sum()
    out["t_chg"] = dt[chg].sum()
    out["t_dis"] = dt[dis].sum()
    out["t_cv"] = dt[chg & (V >= CV_VOLTAGE)].sum()
    out["t_rest"] = dt[np.abs(I) <= I_EPS].sum()
    if dis.sum() > 2:
        qd_seg = Qd[dis]
        out["e_dis"] = abs(np.trapz(V[dis], qd_seg))
        span = np.nanmax(qd_seg) - np.nanmin(qd_seg)
        out["v_dis_avg"] = out["e_dis"] / span if span > 0 else np.nan
    if chg.sum() > 2:
        out["e_chg"] = abs(np.trapz(V[chg], Qc[chg]))
        w = dt[chg]
        if w.sum() > 0:
            mean = np.sum(I[chg] * w) / w.sum()
            out["i_chg_mean"] = mean
            out["i_chg_std"] = np.sqrt(np.sum(w * (I[chg] - mean) ** 2) / w.sum())
    out["i_max"] = np.nanmax(np.abs(I))
    if len(T) and np.isfinite(T).any():
        out["temp_mean"] = np.nanmean(T)
        out["temp_max"] = np.nanmax(T)
        out["temp_min"] = np.nanmin(T)
    return out


def build_cell_cache(raw_dir: Path, cell_id: str, t_max: int = T_MAX) -> dict:
    cell = load_cell(raw_dir, cell_id)
    cycles = cell["cycle_data"]
    nominal = float(cell["nominal_capacity_in_Ah"])
    qd_full = np.array([max(c["discharge_capacity_in_Ah"]) for c in cycles], dtype=float)

    n = min(t_max, len(cycles))
    sig = {k: np.full(n, np.nan) for k in SCALAR_SIGNALS}
    qdlin = np.full((n, QDLIN_POINTS), np.nan, dtype=np.float32)
    cycle_number = np.zeros(n, dtype=int)
    for i in range(n):
        c = cycles[i]
        cycle_number[i] = int(c["cycle_number"])
        for k, v in cycle_signals(c).items():
            sig[k][i] = v
        if c.get("Qdlin") is not None:
            q = np.asarray(c["Qdlin"], dtype=np.float32)
            if q.shape == (QDLIN_POINTS,):
                qdlin[i] = q

    return {
        "cell_id": cell_id,
        "nominal_capacity_ah": nominal,
        "min_voltage_v": float(cell["min_voltage_limit_in_V"]),
        "max_voltage_v": float(cell["max_voltage_limit_in_V"]),
        "n_cycles_recorded": len(cycles),
        "label": batteryml_rul_label(qd_full, nominal),
        "legacy_label": legacy_notebook_label(len(cycles)),
        "qd_full": qd_full,            # label/plots only - never a feature input
        "cycle_number": cycle_number,
        "qdlin": qdlin,
        **sig,
    }


def save_cell_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in cache.items()})


def load_cell_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    for k in ("cell_id",):
        out[k] = str(out[k])
    for k in ("nominal_capacity_ah", "min_voltage_v", "max_voltage_v", "label", "legacy_label"):
        out[k] = float(out[k])
    out["n_cycles_recorded"] = int(out["n_cycles_recorded"])
    return out


def load_all(cache_dir: Path, cells) -> dict[str, dict]:
    cache_dir = Path(cache_dir)
    missing = [c for c in cells if not (cache_dir / f"{c}.npz").exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} cells missing from cache {cache_dir} (e.g. {missing[:3]}). "
            "Run scripts/00_build_cache.py first.")
    return {c: load_cell_cache(cache_dir / f"{c}.npz") for c in cells}
