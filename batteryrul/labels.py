"""Cycle-life labels.

``batteryml_rul_label`` is a line-for-line port of BatteryML's
``RULLabelAnnotator`` (batteryml/label/rul.py) with its defaults
(eol_soh=0.8, pad_eol=True, min_rul_limit=100). The label is computed from the
*whole* cycling history and is used only as the regression target, never as an
input.
"""
from __future__ import annotations

import math


def batteryml_rul_label(qd_max_per_cycle, nominal_capacity_ah: float,
                        eol_soh: float = 0.8, pad_eol: bool = True,
                        min_rul_limit: float = 100.0) -> float:
    """Return BatteryML's MATR cycle-life label (NaN if the cell is excluded).

    BatteryML starts the counter at 1 and increments it *before* testing each
    cycle, so a cell whose first sub-threshold cycle sits at 0-based index k
    gets label k + 2. We keep that convention so numbers are comparable with
    the BatteryML benchmark table.
    """
    label, found_eol = 1, False
    threshold = nominal_capacity_ah * eol_soh
    for qd in qd_max_per_cycle:
        label += 1
        if qd <= threshold:
            found_eol = True
            break
    if not found_eol:
        label = label + 1 if pad_eol else math.nan
    if label <= min_rul_limit:
        label = math.nan
    return float(label)


def legacy_notebook_label(n_cycles: int) -> float:
    """Label used by the original notebook: the number of recorded cycles.

    This is *not* the BatteryML definition: MATR cells were often cycled past
    80 % SOH, so it overstates cycle life for many cells.
    """
    return float(n_cycles)
