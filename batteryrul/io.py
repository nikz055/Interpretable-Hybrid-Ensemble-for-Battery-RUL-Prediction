"""Reading BatteryML-processed MATR cells.

BatteryML (``batteryml preprocess MATR ...``) writes one pickle per cell named
``MATR_<cell>.pkl`` holding ``BatteryData.to_dict()``. We read that dict
directly so the project does not need BatteryML (or numba) installed, while
keeping exactly the same on-disk format and field names.

Units in the MATR pickles (inherited from Severson et al., 2019):
    voltage_in_V              V
    current_in_A              A   (charge > 0, discharge < 0)
    charge/discharge capacity Ah
    time_in_s                 *minutes* (the raw MATR ``t`` field; BatteryML
                              stores it unchanged despite the field name)
    temperature_in_C          degC
    internal_resistance_in_ohm  Ohm (one value per cycle)
    Qdlin                     Ah, discharge capacity on a fixed 1000-point
                              voltage grid from 3.6 V down to 2.0 V
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

DEFAULT_RAW_DIR = r"E:\Battery\Preprocessed\MATR"


def resolve_raw_dir(cli_value: str | None = None, config_value: str | None = None) -> Path:
    """CLI flag > ``MATR_DIR`` env var > config value > default."""
    for candidate in (cli_value, os.environ.get("MATR_DIR"), config_value, DEFAULT_RAW_DIR):
        if candidate:
            return Path(candidate)
    raise ValueError("No MATR data directory given")


def cell_path(raw_dir: Path, cell_id: str) -> Path:
    return Path(raw_dir) / f"MATR_{cell_id}.pkl"


def load_cell(raw_dir: Path, cell_id: str) -> dict:
    path = cell_path(raw_dir, cell_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing processed cell {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def available_cells(raw_dir: Path) -> list[str]:
    return sorted(p.stem.split("_", 1)[1] for p in Path(raw_dir).glob("MATR_*.pkl"))
