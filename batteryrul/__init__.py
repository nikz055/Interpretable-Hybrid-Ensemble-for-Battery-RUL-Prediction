"""Leakage-safe, BatteryML-aligned early-life cycle-life prediction on MATR-1."""

import os as _os

__version__ = "1.1.0"

# joblib/loky otherwise shells out to count physical cores on Windows and
# prints a (harmless) traceback when that fails.
_os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_os.cpu_count() or 1))
