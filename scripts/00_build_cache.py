"""Step 0: read BatteryML-processed MATR-1 pickles once and cache per-cycle signals.

    python scripts/00_build_cache.py [--data-dir E:/Battery/Preprocessed/MATR] [--jobs 3]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from joblib import Parallel, delayed  # noqa: E402

from batteryrul.cache import build_cell_cache, save_cell_cache  # noqa: E402
from batteryrul.config import load_config, repo_path  # noqa: E402
from batteryrul.io import resolve_raw_dir  # noqa: E402
from batteryrul.splits import MATR1_CELLS  # noqa: E402


def _one(raw_dir, cell, out, t_max):
    if out.exists():
        return cell, "cached"
    save_cell_cache(build_cell_cache(raw_dir, cell, t_max), out)
    return cell, "built"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--jobs", type=int, default=3, help="parallel readers (each b1 cell needs ~1-2 GB RAM)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_dir = resolve_raw_dir(args.data_dir, cfg["data"]["raw_dir"])
    cache_dir = repo_path(cfg["data"]["cache_dir"])
    t_max = int(cfg["data"]["t_max"])
    print(f"Reading {len(MATR1_CELLS)} MATR-1 cells from {raw_dir} -> {cache_dir}")

    t0 = time.time()
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(_one)(raw_dir, c, cache_dir / f"{c}.npz", t_max) for c in MATR1_CELLS)
    built = sum(s == "built" for _, s in results)
    print(f"Done: {built} built, {len(results) - built} already cached, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
