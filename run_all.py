"""Reproduce the whole study with one command.

    python run_all.py --data-dir E:/Battery/Preprocessed/MATR          # local
    python run_all.py --data-dir /kaggle/input/<dataset>/MATR --jobs 2  # Kaggle
    python run_all.py --from-step 4      # re-run models/report from committed features only

Steps
  0 build per-cycle signal cache from BatteryML-processed MATR pickles   (needs data)
  1 split manifests + label table                                         (needs cache)
  2 BatteryML baseline reproduction                                       (needs cache)
  3 feature matrices A/B/C x T                                            (needs cache)
  a legacy notebook audit                                                 (needs cache)
  4 experiments 1-4                                                       (features only)
  5 experiment 5 robustness                                               (needs cache)
  6 SHAP analysis                                                         (features only)
  7 tables, figures, REPORT.md                                            (results only)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("0", "scripts/00_build_cache.py"),
    ("1", "scripts/01_make_splits.py"),
    ("2", "scripts/02_batteryml_baselines.py"),
    ("3", "scripts/03_build_features.py"),
    ("a", "scripts/audit_legacy_notebook.py"),
    ("4", "scripts/04_run_experiments.py"),
    ("5", "scripts/05_robustness.py"),
    ("6", "scripts/06_shap.py"),
    ("7", "scripts/07_make_report.py"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--from-step", default="0", help="0-7 (step 'a' runs with step 3)")
    args = ap.parse_args()
    order = [s for s, _ in STEPS]
    start = order.index(args.from_step)
    for key, script in STEPS[start:]:
        cmd = [sys.executable, str(ROOT / script)]
        if args.config:
            cmd += ["--config", args.config]
        if key == "0":
            cmd += ["--jobs", str(args.jobs)] + (["--data-dir", args.data_dir] if args.data_dir else [])
        t0 = time.time()
        print(f"\n===== step {key}: {script}", flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
        print(f"===== step {key} finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
