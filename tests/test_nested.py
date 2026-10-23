import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batteryrul.nested import conformal_quantile, recall_cutoff  # noqa: E402


def _load_step14():
    spec = importlib.util.spec_from_file_location("step14", ROOT / "scripts" / "14_test_termination.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_conformal_quantile_finite_sample_rank():
    s = np.arange(1, 20)                    # n = 19 -> k = ceil(20 * 0.9) = 18
    assert conformal_quantile(s, 0.1) == 18
    assert math.isinf(conformal_quantile(np.arange(5), 0.05))   # too few scores -> include everything


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    hits = []
    for _ in range(2000):
        x = rng.normal(size=41)
        q = conformal_quantile(np.abs(x[:40]), 0.1)
        hits.append(abs(x[40]) <= q)
    assert 0.88 <= np.mean(hits) <= 0.94


def test_recall_cutoff_reaches_target():
    score = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0])
    pos = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0], bool)
    c = recall_cutoff(score, pos, 0.75)
    assert (score[pos] <= c).mean() >= 0.75 and c == 4.0


def test_termination_simulator_bookkeeping():
    sim_mod = _load_step14()
    rows = []
    # cell a: decided FAIL at T=10; cell b: never decided -> runs to end of life (900)
    for T, fa, pa in [(5, True, True), (10, True, False), (20, True, False)]:
        rows.append({"cell_id": "a", "T": T, "y_true": 400.0, "early_failure": True,
                     "fail_in_set_a0.1": fa, "pass_in_set_a0.1": pa})
    for T in (5, 10, 20):
        rows.append({"cell_id": "b", "T": T, "y_true": 900.0, "early_failure": False,
                     "fail_in_set_a0.1": True, "pass_in_set_a0.1": True})
    sim = sim_mod.simulate(pd.DataFrame(rows), "0.1", [5, 10, 20]).set_index("cell_id")
    assert sim.loc["a", "decision"] == "FAIL" and sim.loc["a", "stopped_at_cycle"] == 10
    assert sim.loc["b", "stopped_at_cycle"] == 900 and not sim.loc["b", "decided_early"]
    s = sim_mod.summary(sim.reset_index())
    assert abs(s["test_time_saved"] - (1 - 910 / 1300)) < 1e-12 and s["wrong_decisions"] == 0
