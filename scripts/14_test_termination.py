"""Step 14 (v2): adaptive termination of battery durability tests.

A validation lab normally cycles every cell to end of life (here 300-2237
cycles, mean ~700) to know whether it meets a life specification. Using the
step-13 conformal pass/fail sets, each test cell is checked at cycles
5, 10, 20, 30, 50 and 100:

  * set == {fail} -> stop the test, declare FAIL (early failure, life <= 488)
  * set == {pass} -> stop the test, declare PASS
  * otherwise      -> keep cycling to the next checkpoint
  * still undecided after cycle 100 -> run to end of life (decision = truth)

Several policies (which checkpoints may stop a test) are compared; checking
at many checkpoints and stopping at the first confident call compounds errors.
Reported per policy and error level alpha: average cycles per cell, % test time saved vs
running every cell to end of life, share of cells decided by each checkpoint,
wrong decisions (missed failures and false failures). All numbers come from
models that never saw the cell (outer folds of step 13).

Outputs: results/runs/v2/termination_{summary,by_checkpoint,cells}.csv,
results/figures/v2_termination.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from batteryrul.config import load_config, repo_path  # noqa: E402


def simulate(d: pd.DataFrame, alpha: str, checkpoints: list[int]) -> pd.DataFrame:
    """One row per cell: decision, cycle at which the test stopped, correctness."""
    out = []
    for cell, g in d.groupby("cell_id"):
        g = g.set_index("T")
        y, fail = float(g.y_true.iloc[0]), bool(g.early_failure.iloc[0])
        decision, at = None, None
        for T in checkpoints:
            fs, ps = bool(g.loc[T, f"fail_in_set_a{alpha}"]), bool(g.loc[T, f"pass_in_set_a{alpha}"])
            if fs and not ps:
                decision, at = "FAIL", T
                break
            if ps and not fs:
                decision, at = "PASS", T
                break
        if decision is None:                                   # run to end of life
            decision, at, early = ("FAIL" if fail else "PASS"), y, False
        else:
            early = True
        out.append({"cell_id": cell, "true_life": y, "early_failure": fail, "decision": decision,
                    "stopped_at_cycle": at, "decided_early": early,
                    "correct": (decision == "FAIL") == fail})
    return pd.DataFrame(out)


def summary(sim: pd.DataFrame) -> dict:
    n = len(sim)
    return {"avg_test_cycles": sim.stopped_at_cycle.mean(),
            "full_test_cycles": sim.true_life.mean(),
            "test_time_saved": 1 - sim.stopped_at_cycle.sum() / sim.true_life.sum(),
            "decided_early": sim.decided_early.mean(),
            "wrong_decisions": (~sim.correct).sum() / n,
            "missed_failures": int(((sim.decision == "PASS") & sim.early_failure).sum()),
            "false_failures": int(((sim.decision == "FAIL") & ~sim.early_failure).sum()),
            "false_alarm_rate": ((sim.decision == "FAIL") & ~sim.early_failure).sum() / max((~sim.early_failure).sum(), 1),
            "missed_failure_rate": ((sim.decision == "PASS") & sim.early_failure).sum() / max(sim.early_failure.sum(), 1),
            "n_cells": n, "n_failures": int(sim.early_failure.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    v2 = cfg["experiments"]["v2"]
    res = repo_path(cfg["output"]["results_dir"])
    out, figs = res / "runs" / "v2", res / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(out / "predictions.csv")
    cps = sorted(v2["horizons"])
    alphas = [f"{a:g}" for a in v2["alphas"]]
    rng = np.random.default_rng(3)

    rows, cp_rows, cell_rows = [], [], []
    for split, ds in pred.groupby("split"):
        for pol, cps_p in v2["policies"].items():
            dp = ds[ds["T"].isin(cps_p)]
            for a in alphas:
                sims = {r: simulate(g, a, cps_p) for r, g in dp.groupby("repeat")}
                S = pd.DataFrame([summary(s) for s in sims.values()])
                mean = S.mean(numeric_only=True)
                cells = sorted(next(iter(sims.values())).cell_id)
                aligned = [s.set_index("cell_id").loc[cells].reset_index() for s in sims.values()]
                B = []
                for _ in range(args.n_boot):
                    idx = rng.integers(0, len(cells), len(cells))
                    B.append(pd.DataFrame([summary(s.iloc[idx]) for s in aligned]).mean(numeric_only=True))
                B = pd.DataFrame(B)
                for k, v in mean.items():
                    rows.append({"split": split, "policy": pol, "checkpoints": str(cps_p), "alpha": float(a), "metric": k,
                                 "mean": v, "lo95": B[k].quantile(0.025), "hi95": B[k].quantile(0.975)})
                for T in cps_p:
                    frac = np.mean([((s.stopped_at_cycle <= T) & s.decided_early).mean() for s in sims.values()])
                    cp_rows.append({"split": split, "policy": pol, "alpha": float(a), "checkpoint": T,
                                    "cum_share_decided": frac})
                if split == "resampling":
                    for r, s in sims.items():
                        cell_rows.append(s.assign(policy=pol, alpha=float(a), repeat=r))
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "termination_summary.csv", index=False)
    cp = pd.DataFrame(cp_rows)
    cp.to_csv(out / "termination_by_checkpoint.csv", index=False)
    pd.concat(cell_rows).to_csv(out / "termination_cells.csv", index=False)

    # forced yes/no alarms (no "keep testing" option) from step 13, for comparison
    met = pd.read_csv(out / "metrics.csv")
    forced = []
    for (split, T), d in met[met.metric.str.startswith("op_base_r")].groupby(["split", "T"]):
        for rr in sorted({m.split("_")[2][1:] for m in d.metric}):
            g = d.set_index("metric")
            forced.append({"split": split, "policy": f"forced_alarm_{T}", "recall_target": float(rr),
                           "false_alarm_rate": g.loc[f"op_base_r{rr}_false_alarm_rate", "mean"],
                           "fa_lo95": g.loc[f"op_base_r{rr}_false_alarm_rate", "lo95"],
                           "fa_hi95": g.loc[f"op_base_r{rr}_false_alarm_rate", "hi95"],
                           "missed_failure_rate": 1 - g.loc[f"op_base_r{rr}_recall", "mean"]})
    pd.DataFrame(forced).to_csv(out / "forced_alarms.csv", index=False)

    # ---- figure: false alarms vs missed failures, by policy -------------------
    R = summ[summ.split == "resampling"].pivot_table(index=["policy", "alpha"], columns="metric", values="mean").reset_index()
    F = pd.DataFrame(forced)
    F = F[F.split == "resampling"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for pol, g in R.groupby("policy"):
        ax.plot(100 * g.missed_failure_rate, 100 * g.false_alarm_rate, marker="o", label=f"{pol} (keep-testing option)")
        for _, r in g.iterrows():
            ax.annotate(f"α={r.alpha:g}, saves {100 * r.test_time_saved:.0f}%", (100 * r.missed_failure_rate, 100 * r.false_alarm_rate),
                        fontsize=7, xytext=(4, 3), textcoords="offset points")
    for T, g in F.groupby(F.policy):
        ax.plot(100 * g.missed_failure_rate, 100 * g.false_alarm_rate, ls=":", marker="s", label=f"{T} (yes/no only)")
    ax.set_xlabel("missed early failures (%)")
    ax.set_ylabel("false alarms on healthy cells (%)")
    ax.set_title("Early-failure decisions: false alarms vs missed failures")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "v2_termination.png", dpi=150)
    plt.close(fig)
    key = ["false_alarm_rate", "missed_failure_rate", "decided_early", "test_time_saved", "wrong_decisions"]
    for split in ("resampling", "locked"):
        t = summ[(summ.split == split) & summ.metric.isin(key)].pivot_table(index=["policy", "alpha"], columns="metric", values="mean")
        print(f"== {split}")
        print(t[key].round(3).to_string())
    print(pd.DataFrame(forced).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
