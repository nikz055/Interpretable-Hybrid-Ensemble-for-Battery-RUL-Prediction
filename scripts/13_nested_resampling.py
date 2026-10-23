"""Step 13 (v2): very-early warning evaluated by repeated nested battery-level resampling.

Headline protocol: 10 repeats x grouped, life-stratified 5-fold over all 83
MATR-1 cells (every cell is predicted once per repeat by models that never saw
it), with all model/feature/hyper-parameter selection, conformal calibration
and alarm thresholds done by inner CV on the outer-training cells only.
Secondary: the original BatteryML 41/42 split through the identical procedure.

Reads committed feature CSVs only (no raw data needed).

Outputs (results/runs/v2/):
    predictions.csv     one row per (split, repeat, fold, T, cell)
    selections.csv      the pipeline chosen in every outer fold
    metrics.csv         per-T metrics with 95 % cell-bootstrap CIs, resampling and locked split
    comparisons.csv     paired new-vs-baseline differences with 95 % CIs
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from batteryrul.config import load_config, repo_path  # noqa: E402
from batteryrul.nested import run_outer_fold  # noqa: E402
from batteryrul.splits import assert_disjoint, fold_indices, make_cv_folds, read_manifests  # noqa: E402

warnings.filterwarnings("ignore")


def _task(T, Xs, Q, y, cells, trn, tst, inner, v2, tag):
    warnings.filterwarnings("ignore")
    return run_outer_fold(T, Xs, Q, y, cells, trn, tst, inner, v2, tag)


def reg_metrics(y, p):
    e = p - y
    return {"RMSE": np.sqrt(np.mean(e ** 2)), "MAE": np.mean(np.abs(e)), "MedAE": np.median(np.abs(e)),
            "MAPE": np.mean(np.abs(e) / y)}


def clf_metrics(f, score, flag=None):
    m = {"ROC_AUC": roc_auc_score(f, score), "PR_AUC": average_precision_score(f, score)}
    if flag is not None:
        tp, fp = (flag & f).sum(), (flag & ~f).sum()
        m.update(recall=tp / max(f.sum(), 1), precision=tp / max(tp + fp, 1), false_alarm_rate=fp / max((~f).sum(), 1))
    return m


def metric_table(df: pd.DataFrame) -> dict:
    """All metrics for one repeat (or the locked split) at one T."""
    y, f = df.y_true.to_numpy(), df.early_failure.to_numpy(bool)
    m = {}
    for key, col in (("new", "pred_life"), ("base", "base_pred_life"), ("dummy", "dummy_pred_life"),
                     ("batteryml", "batteryml_plsr_qdlinear")):
        if col in df and df[col].notna().all():
            m.update({f"reg_{key}_{k}": v for k, v in reg_metrics(y, df[col].to_numpy()).items()})
    for key, lo, hi, p in (("new", "lo90", "hi90", "pred_life"), ("base", "base_lo90", "base_hi90", "base_pred_life")):
        m[f"reg_{key}_coverage90"] = np.mean((y >= df[lo]) & (y <= df[hi]))
        m[f"reg_{key}_rel_width90"] = np.mean((df[hi] - df[lo]) / df[p])
    m.update({f"clf_new_{k}": v for k, v in clf_metrics(f, df.p_fail.to_numpy(), df.alarm.to_numpy(bool)).items()})
    m["clf_new_Brier"] = np.mean((df.p_fail - f) ** 2)
    m.update({f"clf_base_{k}": v for k, v in clf_metrics(f, -df.base_pred_life.to_numpy(), df.base_alarm.to_numpy(bool)).items()})
    for rr in [c[len("alarm_r"):] for c in df.columns if c.startswith("alarm_r")]:
        for key, col in (("new", f"alarm_r{rr}"), ("base", f"base_alarm_r{rr}")):
            flag = df[col].to_numpy(bool)
            tp, fp = (flag & f).sum(), (flag & ~f).sum()
            m[f"op_{key}_r{rr}_recall"] = tp / max(f.sum(), 1)
            m[f"op_{key}_r{rr}_false_alarm_rate"] = fp / max((~f).sum(), 1)
            m[f"op_{key}_r{rr}_precision"] = tp / max(tp + fp, 1)
    for a in [c[len("fail_in_set_a"):] for c in df.columns if c.startswith("fail_in_set_a")]:
        fs, ps = df[f"fail_in_set_a{a}"].to_numpy(bool), df[f"pass_in_set_a{a}"].to_numpy(bool)
        decided = fs ^ ps
        wrong = decided & ((fs & ~f) | (ps & f))
        m[f"conf_a{a}_decided"] = decided.mean()
        m[f"conf_a{a}_error_among_decided"] = wrong.sum() / max(decided.sum(), 1)
        m[f"conf_a{a}_coverage"] = np.mean(np.where(f, fs, ps))
        # three-way decision at this single checkpoint: alarm / clear / keep testing
        m[f"conf_a{a}_false_alarm_rate"] = (fs & ~ps & ~f).sum() / max((~f).sum(), 1)
        m[f"conf_a{a}_failures_caught"] = (fs & ~ps & f).sum() / max(f.sum(), 1)
        m[f"conf_a{a}_failures_cleared"] = (ps & ~fs & f).sum() / max(f.sum(), 1)
        m[f"conf_a{a}_deferred"] = 1 - decided.mean()
    return m


def summarise(pred: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Mean over repeats + 95 % CI from a cell bootstrap of the repeat-averaged metric."""
    rows = []
    rng = np.random.default_rng(seed)
    for (split, T), d in pred.groupby(["split", "T"]):
        reps = [g.sort_values("cell_id").reset_index(drop=True) for _, g in d.groupby("repeat")]
        base = pd.DataFrame([metric_table(g) for g in reps]).mean()
        n = len(reps[0])
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            try:
                boots.append(pd.DataFrame([metric_table(g.iloc[idx]) for g in reps]).mean())
            except ValueError:                                     # resample with one class only
                continue
        B = pd.DataFrame(boots)
        for k, v in base.items():
            rows.append({"split": split, "T": T, "metric": k, "mean": v,
                         "lo95": B[k].quantile(0.025), "hi95": B[k].quantile(0.975), "n_repeats": len(reps)})
    return pd.DataFrame(rows)


def paired(pred: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """New vs baseline on the same cells: seed/repeat-averaged differences with cell-bootstrap CIs."""
    rng = np.random.default_rng(seed)
    out = []
    for (split, T), d in pred.groupby(["split", "T"]):
        reps = [g.sort_values("cell_id").reset_index(drop=True) for _, g in d.groupby("repeat")]
        n = len(reps[0])

        def diffs(idx):
            r = []
            for g in reps:
                g = g.iloc[idx]
                y, f = g.y_true.to_numpy(), g.early_failure.to_numpy(bool)
                en, eb = np.abs(g.pred_life - y), np.abs(g.base_pred_life - y)
                r.append({"dMAE": en.mean() - eb.mean(),
                          "dRMSE": np.sqrt(np.mean((g.pred_life - y) ** 2)) - np.sqrt(np.mean((g.base_pred_life - y) ** 2)),
                          "dPR_AUC": average_precision_score(f, g.p_fail) - average_precision_score(f, -g.base_pred_life),
                          "dROC_AUC": roc_auc_score(f, g.p_fail) - roc_auc_score(f, -g.base_pred_life)})
            return pd.DataFrame(r).mean()
        est = diffs(np.arange(n))
        boots = []
        for _ in range(n_boot):
            try:
                boots.append(diffs(rng.integers(0, n, n)))
            except ValueError:
                continue
        B = pd.DataFrame(boots)
        for k in est.index:
            out.append({"split": split, "T": T, "difference_new_minus_baseline": k, "estimate": est[k],
                        "lo95": B[k].quantile(0.025), "hi95": B[k].quantile(0.975),
                        "P(new better)": float((B[k] < 0).mean() if k.startswith(("dMAE", "dRMSE")) else (B[k] > 0).mean())})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    cfg = load_config(args.config)
    v2 = cfg["experiments"]["v2"]
    res = repo_path(cfg["output"]["results_dir"])
    out = res / "runs" / "v2"
    out.mkdir(parents=True, exist_ok=True)
    train, test, manifest = read_manifests(repo_path(cfg["split"]["manifest_dir"]))
    cells = train + test
    y = pd.read_csv(res / "features" / "features_C_T100.csv", index_col="cell_id").loc[cells, "cycle_life"].to_numpy()
    labels = pd.Series(y, index=cells)
    Q = pd.read_csv(res / "features" / "qdlinear_T100.csv", index_col="cell_id").loc[cells].to_numpy(float)

    def X_at(T):
        return {fs: pd.read_csv(res / "features" / f"features_{fs}_T{T}.csv", index_col="cell_id")
                .drop(columns=["partition", "cycle_life"]).loc[cells].to_numpy(float) for fs in ("A", "C", "D")}

    tasks = []
    for T in v2["horizons"]:
        Xs = X_at(T)
        QT = Q if T == 100 else None
        for r in range(v2["repeats"]):
            folds = make_cv_folds(cells, labels, v2["n_folds"], v2["seed"] + r)
            for k, trn, tst in fold_indices(cells, folds):
                tr_cells = [cells[i] for i in trn]
                assert_disjoint(outer_train=tr_cells, outer_test=[cells[i] for i in tst])
                inner = [(a, b) for _, a, b in fold_indices(
                    tr_cells, make_cv_folds(tr_cells, labels, v2["n_folds"], 10_000 + 100 * r + k))]
                tasks.append((T, Xs, QT, y, cells, trn, tst, inner, v2, {"split": "resampling", "repeat": r, "fold": k}))
        # locked BatteryML split, inner folds = the committed manifest
        trn = np.arange(len(train))
        tst = np.arange(len(train), len(cells))
        inner = [(a, b) for _, a, b in fold_indices(train, manifest)]
        tasks.append((T, Xs, QT, y, cells, trn, tst, inner, v2, {"split": "locked", "repeat": 0, "fold": 0}))

    t0 = time.time()
    print(f"{len(tasks)} outer folds over horizons {v2['horizons']} on {args.jobs} workers ...", flush=True)
    results = Parallel(n_jobs=args.jobs, verbose=2)(delayed(_task)(*t) for t in tasks)
    pred = pd.DataFrame([r for rows, _ in results for r in rows])
    sel = pd.DataFrame([s for _, s in results])
    pred.to_csv(out / "predictions.csv", index=False)
    sel.to_csv(out / "selections.csv", index=False)
    chk = pred[pred.split == "resampling"].groupby(["T", "repeat"]).cell_id.agg(["count", "nunique"])
    assert (chk["count"] == len(cells)).all() and (chk["nunique"] == len(cells)).all(), "each cell once per repeat"
    print(f"fitted in {time.time() - t0:.0f}s; computing metrics ...", flush=True)

    met = summarise(pred, args.n_boot, 1)
    met.to_csv(out / "metrics.csv", index=False)
    comp = paired(pred, args.n_boot, 2)
    comp.to_csv(out / "comparisons.csv", index=False)

    show = ["reg_new_MAE", "reg_base_MAE", "reg_new_RMSE", "reg_base_RMSE", "reg_new_coverage90",
            "clf_new_ROC_AUC", "clf_new_PR_AUC", "clf_base_PR_AUC", "clf_new_recall", "clf_new_false_alarm_rate",
            "conf_a0.1_decided", "conf_a0.1_error_among_decided"]
    for split in ("resampling", "locked"):
        t = met[(met.split == split) & met.metric.isin(show)].pivot(index="metric", columns="T", values="mean").loc[show]
        print(f"\n== {split} (mean)\n{t.round(3).to_string()}")
    print("\n" + comp[comp.split == "resampling"].round(3).to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
