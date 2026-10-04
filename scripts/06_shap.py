"""Step 6: SHAP interpretability for the main configuration (held-out cells only).

1. Global importance (mean |SHAP|) for RF, XGB (TreeSHAP), MLP and the final
   ensemble (KernelSHAP) on the 42 test cells; top-10 and rank agreement.
2. Dependence of key physical features.
3. Local explanations: correct short-life, correct long-life, worst under-,
   worst over-prediction of the final ensemble.
4. Life-group importance (short/medium/long, tertiles of *training* labels).
5. Stability: TreeSHAP rankings across 10 seeds (test cells) and across the 5
   CV folds (each fold model explains only its own held-out training cells).
6. Physical plausibility: learned sign vs. degradation prior; dominance check.

Outputs: results/runs/shap/*.csv, results/figures/shap_*.png
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402

from batteryrul.config import load_config, repo_path  # noqa: E402
from batteryrul.interpret import kernel_shap_log, sign_consistency, tree_shap_log  # noqa: E402
from batteryrul.metrics import life_groups, tertile_edges  # noqa: E402
from batteryrul.models import make_estimator  # noqa: E402
from batteryrul.splits import fold_indices, read_manifests  # noqa: E402

warnings.filterwarnings("ignore")
KEY_FEATURES = ["qd_retention", "qd_slope", "qd_slope_late", "v_dis_avg_std", "dv_dis_avg",
                "dq_var_log", "ce_std", "ce_mean", "temp_mean", "ir_change"]


def topk(imp: pd.Series, k=10):
    return imp.sort_values(ascending=False).index[:k].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    scfg = cfg["experiments"]["shap"]
    res = repo_path(cfg["output"]["results_dir"])
    out, figs = res / "runs" / "shap", res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    main_cfg = json.loads((res / "runs" / "main.json").read_text())
    run_dir = res / "runs" / main_cfg["tag"]
    fs, T, final = main_cfg["feature_set"], main_cfg["horizon"], main_cfg["final_ensemble"]
    best = json.loads((run_dir / "best_params.json").read_text())
    wq = cfg["preprocessing"]["winsor_quantiles"]
    train, test, folds = read_manifests(repo_path(cfg["split"]["manifest_dir"]))
    df = pd.read_csv(res / "features" / f"features_{fs}_T{T}.csv", index_col="cell_id")
    cols = [c for c in df.columns if c not in ("partition", "cycle_life")]
    Xtr, ytr = df.loc[train, cols].to_numpy(float), df.loc[train, "cycle_life"].to_numpy()
    Xte, yte = df.loc[test, cols].to_numpy(float), df.loc[test, "cycle_life"].to_numpy()

    seed0 = cfg["seeds"][0]
    models = {m: joblib.load(run_dir / "models" / f"{m}_seed{seed0}.joblib") for m in ["rf", "xgb", "mlp"]}
    stacker = joblib.load(run_dir / "models" / f"{final}_seed{seed0}.joblib")
    members = final[len("stack_"):].split("_")
    for m in members:
        models.setdefault(m, joblib.load(run_dir / "models" / f"{m}_seed{seed0}.joblib"))

    def ens_predict(X):
        return stacker.predict(np.column_stack([models[m].predict(X) for m in members]))

    # ---- 1. global importance on test cells ---------------------------------
    sv, base = {}, {}
    sv["rf"], base["rf"] = tree_shap_log(models["rf"], Xte)
    sv["xgb"], base["xgb"] = tree_shap_log(models["xgb"], Xte)
    print("TreeSHAP done; KernelSHAP for MLP and ensemble ...")
    sv["mlp"], base["mlp"] = kernel_shap_log(models["mlp"].predict, Xtr, Xte,
                                             scfg["kernel_background"], scfg["kernel_nsamples"])
    sv["ensemble"], base["ensemble"] = kernel_shap_log(ens_predict, Xtr, Xte,
                                                       scfg["kernel_background"], scfg["kernel_nsamples"])
    imp = pd.DataFrame({m: np.abs(v).mean(0) for m, v in sv.items()}, index=cols)
    imp_norm = imp / imp.sum()
    imp.to_csv(out / "global_importance.csv")
    for m, v in sv.items():
        pd.DataFrame(v, index=test, columns=cols).to_csv(out / f"shap_values_{m}_test.csv")
    rank_agree = pd.DataFrame(index=imp.columns, columns=imp.columns, dtype=float)
    for a in imp.columns:
        for b in imp.columns:
            rank_agree.loc[a, b] = spearmanr(imp[a], imp[b]).correlation
    rank_agree.to_csv(out / "importance_rank_agreement.csv")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharex=False)
    for ax, m in zip(axes, imp.columns):
        s = imp_norm[m].sort_values().iloc[-10:]
        ax.barh(s.index, s.values, color="#3b6ea5")
        ax.set_title(f"{m}: top-10 mean |SHAP| (share)")
    fig.tight_layout()
    fig.savefig(figs / "shap_global_top10.png", dpi=150)
    plt.close(fig)

    shap.summary_plot(sv["ensemble"], pd.DataFrame(Xte, columns=cols), max_display=12, show=False)
    plt.title(f"Final ensemble ({final}) - SHAP on held-out cells [ln cycles]")
    plt.tight_layout()
    plt.savefig(figs / "shap_beeswarm_ensemble.png", dpi=150)
    plt.close()

    # ---- 2. dependence -----------------------------------------------------
    present = [f for f in KEY_FEATURES if f in cols]
    top_ens = topk(imp["ensemble"], 4)
    dep = list(dict.fromkeys(top_ens + present))[:8]
    n = len(dep)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.2 * ((n + 1) // 2), 7.5))
    for ax, f in zip(np.ravel(axes), dep):
        j = cols.index(f)
        for m, mk in (("ensemble", "o"), ("xgb", "^"), ("rf", "s")):
            ax.scatter(Xte[:, j], sv[m][:, j], s=16, marker=mk, alpha=0.7, label=m)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xlabel(f)
        ax.set_ylabel("SHAP [ln cycles]")
    np.ravel(axes)[0].legend(fontsize=7)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figs / "shap_dependence.png", dpi=150)
    plt.close(fig)

    # ---- 3. local explanations ---------------------------------------------
    pred = ens_predict(Xte)
    err = pred - yte
    edges = tertile_edges(ytr)
    grp = life_groups(yte, edges)
    rel = np.abs(err) / yte
    cases = {
        "correct_short_life": int(np.where(grp == "short")[0][np.argmin(rel[grp == "short"])]),
        "correct_long_life": int(np.where(grp == "long")[0][np.argmin(rel[grp == "long"])]),
        "worst_underprediction": int(np.argmin(err)),
        "worst_overprediction": int(np.argmax(err)),
    }
    local_rows = []
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, (case, i) in zip(axes, cases.items()):
        contrib = pd.Series(sv["ensemble"][i], index=cols)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:8])
        ax.barh(top.index[::-1], top.values[::-1], color=["#c0392b" if v < 0 else "#27ae60" for v in top.values[::-1]])
        ax.set_title(f"{case}\n{test[i]}: true {yte[i]:.0f}, pred {pred[i]:.0f}", fontsize=9)
        ax.set_xlabel("SHAP [ln cycles]")
        local_rows.append({"case": case, "cell_id": test[i], "true": yte[i], "pred": pred[i],
                           "error": err[i], "group": grp[i],
                           "top_contributions": json.dumps({k: round(float(v), 4) for k, v in top.items()})})
    fig.tight_layout()
    fig.savefig(figs / "shap_local_cases.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(local_rows).to_csv(out / "local_explanations.csv", index=False)

    # ---- 4. life groups ------------------------------------------------------
    grp_rows = []
    for g in ("short", "medium", "long"):
        mask = grp == g
        s = pd.Series(np.abs(sv["ensemble"][mask]).mean(0), index=cols)
        grp_rows.append({"group": g, "n_cells": int(mask.sum()), "edges_from_train": json.dumps(edges),
                         "RMSE": float(np.sqrt(np.mean(err[mask] ** 2))), "MAE": float(np.mean(np.abs(err[mask]))),
                         "bias": float(np.mean(err[mask])), "top5": ", ".join(topk(s, 5))})
    pd.DataFrame(grp_rows).to_csv(out / "life_group_analysis.csv", index=False)

    # ---- 5. stability --------------------------------------------------------
    stab = []
    for seed in cfg["seeds"]:
        for m in ("rf", "xgb"):
            est = make_estimator(m, best[m], seed, wq).fit(Xtr, ytr)
            v, _ = tree_shap_log(est, Xte)
            stab.append({"source": f"seed{seed}", "model": m, **dict(zip(cols, np.abs(v).mean(0)))})
    for k, trn, val in fold_indices(train, folds):
        for m in ("rf", "xgb"):
            est = make_estimator(m, best[m], seed0, wq).fit(Xtr[trn], ytr[trn])
            v, _ = tree_shap_log(est, Xtr[val])
            stab.append({"source": f"fold{k}", "model": m, **dict(zip(cols, np.abs(v).mean(0)))})
    stab = pd.DataFrame(stab)
    stab.to_csv(out / "stability_importance.csv", index=False)
    stab_rows = []
    for m in ("rf", "xgb"):
        S = stab[stab.model == m].set_index("source")[cols]
        tops = [topk(r, 10) for _, r in S.iterrows()]
        freq = pd.Series({f: np.mean([f in t for t in tops]) for f in cols})
        ranks = S.rank(axis=1, ascending=False)
        taus = [kendalltau(ranks.iloc[i], ranks.iloc[j]).correlation
                for i in range(len(ranks)) for j in range(i + 1, len(ranks))]
        for f in cols:
            stab_rows.append({"model": m, "feature": f, "top10_frequency": freq[f],
                              "median_rank": float(ranks[f].median()), "rank_IQR": float(ranks[f].quantile(.75) - ranks[f].quantile(.25)),
                              "mean_pairwise_kendall_tau": float(np.mean(taus))})
    stab_tab = pd.DataFrame(stab_rows)
    stab_tab.to_csv(out / "stability_summary.csv", index=False)

    # ---- 6. plausibility & dominance ----------------------------------------
    pl_rows = []
    for f in topk(imp["ensemble"], 10):
        j = cols.index(f)
        row = {"feature": f, "ensemble_share": float(imp_norm.loc[f, "ensemble"]),
               "rf_share": float(imp_norm.loc[f, "rf"]), "xgb_share": float(imp_norm.loc[f, "xgb"]),
               "mlp_share": float(imp_norm.loc[f, "mlp"])}
        row.update(sign_consistency(f, Xte[:, j], sv["ensemble"][:, j]))
        st = stab_tab[stab_tab.feature == f]
        row["min_top10_frequency_rf_xgb"] = float(st.top10_frequency.min())
        row["stable"] = bool(st.top10_frequency.min() >= 0.8)
        pl_rows.append(row)
    pl = pd.DataFrame(pl_rows)
    pl.to_csv(out / "plausibility.csv", index=False)
    dominant = pl[pl.ensemble_share > 0.4]
    summary = {"main_config": main_cfg["tag"], "final_ensemble": final,
               "top10_ensemble": topk(imp["ensemble"], 10),
               "rank_agreement_spearman": rank_agree.round(3).to_dict(),
               "dominant_features_gt40pct": dominant.feature.tolist(),
               "local_cases": {r["case"]: r["cell_id"] for r in local_rows}}
    (out / "shap_summary.json").write_text(json.dumps(summary, indent=2))
    print(pl.round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
