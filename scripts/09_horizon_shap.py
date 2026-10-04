"""Step 9: which early-cycle features carry the life signal, as a function of the
number of cycles observed (T = 20, 50, 100)?

For every horizon T and every explained model (PLSR, RF, XGBoost, MLP) with
the hyper-parameters tuned for that horizon (results/runs/T{T}_C):
  * 10 seeds: fit on the 41 training cells, explain the 42 held-out test cells;
  * 5 CV folds (seed 0): each fold model explains only its own held-out cells.
SHAP is in ln(cycle life); exact for PLSR (linear) and RF/XGB (TreeSHAP),
KernelSHAP for the MLP. Feature importance = mean |SHAP|; group importance =
mean |sum of member SHAP| (additivity). "Consensus" = mean over models of each
model's normalised share, so no single model dominates the conclusion.

Outputs: results/runs/early_warning/{feature_importance, group_shap_cells,
feature_shap_seed0, group_importance_by_horizon, feature_stability_by_horizon}.csv
and results/figures/early_*.png
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from batteryrul.config import load_config, repo_path  # noqa: E402
from batteryrul.features.groups import (GROUP_LABEL, MECHANISM_GROUPS, PROTOCOL_GROUP,  # noqa: E402
                                        feature_group, groups_for)
from batteryrul.interpret import EXPECTED_SIGN, explain_log, group_shap  # noqa: E402
from batteryrul.models import make_estimator  # noqa: E402
from batteryrul.splits import fold_indices, read_manifests  # noqa: E402

warnings.filterwarnings("ignore")
DETERMINISTIC = {"plsr", "ridge"}


def load_xy(res, fs, T, train, test):
    df = pd.read_csv(res / "features" / f"features_{fs}_T{T}.csv", index_col="cell_id")
    cols = [c for c in df.columns if c not in ("partition", "cycle_life")]
    return (cols, df.loc[train, cols].to_numpy(float), df.loc[train, "cycle_life"].to_numpy(),
            df.loc[test, cols].to_numpy(float), df.loc[test, "cycle_life"].to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ecfg, scfg = cfg["experiments"]["early_warning"], cfg["experiments"]["shap"]
    res = repo_path(cfg["output"]["results_dir"])
    out, figs = res / "runs" / "early_warning", res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    train, test, folds = read_manifests(repo_path(cfg["split"]["manifest_dir"]))
    wq = cfg["preprocessing"]["winsor_quantiles"]
    fs = ecfg["feature_set"]

    imp_rows, grp_rows, cell_rows = [], [], []
    for T in ecfg["horizons"]:
        cols, Xtr, ytr, Xte, yte = load_xy(res, fs, T, train, test)
        groups = groups_for(cols)
        best = json.loads((res / "runs" / f"T{T}_{fs}" / "best_params.json").read_text())
        for m in ecfg["models"]:
            seeds = [ecfg["shap_seeds"][0]] if m in DETERMINISTIC else ecfg["shap_seeds"]
            runs = []
            for s in seeds:
                est = make_estimator(m, best[m], s, wq).fit(Xtr, ytr)
                sv, _ = explain_log(m, est, Xtr, Xte, scfg["kernel_background"], scfg["kernel_nsamples"], s)
                runs.append((f"seed{s}", "test", sv, test, Xte))
                if s == seeds[0]:
                    for i, c in enumerate(test):
                        for j, f in enumerate(cols):
                            cell_rows.append((T, m, c, f, feature_group(f), sv[i, j], Xte[i, j]))
            for k, trn, val in fold_indices(train, folds):
                est = make_estimator(m, best[m], seeds[0], wq).fit(Xtr[trn], ytr[trn])
                sv, _ = explain_log(m, est, Xtr[trn], Xtr[val], scfg["kernel_background"], scfg["kernel_nsamples"], k)
                runs.append((f"fold{k}", "oof", sv, [train[i] for i in val], Xtr[val]))
            for source, part, sv, cells, _ in runs:
                mabs = np.abs(sv).mean(0)
                for j, f in enumerate(cols):
                    imp_rows.append((T, m, source, part, f, feature_group(f), mabs[j], mabs[j] / mabs.sum()))
                for g, v in group_shap(sv, groups).items():
                    for c, val in zip(cells, v):
                        grp_rows.append((T, m, source, part, c, g, val))
            print(f"T={T:3d} {m:5s}: {len(runs)} explanation runs")

    imp = pd.DataFrame(imp_rows, columns=["T", "model", "source", "partition", "feature", "group", "mean_abs_shap", "share"])
    grp = pd.DataFrame(grp_rows, columns=["T", "model", "source", "partition", "cell_id", "group", "shap"])
    cells = pd.DataFrame(cell_rows, columns=["T", "model", "cell_id", "feature", "group", "shap", "value"])
    imp.to_csv(out / "feature_importance.csv", index=False)
    grp.to_csv(out / "group_shap_cells.csv", index=False)
    cells.to_csv(out / "feature_shap_seed0.csv", index=False)

    # ---- group importance by horizon --------------------------------------
    g_imp = grp.assign(abs_shap=grp.shap.abs()).groupby(["T", "model", "source", "group"]).abs_shap.mean().reset_index()
    g_imp["share"] = g_imp.abs_shap / g_imp.groupby(["T", "model", "source"]).abs_shap.transform("sum")
    deg = g_imp[g_imp.group != PROTOCOL_GROUP].copy()
    deg["share_degradation_only"] = deg.abs_shap / deg.groupby(["T", "model", "source"]).abs_shap.transform("sum")
    g_imp = g_imp.merge(deg[["T", "model", "source", "group", "share_degradation_only"]], how="left")
    per_model = g_imp.groupby(["T", "model", "group"]).agg(share_mean=("share", "mean"), share_std=("share", "std"),
                                                          deg_share_mean=("share_degradation_only", "mean")).reset_index()
    cons = per_model.groupby(["T", "group"]).agg(consensus_share=("share_mean", "mean"),
                                                 between_model_std=("share_mean", "std"),
                                                 consensus_degradation_share=("deg_share_mean", "mean")).reset_index()
    wide = per_model.pivot_table(index=["T", "group"], columns="model", values="share_mean").add_prefix("share_").reset_index()
    cons = cons.merge(wide, on=["T", "group"])
    cons["group_rank"] = cons.groupby("T").consensus_share.rank(ascending=False).astype(int)
    cons.sort_values(["T", "group_rank"]).to_csv(out / "group_importance_by_horizon.csv", index=False)

    # ---- feature stability and sign by horizon -----------------------------
    stab_rows = []
    for T, d in imp.groupby("T"):
        tops = d.groupby(["model", "source"]).apply(lambda x: set(x.nlargest(10, "share").feature))
        model_share = d.groupby(["model", "feature"]).share.mean().unstack(0)
        cons_share = model_share.mean(1)
        cT = cells[cells["T"] == T]
        for f in d.feature.unique():
            signs = []
            for m, cm in cT[cT.feature == f].groupby("model"):
                if cm.value.std() > 0:
                    rho = spearmanr(cm.value, cm.shap).correlation
                    signs.append("+" if rho > 0.3 else "-" if rho < -0.3 else "~")
            learned = max(set(signs), key=signs.count) if signs else "~"
            exp = EXPECTED_SIGN.get(f, "?")
            verdict = ("not assessed" if exp == "?" else "weak" if learned == "~"
                       else "consistent" if learned == exp else "INCONSISTENT")
            stab_rows.append({"T": T, "feature": f, "group": feature_group(f),
                              "consensus_share": cons_share[f],
                              "top10_frequency": float(np.mean([f in t for t in tops])),
                              "learned_sign_majority": learned, "sign_agreement_models": f"{signs.count(learned)}/{len(signs)}",
                              "expected_sign": exp, "plausibility": verdict})
    stab = pd.DataFrame(stab_rows)
    stab["rank"] = stab.groupby("T").consensus_share.rank(ascending=False).astype(int)
    stab.sort_values(["T", "rank"]).to_csv(out / "feature_stability_by_horizon.csv", index=False)

    # ---- figures -----------------------------------------------------------
    order = list(MECHANISM_GROUPS)
    Ts = ecfg["horizons"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    width = 0.8 / len(Ts)
    for i, T in enumerate(Ts):
        c = cons[cons["T"] == T].set_index("group").reindex(order)
        ax.bar(np.arange(len(order)) + (i - (len(Ts) - 1) / 2) * width, c.consensus_share, width,
               yerr=c.between_model_std, capsize=2, label=f"T = {T} cycles")
    ax.set_xticks(np.arange(len(order)), [GROUP_LABEL[g].replace(" (", "\n(") for g in order], fontsize=8)
    ax.set_ylabel("Share of total |SHAP| (consensus of 4 models)")
    ax.set_title("Which mechanism carries the cycle-life signal, by cycles observed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "early_group_importance_by_horizon.png", dpi=150)
    plt.close(fig)

    top = sorted(set().union(*[set(stab[stab["T"] == T].nsmallest(8, "rank").feature) for T in Ts]))
    H = stab.pivot(index="feature", columns="T", values="consensus_share").loc[top]
    H = H.loc[H.mean(1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(6.5, 0.38 * len(H) + 1.5))
    im = ax.imshow(H.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(Ts)), [f"T={T}" for T in Ts])
    ax.set_yticks(range(len(H)), [f"{f}  [{feature_group(f).split('_')[0]}]" for f in H.index], fontsize=8)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            ax.text(j, i, f"{H.iat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w")
    fig.colorbar(im, ax=ax, label="consensus share of |SHAP|")
    ax.set_title("Top early-life features by horizon")
    fig.tight_layout()
    fig.savefig(figs / "early_top_features_heatmap.png", dpi=150)
    plt.close(fig)

    for T in Ts:
        c = cons[cons["T"] == T].sort_values("group_rank")
        print(f"T={T}: " + ", ".join(f"{g} {s:.2f}" for g, s in zip(c.group, c.consensus_share)))


if __name__ == "__main__":
    main()
