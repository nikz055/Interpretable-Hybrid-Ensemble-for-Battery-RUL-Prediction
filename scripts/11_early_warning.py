"""Step 11: predictive-maintenance view - early-failure alarms, protocol effect,
and per-cell maintenance reports.

Early failure := cycle life <= the lower tertile of the TRAINING labels.
For every horizon T:
  * regression-based alarms: flag a cell when its predicted life <= c, where c
    is the smallest cutoff giving >= target recall on the OOF predictions of
    the training cells (test labels never touch the cutoff);
  * direct logistic classifiers (class-balanced, C tuned by grouped CV on
    ROC-AUC) on (i) all features, (ii) degradation features only (no
    charging-protocol proxies), (iii) protocol metadata only (C1, Q1, C2).
Metrics on the 42 test cells: recall, precision, F1, false-alarm rate,
balanced accuracy, ROC-AUC, PR-AUC (average precision).

Also: errors on test cells whose charging protocol was / was not seen in
training, and a per-cell maintenance report at the report horizon that lists
which mechanism groups pulled each cell's predicted life down (consensus group
SHAP from step 9).
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
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from batteryrul.config import load_config, repo_path  # noqa: E402
from batteryrul.features.groups import GROUP_LABEL, PROTOCOL_GROUP, groups_for  # noqa: E402
from batteryrul.models import Winsorizer, repeated_group_folds  # noqa: E402
from batteryrul.splits import fold_indices, read_manifests  # noqa: E402

warnings.filterwarnings("ignore")
REG_MODELS = ["plsr", "ridge", "rf", "xgb", "mlp", "mean_rf_xgb_mlp", "stack_rf_xgb_mlp"]
PRETTY = {"plsr": "PLSR", "ridge": "Ridge", "rf": "RF", "xgb": "XGBoost", "mlp": "MLP",
          "mean_rf_xgb_mlp": "Equal-weight ensemble", "stack_rf_xgb_mlp": "Constrained stack",
          "logit_all": "Logistic (all features)", "logit_degradation": "Logistic (degradation only)",
          "logit_protocol": "Logistic (protocol metadata only)"}


def recall_cutoff(score_oof, y_fail, target):
    """Smallest cutoff c with recall(score <= c) >= target on OOF (higher precision)."""
    for c in np.sort(np.unique(score_oof)):
        flag = score_oof <= c
        if flag[y_fail].mean() >= target:
            return float(c)
    return float(np.max(score_oof))


def alarm_metrics(y_fail, flag, risk):
    tp, fp = int((flag & y_fail).sum()), int((flag & ~y_fail).sum())
    fn, tn = int((~flag & y_fail).sum()), int((~flag & ~y_fail).sum())
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    fpr = fp / max(fp + tn, 1)
    return {"recall": rec, "precision": prec, "F1": 2 * prec * rec / max(prec + rec, 1e-12),
            "false_alarm_rate": fpr, "balanced_acc": (rec + 1 - fpr) / 2, "n_alarms": tp + fp,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "ROC_AUC": roc_auc_score(y_fail, risk), "PR_AUC": average_precision_score(y_fail, risk)}


def logistic(Xtr, ytr_fail, splits_tune, seed):
    pipe = Pipeline([("winsor", Winsorizer()), ("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=seed))])
    gs = GridSearchCV(pipe, {"clf__C": list(np.logspace(-3, 2, 11))}, cv=splits_tune, scoring="roc_auc", n_jobs=-1)
    return gs.fit(Xtr, ytr_fail).best_estimator_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ecfg = cfg["experiments"]["early_warning"]
    res = repo_path(cfg["output"]["results_dir"])
    out, figs = res / "runs" / "early_warning", res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    train, test, folds = read_manifests(repo_path(cfg["split"]["manifest_dir"]))
    oof_splits = [(trn, val) for _, trn, val in fold_indices(train, folds)]
    fs, target = ecfg["feature_set"], ecfg["target_recall"]
    proto = pd.read_csv(res / "tables" / "protocols.csv").set_index("cell_id")

    labels = pd.read_csv(repo_path(cfg["split"]["manifest_dir"]) / "train_cells.csv").set_index("cell_id").cycle_life
    L_star = float(np.quantile(labels.loc[train], ecfg["failure_quantile"]))
    print(f"early failure := cycle life <= {L_star:.1f} cycles (training-label quantile {ecfg['failure_quantile']:.3f})")

    rows, scores_tab = [], []
    for T in ecfg["horizons"]:
        df = pd.read_csv(res / "features" / f"features_{fs}_T{T}.csv", index_col="cell_id")
        cols = [c for c in df.columns if c not in ("partition", "cycle_life")]
        ytr, yte = df.loc[train, "cycle_life"].to_numpy(), df.loc[test, "cycle_life"].to_numpy()
        ftr, fte = ytr <= L_star, yte <= L_star
        preds = pd.read_csv(res / "runs" / f"T{T}_{fs}" / "predictions.csv")

        # regression-based alarms, per seed
        for m in REG_MODELS:
            pm = preds[preds.model == m]
            for s, g in pm.groupby("seed"):
                oof = g[g.partition == "oof"].set_index("cell_id").loc[train, "y_pred"].to_numpy()
                te = g[g.partition == "test"].set_index("cell_id").loc[test, "y_pred"].to_numpy()
                c = recall_cutoff(oof, ftr, target)
                r = alarm_metrics(fte, te <= c, -te)
                rows.append({"T": T, "model": m, "kind": "regression", "seed": s, "cutoff_cycles": c,
                             "oof_PR_AUC": average_precision_score(ftr, -oof), **r})
                if s == pm.seed.min() or m in ("plsr", "ridge"):
                    pass
            # seed-averaged scores for the maintenance report
            avg_oof = pm[pm.partition == "oof"].groupby("cell_id").y_pred.mean().loc[train].to_numpy()
            avg_te = pm[pm.partition == "test"].groupby("cell_id").y_pred.mean().loc[test].to_numpy()
            scores_tab.append((T, m, "regression", avg_oof, avg_te, recall_cutoff(avg_oof, ftr, target)))

        # direct classifiers
        groups = groups_for(cols)
        deg_cols = [c for j, c in enumerate(cols) if j not in groups[PROTOCOL_GROUP]]
        tune = repeated_group_folds(train, pd.Series(ytr, index=train), cfg["split"]["n_folds"],
                                    cfg["tuning"]["repeats"], cfg["tuning"]["seed"])
        for name, Xtr, Xte in [("logit_all", df.loc[train, cols].to_numpy(float), df.loc[test, cols].to_numpy(float)),
                               ("logit_degradation", df.loc[train, deg_cols].to_numpy(float), df.loc[test, deg_cols].to_numpy(float)),
                               ("logit_protocol", proto.loc[train, ["c1", "soc_switch", "c2"]].to_numpy(float),
                                proto.loc[test, ["c1", "soc_switch", "c2"]].to_numpy(float))]:
            est = logistic(Xtr, ftr, tune, 0)
            oof = np.zeros(len(train))
            for trn, val in oof_splits:
                from sklearn.base import clone
                oof[val] = clone(est).fit(Xtr[trn], ftr[trn]).predict_proba(Xtr[val])[:, 1]
            p_te = est.fit(Xtr, ftr).predict_proba(Xte)[:, 1]
            c = recall_cutoff(-oof, ftr, target)                # score = -probability
            r = alarm_metrics(fte, -p_te <= c, p_te)
            rows.append({"T": T, "model": name, "kind": "classifier", "seed": 0, "cutoff_cycles": np.nan,
                         "prob_threshold": -c, "C": est.named_steps["clf"].C,
                         "oof_PR_AUC": average_precision_score(ftr, oof), **r})
            scores_tab.append((T, name, "classifier", -oof, -p_te, c))
        print(f"T={T}: done ({int(fte.sum())} early failures among {len(test)} test cells)")

    raw = pd.DataFrame(rows)
    raw.to_csv(out / "early_warning_raw.csv", index=False)
    metrics = ["recall", "precision", "F1", "false_alarm_rate", "balanced_acc", "ROC_AUC", "PR_AUC", "n_alarms", "oof_PR_AUC"]
    summ = raw.groupby(["T", "model", "kind"])[metrics].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(out / "early_warning_summary.csv", index=False)

    # ---- protocol seen vs unseen (seed-averaged regression predictions) ----
    seen = proto.loc[test, "protocol_seen_in_train"].to_numpy(bool)
    tr_proto = proto.loc[train, "protocol"]
    lookup = np.array([labels.loc[tr_proto[tr_proto == proto.loc[c, "protocol"]].index].mean() if s else np.nan
                       for c, s in zip(test, seen)])
    pr_rows = []
    for T in ecfg["horizons"]:
        preds = pd.read_csv(res / "runs" / f"T{T}_{fs}" / "predictions.csv")
        y = preds[preds.partition == "test"].drop_duplicates("cell_id").set_index("cell_id").loc[test, "y_true"].to_numpy()
        candidates = {m: preds[(preds.model == m) & (preds.partition == "test")].groupby("cell_id").y_pred.mean().loc[test].to_numpy()
                      for m in REG_MODELS}
        candidates["protocol_lookup"] = lookup
        normal = y <= 1100                                     # excludes the 3.6C cells (life ~2000) that confound the comparison
        for m, p in candidates.items():
            for grp_name, mask in (("seen", seen), ("unseen", ~seen), ("seen_life<=1100", seen & normal),
                                   ("unseen_life<=1100", ~seen & normal)):
                ok = mask & np.isfinite(p)
                if ok.sum() == 0:
                    continue
                e = p[ok] - y[ok]
                pr_rows.append({"T": T, "model": m, "group": grp_name, "n": int(ok.sum()),
                                "mean_life": float(y[ok].mean()), "MAE": float(np.mean(np.abs(e))),
                                "RMSE": float(np.sqrt(np.mean(e ** 2))), "MAPE_pct": float(100 * np.mean(np.abs(e) / y[ok]))})
    pd.DataFrame(pr_rows).to_csv(out / "protocol_seen_unseen.csv", index=False)

    # ---- maintenance model + per-cell report at the report horizon ----------
    Tr = ecfg["report_horizon"]
    cand = summ[summ["T"] == Tr].sort_values("oof_PR_AUC_mean", ascending=False)
    chosen = cand.iloc[0].model                                  # chosen on training OOF only
    _, _, kind, s_oof, s_te, cut = next(x for x in scores_tab if x[0] == Tr and x[1] == chosen)
    flags = s_te <= cut
    grp = pd.read_csv(out / "group_shap_cells.csv")
    g = grp[(grp["T"] == Tr) & (grp.partition == "test")]
    cons = g.groupby(["model", "cell_id", "group"]).shap.mean().groupby(["cell_id", "group"]).mean().unstack()
    feat = pd.read_csv(out / "feature_shap_seed0.csv")
    feat = feat[feat["T"] == Tr].groupby(["cell_id", "feature", "group"]).shap.mean().reset_index()
    preds = pd.read_csv(res / "runs" / f"T{Tr}_{fs}" / "predictions.csv")
    cons_life = (preds[(preds.partition == "test") & preds.model.isin(ecfg["models"])]
                 .groupby(["model", "cell_id"]).y_pred.mean().groupby("cell_id").mean())
    ytrue = preds[preds.partition == "test"].drop_duplicates("cell_id").set_index("cell_id").y_true
    rep = []
    for i, c in enumerate(test):
        contrib = cons.loc[c]
        deg = contrib.drop(PROTOCOL_GROUP).sort_values()
        down = [(k, v) for k, v in deg.items() if v < 0][:3]
        top_group = down[0][0] if down else deg.index[0]
        fg = feat[(feat.cell_id == c) & (feat.group == top_group)].nsmallest(3, "shap")
        rep.append({"cell_id": c, "true_life": ytrue[c], "true_early_failure": bool(ytrue[c] <= L_star),
                    "consensus_pred_life": round(cons_life[c], 1),
                    "alarm": bool(flags[i]), "alarm_model": chosen,
                    "risk_score": round(float(-s_te[i]), 3) if kind == "classifier" else round(float(s_te[i]), 1),
                    "outcome": ("true alarm" if flags[i] and ytrue[c] <= L_star else "false alarm" if flags[i]
                                else "missed failure" if ytrue[c] <= L_star else "correctly not flagged"),
                    "life_pulled_down_by": "; ".join(f"{GROUP_LABEL[k]} {100 * (np.exp(v) - 1):+.0f}%" for k, v in down),
                    "protocol_proxy_effect": f"{100 * (np.exp(contrib[PROTOCOL_GROUP]) - 1):+.0f}%",
                    "key_features_in_top_group": ", ".join(fg.feature)})
    rep = pd.DataFrame(rep).sort_values(["alarm", "consensus_pred_life"], ascending=[False, True])
    rep.to_csv(out / f"maintenance_report_T{Tr}.csv", index=False)

    # ---- figures -------------------------------------------------------------
    show = ["plsr", "mlp", "stack_rf_xgb_mlp", "logit_all", "logit_degradation", "logit_protocol"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for m in show:
        d = summ[summ.model == m].sort_values("T")
        for ax, k in zip(axes, ["recall", "false_alarm_rate", "PR_AUC"]):
            ax.errorbar(d["T"], d[f"{k}_mean"], yerr=d[f"{k}_std"].fillna(0), marker="o", capsize=2, label=PRETTY[m])
    prev = float(np.mean(ytrue.loc[test].to_numpy() <= L_star))
    axes[2].axhline(prev, ls=":", color="grey", label=f"no-skill ({prev:.2f})")
    for ax, t in zip(axes, [f"Recall (alarm set for ≥{target:.0%} OOF recall)", "False-alarm rate", "PR-AUC (average precision)"]):
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("cycles observed T")
        ax.set_xticks(ecfg["horizons"])
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "early_warning_vs_horizon.png", dpi=150)
    plt.close(fig)

    pick = pd.concat([rep[rep.outcome == "true alarm"].head(3), rep[rep.outcome == "false alarm"].head(1),
                      rep[rep.outcome == "missed failure"].head(1),
                      rep[rep.outcome == "correctly not flagged"].tail(1)])
    fig, axes = plt.subplots(1, len(pick), figsize=(3.6 * len(pick), 4.2), sharex=True)
    for ax, (_, r) in zip(np.atleast_1d(axes), pick.iterrows()):
        v = 100 * (np.exp(cons.loc[r.cell_id]) - 1)
        v = v.reindex(v.abs().sort_values().index)
        ax.barh([GROUP_LABEL[k].split(" (")[0] for k in v.index], v.values,
                color=["#c0392b" if x < 0 else "#27ae60" for x in v.values])
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(f"{r.cell_id}: {r.outcome}\ntrue {r.true_life:.0f}, predicted {r.consensus_pred_life:.0f}", fontsize=9)
        ax.set_xlabel("effect on predicted life (%)")
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(f"Maintenance reports after {Tr} cycles: which mechanisms pull predicted life down", fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / "early_maintenance_cards.png", dpi=150)
    plt.close(fig)

    meta = {"early_failure_threshold_cycles": L_star, "target_recall_oof": target,
            "n_test_early_failures": int((ytrue.loc[test] <= L_star).sum()),
            "n_train_early_failures": int((labels.loc[train] <= L_star).sum()),
            "report_horizon": Tr, "maintenance_model_selected_by_oof_PR_AUC": chosen}
    (out / "early_warning_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(summ[summ.model.isin(show)][["T", "model", "recall_mean", "precision_mean", "false_alarm_rate_mean",
                                        "PR_AUC_mean", "ROC_AUC_mean", "oof_PR_AUC_mean"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
