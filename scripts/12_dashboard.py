"""Step 12: interactive Plotly dashboard of the study (docs/index.html).

Built only from committed result files, so it needs no raw data:

    python scripts/12_dashboard.py

The page is self-contained apart from plotly.js (loaded from the Plotly CDN)
and is served by GitHub Pages from the repository's docs/ folder.

Design notes: colours are the validated default data-viz palette (categorical
slots 1-6, an ordinal blue ramp for the three horizons, blue/red for
diverging effects), re-stepped for dark mode rather than inverted; every chart
has a hover layer and a table view; controls sit in one row above the chart
they scope.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
from plotly.offline import get_plotlyjs_version  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from batteryrul.config import load_config, repo_path  # noqa: E402
from batteryrul.features.groups import GROUP_LABEL, MECHANISM_GROUPS, PROTOCOL_GROUP  # noqa: E402

REPO_URL = "https://github.com/nikz055/Interpretable-Hybrid-Ensemble-for-Battery-RUL-Prediction"

# ---- validated palette (light / dark steps of the same hues) ----------------
PAL = {
    "light": {"surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "ink2": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "cat": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
              "ord": ["#86b6ef", "#2a78d6", "#104281"], "pos": "#2a78d6", "neg": "#e34948",
              "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]},
    "dark": {"surface": "#1a1a19", "page": "#0d0d0d", "ink": "#f0efec", "ink2": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
             "cat": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
             "ord": ["#cde2fb", "#5598e7", "#184f95"], "pos": "#3987e5", "neg": "#e66767",
             "seq": ["#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]},
}
L = PAL["light"]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
HORIZONS = [20, 50, 100]
GROUP_ORDER = list(MECHANISM_GROUPS)
SHORT = {g: GROUP_LABEL[g].split(" (")[0] for g in GROUP_ORDER}
MODEL_NAME = {"consensus": "Consensus of 4 models", "plsr": "PLSR", "rf": "Random forest", "xgb": "XGBoost",
              "mlp": "MLP", "ridge": "Ridge", "stack_rf_xgb_mlp": "Stacked ensemble",
              "mean_rf_xgb_mlp": "Equal-weight ensemble", "logit_all": "Logistic, all features",
              "logit_degradation": "Logistic, degradation only", "logit_protocol": "Logistic, protocol only"}


def meta(role, i=0, opt=None):
    m = {"role": role, "i": i}
    if opt is not None:
        m["opt"] = opt
    return m


def base_layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=16, t=16, b=8), font=dict(family=FONT, size=13, color=L["ink2"]),
        paper_bgcolor=L["surface"], plot_bgcolor=L["surface"], barcornerradius=4,
        hoverlabel=dict(bgcolor=L["surface"], bordercolor=L["grid"], font=dict(family=FONT, color=L["ink"])),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""), **kw)
    fig.update_xaxes(gridcolor=L["grid"], linecolor=L["axis"], zerolinecolor=L["axis"], tickfont_color=L["muted"],
                     automargin=True, showline=True)
    fig.update_yaxes(gridcolor=L["grid"], linecolor=L["axis"], zerolinecolor=L["axis"], tickfont_color=L["muted"],
                     automargin=True)
    return fig


def table_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, classes="tbl", border=0, float_format=lambda v: f"{v:,.3g}", escape=True)


# ---------------------------------------------------------------- figures ----
def fig_group_importance(ew: Path):
    g = pd.read_csv(ew / "group_importance_by_horizon.csv")
    fig = go.Figure()
    opts = ["consensus", "plsr", "rf", "xgb", "mlp"]
    for o in opts:
        col = "consensus_share" if o == "consensus" else f"share_{o}"
        for k, T in enumerate(HORIZONS):
            d = g[g["T"] == T].set_index("group").reindex(GROUP_ORDER)
            fig.add_bar(x=[SHORT[x] for x in d.index], y=d[col], name=f"{T} cycles", visible=(o == "consensus"),
                        marker_color=L["ord"][k], meta=meta("ord", k, o), showlegend=True,
                        customdata=np.stack([d["between_model_std"], d["share_plsr"], d["share_rf"],
                                             d["share_xgb"], d["share_mlp"]], axis=1),
                        hovertemplate=(f"<b>%{{x}}</b> · after {T} cycles<br>{MODEL_NAME[o]}: %{{y:.0%}}"
                                       "<br>PLSR %{customdata[1]:.0%} · RF %{customdata[2]:.0%} · "
                                       "XGB %{customdata[3]:.0%} · MLP %{customdata[4]:.0%}<extra></extra>"))
    base_layout(fig, 420, bargap=0.25, bargroupgap=0.06)
    fig.update_yaxes(title_text="Share of SHAP attribution", tickformat=".0%", rangemode="tozero")
    tab = g.assign(group=g.group.map(SHORT))[["T", "group", "consensus_share", "between_model_std",
                                              "share_plsr", "share_rf", "share_xgb", "share_mlp"]]
    controls = {"label": "Model", "options": [(o, MODEL_NAME[o]) for o in opts]}
    return fig, tab, controls


def fig_feature_heatmap(ew: Path):
    s = pd.read_csv(ew / "feature_stability_by_horizon.csv")
    top = sorted(set().union(*[set(s[s["T"] == T].nsmallest(8, "rank").feature) for T in HORIZONS]))
    piv = s.pivot(index="feature", columns="T", values="consensus_share").loc[top]
    piv = piv.loc[piv.mean(1).sort_values().index]               # strongest at the top of the chart
    info = s.set_index(["feature", "T"])
    cd = [[[f"{info.loc[(f, T), 'top10_frequency']:.0%}", info.loc[(f, T), "learned_sign_majority"],
            info.loc[(f, T), "plausibility"], GROUP_LABEL[info.loc[(f, T), "group"]]] for T in HORIZONS]
          for f in piv.index]
    fig = go.Figure(go.Heatmap(
        z=piv.to_numpy(), x=[f"{T} cycles" for T in HORIZONS], y=list(piv.index), customdata=cd,
        colorscale=[[i / 6, c] for i, c in enumerate(L["seq"])], xgap=2, ygap=2, meta=meta("seq"),
        colorbar=dict(title="share", tickformat=".0%", outlinewidth=0, thickness=10),
        hovertemplate=("<b>%{y}</b> · after %{x}<br>share of attribution %{z:.1%}<br>top-10 in %{customdata[0]} of runs"
                       "<br>learned sign %{customdata[1]} · %{customdata[2]}<br>%{customdata[3]}<extra></extra>")))
    base_layout(fig, 40 + 26 * len(piv))
    fig.update_xaxes(side="top", showline=False, showgrid=False)
    fig.update_yaxes(showgrid=False)
    tab = s[s.feature.isin(top)][["T", "feature", "group", "consensus_share", "top10_frequency",
                                  "learned_sign_majority", "expected_sign", "plausibility"]].sort_values(["T", "consensus_share"], ascending=[True, False])
    return fig, tab, None


def fig_ablation(ew: Path):
    a = pd.read_csv(ew / "group_ablation_consensus.csv")
    fig = go.Figure()
    for o, prefix, col, ylab in (("drop", "drop:", "d_cv_mae", "Δ"), ("only", "only:", "cv_mae", "")):
        for k, T in enumerate(HORIZONS):
            d = a[(a["T"] == T) & a.config.str.startswith(prefix)].copy()
            d["group"] = d.config.str[len(prefix):]
            d = d.set_index("group").reindex(GROUP_ORDER)
            fig.add_bar(x=[SHORT[x] for x in d.index], y=d[col], name=f"{T} cycles", visible=(o == "drop"),
                        marker_color=L["ord"][k], meta=meta("ord", k, o),
                        hovertemplate=(f"<b>%{{x}}</b> · after {T} cycles<br>"
                                       + ("CV MAE change when removed: %{y:+.1f} cycles" if o == "drop"
                                          else "CV MAE using only this group: %{y:.1f} cycles") + "<extra></extra>"))
    base_layout(fig, 400, bargap=0.25, bargroupgap=0.06)
    fig.update_yaxes(title_text="cycles", zeroline=True)
    fig.layout.meta = {"ytitle": {"drop": "Δ CV MAE (cycles)", "only": "CV MAE (cycles)"}}
    fig.update_yaxes(title_text=fig.layout.meta["ytitle"]["drop"])
    tab = a[["T", "config", "cv_mae", "d_cv_mae", "cv_rmse", "d_cv_rmse"]]
    controls = {"label": "View", "options": [("drop", "Remove one group (accuracy lost)"),
                                             ("only", "Use only one group (accuracy kept)")]}
    return fig, tab, controls


def fig_alarms(ew: Path):
    w = pd.read_csv(ew / "early_warning_summary.csv")
    meta_json = json.loads((ew / "early_warning_meta.json").read_text())
    models = ["plsr", "mlp", "stack_rf_xgb_mlp", "logit_all", "logit_degradation", "logit_protocol"]
    panels = [("recall", "Failures caught (recall)"), ("false_alarm_rate", "False-alarm rate"),
              ("PR_AUC", "Ranking quality (PR-AUC)")]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[p[1] for p in panels], horizontal_spacing=0.07)
    for i, m in enumerate(models):
        d = w[w.model == m].sort_values("T")
        for j, (k, title) in enumerate(panels):
            fig.add_scatter(x=d["T"], y=d[f"{k}_mean"], error_y=dict(array=d[f"{k}_std"].fillna(0), thickness=1, width=0),
                            mode="lines+markers", name=MODEL_NAME[m], legendgroup=m, showlegend=(j == 0),
                            line=dict(color=L["cat"][i], width=2), marker=dict(size=8, color=L["cat"][i],
                                                                              line=dict(color=L["surface"], width=2)),
                            meta=meta("cat", i), row=1, col=j + 1,
                            hovertemplate=f"<b>{MODEL_NAME[m]}</b><br>after %{{x}} cycles<br>{title}: %{{y:.2f}}<extra></extra>")
    prev = meta_json["n_test_early_failures"] / 42
    fig.add_hline(y=prev, line=dict(color=L["muted"], width=1), row=1, col=3,
                  annotation_text=f"chance {prev:.2f}", annotation_position="bottom right")
    base_layout(fig, 400)
    fig.update_xaxes(tickvals=HORIZONS, title_text="cycles observed")
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig.update_yaxes(tickformat=".2f", row=1, col=3)
    fig.update_layout(legend=dict(yanchor="top", y=-0.28), margin=dict(t=40, b=8), height=460)
    tab = w[w.model.isin(models)][["T", "model", "recall_mean", "precision_mean", "false_alarm_rate_mean",
                                   "ROC_AUC_mean", "PR_AUC_mean", "oof_PR_AUC_mean"]]
    return fig, tab, None


def fig_feature_explorer(res: Path, threshold: float):
    df = pd.read_csv(res / "features" / "features_C_T50.csv")
    feats = [("v_dis_avg_std", "Discharge-voltage variability, cycles 1-50 (V)"),
             ("dv_dis_avg", "Discharge-voltage drift, cycle 2 → 50 (V)"),
             ("dq_var_log", "ΔQ(V) variance, cycle 10 → 50 (log10 Ah²)"),
             ("ir_ref", "Internal resistance at cycle 2 (Ω)"),
             ("qd_retention", "Capacity retention at cycle 50 (–)"),
             ("i_chg_std", "Charge-current variability (A) — protocol proxy")]
    fig = go.Figure()
    early = df.cycle_life <= threshold
    for f, label in feats:
        for k, (mask, name) in enumerate(((~early, "Normal life"), (early, f"Early failure (≤ {threshold:.0f} cycles)"))):
            d = df[mask]
            fig.add_scatter(x=d[f], y=d.cycle_life, mode="markers", name=name, visible=(f == feats[0][0]),
                            marker=dict(size=9, color=[L["pos"], L["neg"]][k], line=dict(color=L["surface"], width=2)),
                            meta=meta("two", k, f), customdata=np.stack([d.cell_id, d.partition], axis=1),
                            hovertemplate=(f"<b>%{{customdata[0]}}</b> ({'%{customdata[1]}'})<br>{label}: %{{x:.4g}}"
                                           "<br>cycle life %{y:.0f}<extra></extra>"))
    fig.add_hline(y=threshold, line=dict(color=L["muted"], width=1))
    # annotations on a log axis are positioned in log10 units
    fig.add_annotation(x=0, xref="paper", y=np.log10(threshold), yref="y", text=f"early-failure threshold {threshold:.0f}",
                       showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=12, color=L["ink2"]))
    base_layout(fig, 430)
    ticks = [300, 400, 500, 700, 1000, 1500, 2000]
    fig.update_yaxes(type="log", title_text="Cycle life (log scale)", tickvals=ticks, ticktext=[str(t) for t in ticks])
    fig.layout.meta = {"xtitle": dict(feats)}
    fig.update_xaxes(title_text=feats[0][1])
    tab = df[["cell_id", "partition", "cycle_life"] + [f for f, _ in feats]]
    controls = {"label": "Feature after 50 cycles", "options": feats}
    return fig, tab, controls


def fig_maintenance(ew: Path, T: int):
    rep = pd.read_csv(ew / f"maintenance_report_T{T}.csv")
    grp = pd.read_csv(ew / "group_shap_cells.csv")
    g = grp[(grp["T"] == T) & (grp.partition == "test")]
    cons = g.groupby(["model", "cell_id", "group"]).shap.mean().groupby(["cell_id", "group"]).mean().unstack()
    pct = (100 * (np.exp(cons) - 1)).reindex(columns=GROUP_ORDER)
    fig = go.Figure()
    titles = {}
    order = list(reversed(GROUP_ORDER))                           # first group at the top
    for _, r in rep.iterrows():
        v = pct.loc[r.cell_id, order]
        fig.add_bar(x=v.values, y=[SHORT[x] for x in order], orientation="h", visible=(r.cell_id == rep.cell_id.iloc[0]),
                    marker_color=[L["pos"] if x >= 0 else L["neg"] for x in v.values], meta=meta("div", 0, r.cell_id),
                    showlegend=False, hovertemplate="<b>%{y}</b><br>effect on predicted life: %{x:+.1f}%<extra></extra>")
        titles[r.cell_id] = (f"{r.cell_id}: {r.outcome} · true life {r.true_life:.0f} · predicted {r.consensus_pred_life:.0f}"
                             f" · {'ALARM' if r.alarm else 'no alarm'}")
    base_layout(fig, 360)
    lim = float(np.nanmax(np.abs(pct.to_numpy()))) * 1.08        # one fixed scale so cells are comparable
    fig.update_xaxes(title_text="Effect on predicted cycle life (%) — red pulls life down", zeroline=True,
                     zerolinewidth=1, ticksuffix="%", range=[-lim, lim])
    fig.layout.meta = {"caption": titles}
    opts = [(c, f"{c} — {o}") for c, o in zip(rep.cell_id, rep.outcome)]
    tab = pct.reset_index().merge(rep[["cell_id", "true_life", "consensus_pred_life", "alarm", "outcome"]], on="cell_id")
    tab.columns = [SHORT.get(c, c) for c in tab.columns]
    controls = {"label": "Test cell", "options": opts}
    return fig, tab.round(1), controls, titles


def fig_benchmark(res: Path):
    s = pd.read_csv(res / "runs" / "T100_C" / "metrics_summary.csv")
    t = s[s.partition == "test"].set_index("model")
    names = ["rf", "xgb", "mean_rf_xgb_mlp", "ridge", "stack_rf_xgb_mlp", "plsr", "mlp"]
    d = t.loc[names].sort_values("RMSE_mean", ascending=False)
    fig = go.Figure(go.Bar(x=d.RMSE_mean, y=[MODEL_NAME[n] for n in d.index], orientation="h",
                           error_x=dict(array=d.RMSE_std, thickness=1, width=0, color=L["muted"]),
                           marker_color=L["cat"][0], meta=meta("cat", 0),
                           customdata=np.stack([d.RMSE_std, d.MAE_mean], axis=1),
                           hovertemplate="<b>%{y}</b><br>test RMSE %{x:.1f} ± %{customdata[0]:.1f} cycles (10 seeds)"
                                         "<br>test MAE %{customdata[1]:.1f}<extra></extra>"))
    bml = pd.read_csv(res / "tables" / "batteryml_reproduction.csv").set_index("model").loc["PLSR (QdLinear)", "RMSE"]
    fig.add_vline(x=bml, line=dict(color=L["muted"], width=1),
                  annotation_text=f"BatteryML PLSR baseline {bml:.1f}", annotation_position="bottom right")
    base_layout(fig, 360, bargap=0.35)
    fig.update_xaxes(title_text="Test RMSE (cycles), 42 locked test cells, after 100 cycles", rangemode="tozero")
    tab = d[["RMSE_mean", "RMSE_std", "MAE_mean", "R2_mean"]].reset_index()
    return fig, tab.round(2), None


def fig_pred_true(res: Path):
    p = pd.read_csv(res / "predictions" / "test_predictions_main_seed_avg.csv")
    opts = ["plsr", "mlp", "stack_rf_xgb_mlp", "xgb", "rf"]
    fig = go.Figure()
    hi = float(max(p.y_true.max(), p[opts].max().max())) * 1.05
    for o in opts:
        err = p[o] - p.y_true
        fig.add_scatter(x=p.y_true, y=p[o], mode="markers", name=MODEL_NAME[o], visible=(o == opts[0]), showlegend=False,
                        marker=dict(size=9, color=L["cat"][0], line=dict(color=L["surface"], width=2)), meta=meta("cat", 0, o),
                        customdata=np.stack([p.cell_id, err], axis=1),
                        hovertemplate="<b>%{customdata[0]}</b><br>true %{x:.0f} · predicted %{y:.0f}<br>error %{customdata[1]:+.0f} cycles<extra></extra>")
    fig.add_shape(type="line", x0=0, y0=0, x1=hi, y1=hi, line=dict(color=L["muted"], width=1))
    base_layout(fig, 430)
    fig.update_xaxes(title_text="True cycle life", range=[0, hi])
    fig.update_yaxes(title_text="Predicted cycle life (mean of 10 seeds)", range=[0, hi])
    controls = {"label": "Model", "options": [(o, MODEL_NAME[o]) for o in opts]}
    return fig, p[["cell_id", "y_true"] + opts].round(1), controls


def fig_robustness(res: Path):
    r = pd.read_csv(res / "runs" / "robustness" / "robustness_summary.csv")
    conds = [("gaussian_noise", "Sensor noise (1 mAh, 5 mV)"), ("missing_10pct", "10% values missing"),
             ("missing_20pct", "20% values missing"), ("sparse_every5", "Only every 5th cycle")]
    models = ["ridge", "plsr", "rf", "xgb", "mlp", "stack_rf_xgb_mlp"]
    fig = go.Figure()
    for k, (c, label) in enumerate(conds):
        d = r[r.condition == c].set_index("model").loc[models]
        fig.add_bar(x=[MODEL_NAME[m] for m in models], y=d.dRMSE_mean, name=label, marker_color=L["cat"][k],
                    meta=meta("cat", k), error_y=dict(array=d.dRMSE_std, thickness=1, width=0, color=L["muted"]),
                    hovertemplate=f"<b>%{{x}}</b> · {label}<br>RMSE change %{{y:+.1f}} cycles<extra></extra>")
    base_layout(fig, 400, bargap=0.25, bargroupgap=0.06)
    fig.update_yaxes(title_text="Change in test RMSE vs clean input (cycles)", zeroline=True)
    tab = r[r.condition != "clean"][["condition", "model", "dRMSE_mean", "dRMSE_std", "dMAE_mean", "dMAE_std"]]
    return fig, tab.round(1), None


def fig_v2_tradeoff(res: Path):
    """False alarms vs missed failures: keep-testing policies vs forced yes/no alarms."""
    v2 = res / "runs" / "v2"
    s = pd.read_csv(v2 / "termination_summary.csv")
    s = s[s.split == "resampling"]
    W = s.pivot_table(index=["policy", "alpha"], columns="metric", values="mean").reset_index()
    LO = s.pivot_table(index=["policy", "alpha"], columns="metric", values="lo95").reset_index()
    HI = s.pivot_table(index=["policy", "alpha"], columns="metric", values="hi95").reset_index()
    f = pd.read_csv(v2 / "forced_alarms.csv")
    f = f[(f.split == "resampling") & f.policy.isin(["forced_alarm_50", "forced_alarm_100"])]
    series = [("single_50", "Decide at cycle 50, else keep testing"), ("single_100", "Decide at cycle 100, else keep testing"),
              ("from_50", "Check at 50 and 100"), ("every_checkpoint", "Check at every checkpoint (5-100)")]
    fig = go.Figure()
    for i, (pol, label) in enumerate(series):
        d, lo, hi = (x[x.policy == pol].sort_values("alpha") for x in (W, LO, HI))
        fig.add_scatter(x=100 * d.missed_failure_rate, y=100 * d.false_alarm_rate, mode="lines+markers", name=label,
                        line=dict(color=L["cat"][i], width=2), marker=dict(size=9, color=L["cat"][i], line=dict(color=L["surface"], width=2)),
                        meta=meta("cat", i),
                        customdata=np.stack([d.alpha, 100 * d.test_time_saved, 100 * d.decided_early,
                                             100 * lo.false_alarm_rate.values, 100 * hi.false_alarm_rate.values,
                                             100 * lo.missed_failure_rate.values, 100 * hi.missed_failure_rate.values], axis=1),
                        hovertemplate=(f"<b>{label}</b><br>α = %{{customdata[0]}}"
                                       "<br>false alarms %{y:.0f}% (95% CI %{customdata[3]:.0f}-%{customdata[4]:.0f}%)"
                                       "<br>missed failures %{x:.0f}% (95% CI %{customdata[5]:.0f}-%{customdata[6]:.0f}%)"
                                       "<br>test time saved %{customdata[1]:.0f}% · decided early %{customdata[2]:.0f}%<extra></extra>"))
    for j, (pol, label) in enumerate([("forced_alarm_50", "Forced yes/no alarm at 50"), ("forced_alarm_100", "Forced yes/no alarm at 100")]):
        d = f[f.policy == pol].sort_values("recall_target")
        k = 4 + j
        fig.add_scatter(x=100 * d.missed_failure_rate, y=100 * d.false_alarm_rate, mode="lines+markers", name=label,
                        line=dict(color=L["cat"][k], width=2), marker=dict(size=9, symbol="square", color=L["cat"][k],
                                                                         line=dict(color=L["surface"], width=2)),
                        meta=meta("cat", k), customdata=d.recall_target,
                        hovertemplate=f"<b>{label}</b><br>alarm tuned for %{{customdata:.0%}} recall<br>false alarms %{{y:.0f}}% · missed failures %{{x:.0f}}%<extra></extra>")
    base_layout(fig, 460)
    fig.update_xaxes(title_text="Missed early failures (%)", rangemode="tozero", ticksuffix="%")
    fig.update_yaxes(title_text="False alarms on healthy cells (%)", rangemode="tozero", ticksuffix="%")
    fig.update_layout(legend=dict(yanchor="top", y=-0.2), margin=dict(t=16))
    tab = s[s.metric.isin(["false_alarm_rate", "missed_failure_rate", "decided_early", "test_time_saved"])][
        ["policy", "checkpoints", "alpha", "metric", "mean", "lo95", "hi95"]].round(3)
    return fig, tab, None


def fig_v2_horizon(res: Path):
    m = pd.read_csv(res / "runs" / "v2" / "metrics.csv")
    m = m[m.split == "resampling"]
    fig = go.Figure()
    for i, (key, label) in enumerate([("reg_base_MAE", "v1: PLSR on engineered features"),
                                      ("reg_new_MAE", "v2: pipeline chosen by nested CV")]):
        d = m[m.metric == key].sort_values("T")
        fig.add_scatter(x=d["T"], y=d["mean"], mode="lines+markers", name=label,
                        error_y=dict(type="data", symmetric=False, array=d.hi95 - d["mean"], arrayminus=d["mean"] - d.lo95,
                                     thickness=1, width=0),
                        line=dict(color=L["cat"][i], width=2), marker=dict(size=9, color=L["cat"][i], line=dict(color=L["surface"], width=2)),
                        meta=meta("cat", i), customdata=np.stack([d.lo95, d.hi95], axis=1),
                        hovertemplate=f"<b>{label}</b><br>after %{{x}} cycles: MAE %{{y:.1f}} cycles<br>95% CI %{{customdata[0]:.1f}}-%{{customdata[1]:.1f}}<extra></extra>")
    base_layout(fig, 400)
    fig.update_xaxes(type="log", tickvals=[5, 10, 20, 30, 50, 100], title_text="cycles observed")
    fig.update_yaxes(title_text="Cycle-life MAE (cycles)", rangemode="tozero")
    tab = m[m.metric.isin(["reg_new_MAE", "reg_base_MAE", "reg_new_RMSE", "reg_base_RMSE", "reg_new_coverage90"])][
        ["T", "metric", "mean", "lo95", "hi95"]].round(3)
    return fig, tab, None


# ------------------------------------------------------------------- page ----
CSS = """
:root { color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9;
  --border:rgba(11,11,11,0.10); --accent:#2a78d6; --neg:#e34948; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) { color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a;
  --border:rgba(255,255,255,0.10); --accent:#3987e5; --neg:#e66767; } }
:root[data-theme="dark"] { color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a;
  --border:rgba(255,255,255,0.10); --accent:#3987e5; --neg:#e66767; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
.wrap { max-width:1180px; margin:0 auto; padding:24px 16px 64px; }
header { display:flex; gap:16px; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; }
h1 { font-size:26px; line-height:1.25; margin:0 0 6px; }
h2 { font-size:19px; margin:0 0 4px; }
.sub { color:var(--ink2); margin:0; max-width:780px; }
.links a, .links button { color:var(--accent); text-decoration:none; margin-right:14px; font:inherit; }
.links button { background:none; border:1px solid var(--border); border-radius:6px; padding:3px 10px; cursor:pointer; color:var(--ink2); }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; margin:22px 0 8px; }
.tile { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }
.tile .v { font-size:28px; font-weight:650; line-height:1.15; }
.tile .k { color:var(--ink2); font-size:13px; margin-top:4px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:18px 18px 10px; margin-top:18px; }
.card p.cap { color:var(--ink2); margin:2px 0 10px; max-width:900px; }
.controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:8px 0 4px; color:var(--ink2); font-size:14px; }
select { font:inherit; color:var(--ink); background:var(--surface); border:1px solid var(--border); border-radius:6px; padding:5px 8px; max-width:100%; }
.caption { font-weight:600; color:var(--ink); margin:6px 0 0; font-size:14px; }
details { margin:4px 0 6px; color:var(--ink2); font-size:13px; }
summary { cursor:pointer; }
.tblwrap { overflow-x:auto; max-height:340px; margin-top:6px; }
table.tbl { border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:12.5px; color:var(--ink); }
table.tbl th, table.tbl td { padding:4px 10px; border-bottom:1px solid var(--grid); text-align:right; white-space:nowrap; }
table.tbl th { position:sticky; top:0; background:var(--surface); color:var(--ink2); font-weight:600; }
.note { color:var(--ink2); font-size:13.5px; margin-top:26px; }
code { background:var(--grid); padding:1px 5px; border-radius:4px; font-size:13px; }
"""

JS = r"""
const PAL = __PAL__;
function mode() {
  const t = document.documentElement.dataset.theme;
  if (t === 'light' || t === 'dark') return t;
  return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}
function colorFor(P, m, tr) {
  if (!m) return null;
  if (m.role === 'cat') return P.cat[m.i];
  if (m.role === 'ord') return P.ord[m.i];
  if (m.role === 'two') return m.i === 0 ? P.pos : P.neg;
  if (m.role === 'div') return tr.x.map(v => v >= 0 ? P.pos : P.neg);
  return null;
}
function theme(gd) {
  const P = PAL[mode()];
  const lay = { paper_bgcolor: P.surface, plot_bgcolor: P.surface, 'font.color': P.ink2,
    'hoverlabel.bgcolor': P.surface, 'hoverlabel.bordercolor': P.grid, 'hoverlabel.font.color': P.ink,
    'legend.font.color': P.ink2 };
  for (const k of Object.keys(gd._fullLayout)) if (/^[xy]axis\d*$/.test(k)) {
    Object.assign(lay, { [k + '.gridcolor']: P.grid, [k + '.linecolor']: P.axis, [k + '.zerolinecolor']: P.axis,
      [k + '.tickfont.color']: P.muted, [k + '.title.font.color']: P.ink2 });
  }
  (gd.layout.shapes || []).forEach((s, i) => { lay[`shapes[${i}].line.color`] = P.muted; });
  (gd.layout.annotations || []).forEach((a, i) => { lay[`annotations[${i}].font.color`] = P.ink2; });
  Plotly.relayout(gd, lay);
  gd.data.forEach((tr, i) => {
    const m = tr.meta; if (!m) return;
    if (m.role === 'seq') {
      Plotly.restyle(gd, { colorscale: [P.seq.map((c, j) => [j / (P.seq.length - 1), c])],
        'colorbar.tickfont.color': P.muted, 'colorbar.title.font.color': P.ink2 }, [i]);
      return;
    }
    const c = colorFor(P, m, tr); if (!c) return;
    const upd = { 'marker.color': [c] };
    if (tr.type === 'scatter') Object.assign(upd, { 'line.color': c, 'marker.line.color': P.surface });
    if (tr.error_y && tr.error_y.array) upd['error_y.color'] = tr.type === 'scatter' ? c : P.muted;
    if (tr.error_x && tr.error_x.array) upd['error_x.color'] = P.muted;
    Plotly.restyle(gd, upd, [i]);
  });
}
function choose(gd, value) {
  const vis = gd.data.map(tr => !(tr.meta && tr.meta.opt !== undefined) || tr.meta.opt === value);
  Plotly.restyle(gd, { visible: vis });
  const lm = gd.layout.meta || {};
  if (lm.ytitle) Plotly.relayout(gd, { 'yaxis.title.text': lm.ytitle[value] });
  if (lm.xtitle) Plotly.relayout(gd, { 'xaxis.title.text': lm.xtitle[value] });
  const cap = document.getElementById(gd.id + '-caption');
  if (cap && lm.caption) cap.textContent = lm.caption[value];
}
const FIGS = JSON.parse(document.getElementById('figs').textContent);
const plots = [];
for (const [id, fig] of Object.entries(FIGS)) {
  const gd = document.getElementById(id);
  Plotly.newPlot(gd, fig.data, fig.layout, { responsive: true, displayModeBar: false }).then(() => {
    plots.push(gd); theme(gd);
    const sel = document.getElementById(id + '-select');
    if (sel) { sel.addEventListener('change', e => choose(gd, e.target.value)); choose(gd, sel.value); }
  });
}
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => plots.forEach(theme));
const btn = document.getElementById('theme-btn');
const cycle = { auto: 'light', light: 'dark', dark: 'auto' };
function label() { const t = document.documentElement.dataset.theme || 'auto'; btn.textContent = 'Theme: ' + t; }
btn.addEventListener('click', () => {
  const next = cycle[document.documentElement.dataset.theme || 'auto'];
  if (next === 'auto') delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = next;
  label(); plots.forEach(theme);
});
label();
"""


def section(fid, title, caption, tab, controls=None, extra_caption=None):
    ctl = ""
    if controls:
        opts = "".join(f'<option value="{html.escape(str(v))}">{html.escape(str(t))}</option>' for v, t in controls["options"])
        ctl = (f'<div class="controls"><label for="{fid}-select">{html.escape(controls["label"])}</label>'
               f'<select id="{fid}-select">{opts}</select></div>')
    cap2 = f'<p class="caption" id="{fid}-caption">{html.escape(extra_caption)}</p>' if extra_caption else ""
    return (f'<section class="card" aria-labelledby="{fid}-h"><h2 id="{fid}-h">{html.escape(title)}</h2>'
            f'<p class="cap">{caption}</p>{ctl}{cap2}<div id="{fid}" role="img" aria-label="{html.escape(title)}"></div>'
            f'<details><summary>Table view</summary><div class="tblwrap">{table_html(tab)}</div></details></section>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="docs/index.html")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = repo_path(cfg["output"]["results_dir"])
    ew = res / "runs" / "early_warning"
    m = json.loads((ew / "early_warning_meta.json").read_text())
    thr, Tr = m["early_failure_threshold_cycles"], m["report_horizon"]
    w = pd.read_csv(ew / "early_warning_summary.csv").set_index(["T", "model"])
    plsr50 = w.loc[(50, "plsr")]
    g = pd.read_csv(ew / "group_importance_by_horizon.csv").set_index(["T", "group"])

    f1, t1, c1 = fig_group_importance(ew)
    f2, t2, _ = fig_feature_heatmap(ew)
    f3, t3, c3 = fig_ablation(ew)
    f4, t4, _ = fig_alarms(ew)
    f5, t5, c5 = fig_feature_explorer(res, thr)
    f6, t6, c6, titles = fig_maintenance(ew, Tr)
    f7, t7, _ = fig_benchmark(res)
    f8, t8, c8 = fig_pred_true(res)
    f9, t9, _ = fig_robustness(res)
    fa, ta, _ = fig_v2_tradeoff(res)
    fb, tb, _ = fig_v2_horizon(res)
    figs = {"g-v2-policy": fa, "g-v2-mae": fb, "g-groups": f1, "g-heat": f2, "g-ablation": f3, "g-alarms": f4,
            "g-explorer": f5, "g-maint": f6, "g-bench": f7, "g-pred": f8, "g-robust": f9}

    v2 = res / "runs" / "v2"
    ts = pd.read_csv(v2 / "termination_summary.csv")
    rec = ts[(ts.split == "resampling") & (ts.policy == "single_50") & (ts.alpha == 0.1)].set_index("metric")["mean"]
    forced = pd.read_csv(v2 / "forced_alarms.csv")
    f50 = forced[(forced.split == "resampling") & (forced.policy == "forced_alarm_50") & (forced.recall_target == 0.9)].iloc[0]
    mv2 = pd.read_csv(v2 / "metrics.csv").query("split == 'resampling' and T == 50").set_index("metric")["mean"]
    tiles = [
        (f"{rec['false_alarm_rate']:.0%}", f"False alarms (forced alarm: {f50.false_alarm_rate:.0%})"),
        (f"{rec['missed_failure_rate']:.0%}", f"Missed failures (forced alarm: {f50.missed_failure_rate:.0%})"),
        (f"{rec['test_time_saved']:.0%}", "Test time saved"),
        (f"{mv2['reg_new_MAE']:.0f} cycles", f"Life prediction error, MAE (baseline: {mv2['reg_base_MAE']:.0f})"),
    ]
    tiles_html = "".join(f'<div class="tile"><div class="v">{v}</div><div class="k">{html.escape(k)}</div></div>' for v, k in tiles)

    body = [
        section("g-v2-policy", "1 · Early-warning decisions",
                f"False alarms versus missed failures for each decision policy. Continuing to test uncertain cells at "
                f"cycle 50 lowers both and saves {rec['test_time_saved']:.0%} of test time.", ta),
        section("g-v2-mae", "2 · Life prediction error",
                "Mean absolute error by cycles observed, with 95% confidence intervals.", tb),
        section("g-groups", "3 · Failure indicators by mechanism",
                "Share of SHAP attribution. Discharge-voltage drift dominates from cycle 50.", t1, c1),
        section("g-heat", "4 · Top features",
                "Consensus SHAP importance by cycles observed. Hover for stability and physical consistency.", t2),
        section("g-ablation", "5 · Retraining check",
                "Accuracy change when a feature group is removed or used alone.", t3, c3),
        section("g-alarms", "6 · Alarm performance",
                f"Recall, false-alarm rate and PR-AUC by cycles observed. Early failure: life of {thr:.0f} cycles or less.", t4),
        section("g-explorer", "7 · Feature explorer",
                "How well a single feature at cycle 50 separates early-failing cells (red).", t5, c5),
        section("g-maint", "8 · Cell-level explanations",
                f"Effect of each mechanism on a cell's predicted life after {Tr} cycles. Red shortens, blue extends.",
                t6, c6, extra_caption=next(iter(titles.values()))),
        section("g-bench", "9 · Model benchmark",
                "Test RMSE after 100 cycles. PLSR matched the stacked ensemble.", t7),
        section("g-pred", "10 · Predicted versus actual life",
                "Points on the diagonal are exact predictions.", t8, c8),
        section("g-robust", "11 · Robustness",
                "Change in test RMSE under noisy, missing or sparse input data.", t9),
    ]
    fig_json = {k: json.loads(pio.to_json(v, validate=False)) for k, v in figs.items()}
    payload = json.dumps(fig_json, separators=(",", ":")).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Early Battery Failure Prediction</title>
<meta name="description" content="Interpretable early-life battery failure prediction for predictive maintenance.">
<style>{CSS}</style>
<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js" charset="utf-8"></script>
</head><body><div class="wrap">
<header><div><h1>Interpretable Early-Life Battery Failure Prediction for Predictive Maintenance</h1>
<p class="sub">Identifies lithium-ion cells likely to fail early from their first 50 cycles.
83 LFP cells, BatteryML MATR-1 benchmark.</p></div>
<div class="links"><a href="{REPO_URL}">GitHub repository</a><a href="{REPO_URL}/blob/main/results/REPORT.md">Results report</a>
<button id="theme-btn" type="button">Theme: auto</button></div></header>
<div class="tiles">{tiles_html}</div>
{''.join(body)}
<p class="note">SHAP indicates association, not causation. Single dataset of 83 cells.
<a href="{REPO_URL}/blob/main/docs/TECHNICAL_DETAILS.md">Methods and limitations</a>.</p>
</div>
<script type="application/json" id="figs">{payload}</script>
<script>{JS.replace("__PAL__", json.dumps(PAL))}</script>
</body></html>"""
    out = repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
