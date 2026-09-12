"""Streamlit demo for Silent Churn: merchant retention analytics for a UPI network.

This is a presentation layer only. It performs no analysis: every number shown
here is read from the artefacts already committed under `reports/`, which are
produced by the SQL warehouse and the scripts in `src/`. Nothing is recomputed,
no model is fitted at load time, and the DuckDB warehouse is not required --
that is what makes the app deployable from a clean clone with no pipeline run.

Sources
-------
reports/bi_exports/*.csv          warehouse marts (src/export_for_bi.py)
reports/*.json                    scorecards (evaluate_churn_definition, churn_drivers,
                                  segmentation, forecast)
reports/churn_driver_coefficients.csv
reports/segment_profile.csv
reports/figures/*.png             static figures reused where the underlying
                                  per-merchant features are not committed
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
BI = REPORTS / "bi_exports"
FIGURES = REPORTS / "figures"

INK = "#1F3864"
ACCENT = "#C0504D"
MUTED = "#8C9BB5"
GOOD = "#2E7D5B"

st.set_page_config(
    page_title="Silent Churn — UPI Merchant Retention",
    page_icon="📉",
    layout="wide",
)


# ---------------------------------------------------------------- loading ---
@st.cache_data(show_spinner=False)
def csv(name: str) -> pd.DataFrame:
    path = BI / f"{name}.csv"
    if not path.exists():
        path = REPORTS / f"{name}.csv"
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def scorecard(name: str) -> dict:
    return json.loads((REPORTS / f"{name}.json").read_text())


def inr(v: float) -> str:
    """Compact rupee formatting, matching the units used in the memo."""
    v = float(v)
    if abs(v) >= 1e9:
        return f"₹{v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"₹{v / 1e6:.1f}M"
    if abs(v) >= 1e3:
        return f"₹{v / 1e3:.0f}K"
    return f"₹{v:,.0f}"


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    return fig


def chart(fig: go.Figure, height: int = 380) -> None:
    st.plotly_chart(style(fig, height), width="stretch")


# ------------------------------------------------------------------ header ---
st.title("Silent Churn")
st.caption(
    "Merchant retention analytics for a simulated UPI acceptance network — "
    "24 months, 6,000 merchants, 7.4M payment attempts. "
    "Every figure below is read from committed analysis outputs; nothing is recomputed here."
)

health = csv("network_health_monthly")
banks = csv("bank_month_sr")
definition = scorecard("churn_definition_scorecard")
model = scorecard("churn_model_scorecard")
forecast = scorecard("forecast_scorecard")
segments = scorecard("segmentation_scorecard")
rar = csv("revenue_at_risk").set_index("status")

tab_overview, tab_churn, tab_segments, tab_model = st.tabs(
    ["Overview", "Churn & cohorts", "Merchant segmentation", "ML & forecast"]
)

# ---------------------------------------------------------------- overview ---
with tab_overview:
    scored = definition["merchants_scored"]
    confirmed = rar.loc["confirmed_churn"]
    at_risk = rar.loc["at_risk"]

    c = st.columns(5)
    c[0].metric("Payment attempts", f"{banks['attempted_txns'].sum() / 1e6:.1f}M")
    c[1].metric("Total TPV", inr(health["tpv_inr"].sum()))
    c[2].metric("Active merchants", f"{int(health['active_merchants'].iloc[-1]):,}")
    c[3].metric(
        "Confirmed churn rate",
        f"{confirmed['merchants'] / scored:.1%}",
        help="65 days silent. Precision 0.977 against the simulator's answer key.",
    )
    c[4].metric(
        "Revenue at risk (annualised)",
        inr(at_risk["annualised_revenue_at_risk"]),
        help=(
            "30–65 days silent and still recoverable. "
            f"Already lost to confirmed churn: {inr(confirmed['annualised_revenue_at_risk'])}. "
            "The two are separate numbers and are not additive."
        ),
    )

    st.divider()
    st.subheader("Growth hid the retention problem")

    h = health.copy()
    m3 = csv("cohort_m3_retention")
    fig = go.Figure()
    fig.add_bar(
        x=h["month"], y=h["active_merchants"], name="Active merchants",
        marker_color=MUTED, opacity=0.55,
    )
    fig.add_scatter(
        x=m3["cohort_month"], y=m3["m3_logo_retention"], name="Month-3 cohort retention",
        yaxis="y2", mode="lines+markers", line=dict(color=ACCENT, width=3),
    )
    fig.update_layout(
        yaxis=dict(title="Active merchants"),
        yaxis2=dict(title="M3 retention", overlaying="y", side="right", tickformat=".0%"),
    )
    chart(fig)
    st.caption(
        "The active base rose 3,338 → 4,270 and TPV ₹48M → ₹111M straight through a "
        "six-month issuer incident, because onboarding outran attrition every month. "
        "Month-3 retention fell underneath it."
    )

    st.subheader("Key business findings")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("**1 · Churn is overstated by 17%**")
        st.markdown(
            "The standard 30-day silence rule flags "
            f"{definition['conventional_30d']['flagged']:,.0f} merchants. "
            f"{definition['conventional_30d']['fp']:,.0f} of them are still alive, and "
            f"{definition['false_positive_profile']['share_dormant']:.0%} of those are "
            "dormant-but-alive — quiet for a month, then back. Win-back spend goes to "
            "merchants who never left."
        )
    with f2:
        st.markdown("**2 · An issuer incident cost merchants, unattributed**")
        st.markdown(
            "Network success rate fell 93.0% → 89.4% and recovered. Merchants in the "
            "top issuer-exposure decile churned at 19.4% against 14.0% in the bottom — "
            "a 39% relative increase, with success rate falling monotonically across "
            "deciles."
        )
    with f3:
        st.markdown("**3 · The obvious culprit is the wrong one**")
        st.markdown(
            "Ranked by average success rate the worst issuer is a small regional bank. "
            "Ranked by volume-weighted contribution to network success rate it causes "
            "about one-ninth the damage of a large public-sector issuer that ranks only "
            "second on raw rate."
        )

    culprit = csv("incident_culprit_ranking").sort_values("cumulative_rate_effect")
    fig = go.Figure()
    fig.add_bar(
        x=culprit["cumulative_rate_effect"], y=culprit["issuer_bank"],
        orientation="h", marker_color=ACCENT, name="Cumulative rate effect",
        customdata=culprit[["avg_success_rate", "avg_volume_share"]],
        hovertemplate=(
            "%{y}<br>rate effect %{x:.4f}"
            "<br>avg SR %{customdata[0]:.1%}"
            "<br>volume share %{customdata[1]:.1%}<extra></extra>"
        ),
    )
    fig.update_layout(
        title="Contribution to network success-rate damage, by issuer",
        xaxis_title="Cumulative rate effect (volume-weighted)",
        hovermode="closest",
    )
    chart(fig, height=420)

    with st.expander("Issuer ranking table — the two rankings disagree"):
        st.dataframe(
            culprit[
                [
                    "issuer_bank", "avg_success_rate", "avg_volume_share",
                    "cumulative_rate_effect", "rank_by_worst_sr", "rank_by_network_impact",
                ]
            ].sort_values("rank_by_network_impact"),
            hide_index=True,
            width="stretch",
        )

# ----------------------------------------------------------- churn/cohorts ---
with tab_churn:
    st.subheader("Churn definitions scored against a known answer key")
    st.caption(
        "The simulator records each merchant's true churn month. Treating a churn rule "
        "as a classifier makes the definitions directly comparable."
    )

    rules = {
        "Conventional 30d": definition["conventional_30d"],
        "Best fixed window (90d)": definition["best_fixed_window"],
        "Adaptive (×6 p90 gap)": definition["best_adaptive"],
        "Two-tier: CONFIRMED (65d)": definition["recommended_two_tier"]["confirmed"],
    }
    comp = pd.DataFrame(
        [
            {"rule": name, "precision": r["precision"], "recall": r["recall"], "f1": r["f1"]}
            for name, r in rules.items()
        ]
    )
    fig = go.Figure()
    for metric, colour in [("precision", INK), ("recall", ACCENT), ("f1", MUTED)]:
        fig.add_bar(x=comp["rule"], y=comp[metric], name=metric.upper(), marker_color=colour)
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.05], tickformat=".0%"))
    chart(fig)

    left, right = st.columns([3, 2])
    with left:
        sweep = pd.DataFrame(definition["fixed_sweep"])
        fig = go.Figure()
        for metric, colour in [("precision", INK), ("recall", ACCENT), ("f1", GOOD)]:
            fig.add_scatter(
                x=sweep["window_days"], y=sweep[metric], name=metric.upper(),
                mode="lines+markers", line=dict(color=colour, width=2),
            )
        for day, label in [(30, "AT RISK"), (65, "CONFIRMED")]:
            fig.add_vline(x=day, line_dash="dot", line_color="#999",
                          annotation_text=label, annotation_position="top")
        fig.update_layout(
            title="Precision / recall as the silence window widens",
            xaxis_title="Silence window (days)", yaxis=dict(tickformat=".0%"),
        )
        chart(fig)
    with right:
        st.markdown("**Why one number cannot do both jobs**")
        st.markdown(
            "A dormant merchant and a churned merchant are behaviourally identical at "
            "the instant you look at them, so no recency-only rule escapes the precision "
            f"ceiling of {definition['precision_ceiling']:.1%}. The fix is structural: "
            "two metrics with a confirmation lag."
        )
        st.markdown(
            f"- **AT RISK** — 30 days silent — recall "
            f"{definition['recommended_two_tier']['at_risk']['recall']:.3f} — drives outreach\n"
            f"- **CONFIRMED** — 65 days silent — precision "
            f"{definition['recommended_two_tier']['confirmed']['precision']:.3f} — drives reporting"
        )
        fp = definition["false_positive_profile"]
        st.markdown(
            f"Of the {fp['false_positives_at_30d']} merchants wrongly flagged at 30 days, "
            f"{fp['of_which_dormant_but_alive']} were dormant-but-alive. Their median p90 "
            f"activity gap is {fp['median_p90_gap_fp']:.0f} days against "
            f"{fp['median_p90_gap_population']:.0f} for the population."
        )

    st.divider()
    st.subheader("Cohort retention")

    cohorts = csv("cohort_retention")
    metric_label = st.radio(
        "Retention measure", ["Logo retention", "TPV retention"],
        horizontal=True, label_visibility="collapsed",
    )
    col = "logo_retention" if metric_label == "Logo retention" else "tpv_retention"
    triangle = cohorts.pivot(index="cohort_month", columns="month_index", values=col)
    fig = go.Figure(
        go.Heatmap(
            z=triangle.values,
            x=[f"M{i}" for i in triangle.columns],
            y=triangle.index,
            colorscale="RdYlGn",
            zmid=1.0 if col == "tpv_retention" else None,
            hovertemplate="cohort %{y}<br>%{x}<br>" + metric_label.lower() + " %{z:.1%}<extra></extra>",
            colorbar=dict(tickformat=".0%"),
        )
    )
    fig.update_layout(title=f"{metric_label} by cohort × month index", hovermode="closest")
    chart(fig, height=520)
    st.caption(
        "Cohorts after the first are small (the Aug-2024 cohort is the initial base of "
        "3,338 merchants), so later rows are noisier. Month-3 retention is the "
        "comparable summary used on the Overview tab."
    )

    st.divider()
    st.subheader("Churn flows and issuer exposure")

    flows = health[health["is_censored_boundary"] == 0]
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_bar(x=flows["month"], y=flows["new_merchants"], name="New", marker_color=GOOD)
        fig.add_bar(
            x=flows["month"], y=flows["reactivated_merchants"], name="Reactivated",
            marker_color=MUTED,
        )
        fig.add_bar(
            x=flows["month"], y=-flows["went_silent_merchants"], name="Went silent",
            marker_color=ACCENT,
        )
        fig.add_scatter(
            x=flows["month"], y=flows["net_merchant_adds"], name="Net adds",
            mode="lines+markers", line=dict(color=INK, width=2),
        )
        fig.update_layout(
            title="Merchant flows per month (censored boundary months excluded)",
            barmode="relative",
        )
        chart(fig)
    with right:
        exposure = csv("exposure_vs_churn")
        fig = go.Figure()
        fig.add_bar(
            x=exposure["exposure_decile"], y=exposure["churn_rate"],
            name="Churn rate", marker_color=ACCENT,
        )
        fig.add_scatter(
            x=exposure["exposure_decile"], y=exposure["avg_sr_at_cutoff"],
            name="Success rate", yaxis="y2", mode="lines+markers",
            line=dict(color=INK, width=2),
        )
        fig.update_layout(
            title="Churn by degraded-issuer exposure decile",
            xaxis_title="Exposure decile (1 = least exposed)",
            yaxis=dict(title="Churn rate", tickformat=".0%"),
            yaxis2=dict(title="Success rate", overlaying="y", side="right", tickformat=".0%"),
        )
        chart(fig)
    st.caption(
        "Observational, not identified: exposure correlates with city tier, which "
        "correlates with merchant size and competitive density. The gradient survives "
        "controls in the driver model, but controls are not identification."
    )

# ------------------------------------------------------------ segmentation ---
with tab_segments:
    profile = csv("segment_profile").sort_values("merchants", ascending=False)
    base_rate = model["metrics"]["base_rate"]

    st.subheader("Behavioural segmentation")
    st.caption(
        f"k-means over merchant features, k chosen by silhouette across 3–8. "
        f"k={segments['chosen_k']} won at "
        f"{max(s['silhouette'] for s in segments['k_selection']):.3f} — a modest score, "
        "reported rather than hidden: this merchant base does not have crisp natural clusters."
    )

    cols = st.columns(len(profile))
    for col, (_, row) in zip(cols, profile.iterrows()):
        col.metric(
            row["segment_name"],
            f"{int(row['merchants']):,} merchants",
            f"{row['churn_rate']:.1%} churn",
            delta_color="inverse",
        )

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_bar(
            x=profile["segment_name"], y=profile["merchants"],
            name="Merchants", marker_color=INK,
        )
        fig.add_scatter(
            x=profile["segment_name"], y=profile["tpv_share"], name="TPV share",
            yaxis="y2", mode="markers+lines", marker=dict(size=12, color=ACCENT),
        )
        fig.update_layout(
            title="Cluster size vs share of TPV",
            yaxis=dict(title="Merchants"),
            yaxis2=dict(title="TPV share", overlaying="y", side="right", tickformat=".0%"),
        )
        chart(fig)
    with right:
        fig = go.Figure()
        fig.add_bar(
            x=profile["segment_name"], y=profile["churn_rate"],
            marker_color=ACCENT, name="Churn rate",
        )
        fig.add_hline(
            y=base_rate, line_dash="dash", line_color=INK,
            annotation_text=f"base {base_rate:.1%}", annotation_position="top left",
        )
        fig.update_layout(
            title="Churn rate by segment", yaxis=dict(tickformat=".0%"), hovermode="closest"
        )
        chart(fig)

    st.info(
        "3.4× churn spread across segments — but the highest-churn segment holds only "
        f"{profile['tpv_share'].min():.1%} of TPV, which is why the memo recommends "
        "targeting by predicted risk rather than by segment.",
        icon="💡",
    )

    st.markdown("**Segment profile**")
    st.dataframe(
        profile[
            [
                "segment_name", "merchants", "tpv_share", "churn_rate",
                "avg_monthly_tpv", "avg_monthly_txns", "success_rate_3m_at_cutoff",
                "sr_trend", "p90_gap_days", "degraded_issuer_exposure",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "segment_name": "Segment",
            "merchants": st.column_config.NumberColumn("Merchants", format="%d"),
            "tpv_share": st.column_config.NumberColumn("TPV share", format="%.1f%%"),
            "churn_rate": st.column_config.NumberColumn("Churn", format="%.3f"),
            "avg_monthly_tpv": st.column_config.NumberColumn("Avg monthly TPV", format="₹%.0f"),
            "avg_monthly_txns": st.column_config.NumberColumn("Avg monthly txns", format="%.1f"),
            "success_rate_3m_at_cutoff": st.column_config.NumberColumn("SR (3m)", format="%.4f"),
            "sr_trend": st.column_config.NumberColumn("SR trend", format="%.4f"),
            "p90_gap_days": st.column_config.NumberColumn("p90 gap (d)", format="%.1f"),
            "degraded_issuer_exposure": st.column_config.NumberColumn("Exposure", format="%.3f"),
        },
    )
    st.caption("TPV share is a fraction, shown here on its raw 0–1 scale in the table.")

    with st.expander("k selection and clusters in PCA space"):
        k_sel = pd.DataFrame(segments["k_selection"])
        fig = go.Figure()
        fig.add_scatter(
            x=k_sel["k"], y=k_sel["silhouette"], mode="lines+markers",
            line=dict(color=INK, width=2), name="Silhouette",
        )
        fig.add_vline(x=segments["chosen_k"], line_dash="dash", line_color=ACCENT)
        fig.update_layout(title="Silhouette by k", xaxis_title="k")
        chart(fig, height=300)
        st.image(
            str(FIGURES / "segmentation.png"),
            caption="Committed figure from src/segmentation.py — the PCA scatter needs "
                    "per-merchant features, which are generated locally and not committed.",
            width="stretch",
        )

# -------------------------------------------------------------- ml/forecast ---
with tab_model:
    metrics = model["metrics"]
    st.subheader("Churn driver model")
    st.caption(
        "Inference first, prediction second — a retention team needs signed, arguable "
        "effects, not the highest available AUC. Both models are scored on the same "
        "temporally held-out merchants."
    )

    c = st.columns(4)
    c[0].metric("Logistic ROC-AUC", f"{metrics['logistic']['roc_auc']:.3f}")
    c[1].metric(
        "GBM benchmark ROC-AUC",
        f"{metrics['gbm_benchmark']['roc_auc']:.3f}",
        f"{metrics['gbm_benchmark']['roc_auc'] - metrics['logistic']['roc_auc']:+.3f} vs logistic",
    )
    c[2].metric("Logistic PR-AUC", f"{metrics['logistic']['pr_auc']:.3f}",
                help=f"Base churn rate in the test window: {metrics['base_rate']:.1%}")
    c[3].metric("Test merchants", f"{metrics['n_test']:,}",
                help=f"Trained on {metrics['n_train']:,} merchants from an earlier window.")
    st.markdown(
        "The non-linear benchmark is **worse**, so the interpretable model ships. "
        "That is reported as-is rather than swapped for a better-looking number."
    )

    left, right = st.columns(2)
    with left:
        deciles = pd.DataFrame(model["decile_lift"])
        fig = go.Figure()
        fig.add_bar(
            x=deciles["decile"], y=deciles["churn_rate"],
            marker_color=[ACCENT if d == deciles["decile"].max() else INK for d in deciles["decile"]],
            name="Churn rate",
        )
        fig.add_hline(
            y=metrics["base_rate"], line_dash="dash", line_color="#666",
            annotation_text=f"base {metrics['base_rate']:.1%}",
        )
        top = deciles.iloc[-1]
        fig.update_layout(
            title=(f"Risk decile lift — top decile churns at {top['churn_rate']:.1%} "
                   f"({top['lift_vs_base']:.1f}× base)"),
            xaxis_title="Predicted-risk decile", yaxis=dict(tickformat=".0%"),
            hovermode="closest",
        )
        chart(fig)
    with right:
        coefs = csv("churn_driver_coefficients")
        coefs["driver"] = coefs["feature"].str.replace(r"^(num|cat)__", "", regex=True)
        only_sig = st.checkbox("Significant drivers only", value=True)
        shown = coefs[coefs["significant"]] if only_sig else coefs
        shown = shown.reindex(shown["coef"].abs().sort_values().index).tail(12)
        fig = go.Figure()
        fig.add_scatter(
            x=shown["odds_ratio"], y=shown["driver"], mode="markers",
            marker=dict(size=11, color=[ACCENT if o > 1 else GOOD for o in shown["odds_ratio"]]),
            error_x=dict(
                type="data",
                array=shown["ci_high"] - shown["odds_ratio"],
                arrayminus=shown["odds_ratio"] - shown["ci_low"],
                color="#aaa",
            ),
            name="Odds ratio",
        )
        fig.add_vline(x=1.0, line_dash="dash", line_color="#666")
        fig.update_layout(
            title="Odds ratios with 95% CI", xaxis_title="Odds ratio (log scale)",
            xaxis_type="log", hovermode="closest",
        )
        chart(fig)

    with st.expander("Coefficient recovery against the simulator's planted hazard"):
        rec = pd.DataFrame(model["coefficient_recovery"])
        st.markdown(
            f"**{int(rec['sign_matches'].sum())} of {len(rec)} signs recovered.** "
            "Magnitudes are not comparable — the planted coefficients act per "
            "merchant-month on raw units, the recovered ones on standardised features "
            "over the observation window. The sign test is the meaningful one."
        )
        st.dataframe(rec, hide_index=True, width="stretch")

    st.divider()
    st.subheader("TPV forecast")

    hist = health[["month", "tpv_inr"]].copy()
    fc = pd.DataFrame(forecast["forecast"])
    bridge = pd.concat([hist.tail(1), fc[["month", "tpv_inr"]]])

    fig = go.Figure()
    fig.add_scatter(
        x=list(fc["month"]) + list(fc["month"])[::-1],
        y=list(fc["hi80"]) + list(fc["lo80"])[::-1],
        fill="toself", fillcolor="rgba(192,80,77,0.15)", line=dict(width=0),
        hoverinfo="skip", name="80% interval",
    )
    fig.add_scatter(
        x=hist["month"], y=hist["tpv_inr"], name="Actual TPV",
        mode="lines+markers", line=dict(color=INK, width=2.5),
    )
    fig.add_scatter(
        x=bridge["month"], y=bridge["tpv_inr"], name=f"Forecast ({forecast['best_model']})",
        mode="lines+markers", line=dict(color=ACCENT, width=2.5, dash="dash"),
    )
    fig.update_layout(
        title=f"Monthly TPV, {forecast['horizon_months']}-month horizon",
        yaxis_title="TPV (INR)",
    )
    chart(fig, height=420)

    left, right = st.columns([2, 3])
    with left:
        scores = pd.DataFrame(forecast["scores"]).sort_values("mape_pct", ascending=False)
        fig = go.Figure()
        fig.add_bar(
            x=scores["mape_pct"], y=scores["model"], orientation="h",
            marker_color=[ACCENT if m == forecast["best_model"] else MUTED for m in scores["model"]],
        )
        fig.update_layout(
            title=f"Holdout MAPE ({forecast['holdout_months']}-month holdout)",
            xaxis_title="MAPE %", hovermode="closest",
        )
        chart(fig, height=340)
    with right:
        st.markdown("**The simplest model wins**")
        st.markdown(
            f"Best out-of-sample fit: **{forecast['best_model']}** at "
            f"{forecast['best_mape_pct']:.2f}% MAPE, against a naive benchmark of "
            f"{forecast['naive_mape_pct']:.2f}%. A forecast without a benchmark is "
            "unfalsifiable, so every model is scored on the same holdout."
        )
        st.markdown(
            "The SARIMAX variant carrying an intervention dummy did **worse** out of "
            "sample. The holdout sits entirely post-recovery, where the dummy is zero "
            "throughout, so it contributed parameter noise and nothing else. Reported "
            "as-is rather than quietly dropped."
        )
        st.warning(forecast["caveat"], icon="⚠️")

# ------------------------------------------------------------------ sidebar ---
with st.sidebar:
    st.markdown("### Silent Churn")
    st.markdown(
        "SQL-first retention analysis on a simulated Indian UPI acceptance network.\n\n"
        "**Stack** — DuckDB + partitioned Parquet, ANSI SQL marts, scikit-learn, "
        "statsmodels, Streamlit."
    )
    st.markdown(
        "[Repository](https://github.com/Shxdhanshu/upi-merchant-churn-analytics) · "
        "[Stakeholder memo]"
        "(https://github.com/Shxdhanshu/upi-merchant-churn-analytics/blob/main/reports/memo.md)"
    )
    st.divider()
    st.markdown("**Every churn number states its window**")
    st.markdown(
        "`AT RISK` = 30 days silent. `CONFIRMED CHURN` = 65 days silent. "
        "A dashboard that says *churn* with no window is the problem this project "
        "exists to document."
    )
    st.divider()
    st.caption(
        "Data is synthetic, generated by `src/generate_data.py` from a documented "
        "data-generating process. Real merchant-level payments data is not public, and "
        "scoring a churn definition requires a known answer key."
    )
