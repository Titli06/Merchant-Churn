"""Behavioural segmentation of the merchant base.

RFM alone is not enough here. Recency, frequency and monetary value describe
how much a merchant transacts, not whether the platform is working for them.
A merchant with healthy volume and a collapsing success rate looks identical to
a healthy merchant on RFM and is a completely different commercial situation.
So the feature set adds payment quality (success rate and its trend), service
friction (settlement delay, unresolved tickets) and rhythm (p90 activity gap).

k is chosen by silhouette across a stated range rather than assumed. Features
are standardised first, because k-means minimises Euclidean distance and TPV in
rupees would otherwise dominate every other dimension by three orders of
magnitude -- the most common way k-means "finds" segments that are just size
buckets wearing a hat.

Segments are named from their profile, then scored against realised churn. A
segmentation nobody can act on differently per segment is decoration.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from warehouse import connect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
FIG = OUT / "figures"

# One volume feature, not three. avg_monthly_tpv, avg_monthly_txns and
# avg_ticket_inr are near-collinear; including all three triples the weight
# k-means puts on "how big is this merchant" and drowns out everything about
# how well the platform is serving them -- which is the thing worth segmenting
# on. Exposure and service-quality features carry the rest.
FEATURES = [
    "recency_days", "avg_monthly_tpv", "success_rate_3m_at_cutoff", "sr_trend",
    "avg_settlement_delay_days", "tickets_unresolved", "p90_gap_days",
    "tenure_months", "tpv_cv", "degraded_issuer_exposure",
]

# Skewed money/count features get a log1p before scaling. Without it a handful
# of very large merchants sit so far out that k-means spends a cluster on them.
LOG_FEATURES = ["avg_monthly_tpv", "recency_days", "p90_gap_days"]


def name_segment(row: pd.Series, medians: pd.Series) -> str:
    hi_vol = row["avg_monthly_tpv"] > medians["avg_monthly_tpv"]
    bad_sr = row["success_rate_3m_at_cutoff"] < medians["success_rate_3m_at_cutoff"]
    falling = row["sr_trend"] < 0
    stale = row["recency_days"] > medians["recency_days"]
    young = row["tenure_months"] < medians["tenure_months"]

    if hi_vol and bad_sr and falling:
        return "High-value, payments-impaired"
    if hi_vol and not bad_sr:
        return "Core healthy volume"
    if stale and not hi_vol:
        return "Fading long-tail"
    if young and not stale:
        return "New and ramping"
    if bad_sr:
        return "Low-volume, poor experience"
    return "Steady long-tail"


def main() -> None:
    with open(ROOT / "config" / "params.yml") as fh:
        cfg = yaml.safe_load(fh)["analysis"]
    k_range = cfg["kmeans_k_range"]

    con = connect(read_only=True)
    df = con.execute("SELECT * FROM merchant_features").df()
    con.close()

    df = df[df["success_rate_3m_at_cutoff"].notna()].copy()
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    Xt = X.copy()
    for c in LOG_FEATURES:
        Xt[c] = np.log1p(np.clip(Xt[c], 0, None))

    Z = StandardScaler().fit_transform(Xt)

    sil = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=7).fit(Z)
        sil.append({"k": k,
                    "silhouette": round(float(silhouette_score(Z, km.labels_, sample_size=4000,
                                                               random_state=7)), 4),
                    "inertia": round(float(km.inertia_), 1)})
    sil_df = pd.DataFrame(sil)
    best_k = int(sil_df.loc[sil_df["silhouette"].idxmax(), "k"])

    km = KMeans(n_clusters=best_k, n_init=25, random_state=7).fit(Z)
    df["segment_id"] = km.labels_

    profile = df.groupby("segment_id").agg(
        merchants=("merchant_id", "size"),
        avg_monthly_tpv=("avg_monthly_tpv", "median"),
        avg_monthly_txns=("avg_monthly_txns", "median"),
        success_rate_3m_at_cutoff=("success_rate_3m_at_cutoff", "median"),
        sr_trend=("sr_trend", "median"),
        recency_days=("recency_days", "median"),
        tenure_months=("tenure_months", "median"),
        p90_gap_days=("p90_gap_days", "median"),
        tickets_unresolved=("tickets_unresolved", "median"),
        degraded_issuer_exposure=("degraded_issuer_exposure", "median"),
        churn_rate=("churned", "mean"),
    ).reset_index()

    medians = profile[FEATURES[:len(FEATURES)]].median() if False else df[FEATURES].median()
    profile["segment_name"] = profile.apply(lambda r: name_segment(r, medians), axis=1)
    # Disambiguate any duplicate names so downstream joins stay unique.
    seen: dict[str, int] = {}
    names = []
    for n in profile["segment_name"]:
        seen[n] = seen.get(n, 0) + 1
        names.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    profile["segment_name"] = names

    profile["tpv_share"] = (
        df.groupby("segment_id")["avg_monthly_tpv"].sum()
          / df["avg_monthly_tpv"].sum()).values.round(4)
    profile = profile.sort_values("churn_rate", ascending=False).reset_index(drop=True)

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    profile.round(4).to_csv(OUT / "segment_profile.csv", index=False)
    df[["merchant_id", "segment_id"]].to_csv(OUT / "merchant_segment_assignment.csv", index=False)
    with open(OUT / "segmentation_scorecard.json", "w") as fh:
        json.dump({"k_selection": sil, "chosen_k": best_k,
                   "segments": profile.round(4).to_dict(orient="records")}, fh, indent=2, default=float)

    # ---- figure -----------------------------------------------------------
    pca = PCA(n_components=2, random_state=7).fit(Z)
    P = pca.transform(Z)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot(sil_df["k"], sil_df["silhouette"], marker="o", c="#2c3e50")
    ax.axvline(best_k, ls="--", c="crimson", lw=1.2)
    ax.set_xlabel("k"); ax.set_ylabel("Silhouette")
    ax.set_title(f"k selection (chosen k={best_k})", fontsize=10); ax.grid(alpha=0.25)

    ax = axes[1]
    sample = np.random.default_rng(7).choice(len(P), size=min(3500, len(P)), replace=False)
    sc = ax.scatter(P[sample, 0], P[sample, 1], c=km.labels_[sample], s=6, cmap="tab10", alpha=0.65)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
    ax.set_title("Segments in PCA space", fontsize=10); ax.grid(alpha=0.2)

    ax = axes[2]
    order = profile.sort_values("churn_rate")
    ax.barh(order["segment_name"], order["churn_rate"], color="#c0392b", alpha=0.85)
    ax.axvline(df["churned"].mean(), ls="--", c="k", lw=1,
               label=f"base {df['churned'].mean():.1%}")
    ax.set_xlabel("Churn rate"); ax.set_title("Churn by segment", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(FIG / "segmentation.png", dpi=150)
    plt.close(fig)

    print(f"Merchants segmented: {len(df):,}   chosen k = {best_k}\n")
    print(sil_df.to_string(index=False))
    print("\nSegment profile:")
    cols = ["segment_name", "merchants", "tpv_share", "avg_monthly_tpv",
            "success_rate_3m_at_cutoff", "sr_trend", "recency_days",
            "degraded_issuer_exposure", "churn_rate"]
    print(profile[cols].to_string(index=False,
          float_format=lambda v: f"{v:10.4f}"))
    print("\nWritten: reports/segment_profile.csv, reports/segmentation_scorecard.json, "
          "reports/figures/segmentation.png")


if __name__ == "__main__":
    main()
