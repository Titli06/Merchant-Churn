"""Evaluate churn *definitions* as classifiers, and recommend an operating point.

This is the only module allowed to read `_ground_truth_lifecycle.parquet`.
Nothing feeding the dashboard, the driver model, or the forecast touches it.

Why this exists
---------------
Most teams choose "no activity in 30 days" in a meeting and never measure it.
A definition is a classifier: behaviour in, label out. Given a counterfactual we
can measure what the choice costs, sweep the threshold, and pick an operating
point on purpose rather than by convention.

The headline result is not "my clever rule wins". It is that recency-only rules
of *any* threshold hit a precision ceiling, because dormant-but-alive merchants
and churned merchants are behaviourally identical at the observation instant.
The fix is structural -- a two-tier metric with a confirmation lag -- not a
better threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from warehouse import connect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
FIG = OUT / "figures"


def score(pred: np.ndarray, actual: np.ndarray) -> dict:
    tp = int(((pred == 1) & (actual == 1)).sum())
    fp = int(((pred == 1) & (actual == 0)).sum())
    fn = int(((pred == 0) & (actual == 1)).sum())
    tn = int(((pred == 0) & (actual == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "flagged": tp + fp,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def main() -> None:
    con = connect(read_only=True)
    status = con.execute("SELECT * FROM merchant_churn_status").df()
    con.close()

    gt = pd.read_parquet(ROOT / "data" / "raw" / "_ground_truth_lifecycle.parquet")
    df = status.merge(gt, on="merchant_id", how="inner")
    actual = (df["true_churn_month_index"] >= 0).to_numpy().astype(int)
    days = df["days_since_last_success"].to_numpy()
    p90 = df["p90_gap_days"].fillna(15).to_numpy()
    enough_history = (df["baseline_gap_count"].fillna(0) >= 5).to_numpy()

    # ---- 1. fixed-window sweep -------------------------------------------
    fixed = pd.DataFrame([
        {"window_days": w, **score((days >= w).astype(int), actual)}
        for w in range(10, 121, 5)
    ])

    # ---- 2. adaptive-multiplier sweep ------------------------------------
    adaptive_rows = []
    for mult in np.arange(1.0, 6.01, 0.25):
        thresh = np.maximum(21, mult * p90)
        pred = np.where(enough_history, days > thresh, days >= 30).astype(int)
        adaptive_rows.append({"multiplier": round(float(mult), 2),
                              "median_threshold_days": float(np.median(thresh)),
                              **score(pred, actual)})
    adaptive = pd.DataFrame(adaptive_rows)

    best_fixed = fixed.loc[fixed["f1"].idxmax()]
    best_adaptive = adaptive.loc[adaptive["f1"].idxmax()]
    conventional = fixed[fixed["window_days"] == 30].iloc[0]

    # ---- 3. two-tier metric ----------------------------------------------
    # AT RISK at 30d (timely, drives outreach); CONFIRMED at the smallest window
    # reaching 95% precision (drives reporting and the retention denominator).
    hi_prec = fixed[fixed["precision"] >= 0.95]
    confirm_window = int(hi_prec["window_days"].min()) if len(hi_prec) \
        else int(fixed.loc[fixed["precision"].idxmax(), "window_days"])
    at_risk = score((days >= 30).astype(int), actual)
    confirmed = score((days >= confirm_window).astype(int), actual)

    # ---- 4. where do the errors live? ------------------------------------
    fp_mask = (days >= 30) & (actual == 0)
    d1_fp = df[fp_mask]
    fp_profile = {
        "false_positives_at_30d": int(len(d1_fp)),
        "of_which_dormant_but_alive": int(d1_fp["dormant_at_end"].sum()),
        "share_dormant": round(float(d1_fp["dormant_at_end"].mean()), 4) if len(d1_fp) else 0.0,
        "fp_rate_by_category": (
            df.assign(fp=fp_mask.astype(int))
              .groupby("category")["fp"].mean().round(4).sort_values(ascending=False).to_dict()
        ),
        "median_p90_gap_fp": float(d1_fp["p90_gap_days"].median()),
        "median_p90_gap_population": float(df["p90_gap_days"].median()),
    }

    payload = {
        "merchants_scored": int(len(df)),
        "true_churn_rate": round(float(actual.mean()), 4),
        "conventional_30d": conventional.to_dict(),
        "best_fixed_window": best_fixed.to_dict(),
        "best_adaptive": best_adaptive.to_dict(),
        "recommended_two_tier": {
            "at_risk_window_days": 30,
            "at_risk": at_risk,
            "confirmed_window_days": confirm_window,
            "confirmed": confirmed,
        },
        "precision_ceiling": round(float(fixed["precision"].max()), 4),
        "false_positive_profile": fp_profile,
        "fixed_sweep": fixed.to_dict(orient="records"),
        "adaptive_sweep": adaptive.to_dict(orient="records"),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    with open(OUT / "churn_definition_scorecard.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=float)

    # ---- 5. figure --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    ax.plot(fixed["window_days"], fixed["precision"], marker="o", ms=3, label="Precision")
    ax.plot(fixed["window_days"], fixed["recall"], marker="s", ms=3, label="Recall")
    ax.plot(fixed["window_days"], fixed["f1"], marker="^", ms=3, label="F1")
    ax.axvline(30, ls="--", c="crimson", lw=1)
    ax.annotate("conventional\n30-day rule", xy=(30, 0.45), xytext=(48, 0.30),
                fontsize=8, color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8))
    ax.axvline(confirm_window, ls=":", c="seagreen", lw=1.4)
    ax.annotate(f"confirmation\nwindow ({confirm_window}d)", xy=(confirm_window, 0.62),
                xytext=(confirm_window + 6, 0.46), fontsize=8, color="seagreen",
                arrowprops=dict(arrowstyle="->", color="seagreen", lw=0.8))
    ax.set_xlabel("Inactivity window (days)")
    ax.set_ylabel("Score")
    ax.set_title("Fixed-window churn rule: threshold sweep", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(fixed["recall"], fixed["precision"], marker="o", ms=3, label="Fixed window")
    ax.plot(adaptive["recall"], adaptive["precision"], marker="s", ms=3,
            label="Adaptive (merchant p90 gap)")
    ax.scatter([conventional["recall"]], [conventional["precision"]], s=90,
               facecolors="none", edgecolors="crimson", lw=1.6, zorder=5,
               label="Conventional 30d")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision/recall frontier: both rules hit the same ceiling", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIG / "churn_definition_sweep.png", dpi=150)
    plt.close(fig)

    # ---- console ----------------------------------------------------------
    print(f"Merchants scored: {len(df):,}    true churn rate: {actual.mean():.2%}\n")
    print(f"{'rule':<34}{'flagged':>9}{'FP':>7}{'FN':>7}{'prec':>8}{'recall':>8}{'F1':>8}")
    print("-" * 81)
    for label, r in [
        ("conventional 30-day", conventional),
        (f"best fixed window ({int(best_fixed['window_days'])}d)", best_fixed),
        (f"best adaptive (x{best_adaptive['multiplier']} p90)", best_adaptive),
    ]:
        print(f"{label:<34}{int(r['flagged']):>9,}{int(r['fp']):>7,}{int(r['fn']):>7,}"
              f"{r['precision']:>8.3f}{r['recall']:>8.3f}{r['f1']:>8.3f}")

    print(f"\nPrecision ceiling across ALL fixed windows: {fixed['precision'].max():.3f}")
    print(f"False positives at 30d: {fp_profile['false_positives_at_30d']:,} "
          f"({fp_profile['share_dormant']:.0%} are dormant-but-alive merchants)\n")
    print("Recommended two-tier metric:")
    print(f"  AT RISK    (>=30d silent): {at_risk['flagged']:,} merchants, "
          f"precision {at_risk['precision']:.3f}, recall {at_risk['recall']:.3f}")
    print(f"  CONFIRMED  (>={confirm_window}d silent): {confirmed['flagged']:,} merchants, "
          f"precision {confirmed['precision']:.3f}, recall {confirmed['recall']:.3f}")
    print(f"\nWritten: {(OUT/'churn_definition_scorecard.json').relative_to(ROOT)}, "
          f"{(FIG/'churn_definition_sweep.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
