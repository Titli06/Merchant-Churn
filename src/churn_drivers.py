"""Churn driver model.

Purpose is inference first, prediction second. The question a retention team
actually asks is "what do we change", which needs interpretable, signed effects
with confidence intervals -- not the highest AUC available.

So the primary model is a regularised logistic regression on standardised
features, reported as odds ratios. A gradient-boosted benchmark runs alongside
purely to answer "how much are we giving up by staying linear?". If the gap is
small, the linear model is the right thing to ship, because it can be argued
with in a meeting.

Guards against the usual failure modes:

* Features come from sql/05, which is already split temporally. Nothing here
  can see the label window.
* `avg_monthly_tpv` and friends are computed over the feature window only. The
  version of this model that includes label-window activity scores AUC ~0.99
  and is worthless.
* Categorical encoding drops the first level to avoid the dummy trap, so
  coefficients read as differences from a stated reference category.
* Confidence intervals come from the observed information matrix, so a
  coefficient that is merely noisy does not get reported as a finding.

The final section compares recovered effects against the simulator's planted
hazard coefficients. That comparison is a *validation artifact*: it is printed
and written to the scorecard, and never feeds the model.
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from warehouse import connect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
FIG = OUT / "figures"

NUMERIC = [
    "success_rate_3m_at_cutoff", "sr_trend", "avg_settlement_delay_days",
    "tickets_unresolved", "tickets_payment_failure", "tenure_months",
    "competitor_density", "avg_monthly_txns", "avg_monthly_tpv", "tpv_cv",
    "avg_ticket_inr", "p90_gap_days", "degraded_issuer_exposure",
    "avg_active_days",
]
CATEGORICAL = ["category", "city_tier", "acquisition_channel", "device_type"]

# Sign and rough ordering we expect from the simulator's hazard, for the
# validation section. Keys are model features; values are the planted term.
PLANTED_MAP = {
    "success_rate_3m_at_cutoff": ("sr_shortfall", -1),   # higher SR -> less churn
    "avg_settlement_delay_days": ("settle_delay_days", +1),
    "tickets_unresolved":        ("open_tickets", +1),
    "tenure_months":             ("tenure_months", -1),
    "competitor_density":        ("competitor_density", +1),
    "avg_monthly_txns":          ("log_txn_volume", -1),
}


def load_features() -> pd.DataFrame:
    con = connect(read_only=True)
    df = con.execute("SELECT * FROM merchant_features").df()
    con.close()
    return df


def build_pipeline(C: float = 1.0) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, C=C, penalty="l2")),
    ])


def coefficient_table(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Coefficients with Wald standard errors from the observed information."""
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    names = list(pre.get_feature_names_out())
    Z = pre.transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    Z1 = np.hstack([np.ones((Z.shape[0], 1)), Z])

    p = pipe.predict_proba(X)[:, 1]
    W = np.clip(p * (1 - p), 1e-9, None)
    # Fisher information for logistic regression, plus the L2 penalty term that
    # was actually applied, so the SEs correspond to the fitted estimator.
    ridge = np.eye(Z1.shape[1]) / max(clf.C, 1e-9)
    ridge[0, 0] = 0.0
    try:
        cov = np.linalg.inv(Z1.T * W @ Z1 + ridge)
        se = np.sqrt(np.clip(np.diag(cov)[1:], 0, None))
    except np.linalg.LinAlgError:
        se = np.full(len(names), np.nan)

    coef = clf.coef_[0]
    out = pd.DataFrame({
        "feature": names,
        "coef": coef,
        "std_err": se,
        "odds_ratio": np.exp(coef),
        "ci_low": np.exp(coef - 1.96 * se),
        "ci_high": np.exp(coef + 1.96 * se),
    })
    out["z"] = out["coef"] / out["std_err"].replace(0, np.nan)
    out["significant"] = out["z"].abs() > 1.96
    return out.sort_values("coef", key=np.abs, ascending=False).reset_index(drop=True)


def main() -> None:
    df = load_features()
    # Restrict to merchants who were still transacting at the feature cutoff.
    # Merchants already silent before the cutoff have no trailing-3m success
    # rate, and including them would mean predicting churn for accounts that had
    # already churned -- which inflates every metric and answers no question
    # anyone asked. This is a population definition, not a convenience dropna.
    at_risk_population = df["success_rate_3m_at_cutoff"].notna()
    print(f"Eligible population: {at_risk_population.sum():,} of {len(df):,} merchants "
          f"still active at the feature cutoff\n")
    df = df[at_risk_population].copy()
    X = df[NUMERIC + CATEGORICAL].copy()
    X[NUMERIC] = X[NUMERIC].apply(pd.to_numeric, errors="coerce")
    X[NUMERIC] = X[NUMERIC].fillna(X[NUMERIC].median())
    y = df["churned"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=7)

    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    p_te = pipe.predict_proba(X_te)[:, 1]

    gbm = Pipeline([
        ("pre", ColumnTransformer([
            ("num", "passthrough", NUMERIC),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
        ])),
        ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                               max_depth=4, random_state=7)),
    ])
    gbm.fit(X_tr, y_tr)
    p_gbm = gbm.predict_proba(X_te)[:, 1]

    metrics = {
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "base_rate": round(float(y.mean()), 4),
        "logistic": {
            "roc_auc": round(float(roc_auc_score(y_te, p_te)), 4),
            "pr_auc": round(float(average_precision_score(y_te, p_te)), 4),
            "brier": round(float(brier_score_loss(y_te, p_te)), 4),
        },
        "gbm_benchmark": {
            "roc_auc": round(float(roc_auc_score(y_te, p_gbm)), 4),
            "pr_auc": round(float(average_precision_score(y_te, p_gbm)), 4),
            "brier": round(float(brier_score_loss(y_te, p_gbm)), 4),
        },
    }

    coefs = coefficient_table(pipe, X_tr)

    # ---- validation: do the recovered effects match the planted hazard? ----
    with open(ROOT / "config" / "params.yml") as fh:
        planted = yaml.safe_load(fh)["dgp"]["hazard"]
    recovery = []
    for feat, (term, expected_sign) in PLANTED_MAP.items():
        row = coefs[coefs["feature"] == f"num__{feat}"]
        if row.empty:
            continue
        got = float(row["coef"].iloc[0])
        recovery.append({
            "feature": feat,
            "planted_term": term,
            "planted_coef": planted[term],
            "expected_sign": "+" if expected_sign > 0 else "-",
            "recovered_coef": round(got, 4),
            "recovered_sign": "+" if got > 0 else "-",
            "sign_matches": (got > 0) == (expected_sign > 0),
            "significant": bool(row["significant"].iloc[0]),
        })
    rec = pd.DataFrame(recovery)

    # ---- lift: what does targeting the top decile buy? --------------------
    te = pd.DataFrame({"y": y_te.to_numpy(), "p": p_te})
    te["decile"] = pd.qcut(te["p"], 10, labels=False, duplicates="drop") + 1
    lift = (te.groupby("decile")
              .agg(merchants=("y", "size"), churn_rate=("y", "mean"))
              .reset_index())
    lift["lift_vs_base"] = (lift["churn_rate"] / y_te.mean()).round(2)
    lift["churn_rate"] = lift["churn_rate"].round(4)

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    coefs.to_csv(OUT / "churn_driver_coefficients.csv", index=False)
    payload = {"metrics": metrics,
               "coefficient_recovery": recovery,
               "decile_lift": lift.to_dict(orient="records")}
    with open(OUT / "churn_model_scorecard.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=float)

    # ---- figure -----------------------------------------------------------
    top = coefs[coefs["feature"].str.startswith("num__")].head(10).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    labels = [f.replace("num__", "") for f in top["feature"]]
    colors = ["#c0392b" if c > 0 else "#2980b9" for c in top["coef"]]
    ax.barh(labels, top["coef"], color=colors)
    ax.errorbar(top["coef"], range(len(top)),
                xerr=1.96 * top["std_err"], fmt="none", ecolor="#444", lw=1, capsize=2)
    ax.axvline(0, c="k", lw=0.8)
    ax.set_xlabel("Log-odds coefficient (standardised features)")
    ax.set_title("Churn drivers: red increases churn", fontsize=10)
    ax.grid(alpha=0.25, axis="x")

    ax = axes[1]
    ax.bar(lift["decile"], lift["churn_rate"], color="#34495e")
    ax.axhline(y_te.mean(), ls="--", c="crimson", lw=1.2,
               label=f"base rate {y_te.mean():.1%}")
    ax.set_xlabel("Predicted-risk decile")
    ax.set_ylabel("Realised churn rate")
    ax.set_title("Model lift on held-out merchants", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "churn_drivers.png", dpi=150)
    plt.close(fig)

    # ---- console ----------------------------------------------------------
    print(f"Train {len(X_tr):,} / test {len(X_te):,}   base churn rate {y.mean():.2%}\n")
    print(f"  logistic   ROC-AUC {metrics['logistic']['roc_auc']:.4f}   "
          f"PR-AUC {metrics['logistic']['pr_auc']:.4f}   Brier {metrics['logistic']['brier']:.4f}")
    print(f"  gbm bench  ROC-AUC {metrics['gbm_benchmark']['roc_auc']:.4f}   "
          f"PR-AUC {metrics['gbm_benchmark']['pr_auc']:.4f}   Brier {metrics['gbm_benchmark']['brier']:.4f}")
    gap = metrics["gbm_benchmark"]["roc_auc"] - metrics["logistic"]["roc_auc"]
    print(f"  -> non-linear model buys {gap:+.4f} AUC\n")

    print("Top drivers (standardised log-odds):")
    show = coefs.head(12)[["feature", "coef", "odds_ratio", "ci_low", "ci_high", "significant"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    print("\nCoefficient recovery vs planted hazard:")
    if not rec.empty:
        print(rec[["feature", "planted_coef", "expected_sign", "recovered_coef",
                   "recovered_sign", "sign_matches", "significant"]].to_string(index=False))
        print(f"\n  signs recovered: {int(rec['sign_matches'].sum())}/{len(rec)}")

    print("\nDecile lift:")
    print(lift.to_string(index=False))
    print(f"\nWritten: reports/churn_driver_coefficients.csv, "
          f"reports/churn_model_scorecard.json, reports/figures/churn_drivers.png")


if __name__ == "__main__":
    main()
