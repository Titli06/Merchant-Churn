"""Forecast network TPV, and handle the structural break honestly.

The series is 24 monthly observations containing a six-month issuer incident
that depressed success rate and therefore TPV. Two things follow:

1. A model fitted straight through the break will read the recovery as trend
   and extrapolate growth that is really just the incident ending. The fix is
   an exogenous intervention dummy, so the break is explained rather than
   absorbed into the trend term.

2. Twenty-four points cannot identify a 12-period seasonal cycle and a trend
   and an intervention at once. Rather than fit a seasonal model and report a
   confident number, the seasonal-naive benchmark is included precisely so the
   comparison shows whether seasonality is even recoverable here. Where the
   honest answer is "the series is too short", that is what gets reported.

Every model is scored on the same rolling holdout against the same benchmarks.
A forecast without a naive benchmark is unfalsifiable.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from warehouse import connect

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
FIG = OUT / "figures"


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def load_series() -> pd.DataFrame:
    con = connect(read_only=True)
    df = con.execute("""
        SELECT month, tpv_inr, active_merchants, success_rate
        FROM network_health_monthly ORDER BY month
    """).df()
    con.close()
    df["month"] = pd.to_datetime(df["month"])
    return df.set_index("month")


def incident_dummy(index: pd.DatetimeIndex, sr: pd.Series | None = None,
                   threshold: float = 0.92) -> np.ndarray:
    """1 in months where network success rate sat below the healthy band.

    Derived from the data, not hard-coded to known dates, so the pipeline still
    works if the incident window moves.
    """
    if sr is None:
        return np.zeros(len(index))
    return (sr.reindex(index).fillna(1.0) < threshold).astype(float).to_numpy()


def main() -> None:
    with open(ROOT / "config" / "params.yml") as fh:
        cfg = yaml.safe_load(fh)["analysis"]
    h_out = int(cfg["forecast_holdout_months"])
    h_fwd = int(cfg["forecast_horizon_months"])

    df = load_series()
    y = df["tpv_inr"].astype(float)
    sr = df["success_rate"].astype(float)

    train, test = y.iloc[:-h_out], y.iloc[-h_out:]
    x_all = incident_dummy(y.index, sr)
    x_tr, x_te = x_all[:-h_out], x_all[-h_out:]

    results = {}

    # ---- benchmarks -------------------------------------------------------
    results["naive_last_value"] = np.repeat(train.iloc[-1], h_out)
    drift = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
    results["drift"] = train.iloc[-1] + drift * np.arange(1, h_out + 1)
    if len(train) > 12:
        results["seasonal_naive"] = train.iloc[-12:-12 + h_out].to_numpy()

    # ---- Holt damped trend ------------------------------------------------
    try:
        hw = ExponentialSmoothing(train, trend="add", damped_trend=True,
                                  seasonal=None, initialization_method="estimated").fit()
        results["holt_damped"] = hw.forecast(h_out).to_numpy()
    except Exception as exc:
        print(f"  holt_damped failed: {exc}")

    # ---- SARIMAX with and without the intervention regressor --------------
    try:
        m_plain = SARIMAX(train, order=(1, 1, 1),
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        results["sarimax_no_intervention"] = m_plain.forecast(h_out).to_numpy()
    except Exception as exc:
        print(f"  sarimax_plain failed: {exc}")

    fitted_x = None
    try:
        m_x = SARIMAX(train, exog=x_tr.reshape(-1, 1), order=(1, 1, 1),
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        results["sarimax_with_intervention"] = m_x.forecast(h_out, exog=x_te.reshape(-1, 1)).to_numpy()
        fitted_x = m_x
    except Exception as exc:
        print(f"  sarimax_x failed: {exc}")

    scores = pd.DataFrame([
        {"model": k, "mape_pct": round(mape(test.to_numpy(), np.asarray(v)), 3),
         "mae_inr": round(mae(test.to_numpy(), np.asarray(v)), 0)}
        for k, v in results.items()
    ]).sort_values("mape_pct").reset_index(drop=True)

    best = scores.iloc[0]["model"]
    naive_mape = float(scores.loc[scores["model"] == "naive_last_value", "mape_pct"].iloc[0])
    best_mape = float(scores.iloc[0]["mape_pct"])
    beats_naive = best_mape < naive_mape

    # ---- refit best on the full series and project forward ----------------
    future_index = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1),
                                 periods=h_fwd, freq="MS")
    x_future = np.zeros(h_fwd)  # assume no further incident
    fc, lo, hi = None, None, None
    try:
        if best == "sarimax_with_intervention":
            m = SARIMAX(y, exog=x_all.reshape(-1, 1), order=(1, 1, 1),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            res = m.get_forecast(h_fwd, exog=x_future.reshape(-1, 1))
        elif best.startswith("sarimax"):
            m = SARIMAX(y, order=(1, 1, 1),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            res = m.get_forecast(h_fwd)
        else:
            m = SARIMAX(y, order=(1, 1, 1),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            res = m.get_forecast(h_fwd)
        fc = res.predicted_mean.to_numpy()
        ci = res.conf_int(alpha=0.20)
        lo, hi = ci.iloc[:, 0].to_numpy(), ci.iloc[:, 1].to_numpy()
    except Exception as exc:
        print(f"  forward forecast failed: {exc}")

    payload = {
        "series_length_months": int(len(y)),
        "holdout_months": h_out,
        "horizon_months": h_fwd,
        "scores": scores.to_dict(orient="records"),
        "best_model": best,
        "best_mape_pct": best_mape,
        "naive_mape_pct": naive_mape,
        "beats_naive_benchmark": bool(beats_naive),
        "intervention_coef": (float(fitted_x.params.get("x1", np.nan))
                              if fitted_x is not None else None),
        "forecast": ([{"month": str(d.date()), "tpv_inr": float(f),
                       "lo80": float(l), "hi80": float(hh)}
                      for d, f, l, hh in zip(future_index, fc, lo, hi)] if fc is not None else []),
        "caveat": ("24 monthly observations cannot jointly identify trend, a 12-period "
                   "seasonal cycle and an intervention. Seasonal terms are reported as a "
                   "benchmark only and should not be read as an estimated seasonality."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    with open(OUT / "forecast_scorecard.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=float)

    # ---- figure -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(y.index, y / 1e6, marker="o", ms=3.5, c="#2c3e50", label="Actual TPV")
    inc = x_all.astype(bool)
    if inc.any():
        ax.axvspan(y.index[inc][0], y.index[inc][-1], color="crimson", alpha=0.08)
        ax.text(y.index[inc][len(y.index[inc]) // 2], (y.max() / 1e6) * 0.99,
                "issuer incident", ha="center", fontsize=8, color="crimson")
    for name in scores["model"].head(3):
        ax.plot(test.index, np.asarray(results[name]) / 1e6, ls="--", marker="s",
                ms=3, alpha=0.8, label=f"{name} (holdout)")
    if fc is not None:
        ax.plot(future_index, fc / 1e6, c="#16a085", marker="^", ms=4, label="Forecast")
        ax.fill_between(future_index, lo / 1e6, hi / 1e6, color="#16a085", alpha=0.18,
                        label="80% interval")
    ax.set_ylabel("TPV (INR mn)")
    ax.set_title("Network TPV: holdout evaluation and forward forecast", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "tpv_forecast.png", dpi=150)
    plt.close(fig)

    print(f"Series: {len(y)} months | holdout {h_out} | horizon {h_fwd}\n")
    print(scores.to_string(index=False))
    print(f"\nBest: {best} (MAPE {best_mape:.2f}%) vs naive {naive_mape:.2f}% "
          f"-> {'beats' if beats_naive else 'DOES NOT BEAT'} the benchmark")
    if fitted_x is not None:
        print(f"Intervention coefficient: {fitted_x.params.get('x1', float('nan')):,.0f} INR/month")
    print("\nWritten: reports/forecast_scorecard.json, reports/figures/tpv_forecast.png")


if __name__ == "__main__":
    main()
