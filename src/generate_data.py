"""
Generate the simulated UPI acceptance network.

Design notes
------------
This is not a "random numbers in a CSV" generator. It implements an explicit
data-generating process (DGP) with three properties that make the downstream
analysis worth doing:

1. **Confounding.** Merchant city tier drives both the payer-side issuer-bank
   mix *and* competitor density *and* transaction volume. Tier-3 merchants
   over-index to public-sector and regional rural banks, which have structurally
   lower technical success rates. So "tier 3 churns more" is true but is not the
   mechanism, and a naive tier-level cut leads you to the wrong intervention.

2. **A planted incident.** One large issuer degrades sharply for five months and
   recovers. Nothing in the schema announces this. It has to be recovered from
   failure-reason mix and merchant-level exposure. A decoy (a slow drift at a
   small rural bank) exists so that "find the bank with the worst SR" is not
   sufficient -- the worst-SR bank is not the one causing the churn spike,
   because it carries too little volume to matter.

3. **A hidden ground truth.** The true churn month for every merchant is written
   to `_ground_truth_lifecycle.parquet`. The analysis pipeline never reads it.
   `tests/test_churn_definition.py` uses it to *score competing definitions of
   churn* -- which is the point of the project. See README.

Everything stochastic flows from a single seed in config/params.yml.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SAMPLE = ROOT / "data" / "sample"

FAILURE_REASONS_NORMAL = {
    "INSUFFICIENT_FUNDS": 0.41,
    "INCORRECT_PIN": 0.18,
    "USER_CANCELLED": 0.14,
    "TIMEOUT": 0.11,
    "ISSUER_DOWN": 0.06,
    "LIMIT_EXCEEDED": 0.06,
    "TECHNICAL_DECLINE": 0.04,
}
FAILURE_REASONS_INCIDENT = {
    "INSUFFICIENT_FUNDS": 0.13,
    "INCORRECT_PIN": 0.06,
    "USER_CANCELLED": 0.05,
    "TIMEOUT": 0.31,
    "ISSUER_DOWN": 0.34,
    "LIMIT_EXCEEDED": 0.03,
    "TECHNICAL_DECLINE": 0.08,
}

# Hour-of-day intensity by category (24 weights, normalised at use time).
HOUR_PROFILES = {
    "kirana":      [1,1,1,1,1,2,5,9,12,13,12,11,10,10,11,13,16,20,22,19,12,6,3,2],
    "restaurant":  [2,1,1,1,1,1,2,4,7,9,10,14,20,18,10,8,10,16,24,28,24,14,7,3],
    "pharmacy":    [2,1,1,1,1,2,4,8,13,15,14,12,10,9,10,12,14,16,17,14,9,5,3,2],
    "grocery":     [1,1,1,1,1,2,4,8,12,14,14,12,11,10,11,13,16,19,20,17,10,5,3,2],
    "apparel":     [1,1,1,1,1,1,1,2,5,10,14,16,15,14,14,16,19,22,21,16,9,4,2,1],
    "electronics": [1,1,1,1,1,1,1,2,5,11,16,18,16,14,14,16,18,20,18,13,7,3,2,1],
    "fuel":        [3,2,2,2,3,6,12,18,20,17,14,12,11,11,12,14,17,20,21,18,12,8,5,4],
    "salon":       [1,1,1,1,1,1,2,3,6,10,13,14,13,12,13,15,18,21,20,15,8,4,2,1],
    "services":    [1,1,1,1,1,2,3,6,11,15,16,15,13,12,13,14,15,15,13,9,6,3,2,1],
}

# Month-of-year multiplier: Indian festive season (Oct/Nov) plus a March
# financial-year-end bump and a monsoon dip.
MONTH_SEASONALITY = {
    1: 0.96, 2: 0.94, 3: 1.06, 4: 1.01, 5: 1.00, 6: 0.93,
    7: 0.91, 8: 0.95, 9: 1.04, 10: 1.24, 11: 1.18, 12: 1.05,
}


def load_config(path: Path | None = None) -> dict:
    path = path or (ROOT / "config" / "params.yml")
    with open(path) as fh:
        return yaml.safe_load(fh)


def vectorised_multinomial(rng: np.random.Generator, n: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Draw multinomial counts where every row has its own probability vector.

    numpy's multinomial requires a single shared `p`. Here each merchant has a
    different payer-bank mix, so we decompose into a sequence of conditional
    binomials -- k-1 vectorised binomial draws instead of a Python loop over
    merchants.

    Parameters
    ----------
    n : (m,) int array of trial counts
    p : (m, k) float array, rows sum to 1

    Returns
    -------
    (m, k) int array of counts, rows summing to n.
    """
    m, k = p.shape
    counts = np.zeros((m, k), dtype=np.int64)
    remaining = n.astype(np.int64).copy()
    p_remaining = np.ones(m, dtype=float)

    for j in range(k - 1):
        with np.errstate(divide="ignore", invalid="ignore"):
            cond = np.where(p_remaining > 1e-12, p[:, j] / p_remaining, 0.0)
        cond = np.clip(cond, 0.0, 1.0)
        draw = rng.binomial(remaining, cond)
        counts[:, j] = draw
        remaining -= draw
        p_remaining -= p[:, j]
        p_remaining = np.clip(p_remaining, 0.0, None)

    counts[:, k - 1] = remaining
    return counts


def build_bank_sr_schedule(cfg: dict) -> pd.DataFrame:
    """Monthly technical success rate per issuer bank, including the incident."""
    d = cfg["dgp"]
    banks = list(d["issuer_banks"].keys())
    n_months = d["n_months"]

    sr = np.zeros((n_months, len(banks)))
    for j, b in enumerate(banks):
        sr[:, j] = d["issuer_banks"][b]["base_sr"]

    # Planted incident: ramp down, plateau, ramp up.
    inc = d["incident"]
    bi = banks.index(inc["bank"])
    start, end, ramp = inc["start_month_index"], inc["end_month_index"], inc["ramp_months"]
    for t in range(n_months):
        if t < start - ramp or t > end + ramp:
            factor = 0.0
        elif t < start:
            factor = (t - (start - ramp)) / ramp
        elif t <= end:
            factor = 1.0
        else:
            factor = 1.0 - (t - end) / ramp
        sr[t, bi] -= inc["sr_drop"] * max(0.0, factor)

    # Decoy: slow monotone drift at a small rural bank.
    dec = d["decoy"]
    di = banks.index(dec["bank"])
    for t in range(dec["start_month_index"], n_months):
        progress = (t - dec["start_month_index"]) / max(1, n_months - dec["start_month_index"])
        sr[t, di] -= dec["sr_drop"] * progress

    sr = np.clip(sr, 0.35, 0.995)
    return pd.DataFrame(sr, columns=banks)


def build_merchants(cfg: dict, rng: np.random.Generator, n_merchants: int) -> pd.DataFrame:
    d = cfg["dgp"]
    cats = list(d["categories"].keys())
    cat_p = np.array([d["categories"][c]["share"] for c in cats])
    cat_p = cat_p / cat_p.sum()

    tiers = list(d["city_tiers"].keys())
    tier_p = np.array([d["city_tiers"][t]["share"] for t in tiers])
    tier_p = tier_p / tier_p.sum()

    chans = list(d["acquisition_channels"].keys())
    chan_p = np.array([d["acquisition_channels"][c]["share"] for c in chans])
    chan_p = chan_p / chan_p.sum()

    devs = list(d["devices"].keys())
    dev_p = np.array([d["devices"][c]["share"] for c in devs])
    dev_p = dev_p / dev_p.sum()

    states = [
        "Maharashtra", "Uttar Pradesh", "Karnataka", "Tamil Nadu", "Gujarat",
        "West Bengal", "Rajasthan", "Telangana", "Madhya Pradesh", "Bihar",
        "Odisha", "Kerala", "Punjab", "Haryana", "Andhra Pradesh", "Assam",
    ]
    state_p = np.array([12, 15, 8, 8, 7, 7, 6, 5, 6, 6, 3, 4, 3, 3, 5, 2], dtype=float)
    state_p /= state_p.sum()

    m = pd.DataFrame({
        "merchant_id": [f"MID{100000 + i}" for i in range(n_merchants)],
        "category": rng.choice(cats, n_merchants, p=cat_p),
        "city_tier": rng.choice(tiers, n_merchants, p=tier_p),
        "acquisition_channel": rng.choice(chans, n_merchants, p=chan_p),
        "device_type": rng.choice(devs, n_merchants, p=dev_p),
        "state": rng.choice(states, n_merchants, p=state_p),
    })

    # Onboarding: 55% are on the platform before the observation window opens
    # (left-truncated tenure), the rest join during it. This is what makes the
    # cohort retention triangle non-trivial.
    pre_existing = rng.random(n_merchants) < 0.55
    cohort_month = np.where(
        pre_existing,
        -rng.integers(1, 30, n_merchants),                     # joined before window
        rng.integers(0, d["n_months"] - 2, n_merchants),        # joined during window
    )
    m["cohort_month_index"] = cohort_month

    m["gst_registered"] = (
        rng.random(n_merchants)
        < np.select(
            [m["city_tier"] == "tier_1", m["city_tier"] == "tier_2"],
            [0.71, 0.48], default=0.26,
        )
    )

    # Chronically slow settlement path (a real operational segment).
    m["slow_settlement"] = rng.random(n_merchants) < d["settlement"]["delayed_share"]

    # Competitor density: tier mean plus merchant-level noise.
    tier_dens = m["city_tier"].map({t: d["city_tiers"][t]["competitor_density"] for t in tiers})
    m["competitor_density"] = np.clip(
        tier_dens.to_numpy() + rng.normal(0, 0.09, n_merchants), 0.02, 0.99
    )

    m["acq_quality"] = m["acquisition_channel"].map(
        {c: d["acquisition_channels"][c]["quality"] for c in chans}
    ).to_numpy()

    # Merchant-level volume multiplier (fat-tailed: a few merchants are huge).
    # Fat-tailed merchant size. Mean-normalised (exp(-s^2/2)) so that widening the
    # spread changes heterogeneity without inflating total network volume.
    _s = 1.05
    m["volume_mult"] = np.exp(rng.normal(0, _s, n_merchants) - _s**2 / 2)
    return m


def build_bank_mix(cfg: dict, rng: np.random.Generator, merchants: pd.DataFrame) -> np.ndarray:
    """Per-merchant payer-side issuer mix, tilted by city tier.

    This is the confounder. Tier-3 merchants serve customers who bank with
    public-sector and regional rural banks; tier-1 merchants serve private-bank
    customers. The incident bank (SBI) is public-sector, so the shock lands
    disproportionately on tier-3 -- but tier-3 also has *lower* competitor
    density, which pushes churn the other way. A tier-level cut therefore
    understates the effect.
    """
    d = cfg["dgp"]
    banks = list(d["issuer_banks"].keys())
    base = np.array([d["issuer_banks"][b]["share"] for b in banks])
    base = base / base.sum()

    private = {"HDFC", "ICICI", "AXIS", "KOTAK", "YES", "IDFC", "PAYTM_PB"}
    public = {"SBI", "PNB", "BOB", "CANARA", "UNION"}
    rural = {"RRB_GRAMIN", "AU_SFB"}

    tilt = {
        "tier_1": {"private": 1.55, "public": 0.72, "rural": 0.30},
        "tier_2": {"private": 1.05, "public": 1.02, "rural": 0.80},
        "tier_3": {"private": 0.62, "public": 1.28, "rural": 1.95},
    }

    n = len(merchants)
    k = len(banks)
    mix = np.zeros((n, k))
    concentration = 14.0  # lower => more merchant-to-merchant variation in exposure

    for tier, grp in merchants.groupby("city_tier", sort=False):
        idx = grp.index.to_numpy()
        w = base.copy()
        for j, b in enumerate(banks):
            if b in private:
                w[j] *= tilt[tier]["private"]
            elif b in public:
                w[j] *= tilt[tier]["public"]
            elif b in rural:
                w[j] *= tilt[tier]["rural"]
        w = w / w.sum()
        mix[idx] = rng.dirichlet(w * concentration, size=len(idx))

    return mix


def simulate(
    cfg: dict,
    n_merchants: int | None = None,
    seed: int | None = None,
    txn_writer=None,
) -> dict:
    """Run the simulation.

    If `txn_writer` is supplied it is called as ``txn_writer(month_str, frame)``
    once per month and the transaction frames are *not* accumulated in memory.
    This keeps peak RSS flat regardless of dataset size -- at 6,000 merchants the
    transaction table is ~9M rows, which does not fit comfortably in a small
    container if held as one pandas frame with object dtypes.
    """
    d = cfg["dgp"]
    rng = np.random.default_rng(seed if seed is not None else d["seed"])
    n_merchants = n_merchants or d["n_merchants"]
    n_months = d["n_months"]
    rate_scale = d.get("txn_rate_scale", 0.5)

    banks = list(d["issuer_banks"].keys())
    bank_sr = build_bank_sr_schedule(cfg)
    merchants = build_merchants(cfg, rng, n_merchants)
    mix = build_bank_mix(cfg, rng, merchants)

    months = pd.period_range(d["start_month"], periods=n_months, freq="M")

    cat_rate = merchants["category"].map(
        {c: d["categories"][c]["txn_rate"] for c in d["categories"]}
    ).to_numpy()
    cat_mu = merchants["category"].map(
        {c: d["categories"][c]["ticket_mu"] for c in d["categories"]}
    ).to_numpy()
    cat_sigma = merchants["category"].map(
        {c: d["categories"][c]["ticket_sigma"] for c in d["categories"]}
    ).to_numpy()
    tier_mult = merchants["city_tier"].map(
        {t: d["city_tiers"][t]["txn_mult"] for t in d["city_tiers"]}
    ).to_numpy()
    dev_mult = merchants["device_type"].map(
        {t: d["devices"][t]["txn_mult"] for t in d["devices"]}
    ).to_numpy()

    base_rate = cat_rate * tier_mult * dev_mult * merchants["volume_mult"].to_numpy() * rate_scale

    h = d["hazard"]
    active = np.zeros(n_merchants, dtype=bool)
    ever_active = np.zeros(n_merchants, dtype=bool)
    churn_month = np.full(n_merchants, -1, dtype=int)
    tenure = np.zeros(n_merchants, dtype=float)
    open_tickets = np.zeros(n_merchants, dtype=float)
    trailing_sr = np.full(n_merchants, 0.95)
    dormant_until = np.full(n_merchants, -1, dtype=int)
    dorm_cfg = d.get("dormancy", {})
    dorm_mult = merchants["category"].map(
        dorm_cfg.get("category_mult", {})).fillna(1.0).to_numpy()
    # Low-frequency merchants go dormant more often. This coupling is what
    # makes a single fixed churn window structurally unfair across the base:
    # the merchants most likely to be quiet-but-alive are also the ones whose
    # normal rhythm already includes long gaps.
    dorm_mult = dorm_mult * np.clip(40.0 / (base_rate + 10.0), 0.4, 3.0)

    txn_frames, ticket_frames, settle_frames = [], [], []
    txn_id_counter = 0

    for t, period in enumerate(months):
        # --- activation ------------------------------------------------------
        newly_live = (merchants["cohort_month_index"].to_numpy() <= t) & (~ever_active) & (churn_month < 0)
        active |= newly_live
        ever_active |= newly_live

        # Reactivation of previously churned merchants.
        churned_now = (churn_month >= 0) & (~active)
        if churned_now.any():
            back = churned_now & (rng.random(n_merchants) < d["reactivation_prob"])
            active |= back
            churn_month[back] = -1

        # Dormant merchants are alive but silent this month.
        act_idx = np.flatnonzero(active & (dormant_until <= t))
        if act_idx.size == 0:
            continue

        # --- transaction counts ---------------------------------------------
        season = MONTH_SEASONALITY[period.month]
        # Tenure ramp: merchants take ~4 months to reach steady-state volume.
        ramp = 0.55 + 0.45 * np.clip(tenure[act_idx] / 4.0, 0, 1)
        lam = base_rate[act_idx] * season * ramp
        n_txn = rng.poisson(lam).astype(np.int64)

        live = act_idx[n_txn > 0]
        n_txn = n_txn[n_txn > 0]
        if live.size == 0:
            continue

        # --- allocate transactions across issuer banks -----------------------
        counts = vectorised_multinomial(rng, n_txn, mix[live])
        total = int(counts.sum())

        merch_row = np.repeat(np.repeat(live, counts.shape[1]), counts.ravel())
        bank_idx = np.repeat(np.tile(np.arange(len(banks)), len(live)), counts.ravel())

        # --- amounts ---------------------------------------------------------
        amount = rng.lognormal(cat_mu[merch_row], cat_sigma[merch_row])
        amount = np.round(np.clip(amount, 5, 200000), 2)

        # --- timestamps ------------------------------------------------------
        days_in_month = period.days_in_month
        # Day-of-week weighting: weekends heavier for retail categories.
        day_of_month = rng.integers(1, days_in_month + 1, total)
        base_ts = pd.PeriodIndex([period] * 1).to_timestamp()[0]
        dates = base_ts + pd.to_timedelta(day_of_month - 1, unit="D")
        dow = dates.dayofweek.to_numpy()
        keep = rng.random(total) < np.where(dow >= 5, 1.0, 0.82)
        # Resample rejected days once (cheap approximation of dow weighting).
        redraw = rng.integers(1, days_in_month + 1, total)
        day_of_month = np.where(keep, day_of_month, redraw)

        hour = np.zeros(total, dtype=int)
        cats_row = merchants["category"].to_numpy()[merch_row]
        for c, prof in HOUR_PROFILES.items():
            sel = cats_row == c
            if sel.any():
                p = np.asarray(prof, dtype=float)
                hour[sel] = rng.choice(24, sel.sum(), p=p / p.sum())
        minute = rng.integers(0, 60, total)
        second = rng.integers(0, 60, total)

        ts = (
            base_ts
            + pd.to_timedelta(day_of_month - 1, unit="D")
            + pd.to_timedelta(hour, unit="h")
            + pd.to_timedelta(minute, unit="m")
            + pd.to_timedelta(second, unit="s")
        )

        # --- success / failure ------------------------------------------------
        sr_row = bank_sr.iloc[t].to_numpy()[bank_idx]
        # Large tickets fail slightly more (limits, 2FA friction).
        amt_penalty = np.clip((np.log10(np.maximum(amount, 10)) - 2.4) * 0.021, 0, 0.075)
        p_success = np.clip(sr_row - amt_penalty, 0.05, 0.999)
        success = rng.random(total) < p_success

        # --- failure reasons ---------------------------------------------------
        reason = np.array([""] * total, dtype=object)
        fail_idx = np.flatnonzero(~success)
        if fail_idx.size:
            inc = d["incident"]
            incident_active = inc["start_month_index"] - inc["ramp_months"] <= t <= inc["end_month_index"] + inc["ramp_months"]
            inc_bank_i = banks.index(inc["bank"])
            is_inc_bank = bank_idx[fail_idx] == inc_bank_i
            use_incident = incident_active & is_inc_bank

            keys = list(FAILURE_REASONS_NORMAL.keys())
            p_norm = np.array([FAILURE_REASONS_NORMAL[k] for k in keys])
            p_inc = np.array([FAILURE_REASONS_INCIDENT[k] for k in keys])
            drawn = np.empty(fail_idx.size, dtype=object)
            n_norm = int((~use_incident).sum())
            n_incd = int(use_incident.sum())
            if n_norm:
                drawn[~use_incident] = rng.choice(keys, n_norm, p=p_norm / p_norm.sum())
            if n_incd:
                drawn[use_incident] = rng.choice(keys, n_incd, p=p_inc / p_inc.sum())
            reason[fail_idx] = drawn

        frame = pd.DataFrame({
            "txn_id": np.arange(txn_id_counter, txn_id_counter + total, dtype=np.int64),
            "merchant_id": pd.Categorical(
                merchants["merchant_id"].to_numpy()[merch_row],
                categories=merchants["merchant_id"].to_numpy(),
            ),
            "txn_ts": ts,
            "amount_inr": amount,
            "issuer_bank": pd.Categorical(np.asarray(banks, dtype=object)[bank_idx], categories=banks),
            "status": pd.Categorical(np.where(success, "SUCCESS", "FAILED"), categories=["SUCCESS", "FAILED"]),
            "failure_reason": pd.Categorical(reason, categories=[""] + list(FAILURE_REASONS_NORMAL)),
        })
        txn_id_counter += total
        if txn_writer is not None:
            txn_writer(str(period), frame)
        else:
            txn_frames.append(frame)

        # --- realised merchant-month success rate ------------------------------
        succ_by_merch = np.bincount(merch_row, weights=success.astype(float), minlength=n_merchants)
        cnt_by_merch = np.bincount(merch_row, minlength=n_merchants)
        realised = np.divide(
            succ_by_merch, cnt_by_merch,
            out=np.full(n_merchants, np.nan), where=cnt_by_merch > 0,
        )
        obs = ~np.isnan(realised)
        # Exponentially-weighted trailing SR (what a merchant actually "feels").
        trailing_sr[obs] = 0.45 * trailing_sr[obs] + 0.55 * realised[obs]

        # --- settlement --------------------------------------------------------
        s = d["settlement"]
        delay = np.full(len(live), s["base_delay_days"])
        delay += merchants["slow_settlement"].to_numpy()[live] * s["delayed_extra_days"]
        delay += rng.gamma(1.4, 0.35, len(live))
        settle_frames.append(pd.DataFrame({
            "merchant_id": merchants["merchant_id"].to_numpy()[live],
            "month": str(period),
            "avg_settlement_delay_days": np.round(delay, 3),
        }))

        # --- support tickets ---------------------------------------------------
        st = d["support_tickets"]
        shortfall = np.clip(0.95 - trailing_sr[live], 0, None)
        tick_lam = st["base_rate"] + st["sr_sensitivity"] * shortfall**1.5
        n_tick = rng.poisson(tick_lam)
        if n_tick.sum():
            tick_merch = np.repeat(merchants["merchant_id"].to_numpy()[live], n_tick)
            tick_short = np.repeat(shortfall, n_tick)
            tcat = np.where(
                rng.random(len(tick_merch)) < np.clip(0.25 + 3.0 * tick_short, 0, 0.9),
                "PAYMENT_FAILURE", 
                rng.choice(["SETTLEMENT", "DEVICE", "PRICING", "ONBOARDING", "OTHER"],
                           len(tick_merch), p=[0.28, 0.24, 0.18, 0.13, 0.17]),
            )
            tday = rng.integers(1, days_in_month + 1, len(tick_merch))
            ticket_frames.append(pd.DataFrame({
                "merchant_id": tick_merch,
                "created_at": base_ts + pd.to_timedelta(tday - 1, unit="D"),
                "ticket_category": tcat,
                "resolved": rng.random(len(tick_merch)) < st["resolve_prob"],
            }))
            unresolved = np.bincount(
                np.searchsorted(merchants["merchant_id"].to_numpy(), tick_merch),
                minlength=n_merchants,
            )
            open_tickets = 0.6 * open_tickets
            open_tickets[:len(unresolved)] += unresolved * (1 - st["resolve_prob"])

        # --- dormancy entry -----------------------------------------------------
        if dorm_cfg:
            eligible = active & (dormant_until <= t) & (tenure >= 2)
            p_dorm = np.clip(dorm_cfg["base_prob"] * dorm_mult, 0, 0.5)
            enters = eligible & (rng.random(n_merchants) < p_dorm)
            spell = rng.integers(1, dorm_cfg["max_months"] + 1, n_merchants)
            dormant_until[enters] = t + spell[enters]

        # --- churn hazard for next month ---------------------------------------
        sr_short = np.clip(0.95 - trailing_sr, 0, None)
        delay_full = np.full(n_merchants, s["base_delay_days"])
        delay_full[live] = delay
        log_vol = np.log1p(cnt_by_merch)

        logit = (
            h["intercept"]
            + h["sr_shortfall"] * sr_short
            + h["settle_delay_days"] * (delay_full - 1.0)
            + h["open_tickets"] * open_tickets
            + h["tenure_months"] * tenure
            + h["early_life_bonus"] * (tenure < 3).astype(float)
            + h["competitor_density"] * merchants["competitor_density"].to_numpy()
            + h["acq_quality"] * merchants["acq_quality"].to_numpy()
            + h["log_txn_volume"] * log_vol
            + h["seasonal_amplitude"] * np.sin(2 * np.pi * (t % 12) / 12)
        )
        p_churn = 1.0 / (1.0 + np.exp(-logit))
        churns = active & (rng.random(n_merchants) < p_churn)
        active[churns] = False
        churn_month[churns] = t

        tenure[active] += 1

    transactions = pd.concat(txn_frames, ignore_index=True) if txn_frames else None
    tickets = pd.concat(ticket_frames, ignore_index=True) if ticket_frames else pd.DataFrame()
    settlements = pd.concat(settle_frames, ignore_index=True)

    merchant_out = merchants.drop(columns=["acq_quality", "volume_mult"]).copy()
    merchant_out["onboarded_month"] = [
        str(months[max(0, c)]) if c >= 0 else str(months[0]) for c in merchants["cohort_month_index"]
    ]
    merchant_out["onboarded_before_window"] = merchants["cohort_month_index"] < 0
    merchant_out["competitor_density"] = merchant_out["competitor_density"].round(4)

    ground_truth = pd.DataFrame({
        "merchant_id": merchants["merchant_id"],
        "true_churn_month_index": churn_month,
        "true_churn_month": [str(months[c]) if c >= 0 else None for c in churn_month],
        "still_active_at_end": active,
        "ever_active": ever_active,
        # Alive but silent in the final month(s). These merchants are the exact
        # population a fixed 30-day churn window gets wrong.
        "dormant_at_end": active & (dormant_until >= n_months - 1),
    })

    return {
        "merchants": merchant_out,
        "transactions": transactions,
        "support_tickets": tickets,
        "settlements": settlements,
        "bank_sr_truth": bank_sr.assign(month=[str(m) for m in months]),
        "_ground_truth_lifecycle": ground_truth,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the simulated UPI network.")
    ap.add_argument("--merchants", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=Path, default=RAW)
    args = ap.parse_args()

    cfg = load_config()
    args.out.mkdir(parents=True, exist_ok=True)
    SAMPLE.mkdir(parents=True, exist_ok=True)
    txn_dir = args.out / "transactions"
    txn_dir.mkdir(parents=True, exist_ok=True)
    for stale in txn_dir.glob("*.parquet"):
        stale.unlink()

    # Transactions are written one parquet part per month. DuckDB reads the
    # whole set with read_parquet('.../transactions/*.parquet') and prunes
    # partitions on date filters, so nothing ever needs to be held in memory.
    state = {"rows": 0, "sample": None}

    def write_month(month: str, frame: pd.DataFrame) -> None:
        frame.to_parquet(txn_dir / f"txn_{month}.parquet", index=False)
        state["rows"] += len(frame)
        if state["sample"] is None:
            state["sample"] = frame.head(2000).copy()

    tables = simulate(cfg, n_merchants=args.merchants, seed=args.seed, txn_writer=write_month)

    manifest = {
        "transactions": {
            "rows": state["rows"],
            "layout": "monthly parquet parts under data/raw/transactions/",
            "parts": len(list(txn_dir.glob("*.parquet"))),
        }
    }
    print(f"  {'transactions':<28} {state['rows']:>12,} rows  ->  transactions/*.parquet")

    for name, df in tables.items():
        if df is None or len(df) == 0:
            continue
        path = args.out / f"{name}.parquet"
        df.to_parquet(path, index=False)
        manifest[name] = {"rows": int(len(df)), "cols": list(df.columns)}
        print(f"  {name:<28} {len(df):>12,} rows  ->  {path.name}")

    # Small committed samples so the repo is browsable without running anything.
    if state["sample"] is not None:
        state["sample"].to_csv(SAMPLE / "transactions_sample.csv", index=False)
    tables["merchants"].head(500).to_csv(SAMPLE / "merchants_sample.csv", index=False)

    with open(args.out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nManifest written to {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
