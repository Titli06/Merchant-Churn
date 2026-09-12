# Merchant retention: what the top line is hiding

**To:** Merchant Growth / Payments Ops
**From:** Analytics
**Date:** August 2026
**Window:** Aug 2024 – Jul 2026 · 6,000 merchants · 7.4M payment attempts

---

## The short version

Active merchants grew from 3,338 to 4,270 and monthly TPV grew from ₹48M to
₹111M over the window. Both numbers are real and both are misleading. Onboarding
outran attrition every single month, which is why the top line never flinched
while retention degraded underneath it.

Three findings, in order of how much money is attached:

**1. We are overstating churn by 17%, and the error is systematic.**
The standard "no activity in 30 days" rule flags 1,721 merchants as churned. 294
of them are still alive — and 99% of those are merchants who go quiet for a
month or two and come back. We are firing win-back campaigns at merchants who
never left, and the miscount inflates every retention number we report.

**2. A six-month issuer degradation cost us merchants, and we did not attribute it.**
Network success rate fell from 93.0% to 89.4% between Aug 2025 and Feb 2026, then
recovered. One issuer caused it. Merchants most exposed to that issuer churned at
19.4% versus 14.0% for the least exposed — a 39% relative increase, with success
rate falling monotonically across exposure deciles from 92.8% to 84.7%.

**3. The obvious culprit is the wrong one.**
Ranked by average success rate, the worst issuer on the network is a small
regional bank sitting around 84.6%. It is not the problem. Ranked by actual
impact on network success rate, it contributes roughly one-ninth of the damage
caused by a large public-sector issuer that ranks only second-worst on raw rate.
The small bank is worse; the large one matters. Chasing the first would have
consumed a quarter of ops time for almost no recovered volume.

---

## What it is worth

| | Annualised revenue | What it means |
|---|---|---|
| Active base run rate | ₹5.83M | The baseline |
| Confirmed churned (65d+ silent) | ₹1.24M | Already lost |
| At risk (30–65d silent) | ₹0.34M | Still recoverable |
| **Attributable to issuer exposure** | **₹0.47M** | What a payments fix can claim |

The last row is the honest number. It is a *subset* of the ₹1.58M total, not an
addition to it, and it is much smaller than the headline. Most churn on this
network is ordinary attrition that no issuer fix will touch. Presenting the
₹1.58M as the value of a payments intervention would be an overclaim, and the
first person to check would find it.

---

## Recommendations

**Replace the single churn number with two.**
`AT RISK` at 30 days silent (catches 94% of real churn, drives outreach) and
`CONFIRMED CHURN` at 65 days silent (98% precise, drives reporting and the
retention denominator). We currently use one number for both jobs, and it is
bad at each. Every churn figure on a dashboard should state its window.

**Do not try to fix this with a cleverer threshold.**
We tested per-merchant adaptive thresholds scaled to each merchant's own
activity rhythm. They barely beat the fixed rule, because a dormant merchant and
a churned merchant are behaviourally identical on the day you look at them. No
recency rule escapes that. The answer is the confirmation lag, not more maths.

**Monitor issuer success rate by contribution, not by level.**
Add volume-weighted rate-effect contribution to the payments dashboard and alert
on it. A rate-level alert would have fired continuously on the small regional
bank for the whole window and stayed silent through the incident that actually
cost us merchants.

**Target retention spend by predicted risk, not by segment.**
The driver model puts 32.3% of the highest-risk decile into churn against an
11.7% base — 2.8× lift. Behavioural segmentation separates churn 3.4× across
three segments, but the top segment holds 82% of TPV and only 6.9% churn, so
segment-level targeting spends most of the budget on healthy merchants.

**Watch self-serve onboarding.**
Self-serve acquired merchants churn at 1.85× the odds of field-sales merchants,
controlling for volume, tenure, success rate and location. This is the single
largest effect in the model that we directly control.

---

## What I would want before acting on this

The success-rate effect is measured on observational data across a natural
experiment, not a randomised one. Exposure to the degraded issuer correlates
with city tier, which correlates with merchant size and competitive density. The
model controls for those, and the exposure gradient survives, but "controls for"
is not "rules out". A holdout test on the next incident — or a randomised
win-back trial on the at-risk cohort — would settle whether the ₹0.47M is
recoverable or merely correlated.

Forecast caveat: 24 monthly observations containing a structural break cannot
support a seasonal model. The best out-of-sample fit came from a simple drift
model at 7.1% MAPE against an 11.4% naive benchmark. Anything claiming to have
estimated festive seasonality from this series is overfitting.
