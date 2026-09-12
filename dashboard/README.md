# Dashboard specification

> `app.py` at the repo root implements the subset of this spec that the
> committed `reports/` artefacts can support, as a deployable Streamlit demo.
> This file remains the full specification for a Power BI / Tableau build
> against the warehouse.

The warehouse is the source of truth; this is the spec for the BI layer on top
of it. Build against `data/warehouse.duckdb` (Power BI and Tableau both read
DuckDB via ODBC), or point at Postgres after `src/load_to_postgres.py`.

## Page 1 — Network health

Purpose: show the top line *and* the thing the top line hides, on one screen.

- KPI row: active merchants, TPV, success rate, confirmed churn rate
- Dual-axis line: active merchants (bars, rising) vs month-3 cohort retention
  (line, falling). This pairing is the whole argument — do not split it across
  two pages.
- Waterfall: new + reactivated − went silent = net adds, by month
- Callout: censored boundary months excluded (`is_censored_boundary = 1`)

Source: `network_health_monthly`, `cohort_m3_retention`

## Page 2 — Payment success diagnosis

- Heatmap: issuer x month success rate
- Ranked bar: cumulative rate effect by issuer, with average success rate shown
  alongside. The point of the page is that these two rankings disagree.
- Stacked area: failure reason mix over time for the selected issuer
- Filter: issuer, city tier, category

Source: `bank_month_sr`, `sr_contribution`, `incident_culprit_ranking`,
`failure_reason_mix`

## Page 3 — Retention and risk

- Cohort triangle: logo retention by cohort x month index
- Scatter: issuer exposure decile vs churn rate, sized by merchant count
- Segment table: name, merchants, TPV share, churn rate, median success rate
- Risk decile bar chart from the model

Source: `cohort_retention`, `exposure_vs_churn`, `reports/segment_profile.csv`

## Page 4 — Revenue at risk

- Three separate cards: realised failure loss, run-rate at risk, attributable
  slice. Do not sum them; they answer different questions and are not additive.

Source: `revenue_at_risk`, `realised_failure_loss`

## Metric definitions to surface in tooltips

Every churn number on the dashboard must state its window. `AT RISK` is 30 days
silent; `CONFIRMED CHURN` is 65 days silent. A dashboard that says "churn" with
no window is the problem this project exists to document.
