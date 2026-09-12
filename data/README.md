# Data

`data/raw/` is generated, not committed. Run `make data` (about 30 seconds).

| table | grain | rows |
|---|---|---|
| `transactions/*.parquet` | one row per payment attempt, partitioned by month | ~7.4M |
| `merchants.parquet` | one row per merchant | 6,000 |
| `support_tickets.parquet` | one row per ticket | ~12k |
| `settlements.parquet` | merchant x month | ~87k |
| `bank_sr_truth.parquet` | issuer x month (simulator internals) | 336 |
| `_ground_truth_lifecycle.parquet` | one row per merchant, true churn month | 6,000 |

`data/sample/` holds small committed CSV extracts so the repo is browsable
without running anything.

## The underscore prefix matters

`_ground_truth_lifecycle.parquet` is the simulator's hidden answer key. Exactly
one module reads it: `src/evaluate_churn_definition.py`, which uses it to score
churn *definitions*. Nothing that feeds the warehouse, the driver model, the
segmentation or the forecast touches it. That separation is what makes the
recovered coefficients and the metric scorecard meaningful rather than circular.
