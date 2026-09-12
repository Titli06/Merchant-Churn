-- ===========================================================================
-- 02_churn_definitions.sql   --  the part that actually matters
--
-- "Churn" is not in the data. It is a decision. This file implements three
-- competing definitions and materialises all of them side by side so they can
-- be scored against each other (tests/test_churn_definition.py) instead of
-- one being asserted by fiat.
--
-- D1  NAIVE          no successful transaction in the last 30 days.
-- D2  ADAPTIVE       silent for longer than 2x the merchant's OWN historical
--                    p90 activity gap (floored at 21 days).
-- D3  VALUE_DECAY    D2, or still transacting but trailing-30d TPV has fallen
--                    below 25% of their own trailing-90d run rate.
--
-- Why D1 is wrong, concretely: an electronics dealer who transacts eight times
-- a month has a p90 gap of ~11 days. A wedding-season apparel merchant may
-- legitimately go 40 days between bursts. D1 calls the second one churned every
-- single winter, and never catches the first one until they are long gone. A
-- fixed window imposes one frequency assumption on a merchant base whose
-- transaction frequency spans three orders of magnitude.
--
-- Leakage guard: the baseline gap distribution is computed on data strictly
-- BEFORE the observation window opens. If you compute a merchant's p90 gap over
-- all history including their final silence, their own silence inflates the
-- threshold and the definition stops being able to fire. That bug is easy to
-- write and almost invisible in review.
-- ===========================================================================

CREATE OR REPLACE TABLE merchant_activity_gaps AS
WITH active_days AS (
    SELECT DISTINCT merchant_id, txn_date
    FROM transactions
    WHERE status = 'SUCCESS'
),
gaps AS (
    SELECT
        merchant_id,
        txn_date,
        txn_date - LAG(txn_date) OVER (PARTITION BY merchant_id ORDER BY txn_date) AS gap_days
    FROM active_days
)
SELECT
    g.merchant_id,
    g.txn_date,
    g.gap_days
FROM gaps g
WHERE g.gap_days IS NOT NULL;


CREATE OR REPLACE TABLE merchant_churn_status AS
WITH win AS (
    SELECT observation_date FROM analysis_window
),
-- Baseline computed only on gaps that closed before the observation window.
baseline AS (
    SELECT
        g.merchant_id,
        COUNT(*)                                                        AS baseline_gap_count,
        ROUND(AVG(g.gap_days), 3)                                       AS mean_gap_days,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY g.gap_days)        AS p90_gap_days,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY g.gap_days)        AS p50_gap_days
    FROM merchant_activity_gaps g
    CROSS JOIN win w
    WHERE g.txn_date < w.observation_date - 30
    GROUP BY g.merchant_id
),
last_seen AS (
    SELECT
        t.merchant_id,
        MAX(t.txn_date) FILTER (WHERE t.status = 'SUCCESS')  AS last_success_date,
        MAX(t.txn_date)                                      AS last_attempt_date,
        MIN(t.txn_date)                                      AS first_txn_date,
        COUNT(DISTINCT t.txn_month)                          AS months_with_activity
    FROM transactions t
    GROUP BY t.merchant_id
),
recent_value AS (
    SELECT
        t.merchant_id,
        SUM(t.amount_inr) FILTER (
            WHERE t.status = 'SUCCESS' AND t.txn_date > w.observation_date - 30)  AS tpv_30d,
        SUM(t.amount_inr) FILTER (
            WHERE t.status = 'SUCCESS'
              AND t.txn_date <= w.observation_date - 30
              AND t.txn_date >  w.observation_date - 120)                         AS tpv_prior_90d,
        COUNT(*) FILTER (
            WHERE t.txn_date > w.observation_date - 30)                           AS attempts_30d,
        COUNT(*) FILTER (
            WHERE t.status = 'SUCCESS' AND t.txn_date > w.observation_date - 30)  AS successes_30d
    FROM transactions t
    CROSS JOIN win w
    GROUP BY t.merchant_id
)
SELECT
    m.merchant_id,
    m.category,
    m.city_tier,
    m.acquisition_channel,
    m.device_type,
    m.state,
    w.observation_date,
    ls.first_txn_date,
    ls.last_success_date,
    ls.months_with_activity,
    w.observation_date - ls.last_success_date                    AS days_since_last_success,
    b.baseline_gap_count,
    b.p50_gap_days,
    b.p90_gap_days,
    -- The adaptive threshold, floored so that a hyper-frequent merchant is not
    -- declared churned after a long weekend.
    GREATEST(21, ROUND(2.0 * COALESCE(b.p90_gap_days, 15), 1))    AS adaptive_threshold_days,
    COALESCE(tr.tpv_30d, 0)                                       AS tpv_30d,
    COALESCE(tr.tpv_prior_90d, 0)                                 AS tpv_prior_90d,
    CASE WHEN COALESCE(tr.tpv_prior_90d, 0) > 0
         THEN ROUND(COALESCE(tr.tpv_30d, 0) / (tr.tpv_prior_90d / 3.0), 4)
    END                                                           AS tpv_run_rate_ratio,

    -- ---- D1: naive fixed window ------------------------------------------
    CASE WHEN w.observation_date - ls.last_success_date >= 30
         THEN 1 ELSE 0 END                                        AS churn_d1_naive,

    -- ---- D2: adaptive, merchant-relative ---------------------------------
    CASE
        WHEN b.baseline_gap_count IS NULL OR b.baseline_gap_count < 5
            -- too little history to personalise; fall back to the fixed rule
            THEN CASE WHEN w.observation_date - ls.last_success_date >= 30 THEN 1 ELSE 0 END
        WHEN w.observation_date - ls.last_success_date
             > GREATEST(21, 2.0 * b.p90_gap_days)
            THEN 1
        ELSE 0
    END                                                           AS churn_d2_adaptive,

    -- ---- D3: adaptive + value decay (early warning) -----------------------
    CASE
        WHEN (CASE
                WHEN b.baseline_gap_count IS NULL OR b.baseline_gap_count < 5
                    THEN CASE WHEN w.observation_date - ls.last_success_date >= 30 THEN 1 ELSE 0 END
                WHEN w.observation_date - ls.last_success_date
                     > GREATEST(21, 2.0 * b.p90_gap_days) THEN 1
                ELSE 0 END) = 1 THEN 1
        WHEN COALESCE(tr.tpv_prior_90d, 0) > 0
             AND COALESCE(tr.tpv_30d, 0) / (tr.tpv_prior_90d / 3.0) < 0.25 THEN 1
        ELSE 0
    END                                                           AS churn_d3_value_decay
FROM merchants m
CROSS JOIN win w
LEFT JOIN last_seen ls ON ls.merchant_id = m.merchant_id
LEFT JOIN baseline  b  ON b.merchant_id  = m.merchant_id
LEFT JOIN recent_value tr ON tr.merchant_id = m.merchant_id
WHERE ls.merchant_id IS NOT NULL;


-- ---------------------------------------------------------------------------
-- Headline: how much do the three definitions disagree? If they agreed, the
-- choice would not matter and this file would be pointless.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW churn_definition_comparison AS
SELECT
    COUNT(*)                                                     AS merchants_scored,
    SUM(churn_d1_naive)                                          AS d1_flagged,
    SUM(churn_d2_adaptive)                                       AS d2_flagged,
    SUM(churn_d3_value_decay)                                    AS d3_flagged,
    ROUND(AVG(churn_d1_naive), 4)                                AS d1_rate,
    ROUND(AVG(churn_d2_adaptive), 4)                             AS d2_rate,
    ROUND(AVG(churn_d3_value_decay), 4)                          AS d3_rate,
    SUM(CASE WHEN churn_d1_naive = 1 AND churn_d2_adaptive = 0 THEN 1 ELSE 0 END) AS d1_only,
    SUM(CASE WHEN churn_d2_adaptive = 1 AND churn_d1_naive = 0 THEN 1 ELSE 0 END) AS d2_only,
    SUM(CASE WHEN churn_d3_value_decay = 1 AND churn_d2_adaptive = 0 THEN 1 ELSE 0 END) AS d3_incremental
FROM merchant_churn_status;
