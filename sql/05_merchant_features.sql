-- ===========================================================================
-- 05_merchant_features.sql
--
-- One row per merchant, built for modelling. Two rules govern this file:
--
-- 1. TEMPORAL SPLIT. Features come from the first 18 months only; the label
--    comes from the last 6. Building features over all 24 months and then
--    predicting churn inside that same window is the single most common way a
--    churn model reports an AUC in the high 0.9s and is worthless in
--    production -- "average monthly TPV" computed over a window that includes
--    the merchant's silence is just the label wearing a disguise.
--
-- 2. NO GROUND TRUTH. The label is derived from observed behaviour using the
--    65-day confirmation window recommended in sql/02, not from the simulator's
--    hidden lifecycle table.
--
-- The degraded issuer is identified from sql/04 at query time rather than
-- hard-coded, so the pipeline still works if the incident moves.
-- ===========================================================================

CREATE OR REPLACE VIEW feature_split AS
SELECT
    (SELECT MIN(month) FROM merchant_month)                                  AS window_start,
    (SELECT MIN(month) FROM (SELECT DISTINCT month FROM merchant_month
                             ORDER BY month DESC LIMIT 6))                   AS label_start,
    (SELECT MAX(month) FROM merchant_month)                                  AS window_end,
    (SELECT MAX(txn_date) FROM transactions)                                 AS observation_date;


-- The issuer whose own rate change cost the network the most. Note this is
-- ranked on cumulative rate effect, not on average success rate -- see the
-- header of sql/04 for why those two give different answers.
CREATE OR REPLACE VIEW degraded_issuer AS
SELECT issuer_bank
FROM incident_culprit_ranking
ORDER BY cumulative_rate_effect ASC
LIMIT 1;


CREATE OR REPLACE TABLE merchant_features AS
WITH s AS (SELECT * FROM feature_split),

-- ---- behaviour during the FEATURE window only ---------------------------
feat AS (
    SELECT
        mm.merchant_id,
        COUNT(*)                                              AS active_months,
        SUM(mm.attempted_txns)                                AS attempted_txns,
        SUM(mm.successful_txns)                               AS successful_txns,
        SUM(mm.tpv_inr)                                       AS tpv_inr,
        AVG(mm.tpv_inr)                                       AS avg_monthly_tpv,
        AVG(mm.attempted_txns)                                AS avg_monthly_txns,
        STDDEV_SAMP(mm.tpv_inr)                               AS sd_monthly_tpv,
        SUM(mm.successful_txns) * 1.0
            / NULLIF(SUM(mm.attempted_txns), 0)               AS success_rate,
        AVG(mm.avg_ticket_inr)                                AS avg_ticket_inr,
        AVG(mm.active_days)                                   AS avg_active_days,
        SUM(mm.tickets_raised)                                AS tickets_raised,
        SUM(mm.tickets_unresolved)                            AS tickets_unresolved,
        SUM(mm.tickets_payment_failure)                       AS tickets_payment_failure,
        AVG(mm.avg_settlement_delay_days)                     AS avg_settlement_delay_days,
        MIN(mm.month)                                         AS first_active_month,
        MAX(mm.month)                                         AS last_active_month
    FROM merchant_month mm, s
    WHERE mm.month < s.label_start
    GROUP BY mm.merchant_id
),

-- ---- recency / trend measured at the CUTOFF, not at the window end -------
recent AS (
    SELECT
        d.merchant_id,
        MAX(d.success_rate_3m) FILTER (WHERE d.month = s.label_start - INTERVAL 1 MONTH)
                                                              AS success_rate_3m_at_cutoff,
        MAX(d.tpv_3m)          FILTER (WHERE d.month = s.label_start - INTERVAL 1 MONTH)
                                                              AS tpv_3m_at_cutoff,
        MAX(d.tenure_months)   FILTER (WHERE d.month = s.label_start - INTERVAL 1 MONTH)
                                                              AS tenure_months
    FROM merchant_month_dense d, s
    WHERE d.month < s.label_start
    GROUP BY d.merchant_id
),

-- Success-rate trajectory: last three feature months minus first three.
-- A merchant sitting at a flat 88% is a different problem from one that fell
-- from 95% to 88%, and a single averaged rate cannot tell them apart.
sr_trend AS (
    SELECT
        merchant_id,
        SUM(successful_txns) FILTER (WHERE rn_desc <= 3) * 1.0
            / NULLIF(SUM(attempted_txns) FILTER (WHERE rn_desc <= 3), 0)
        - SUM(successful_txns) FILTER (WHERE rn_asc <= 3) * 1.0
            / NULLIF(SUM(attempted_txns) FILTER (WHERE rn_asc <= 3), 0) AS sr_trend
    FROM (
        SELECT
            mm.merchant_id, mm.successful_txns, mm.attempted_txns,
            ROW_NUMBER() OVER (PARTITION BY mm.merchant_id ORDER BY mm.month)      AS rn_asc,
            ROW_NUMBER() OVER (PARTITION BY mm.merchant_id ORDER BY mm.month DESC) AS rn_desc
        FROM merchant_month mm, s
        WHERE mm.month < s.label_start
    ) t
    GROUP BY merchant_id
),

-- ---- exposure to the degraded issuer ------------------------------------
exposure AS (
    SELECT
        e.merchant_id,
        COALESCE(SUM(e.exposure_share) FILTER (
            WHERE e.issuer_bank = (SELECT issuer_bank FROM degraded_issuer)), 0) AS degraded_issuer_exposure
    FROM merchant_bank_exposure e
    GROUP BY e.merchant_id
),

-- ---- activity rhythm ----------------------------------------------------
gaps AS (
    SELECT
        g.merchant_id,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY g.gap_days) AS p90_gap_days,
        AVG(g.gap_days)                                          AS mean_gap_days
    FROM merchant_activity_gaps g, s
    WHERE g.txn_date < s.label_start
    GROUP BY g.merchant_id
),

-- Day-grain recency. Deriving it from `last_active_month` gives a
-- month-truncated value that is identical for every merchant still transacting
-- at the cutoff -- a constant column, which contributes nothing to a model and
-- silently wastes a slot in any clustering.
recency AS (
    SELECT t.merchant_id, MAX(t.txn_date) AS last_txn_date_feature
    FROM transactions t, s
    WHERE t.status = 'SUCCESS' AND t.txn_month < s.label_start
    GROUP BY t.merchant_id
),

-- ---- LABEL: derived from behaviour in the label window -------------------
label AS (
    SELECT
        f.merchant_id,
        COALESCE(MAX(t.txn_date), DATE '1900-01-01')             AS last_success_in_label,
        CASE WHEN COALESCE(MAX(t.txn_date), DATE '1900-01-01')
                  < s.observation_date - 65 THEN 1 ELSE 0 END    AS churned
    FROM feat f
    CROSS JOIN s
    LEFT JOIN transactions t
           ON t.merchant_id = f.merchant_id
          AND t.status = 'SUCCESS'
    GROUP BY f.merchant_id, s.observation_date
)

SELECT
    m.merchant_id,
    m.category,
    m.city_tier,
    m.acquisition_channel,
    m.device_type,
    m.state,
    m.gst_registered,
    m.competitor_density,

    f.active_months,
    f.attempted_txns,
    f.successful_txns,
    ROUND(f.tpv_inr, 2)                                          AS tpv_inr,
    ROUND(f.avg_monthly_tpv, 2)                                  AS avg_monthly_tpv,
    ROUND(f.avg_monthly_txns, 3)                                 AS avg_monthly_txns,
    -- Coefficient of variation: volatility relative to size, so a large
    -- merchant is not flagged volatile purely for being large.
    ROUND(f.sd_monthly_tpv / NULLIF(f.avg_monthly_tpv, 0), 4)     AS tpv_cv,
    ROUND(f.success_rate, 6)                                     AS success_rate,
    ROUND(r.success_rate_3m_at_cutoff, 6)                        AS success_rate_3m_at_cutoff,
    ROUND(st.sr_trend, 6)                                        AS sr_trend,
    ROUND(f.avg_ticket_inr, 2)                                   AS avg_ticket_inr,
    ROUND(f.avg_active_days, 2)                                  AS avg_active_days,
    f.tickets_raised,
    f.tickets_unresolved,
    f.tickets_payment_failure,
    ROUND(f.avg_settlement_delay_days, 4)                        AS avg_settlement_delay_days,
    COALESCE(r.tenure_months, f.active_months)                   AS tenure_months,
    ROUND(COALESCE(g.p90_gap_days, 15), 2)                       AS p90_gap_days,
    ROUND(COALESCE(g.mean_gap_days, 5), 3)                       AS mean_gap_days,
    ROUND(x.degraded_issuer_exposure, 6)                         AS degraded_issuer_exposure,

    -- ---- RFM, scored within the feature window ---------------------------
    DATE_DIFF('day', rc.last_txn_date_feature,
              CAST((SELECT label_start FROM s) AS DATE))               AS recency_days,
    NTILE(5) OVER (ORDER BY rc.last_txn_date_feature)            AS r_score,
    NTILE(5) OVER (ORDER BY f.attempted_txns)                    AS f_score,
    NTILE(5) OVER (ORDER BY f.tpv_inr)                           AS m_score,

    l.churned
FROM feat f
JOIN merchants m   ON m.merchant_id = f.merchant_id
JOIN label l       ON l.merchant_id = f.merchant_id
LEFT JOIN recent r ON r.merchant_id = f.merchant_id
LEFT JOIN sr_trend st ON st.merchant_id = f.merchant_id
LEFT JOIN exposure x  ON x.merchant_id = f.merchant_id
LEFT JOIN gaps g      ON g.merchant_id = f.merchant_id
LEFT JOIN recency rc  ON rc.merchant_id = f.merchant_id;


-- ---------------------------------------------------------------------------
-- The payoff of sql/04: did exposure to the degraded issuer actually predict
-- churn? Deciles of exposure against realised churn rate, with tier held
-- alongside so the confound is visible rather than hidden.
-- ---------------------------------------------------------------------------
-- Two traps avoided here, both of which silently produce a wrong answer:
--
--   a) NTILE is a window function and is evaluated AFTER GROUP BY, so writing
--      it in the same select as the aggregate buckets each distinct exposure
--      value on its own. The decile has to be assigned in a subquery.
--
--   b) Exposure is measured over the pre-incident baseline months. Merchants
--      onboarded after that window have no exposure record and COALESCE to
--      zero, which parks a pile of young, high-early-churn merchants in the
--      bottom decile and manufactures a spurious *negative* gradient. They are
--      excluded rather than zero-filled.
CREATE OR REPLACE VIEW exposure_vs_churn AS
--   c) Exposure is a *share*, so it is noisy when the denominator is small. A
--      merchant with nine baseline transactions can show 0% or 60% exposure on
--      luck alone, and those merchants are also small, and small merchants
--      churn more for unrelated reasons. Without a minimum denominator the
--      bottom decile fills with them and the gradient comes out U-shaped.
WITH eligible AS (
    SELECT f.*
    FROM merchant_features f
    JOIN (
        SELECT merchant_id, SUM(attempts_pre) AS baseline_attempts
        FROM merchant_bank_exposure
        GROUP BY merchant_id
    ) e ON e.merchant_id = f.merchant_id
    WHERE e.baseline_attempts >= 50
),
bucketed AS (
    SELECT
        NTILE(10) OVER (ORDER BY degraded_issuer_exposure) AS exposure_decile,
        degraded_issuer_exposure,
        churned,
        city_tier,
        success_rate_3m_at_cutoff
    FROM eligible
)
SELECT
    exposure_decile,
    ROUND(MIN(degraded_issuer_exposure), 4)            AS min_exposure,
    ROUND(MAX(degraded_issuer_exposure), 4)            AS max_exposure,
    COUNT(*)                                           AS merchants,
    ROUND(AVG(churned), 4)                             AS churn_rate,
    ROUND(AVG(CASE WHEN city_tier = 'tier_3' THEN 1.0 ELSE 0 END), 4) AS tier3_share,
    ROUND(AVG(success_rate_3m_at_cutoff), 4)           AS avg_sr_at_cutoff
FROM bucketed
GROUP BY exposure_decile
ORDER BY exposure_decile;
