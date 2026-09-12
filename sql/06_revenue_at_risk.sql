-- ===========================================================================
-- 06_revenue_at_risk.sql
--
-- Turns the retention finding into money, which is the only form in which it
-- will get funded.
--
-- Three numbers, deliberately kept separate because they answer different
-- questions and get conflated constantly:
--
--   1. REALISED LOSS      revenue on transactions that failed. Already gone.
--   2. RUN-RATE AT RISK   annualised revenue from merchants currently silent
--                         but not yet confirmed churned. Recoverable.
--   3. ATTRIBUTABLE LOSS  the slice of (2) associated with issuer exposure
--                         rather than with baseline attrition. This is the
--                         only part an issuer-side fix can claim.
--
-- (3) is the honest number and it is much smaller than (2). Presenting (2) as
-- the value of a payments fix is the standard way these decks overpromise.
-- ===========================================================================

CREATE OR REPLACE VIEW revenue_params AS
SELECT 42.0 / 10000.0 AS take_rate;   -- 42 bps, mirrors config/params.yml


-- 1. Revenue lost to failed transactions, by month.
CREATE OR REPLACE TABLE realised_failure_loss AS
SELECT
    n.month,
    ROUND(n.failed_value_inr, 2)                          AS failed_value_inr,
    ROUND(n.failed_value_inr * p.take_rate, 2)            AS revenue_lost_inr,
    n.success_rate,
    -- Counterfactual: value that would have succeeded at the pre-incident
    -- baseline rate. Uses the first six months as the baseline, matching the
    -- window used in sql/04 so the two analyses cannot disagree.
    ROUND(GREATEST(0,
        (SELECT AVG(success_rate) FROM (
            SELECT success_rate FROM network_health_monthly
            ORDER BY month LIMIT 6)) - n.success_rate)
        * (n.failed_value_inr + n.tpv_inr) * p.take_rate, 2) AS excess_revenue_lost_inr
FROM network_health_monthly n
CROSS JOIN revenue_params p
ORDER BY n.month;


-- 2 and 3. Merchant-level run-rate at risk, split by issuer exposure.
CREATE OR REPLACE TABLE revenue_at_risk AS
WITH silent AS (
    SELECT
        c.merchant_id,
        c.category,
        c.city_tier,
        c.days_since_last_success,
        CASE WHEN c.days_since_last_success >= 65 THEN 'confirmed_churn'
             WHEN c.days_since_last_success >= 30 THEN 'at_risk'
             ELSE 'active' END                            AS status
    FROM merchant_churn_status c
),
run_rate AS (
    -- Monthly TPV over each merchant's last three *active* months, so a
    -- merchant's run rate is not deflated by the very silence being measured.
    SELECT
        merchant_id,
        AVG(tpv_inr) AS monthly_tpv_run_rate
    FROM (
        SELECT
            merchant_id, tpv_inr,
            ROW_NUMBER() OVER (PARTITION BY merchant_id ORDER BY month DESC) AS rn
        FROM merchant_month
    ) t
    WHERE rn <= 3
    GROUP BY merchant_id
),
exposure AS (
    SELECT merchant_id, degraded_issuer_exposure
    FROM merchant_features
)
SELECT
    s.status,
    COUNT(*)                                                          AS merchants,
    ROUND(SUM(r.monthly_tpv_run_rate), 2)                             AS monthly_tpv_at_risk,
    ROUND(SUM(r.monthly_tpv_run_rate) * 12
          * (SELECT take_rate FROM revenue_params), 2)                AS annualised_revenue_at_risk,
    ROUND(AVG(e.degraded_issuer_exposure), 4)                         AS avg_issuer_exposure,
    -- Attributable slice: exposure above the network median is the part a
    -- targeted issuer-side fix could plausibly address.
    ROUND(SUM(CASE WHEN e.degraded_issuer_exposure
                        > (SELECT MEDIAN(degraded_issuer_exposure) FROM merchant_features)
                   THEN r.monthly_tpv_run_rate ELSE 0 END) * 12
          * (SELECT take_rate FROM revenue_params), 2)                AS attributable_revenue_at_risk
FROM silent s
JOIN run_rate r ON r.merchant_id = s.merchant_id
LEFT JOIN exposure e ON e.merchant_id = s.merchant_id
GROUP BY s.status
ORDER BY annualised_revenue_at_risk DESC;
