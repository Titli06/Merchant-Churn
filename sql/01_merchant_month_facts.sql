-- ===========================================================================
-- 01_merchant_month_facts.sql
--
-- Builds the two grains everything else reads from:
--   merchant_month       one row per merchant per active month
--   merchant_bank_month  one row per merchant per issuer bank per month
--
-- Design decision worth defending in an interview: TPV is defined on SUCCESSFUL
-- transactions only, while success_rate is defined over ATTEMPTED transactions.
-- Mixing those two denominators is the single most common way merchant
-- dashboards end up lying -- a merchant whose success rate collapses shows
-- falling TPV, and if you compute "average transaction value" over successes
-- only, it can rise at the same time, which reads as healthy.
-- ===========================================================================

CREATE OR REPLACE TABLE merchant_month AS
WITH base AS (
    SELECT
        t.merchant_id,
        t.txn_month,
        COUNT(*)                                                   AS attempted_txns,
        COUNT(*) FILTER (WHERE t.status = 'SUCCESS')               AS successful_txns,
        COUNT(*) FILTER (WHERE t.status = 'FAILED')                AS failed_txns,
        SUM(t.amount_inr) FILTER (WHERE t.status = 'SUCCESS')      AS tpv_inr,
        SUM(t.amount_inr)                                          AS attempted_value_inr,
        COUNT(DISTINCT t.txn_date)                                 AS active_days,
        AVG(t.amount_inr) FILTER (WHERE t.status = 'SUCCESS')      AS avg_ticket_inr,
        MEDIAN(t.amount_inr) FILTER (WHERE t.status = 'SUCCESS')   AS median_ticket_inr,
        MIN(t.txn_date)                                            AS first_txn_date,
        MAX(t.txn_date)                                            AS last_txn_date,
        COUNT(DISTINCT t.issuer_bank)                              AS distinct_issuers
    FROM transactions t
    GROUP BY t.merchant_id, t.txn_month
),
tickets AS (
    SELECT
        merchant_id,
        ticket_month,
        COUNT(*)                                        AS tickets_raised,
        COUNT(*) FILTER (WHERE NOT resolved)            AS tickets_unresolved,
        COUNT(*) FILTER (WHERE ticket_category = 'PAYMENT_FAILURE') AS tickets_payment_failure
    FROM support_tickets
    GROUP BY merchant_id, ticket_month
)
SELECT
    b.merchant_id,
    b.txn_month                                            AS month,
    b.attempted_txns,
    b.successful_txns,
    b.failed_txns,
    ROUND(b.successful_txns * 1.0 / NULLIF(b.attempted_txns, 0), 6) AS success_rate,
    ROUND(b.tpv_inr, 2)                                    AS tpv_inr,
    ROUND(b.attempted_value_inr, 2)                        AS attempted_value_inr,
    -- Value actually lost to failures. This is the number that makes an
    -- SR problem legible to a commercial stakeholder.
    ROUND(b.attempted_value_inr - COALESCE(b.tpv_inr, 0), 2) AS failed_value_inr,
    b.active_days,
    ROUND(b.avg_ticket_inr, 2)                             AS avg_ticket_inr,
    ROUND(b.median_ticket_inr, 2)                          AS median_ticket_inr,
    b.distinct_issuers,
    b.first_txn_date,
    b.last_txn_date,
    COALESCE(k.tickets_raised, 0)                          AS tickets_raised,
    COALESCE(k.tickets_unresolved, 0)                      AS tickets_unresolved,
    COALESCE(k.tickets_payment_failure, 0)                 AS tickets_payment_failure,
    s.avg_settlement_delay_days
FROM base b
LEFT JOIN tickets k
       ON k.merchant_id = b.merchant_id
      AND k.ticket_month = b.txn_month
LEFT JOIN settlements s
       ON s.merchant_id = b.merchant_id
      AND s.settle_month = b.txn_month;


-- ---------------------------------------------------------------------------
-- Merchant x issuer bank x month. This is what makes the incident attributable
-- to specific merchants rather than just visible in aggregate.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE merchant_bank_month AS
SELECT
    merchant_id,
    txn_month                                            AS month,
    issuer_bank,
    COUNT(*)                                             AS attempted_txns,
    COUNT(*) FILTER (WHERE status = 'SUCCESS')           AS successful_txns,
    ROUND(COUNT(*) FILTER (WHERE status = 'SUCCESS') * 1.0 / COUNT(*), 6) AS success_rate,
    ROUND(SUM(amount_inr) FILTER (WHERE status = 'SUCCESS'), 2)           AS tpv_inr,
    ROUND(SUM(amount_inr), 2)                            AS attempted_value_inr
FROM transactions
GROUP BY merchant_id, txn_month, issuer_bank;


-- ---------------------------------------------------------------------------
-- Month-over-month deltas on the merchant grain, using window functions over a
-- densified calendar so that a *missing* month reads as zero rather than
-- silently disappearing from the LAG. This is the subtle bug that makes most
-- retention dashboards overstate health: if you LAG over only the months a
-- merchant transacted, a merchant who went dark for two months looks like they
-- had no gap at all.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE merchant_month_dense AS
WITH calendar AS (
    SELECT DISTINCT month FROM merchant_month
),
spine AS (
    SELECT
        m.merchant_id,
        c.month
    FROM (SELECT DISTINCT merchant_id, MIN(month) AS first_month
          FROM merchant_month GROUP BY merchant_id) m
    CROSS JOIN calendar c
    WHERE c.month >= m.first_month
),
joined AS (
    SELECT
        s.merchant_id,
        s.month,
        COALESCE(mm.attempted_txns, 0)   AS attempted_txns,
        COALESCE(mm.successful_txns, 0)  AS successful_txns,
        COALESCE(mm.tpv_inr, 0)          AS tpv_inr,
        mm.success_rate,
        COALESCE(mm.tickets_unresolved, 0) AS tickets_unresolved,
        mm.avg_settlement_delay_days,
        CASE WHEN mm.merchant_id IS NULL THEN 0 ELSE 1 END AS was_active
    FROM spine s
    LEFT JOIN merchant_month mm
           ON mm.merchant_id = s.merchant_id
          AND mm.month = s.month
)
SELECT
    merchant_id,
    month,
    attempted_txns,
    successful_txns,
    tpv_inr,
    success_rate,
    tickets_unresolved,
    avg_settlement_delay_days,
    was_active,
    LAG(tpv_inr) OVER w                                     AS prev_tpv_inr,
    tpv_inr - LAG(tpv_inr) OVER w                           AS tpv_mom_delta,
    CASE WHEN LAG(tpv_inr) OVER w > 0
         THEN ROUND((tpv_inr - LAG(tpv_inr) OVER w) / LAG(tpv_inr) OVER w, 6)
    END                                                     AS tpv_mom_pct,
    -- Trailing 3-month success rate, weighted by attempts, as a merchant
    -- actually experiences it. AVG of a rate would weight a 3-txn month the
    -- same as a 3,000-txn month.
    ROUND(
        SUM(successful_txns) OVER w3 * 1.0
        / NULLIF(SUM(attempted_txns) OVER w3, 0), 6)        AS success_rate_3m,
    SUM(tpv_inr) OVER w3                                    AS tpv_3m,
    ROW_NUMBER() OVER w                                     AS tenure_months,
    SUM(was_active) OVER w                                  AS active_months_to_date
FROM joined
WINDOW
    w  AS (PARTITION BY merchant_id ORDER BY month),
    w3 AS (PARTITION BY merchant_id ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW);
