-- ===========================================================================
-- 04_incident_diagnosis.sql
--
-- Network success rate falls ~3.7pp for six months and recovers. This file
-- works out what happened, in the order an analyst would actually do it.
--
--   a) bank_month_sr           -- is the drop concentrated in one issuer?
--   b) sr_variance_decomposition -- mix shift, or rate shift within issuers?
--   c) failure_reason_mix      -- does the failure signature say issuer-side?
--   d) merchant_bank_exposure  -- which merchants ate the loss?
--   e) exposure_vs_churn       -- did the ones who ate it leave?
--
-- The trap: the worst-SR bank on the network is NOT the one that caused this.
-- A small regional bank sits ~10pp below everyone else for the whole window and
-- drifts lower. It is the obvious answer to "which bank has the worst success
-- rate" and it is the wrong one, because it carries too little volume to move
-- the network. Ranking by SR finds it; ranking by *volume-weighted SR
-- contribution delta* finds the real one.
-- ===========================================================================

CREATE OR REPLACE TABLE bank_month_sr AS
SELECT
    month,
    issuer_bank,
    SUM(attempted_txns)                                              AS attempted_txns,
    SUM(successful_txns)                                             AS successful_txns,
    ROUND(SUM(successful_txns) * 1.0 / NULLIF(SUM(attempted_txns), 0), 6) AS success_rate,
    ROUND(SUM(attempted_value_inr), 2)                               AS attempted_value_inr,
    ROUND(SUM(attempted_value_inr) - SUM(tpv_inr), 2)                AS failed_value_inr,
    ROUND(SUM(attempted_txns) * 1.0
          / SUM(SUM(attempted_txns)) OVER (PARTITION BY month), 6)   AS volume_share
FROM merchant_bank_month
GROUP BY month, issuer_bank
ORDER BY month, issuer_bank;


-- ---------------------------------------------------------------------------
-- Contribution analysis. Network SR is a volume-weighted average of issuer SRs,
-- so a change in network SR decomposes into (i) issuers changing their own SR
-- at constant mix, and (ii) volume shifting between issuers of differing SR.
-- Comparing each month to a pre-incident baseline separates the two.
--
-- rate_effect = baseline_share x (sr_t - sr_baseline)
-- mix_effect  = (share_t - baseline_share) x (sr_baseline - network_sr_baseline)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE sr_contribution AS
WITH baseline_window AS (
    -- First six months, before anything happens.
    SELECT DISTINCT month FROM bank_month_sr ORDER BY month LIMIT 6
),
baseline AS (
    SELECT
        b.issuer_bank,
        SUM(b.successful_txns) * 1.0 / SUM(b.attempted_txns)  AS sr_baseline,
        SUM(b.attempted_txns) * 1.0
            / SUM(SUM(b.attempted_txns)) OVER ()              AS share_baseline
    FROM bank_month_sr b
    WHERE b.month IN (SELECT month FROM baseline_window)
    GROUP BY b.issuer_bank
),
network_baseline AS (
    SELECT SUM(sr_baseline * share_baseline) AS network_sr_baseline FROM baseline
)
SELECT
    m.month,
    m.issuer_bank,
    m.success_rate,
    b.sr_baseline,
    m.volume_share,
    b.share_baseline,
    ROUND(m.success_rate - b.sr_baseline, 6)                                AS sr_delta,
    ROUND(b.share_baseline * (m.success_rate - b.sr_baseline), 8)           AS rate_effect,
    ROUND((m.volume_share - b.share_baseline)
          * (b.sr_baseline - n.network_sr_baseline), 8)                     AS mix_effect,
    ROUND(m.failed_value_inr, 2)                                            AS failed_value_inr
FROM bank_month_sr m
JOIN baseline b        ON b.issuer_bank = m.issuer_bank
CROSS JOIN network_baseline n
ORDER BY m.month, ABS(b.share_baseline * (m.success_rate - b.sr_baseline)) DESC;


-- Ranked culprit list: who moved the network, not who has the worst rate.
CREATE OR REPLACE VIEW incident_culprit_ranking AS
WITH worst AS (
    SELECT
        issuer_bank,
        ROUND(AVG(success_rate), 6)                     AS avg_success_rate,
        ROUND(AVG(volume_share), 6)                     AS avg_volume_share,
        ROUND(MIN(sr_delta), 6)                         AS worst_sr_delta,
        ROUND(SUM(rate_effect), 6)                      AS cumulative_rate_effect,
        ROUND(SUM(failed_value_inr), 2)                 AS total_failed_value_inr
    FROM sr_contribution
    GROUP BY issuer_bank
)
SELECT
    issuer_bank,
    avg_success_rate,
    avg_volume_share,
    worst_sr_delta,
    cumulative_rate_effect,
    total_failed_value_inr,
    RANK() OVER (ORDER BY avg_success_rate ASC)          AS rank_by_worst_sr,
    RANK() OVER (ORDER BY cumulative_rate_effect ASC)    AS rank_by_network_impact
FROM worst
ORDER BY cumulative_rate_effect ASC;


-- ---------------------------------------------------------------------------
-- Failure signature. Issuer-side outages look different from customer-side
-- declines: timeouts and issuer-down replace insufficient-funds and wrong-PIN.
-- If the mix does not move, the SR drop is demand-side, not infrastructure.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE failure_reason_mix AS
SELECT
    txn_month                                                        AS month,
    issuer_bank,
    failure_reason,
    COUNT(*)                                                         AS failures,
    ROUND(COUNT(*) * 1.0
          / SUM(COUNT(*)) OVER (PARTITION BY txn_month, issuer_bank), 6) AS reason_share
FROM transactions
WHERE status = 'FAILED'
GROUP BY txn_month, issuer_bank, failure_reason;


-- ---------------------------------------------------------------------------
-- Merchant-level exposure to each issuer, measured on the pre-incident window
-- so that exposure is not contaminated by the incident's own effect on volume.
-- (During an outage a merchant's failed transactions still count as attempts,
-- but customers retry on other apps, so post-incident share is endogenous.)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE merchant_bank_exposure AS
WITH pre AS (
    SELECT DISTINCT month FROM merchant_bank_month ORDER BY month LIMIT 6
),
totals AS (
    SELECT merchant_id, SUM(attempted_txns) AS total_attempts
    FROM merchant_bank_month
    WHERE month IN (SELECT month FROM pre)
    GROUP BY merchant_id
)
SELECT
    b.merchant_id,
    b.issuer_bank,
    SUM(b.attempted_txns)                                            AS attempts_pre,
    ROUND(SUM(b.attempted_txns) * 1.0 / NULLIF(t.total_attempts, 0), 6) AS exposure_share
FROM merchant_bank_month b
JOIN totals t ON t.merchant_id = b.merchant_id
WHERE b.month IN (SELECT month FROM pre)
GROUP BY b.merchant_id, b.issuer_bank, t.total_attempts;
