-- ===========================================================================
-- 03_cohort_retention.sql
--
-- Two things live here:
--   cohort_retention        classic triangle, merchant counts + TPV retention
--   network_health_monthly  the aggregate view a leadership deck would show
--
-- The reason both exist: on this network the aggregate is reassuring and the
-- cohort view is not. Active merchants and TPV both rise every month, because
-- onboarding outruns attrition. Retention within cohorts degrades sharply over
-- the same window. Anyone reading only the top line concludes the business is
-- healthy. That gap is the entire argument for cohorting.
-- ===========================================================================

CREATE OR REPLACE TABLE cohort_retention AS
WITH first_activity AS (
    SELECT
        merchant_id,
        MIN(month) AS cohort_month
    FROM merchant_month
    GROUP BY merchant_id
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(*) AS cohort_size
    FROM first_activity
    GROUP BY cohort_month
),
activity AS (
    SELECT
        f.cohort_month,
        mm.month                                                  AS activity_month,
        -- Months elapsed since the cohort's first month.
        (EXTRACT(YEAR FROM mm.month) - EXTRACT(YEAR FROM f.cohort_month)) * 12
          + (EXTRACT(MONTH FROM mm.month) - EXTRACT(MONTH FROM f.cohort_month)) AS month_index,
        COUNT(DISTINCT mm.merchant_id)                            AS active_merchants,
        SUM(mm.tpv_inr)                                           AS tpv_inr
    FROM merchant_month mm
    JOIN first_activity f ON f.merchant_id = mm.merchant_id
    GROUP BY 1, 2, 3
),
first_month_tpv AS (
    SELECT cohort_month, tpv_inr AS cohort_month0_tpv
    FROM activity
    WHERE month_index = 0
)
SELECT
    a.cohort_month,
    a.activity_month,
    CAST(a.month_index AS INTEGER)                                AS month_index,
    s.cohort_size,
    a.active_merchants,
    ROUND(a.active_merchants * 1.0 / s.cohort_size, 4)            AS logo_retention,
    ROUND(a.tpv_inr, 2)                                           AS tpv_inr,
    ROUND(a.tpv_inr / NULLIF(f.cohort_month0_tpv, 0), 4)          AS tpv_retention
FROM activity a
JOIN cohort_sizes s   ON s.cohort_month = a.cohort_month
LEFT JOIN first_month_tpv f ON f.cohort_month = a.cohort_month
ORDER BY a.cohort_month, a.month_index;


-- ---------------------------------------------------------------------------
-- Month-3 retention by cohort: the single number that shows the deterioration.
-- Restricted to cohorts old enough to have a month-3 observation.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW cohort_m3_retention AS
SELECT
    cohort_month,
    cohort_size,
    logo_retention AS m3_logo_retention,
    tpv_retention  AS m3_tpv_retention
FROM cohort_retention
WHERE month_index = 3
ORDER BY cohort_month;


-- ---------------------------------------------------------------------------
-- The aggregate view. Deliberately includes both the flattering top-line
-- numbers and the unflattering rate metrics side by side.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE network_health_monthly AS
WITH monthly AS (
    SELECT
        month,
        COUNT(DISTINCT merchant_id)                       AS active_merchants,
        SUM(tpv_inr)                                      AS tpv_inr,
        SUM(attempted_value_inr)                          AS attempted_value_inr,
        SUM(successful_txns)                              AS successful_txns,
        SUM(attempted_txns)                               AS attempted_txns,
        SUM(failed_value_inr)                             AS failed_value_inr,
        SUM(tickets_raised)                               AS tickets_raised
    FROM merchant_month
    GROUP BY month
),
new_merchants AS (
    SELECT MIN(month) AS month, merchant_id
    FROM merchant_month GROUP BY merchant_id
),
new_counts AS (
    SELECT month, COUNT(*) AS new_merchants FROM new_merchants GROUP BY month
),
-- A merchant is counted as lost in month M if they were active in M-1 and
-- inactive for M and M+1. Requiring two consecutive silent months is the
-- in-period analogue of the confirmation lag recommended in sql/02.
-- Window functions cannot be nested inside an aggregate's FILTER clause, so
-- lag/lead is computed in an inner select and filtered outside it.
transitions AS (
    SELECT
        merchant_id,
        month,
        was_active,
        LAG(was_active)  OVER w AS prev_active,
        LEAD(was_active) OVER w AS next_active,
        ROW_NUMBER()     OVER w AS rn
    FROM merchant_month_dense
    WINDOW w AS (PARTITION BY merchant_id ORDER BY month)
),
churned AS (
    SELECT month, COUNT(*) AS churned_merchants
    FROM transitions
    WHERE was_active = 0 AND prev_active = 1 AND next_active = 0
    GROUP BY month
),
-- Merchants who were silent last month and are transacting again this month.
-- Without this term the merchant-count identity does not close, and net adds
-- read negative in months where the active base is visibly growing. Dormancy
-- is the same blind spot that breaks the 30-day churn rule in sql/02 -- here it
-- breaks the top-line growth arithmetic instead.
reactivated AS (
    SELECT month, COUNT(*) AS reactivated_merchants
    FROM transitions
    WHERE was_active = 1 AND prev_active = 0 AND rn > 1
    GROUP BY month
),
-- Every merchant who stopped transacting this month, whether or not the stop
-- later turns out to be permanent. This is the flow that reconciles the active
-- base; `churned_merchants` deliberately does not, because it waits a month to
-- confirm and therefore excludes single-month dormancy. Reporting only the
-- confirmed number and then wondering why the base does not add up is a very
-- common way to lose an afternoon.
went_silent AS (
    SELECT month, COUNT(*) AS went_silent_merchants
    FROM transitions
    WHERE was_active = 0 AND prev_active = 1
    GROUP BY month
)
SELECT
    m.month,
    m.active_merchants,
    COALESCE(n.new_merchants, 0)                                       AS new_merchants,
    COALESCE(r.reactivated_merchants, 0)                               AS reactivated_merchants,
    COALESCE(g.went_silent_merchants, 0)                               AS went_silent_merchants,
    COALESCE(c.churned_merchants, 0)                                   AS churned_merchants,
    ROUND(m.tpv_inr, 2)                                                AS tpv_inr,
    ROUND(m.successful_txns * 1.0 / NULLIF(m.attempted_txns, 0), 6)    AS success_rate,
    ROUND(m.failed_value_inr, 2)                                       AS failed_value_inr,
    m.tickets_raised,
    ROUND(m.tpv_inr / NULLIF(m.active_merchants, 0), 2)                AS tpv_per_active_merchant,
    -- MoM growth on the headline numbers.
    ROUND((m.active_merchants - LAG(m.active_merchants) OVER (ORDER BY m.month))
          * 1.0 / NULLIF(LAG(m.active_merchants) OVER (ORDER BY m.month), 0), 6) AS merchant_mom_growth,
    ROUND((m.tpv_inr - LAG(m.tpv_inr) OVER (ORDER BY m.month))
          / NULLIF(LAG(m.tpv_inr) OVER (ORDER BY m.month), 0), 6)                AS tpv_mom_growth,
    -- new + reactivated - churned. Positive here while retention degrades is
    -- exactly the masking effect described in the header.
    COALESCE(n.new_merchants, 0)
      + COALESCE(r.reactivated_merchants, 0)
      - COALESCE(g.went_silent_merchants, 0)                           AS net_merchant_adds,
    -- The churn rule needs a following month to confirm, and the first month
    -- has no prior. Both boundaries are censored and must be excluded from any
    -- trend read rather than plotted as a collapse.
    CASE WHEN m.month = (SELECT MIN(month) FROM monthly)
           OR m.month >= (SELECT MAX(month) FROM monthly) THEN 1 ELSE 0 END AS is_censored_boundary
FROM monthly m
LEFT JOIN new_counts  n ON n.month = m.month
LEFT JOIN churned     c ON c.month = m.month
LEFT JOIN reactivated r ON r.month = m.month
LEFT JOIN went_silent  g ON g.month = m.month
ORDER BY m.month;
