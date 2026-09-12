-- ===========================================================================
-- 00_schema.sql  --  bind the raw parquet files as relations
--
-- This is the ONLY file that is engine-specific. Everything from 01 onwards is
-- plain ANSI SQL against these relation names, so the same scripts run against
-- Postgres once `src/load_to_postgres.py` has populated the equivalent tables.
--
-- Note the transactions table is a partitioned parquet set (one part per
-- month). DuckDB prunes parts on a date predicate, so a query filtered to a
-- few months never touches the other files.
-- ===========================================================================

CREATE OR REPLACE VIEW transactions AS
SELECT
    txn_id,
    merchant_id,
    txn_ts,
    CAST(txn_ts AS DATE)          AS txn_date,
    DATE_TRUNC('month', txn_ts)   AS txn_month,
    amount_inr,
    issuer_bank,
    status,
    NULLIF(failure_reason, '')    AS failure_reason
FROM read_parquet('data/raw/transactions/*.parquet');

CREATE OR REPLACE VIEW merchants AS
SELECT * FROM read_parquet('data/raw/merchants.parquet');

CREATE OR REPLACE VIEW support_tickets AS
SELECT
    merchant_id,
    created_at,
    DATE_TRUNC('month', created_at) AS ticket_month,
    ticket_category,
    resolved
FROM read_parquet('data/raw/support_tickets.parquet');

CREATE OR REPLACE VIEW settlements AS
SELECT
    merchant_id,
    CAST(month || '-01' AS DATE) AS settle_month,
    avg_settlement_delay_days
FROM read_parquet('data/raw/settlements.parquet');

-- The observation date anchors every "as of" calculation. Deriving it from the
-- data rather than hard-coding it means the pipeline stays correct if the
-- generator is re-run over a different window.
CREATE OR REPLACE VIEW analysis_window AS
SELECT
    MIN(txn_date) AS first_date,
    MAX(txn_date) AS observation_date,
    DATE_TRUNC('month', MAX(txn_date)) AS observation_month
FROM transactions;
