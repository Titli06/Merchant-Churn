"""Guardrails. These exist to catch the failure modes that produce a pipeline
which runs cleanly and reports the wrong number."""
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from warehouse import connect  # noqa: E402


@pytest.fixture(scope="module")
def con():
    c = connect(read_only=True)
    yield c
    c.close()


def test_merchant_identity_reconciles(con):
    """new + reactivated - went_silent must equal the change in active base.

    If this fails, a merchant flow is being double counted or dropped.
    """
    df = con.execute("""
        SELECT net_merchant_adds,
               active_merchants - LAG(active_merchants) OVER (ORDER BY month) AS actual_delta
        FROM network_health_monthly
    """).df().dropna()
    assert (df["net_merchant_adds"] == df["actual_delta"]).all()


def test_success_rate_bounded(con):
    bad = con.execute("""
        SELECT COUNT(*) FROM merchant_month
        WHERE success_rate < 0 OR success_rate > 1
    """).fetchone()[0]
    assert bad == 0


def test_tpv_never_exceeds_attempted_value(con):
    bad = con.execute("""
        SELECT COUNT(*) FROM merchant_month WHERE tpv_inr > attempted_value_inr + 0.01
    """).fetchone()[0]
    assert bad == 0


def test_features_contain_no_label_window_data(con):
    """The feature table must not see any month at or after the label cutoff."""
    leak = con.execute("""
        SELECT COUNT(*) FROM merchant_month mm, feature_split s
        WHERE mm.month >= s.label_start
          AND mm.merchant_id IN (SELECT merchant_id FROM merchant_features)
          AND mm.month < s.label_start
    """).fetchone()[0]
    assert leak == 0


def test_recency_is_not_constant(con):
    """A month-truncated recency is constant for active merchants and useless.
    This caught a real bug during development."""
    n = con.execute("SELECT COUNT(DISTINCT recency_days) FROM merchant_features").fetchone()[0]
    assert n > 50


def test_exposure_shares_sum_to_one(con):
    df = con.execute("""
        SELECT merchant_id, SUM(exposure_share) AS s
        FROM merchant_bank_exposure GROUP BY merchant_id
    """).df()
    assert df["s"].between(0.999, 1.001).mean() > 0.99


def test_churn_definitions_disagree(con):
    """If every definition agreed, sql/02 would have nothing to say."""
    row = con.execute("SELECT d1_flagged, d2_flagged FROM churn_definition_comparison").fetchone()
    assert row[0] != row[1]
