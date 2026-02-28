"""Tests for sentiment metrics module against real FMP database data."""

from __future__ import annotations

import math
import sqlite3

import pandas as pd
import pytest

from pipeline.config import PipelineConfig
from pipeline.data import load_universe
from pipeline.data import fmp as fmp_module
from pipeline.data.models import CompanyData
from pipeline.metrics.sentiment import SentimentMetrics, compute_sentiment

DB_PATH = "/home/mattm/projects/Pers/financial_db/data/fmp.db"


@pytest.fixture
def db_config() -> PipelineConfig:
    """Config with a valid exchange discovered from the database."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    row = conn.execute(
        """
        SELECT exchange FROM entity
        WHERE isEtf = 0 AND isAdr = 0 AND isFund = 0 AND isActivelyTrading = 1
        GROUP BY exchange
        ORDER BY COUNT(*) DESC
        LIMIT 1
        """
    ).fetchone()
    conn.close()
    if row is None:
        pytest.skip("No eligible exchanges in database")
    return PipelineConfig(exchanges=[row[0]])


@pytest.fixture
def sample_company(db_config: PipelineConfig) -> CompanyData:
    """Load a real company with at least 1 year of price history."""
    universe = load_universe(db_config)
    if not universe:
        pytest.skip("Universe is empty")

    for entity_id, _ in universe[:20]:
        company = fmp_module.load_company(entity_id, db_config)
        if company is not None and len(company.price_history) > 252:
            return company

    pytest.skip("No company with over 1 year of price history found")


class TestComputeSentiment:
    """Tests for compute_sentiment against real database data."""

    def test_returns_sentiment_metrics_dataclass(
        self, sample_company: CompanyData
    ) -> None:
        """compute_sentiment returns a SentimentMetrics instance."""
        result = compute_sentiment(sample_company)
        assert isinstance(result, SentimentMetrics)

    def test_return_6m_is_float_or_none(
        self, sample_company: CompanyData
    ) -> None:
        """return_6m is either a float or None."""
        result = compute_sentiment(sample_company)
        assert result.return_6m is None or isinstance(result.return_6m, float)

    def test_return_12m_is_float_or_none(
        self, sample_company: CompanyData
    ) -> None:
        """return_12m is either a float or None."""
        result = compute_sentiment(sample_company)
        assert result.return_12m is None or isinstance(result.return_12m, float)

    def test_returns_are_finite_with_sufficient_history(
        self, sample_company: CompanyData
    ) -> None:
        """With >1 year of price history, both returns should be finite floats."""
        result = compute_sentiment(sample_company)
        assert result.return_6m is not None, "Expected float for return_6m"
        assert result.return_12m is not None, "Expected float for return_12m"
        assert math.isfinite(result.return_6m)
        assert math.isfinite(result.return_12m)

    def test_returns_are_reasonable(
        self, sample_company: CompanyData
    ) -> None:
        """Returns should be within plausible bounds (not nonsensical)."""
        result = compute_sentiment(sample_company)
        if result.return_6m is not None:
            assert -0.99 <= result.return_6m <= 10.0, (
                f"return_6m={result.return_6m} outside plausible range"
            )
        if result.return_12m is not None:
            assert -0.99 <= result.return_12m <= 10.0, (
                f"return_12m={result.return_12m} outside plausible range"
            )


class TestEdgeCases:
    """Edge case tests using synthetic data."""

    def test_empty_price_history(self) -> None:
        """Empty price history returns SentimentMetrics(None, None)."""
        empty_company = CompanyData(
            symbol="TEST",
            company_name="Test",
            sector="Test",
            exchange="TEST",
            financials=pd.DataFrame(),
            latest_price=0.0,
            market_cap=0.0,
            shares_outstanding=0.0,
            price_history=pd.DataFrame(columns=["date", "close"]),
        )
        result = compute_sentiment(empty_company)
        assert result.return_6m is None
        assert result.return_12m is None

    def test_single_price_point(self) -> None:
        """A single price point yields both returns as None."""
        company = CompanyData(
            symbol="TEST",
            company_name="Test",
            sector="Test",
            exchange="TEST",
            financials=pd.DataFrame(),
            latest_price=100.0,
            market_cap=0.0,
            shares_outstanding=0.0,
            price_history=pd.DataFrame({
                "date": ["2024-01-15"],
                "close": [100.0],
            }),
        )
        result = compute_sentiment(company)
        assert result.return_6m is None
        assert result.return_12m is None

    def test_short_history_returns_none_for_12m(self) -> None:
        """Price history shorter than 12 months yields return_12m = None."""
        dates = pd.date_range("2024-06-01", periods=150, freq="B")
        company = CompanyData(
            symbol="TEST",
            company_name="Test",
            sector="Test",
            exchange="TEST",
            financials=pd.DataFrame(),
            latest_price=110.0,
            market_cap=0.0,
            shares_outstanding=0.0,
            price_history=pd.DataFrame({
                "date": [d.strftime("%Y-%m-%d") for d in dates],
                "close": [100.0 + i * 0.1 for i in range(len(dates))],
            }),
        )
        result = compute_sentiment(company)
        # 150 business days is roughly 7 months — enough for 6m, not 12m
        assert result.return_6m is not None
        assert result.return_12m is None

    def test_zero_reference_close_returns_none(self) -> None:
        """A zero close price at the reference date yields None."""
        dates = pd.date_range("2024-01-01", periods=200, freq="B")
        closes = [50.0] * len(dates)
        # Set all early prices to zero so the 6m reference hits zero
        for i in range(50):
            closes[i] = 0.0
        company = CompanyData(
            symbol="TEST",
            company_name="Test",
            sector="Test",
            exchange="TEST",
            financials=pd.DataFrame(),
            latest_price=50.0,
            market_cap=0.0,
            shares_outstanding=0.0,
            price_history=pd.DataFrame({
                "date": [d.strftime("%Y-%m-%d") for d in dates],
                "close": closes,
            }),
        )
        result = compute_sentiment(company)
        # 12m reference should land in the zero-price zone
        assert result.return_12m is None
