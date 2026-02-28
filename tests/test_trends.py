"""Tests for Phase 2: trend analysis metrics."""

from __future__ import annotations

import math
import sqlite3

import pytest

from pipeline.config import PipelineConfig, TrendsConfig
from pipeline.data import load_universe
from pipeline.data import fmp as fmp_module
from pipeline.data.models import CompanyData
from pipeline.metrics.trends import TrendMetrics, compute_trends

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
def trends_config() -> TrendsConfig:
    """Default trends configuration."""
    return TrendsConfig()


@pytest.fixture
def sample_company(db_config: PipelineConfig) -> CompanyData:
    """Load a company with sufficient data for trend analysis.

    Iterates the universe until a company with at least 12 quarterly
    financials is found.
    """
    universe = load_universe(db_config)
    if not universe:
        pytest.skip("Universe is empty")

    for entity_id, symbol in universe[:50]:
        company = fmp_module.load_company(entity_id, db_config)
        if company is not None and len(company.financials) >= 12:
            return company

    pytest.skip("No company with >= 12 quarters found in first 50 entities")


@pytest.fixture
def sample_metrics(
    sample_company: CompanyData, trends_config: TrendsConfig
) -> TrendMetrics:
    """Computed TrendMetrics for the sample company."""
    result = compute_trends(sample_company, trends_config)
    if result is None:
        pytest.skip(
            f"{sample_company.symbol}: compute_trends returned None"
        )
    return result


class TestComputeTrendsReturn:
    """Tests that compute_trends returns the correct type."""

    def test_returns_trend_metrics(
        self, sample_metrics: TrendMetrics
    ) -> None:
        assert isinstance(sample_metrics, TrendMetrics)


class TestRevenueCagr:
    """Tests for revenue CAGR computation."""

    def test_cagr_is_finite(self, sample_metrics: TrendMetrics) -> None:
        assert math.isfinite(sample_metrics.revenue_cagr)

    def test_cagr_is_reasonable(self, sample_metrics: TrendMetrics) -> None:
        # CAGR should typically be between -90% and +500%
        assert -0.9 < sample_metrics.revenue_cagr < 5.0


class TestRevenueQoqStats:
    """Tests for quarter-over-quarter revenue growth statistics."""

    def test_qoq_mean_is_finite(self, sample_metrics: TrendMetrics) -> None:
        assert math.isfinite(sample_metrics.revenue_qoq_growth_mean)

    def test_qoq_variance_is_finite(
        self, sample_metrics: TrendMetrics
    ) -> None:
        assert math.isfinite(sample_metrics.revenue_qoq_growth_var)

    def test_qoq_variance_is_non_negative(
        self, sample_metrics: TrendMetrics
    ) -> None:
        assert sample_metrics.revenue_qoq_growth_var >= 0


class TestRevenueYoyStd:
    """Tests for year-over-year revenue growth standard deviation."""

    def test_yoy_std_is_finite(self, sample_metrics: TrendMetrics) -> None:
        assert math.isfinite(sample_metrics.revenue_yoy_growth_std)

    def test_yoy_std_is_non_negative(
        self, sample_metrics: TrendMetrics
    ) -> None:
        assert sample_metrics.revenue_yoy_growth_std >= 0


class TestMarginRegression:
    """Tests for operating margin regression outputs."""

    def test_intercept_is_finite(
        self, sample_metrics: TrendMetrics
    ) -> None:
        assert math.isfinite(sample_metrics.margin_intercept)

    def test_slope_is_finite(self, sample_metrics: TrendMetrics) -> None:
        assert math.isfinite(sample_metrics.margin_slope)

    def test_r_squared_between_zero_and_one(
        self, sample_metrics: TrendMetrics
    ) -> None:
        assert 0.0 <= sample_metrics.margin_r_squared <= 1.0


class TestFcfConversion:
    """Tests for FCF conversion regression or fallback."""

    def test_conversion_is_fallback_is_bool(
        self, sample_metrics: TrendMetrics
    ) -> None:
        assert isinstance(sample_metrics.conversion_is_fallback, bool)

    def test_regression_or_fallback_exclusive(
        self, sample_metrics: TrendMetrics
    ) -> None:
        """Regression fields and fallback median are mutually exclusive."""
        if sample_metrics.conversion_is_fallback:
            # Fallback: median set, regression fields None
            assert sample_metrics.conversion_median is not None
            assert sample_metrics.conversion_intercept is None
            assert sample_metrics.conversion_slope is None
            assert sample_metrics.conversion_r_squared is None
        else:
            # Full regression: intercept/slope/R² set, median None
            assert sample_metrics.conversion_intercept is not None
            assert sample_metrics.conversion_slope is not None
            assert sample_metrics.conversion_r_squared is not None
            assert sample_metrics.conversion_median is None

    def test_regression_r_squared_valid(
        self, sample_metrics: TrendMetrics
    ) -> None:
        if not sample_metrics.conversion_is_fallback:
            assert sample_metrics.conversion_r_squared is not None
            assert 0.0 <= sample_metrics.conversion_r_squared <= 1.0

    def test_fallback_median_is_finite(
        self, sample_metrics: TrendMetrics
    ) -> None:
        if sample_metrics.conversion_is_fallback:
            assert sample_metrics.conversion_median is not None
            assert math.isfinite(sample_metrics.conversion_median)


class TestRoicMetrics:
    """Tests for ROIC computation."""

    def test_roic_latest_is_finite(
        self, sample_metrics: TrendMetrics
    ) -> None:
        assert math.isfinite(sample_metrics.roic_latest)

    def test_roic_slope_is_finite(
        self, sample_metrics: TrendMetrics
    ) -> None:
        assert math.isfinite(sample_metrics.roic_slope)

    def test_roic_detrended_std_is_finite(
        self, sample_metrics: TrendMetrics
    ) -> None:
        assert math.isfinite(sample_metrics.roic_detrended_std)

    def test_roic_minimum_is_finite(
        self, sample_metrics: TrendMetrics
    ) -> None:
        assert math.isfinite(sample_metrics.roic_minimum)

    def test_roic_latest_is_positive(
        self, sample_metrics: TrendMetrics
    ) -> None:
        # ROIC excludes quarters with income_before_tax <= 0,
        # so latest must be positive
        assert sample_metrics.roic_latest > 0

    def test_roic_minimum_le_latest(
        self, sample_metrics: TrendMetrics
    ) -> None:
        assert sample_metrics.roic_minimum <= sample_metrics.roic_latest


class TestInsufficientData:
    """Tests for graceful handling of insufficient data."""

    def test_empty_financials_returns_none(
        self, trends_config: TrendsConfig
    ) -> None:
        import pandas as pd

        company = CompanyData(
            symbol="EMPTY",
            company_name="Empty Corp",
            sector="Test",
            exchange="TEST",
            financials=pd.DataFrame(),
            latest_price=0.0,
            market_cap=0.0,
            shares_outstanding=0.0,
            price_history=pd.DataFrame(),
        )
        result = compute_trends(company, trends_config)
        assert result is None

    def test_single_quarter_returns_none(
        self, trends_config: TrendsConfig
    ) -> None:
        import pandas as pd

        company = CompanyData(
            symbol="SINGLE",
            company_name="Single Quarter Corp",
            sector="Test",
            exchange="TEST",
            financials=pd.DataFrame(
                {
                    "date": ["2024-03-31"],
                    "revenue": [1_000_000],
                    "operating_income": [200_000],
                    "free_cash_flow": [150_000],
                    "income_before_tax": [180_000],
                    "total_assets": [5_000_000],
                    "total_current_liabilities": [1_000_000],
                }
            ),
            latest_price=10.0,
            market_cap=1_000_000.0,
            shares_outstanding=100_000.0,
            price_history=pd.DataFrame(),
        )
        result = compute_trends(company, trends_config)
        assert result is None
