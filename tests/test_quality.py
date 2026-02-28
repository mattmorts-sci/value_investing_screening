"""Tests for quality metrics module against real FMP database data."""

from __future__ import annotations

import math
import sqlite3

import pytest

from pipeline.config import PipelineConfig
from pipeline.data import load_universe
from pipeline.data import fmp as fmp_module
from pipeline.data.models import CompanyData
from pipeline.metrics.quality import QualityMetrics, compute_quality

DB_PATH = "/home/mattm/projects/Pers/financial_db/data/fmp.db"

_FSCORE_COMPONENTS = [
    "f_roa_positive",
    "f_ocf_positive",
    "f_roa_improving",
    "f_accruals_negative",
    "f_leverage_decreasing",
    "f_current_ratio_improving",
    "f_no_dilution",
    "f_gross_margin_improving",
    "f_asset_turnover_improving",
]


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
    """Load a real company with at least 4 quarters of data."""
    universe = load_universe(db_config)
    if not universe:
        pytest.skip("Universe is empty")

    for entity_id, _ in universe[:20]:
        company = fmp_module.load_company(entity_id, db_config)
        if company is not None and len(company.financials) >= 4:
            return company

    pytest.skip("No company with at least 4 quarters of data found")


@pytest.fixture
def company_with_8q(db_config: PipelineConfig) -> CompanyData:
    """Load a real company with at least 8 quarters of data."""
    universe = load_universe(db_config)
    if not universe:
        pytest.skip("Universe is empty")

    for entity_id, _ in universe[:50]:
        company = fmp_module.load_company(entity_id, db_config)
        if company is not None and len(company.financials) >= 8:
            return company

    pytest.skip("No company with at least 8 quarters of data found")


class TestComputeQuality:
    """Tests for compute_quality against real database data."""

    def test_returns_quality_metrics_dataclass(
        self, sample_company: CompanyData
    ) -> None:
        """compute_quality returns a QualityMetrics instance."""
        result = compute_quality(sample_company)
        assert isinstance(result, QualityMetrics)

    def test_f_score_is_int_between_0_and_9(
        self, sample_company: CompanyData
    ) -> None:
        """f_score is an integer in the range [0, 9]."""
        result = compute_quality(sample_company)
        assert isinstance(result.f_score, int)
        assert 0 <= result.f_score <= 9

    def test_all_components_are_bool(
        self, sample_company: CompanyData
    ) -> None:
        """All 9 F-Score component fields are boolean."""
        result = compute_quality(sample_company)
        for field_name in _FSCORE_COMPONENTS:
            value = getattr(result, field_name)
            assert isinstance(value, bool), (
                f"{field_name} is {type(value).__name__}, expected bool"
            )

    def test_f_score_equals_sum_of_components(
        self, sample_company: CompanyData
    ) -> None:
        """f_score equals the sum of the 9 component booleans."""
        result = compute_quality(sample_company)
        component_sum = sum(
            getattr(result, name) for name in _FSCORE_COMPONENTS
        )
        assert result.f_score == component_sum

    def test_gross_profitability_type(
        self, sample_company: CompanyData
    ) -> None:
        """gross_profitability is a float or None."""
        result = compute_quality(sample_company)
        assert result.gross_profitability is None or isinstance(
            result.gross_profitability, float
        )

    def test_accruals_ratio_type(
        self, sample_company: CompanyData
    ) -> None:
        """accruals_ratio is a float or None."""
        result = compute_quality(sample_company)
        assert result.accruals_ratio is None or isinstance(
            result.accruals_ratio, float
        )

    def test_metrics_finite_when_total_assets_positive(
        self, sample_company: CompanyData
    ) -> None:
        """When total_assets > 0, gross_profitability and accruals_ratio are finite."""
        latest_ta = float(sample_company.financials["total_assets"].iloc[-1])
        if math.isnan(latest_ta) or latest_ta <= 0:
            pytest.skip("Latest total_assets is zero or NaN")

        result = compute_quality(sample_company)
        assert result.gross_profitability is not None
        assert math.isfinite(result.gross_profitability)
        assert result.accruals_ratio is not None
        assert math.isfinite(result.accruals_ratio)

    def test_yoy_signals_not_all_false_with_8_quarters(
        self, company_with_8q: CompanyData
    ) -> None:
        """With >= 8 quarters, at least some YoY comparison signals could be True.

        Not all 5 YoY signals should default to False when sufficient data
        exists. At least one should have a non-default value (True), or at
        minimum the signals should reflect real comparison rather than all
        defaulting. We verify at least one YoY signal is True, which is
        extremely likely for any real company over a year-on-year period.
        """
        result = compute_quality(company_with_8q)
        yoy_signals = [
            result.f_roa_improving,
            result.f_leverage_decreasing,
            result.f_current_ratio_improving,
            result.f_no_dilution,
            result.f_gross_margin_improving,
            result.f_asset_turnover_improving,
        ]
        # With real data over 8+ quarters, it is extremely unlikely all
        # YoY signals are False. If they are, the comparison logic may
        # be defaulting rather than computing.
        assert any(yoy_signals), (
            f"All YoY signals are False for {company_with_8q.symbol} "
            f"with {len(company_with_8q.financials)} quarters — "
            f"comparison logic may be defaulting"
        )
