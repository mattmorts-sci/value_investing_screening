"""Tests for valuation metrics module against real FMP database data."""

from __future__ import annotations

import math
import sqlite3

import pytest

from pipeline.config import PipelineConfig
from pipeline.data import load_universe
from pipeline.data import fmp as fmp_module
from pipeline.data.models import CompanyData
from pipeline.metrics.valuation import ValuationMetrics, compute_valuation

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
    """Load a real company from the database (no live price fetch)."""
    universe = load_universe(db_config)
    if not universe:
        pytest.skip("Universe is empty")

    # Find a company with enough data
    for entity_id, _ in universe[:20]:
        company = fmp_module.load_company(entity_id, db_config)
        if company is not None and len(company.financials) >= 4:
            return company

    pytest.skip("No company with at least 4 quarters of data found")


class TestComputeValuation:
    """Tests for compute_valuation against real database data."""

    def test_returns_valuation_metrics_dataclass(
        self, sample_company: CompanyData, db_config: PipelineConfig
    ) -> None:
        """compute_valuation returns a ValuationMetrics instance."""
        result = compute_valuation(sample_company, db_config)
        assert isinstance(result, ValuationMetrics)

    def test_ev_is_positive_for_normal_company(
        self, sample_company: CompanyData, db_config: PipelineConfig
    ) -> None:
        """A normal company with market cap should have positive EV."""
        result = compute_valuation(sample_company, db_config)
        assert result.enterprise_value > 0

    def test_ttm_fcf_is_computed(
        self, sample_company: CompanyData, db_config: PipelineConfig
    ) -> None:
        """TTM FCF should be a finite number, not NaN."""
        result = compute_valuation(sample_company, db_config)
        assert math.isfinite(result.ttm_fcf)

    def test_ttm_ebit_is_computed(
        self, sample_company: CompanyData, db_config: PipelineConfig
    ) -> None:
        """TTM EBIT should be a finite number, not NaN."""
        result = compute_valuation(sample_company, db_config)
        assert math.isfinite(result.ttm_ebit)

    def test_yield_ratios_are_floats_when_ev_positive(
        self, sample_company: CompanyData, db_config: PipelineConfig
    ) -> None:
        """FCF/EV and EBIT/EV should be floats when EV > 0."""
        result = compute_valuation(sample_company, db_config)
        if result.enterprise_value > 0:
            assert isinstance(result.fcf_ev, float)
            assert isinstance(result.ebit_ev, float)
            assert math.isfinite(result.fcf_ev)
            assert math.isfinite(result.ebit_ev)

    def test_composite_equals_weighted_sum(
        self, sample_company: CompanyData, db_config: PipelineConfig
    ) -> None:
        """Composite yield must equal weighted sum of FCF/EV and EBIT/EV."""
        result = compute_valuation(sample_company, db_config)
        if result.fcf_ev is not None and result.ebit_ev is not None:
            expected = (
                result.fcf_ev * db_config.fcf_ev_weight
                + result.ebit_ev * db_config.ebit_ev_weight
            )
            assert result.composite_yield is not None
            assert abs(result.composite_yield - expected) < 1e-12

    def test_ev_formula_correct(
        self, sample_company: CompanyData, db_config: PipelineConfig
    ) -> None:
        """EV = market_cap + total_debt - cash_and_cash_equivalents."""
        result = compute_valuation(sample_company, db_config)
        latest = sample_company.financials.iloc[-1]
        total_debt = float(latest["total_debt"]) if not math.isnan(float(latest["total_debt"])) else 0.0
        cash = float(latest["cash_and_cash_equivalents"]) if not math.isnan(float(latest["cash_and_cash_equivalents"])) else 0.0
        expected_ev = sample_company.market_cap + total_debt - cash
        assert abs(result.enterprise_value - expected_ev) < 0.01

    def test_custom_weights(
        self, sample_company: CompanyData, db_config: PipelineConfig
    ) -> None:
        """Composite yield respects custom weight configuration."""
        custom_config = PipelineConfig(
            exchanges=db_config.exchanges,
            fcf_ev_weight=0.8,
            ebit_ev_weight=0.2,
        )
        result = compute_valuation(sample_company, custom_config)
        if result.fcf_ev is not None and result.ebit_ev is not None:
            expected = result.fcf_ev * 0.8 + result.ebit_ev * 0.2
            assert result.composite_yield is not None
            assert abs(result.composite_yield - expected) < 1e-12

    def test_ttm_uses_last_four_quarters(
        self, sample_company: CompanyData, db_config: PipelineConfig
    ) -> None:
        """TTM FCF and EBIT should match the sum of the last 4 rows."""
        result = compute_valuation(sample_company, db_config)
        tail4 = sample_company.financials.tail(4)
        expected_fcf = tail4["free_cash_flow"].sum(skipna=True)
        expected_ebit = tail4["ebit"].sum(skipna=True)
        assert abs(result.ttm_fcf - expected_fcf) < 0.01
        assert abs(result.ttm_ebit - expected_ebit) < 0.01
