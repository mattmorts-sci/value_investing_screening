"""Tests for safety metrics module."""

from __future__ import annotations

import math
import sqlite3

import pandas as pd
import pytest

from pipeline.config import PipelineConfig
from pipeline.data import load_universe
from pipeline.data import fmp as fmp_module
from pipeline.data.models import CompanyData
from pipeline.metrics.safety import SafetyMetrics, compute_safety

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
    """Load a real company from the database for testing."""
    universe = load_universe(db_config)
    if not universe:
        pytest.skip("Universe is empty")
    for entity_id, _ in universe:
        company = fmp_module.load_company(entity_id, db_config)
        if company is not None and len(company.financials) >= 4:
            return company
    pytest.skip("No company with at least 4 quarters of data found")


class TestComputeSafety:
    """Tests for compute_safety against real database data."""

    def test_returns_safety_metrics(self, sample_company: CompanyData) -> None:
        """compute_safety returns a SafetyMetrics dataclass."""
        result = compute_safety(sample_company)
        assert isinstance(result, SafetyMetrics)

    def test_interest_coverage_type(self, sample_company: CompanyData) -> None:
        """Interest coverage is float or None."""
        result = compute_safety(sample_company)
        assert result.interest_coverage is None or isinstance(
            result.interest_coverage, float
        )

    def test_ocf_to_debt_type(self, sample_company: CompanyData) -> None:
        """OCF to debt is float or None."""
        result = compute_safety(sample_company)
        assert result.ocf_to_debt is None or isinstance(result.ocf_to_debt, float)

    def test_interest_coverage_finite_when_present(
        self, sample_company: CompanyData
    ) -> None:
        """When interest_expense is positive, interest_coverage is finite."""
        ttm = sample_company.financials.tail(4)
        ttm_interest = ttm["interest_expense"].sum(skipna=True)

        result = compute_safety(sample_company)

        if pd.notna(ttm_interest) and ttm_interest > 0:
            assert result.interest_coverage is not None
            assert math.isfinite(result.interest_coverage)

    def test_ocf_to_debt_finite_when_present(
        self, sample_company: CompanyData
    ) -> None:
        """When total_debt is non-zero, ocf_to_debt is finite."""
        latest_debt = sample_company.financials.iloc[-1]["total_debt"]

        result = compute_safety(sample_company)

        if pd.notna(latest_debt) and latest_debt != 0:
            assert result.ocf_to_debt is not None
            assert math.isfinite(result.ocf_to_debt)

    def test_ttm_calculation(self, sample_company: CompanyData) -> None:
        """Values match manual TTM calculation from raw data."""
        financials = sample_company.financials
        ttm = financials.tail(4)
        result = compute_safety(sample_company)

        # Manual interest coverage
        ttm_ebit = ttm["ebit"].fillna(0).sum()
        ttm_interest = ttm["interest_expense"].fillna(0).sum()

        if ttm_interest != 0:
            expected_ic = ttm_ebit / ttm_interest
            assert result.interest_coverage is not None
            assert abs(result.interest_coverage - expected_ic) < 1e-9
        else:
            assert result.interest_coverage is None

        # Manual OCF to debt
        ttm_ocf = ttm["operating_cash_flow"].fillna(0).sum()
        latest_debt_raw = financials.iloc[-1]["total_debt"]
        latest_debt = float(latest_debt_raw) if pd.notna(latest_debt_raw) else 0.0

        if latest_debt != 0:
            expected_ocf = ttm_ocf / latest_debt
            assert result.ocf_to_debt is not None
            assert abs(result.ocf_to_debt - expected_ocf) < 1e-9
        else:
            assert result.ocf_to_debt is None


class TestEdgeCases:
    """Tests for edge cases using synthetic data."""

    @staticmethod
    def _make_company(financials_data: dict[str, list[float | None]]) -> CompanyData:
        """Build a CompanyData with synthetic financials."""
        n = len(next(iter(financials_data.values())))
        base = {
            "date": [f"2024-0{i+1}-01" for i in range(n)],
            "fiscal_year": [2024] * n,
            "period": [f"Q{i+1}" for i in range(n)],
            "revenue": [100.0] * n,
            "gross_profit": [50.0] * n,
            "operating_income": [30.0] * n,
            "net_income": [20.0] * n,
            "income_before_tax": [25.0] * n,
            "income_tax_expense": [5.0] * n,
            "weighted_average_shs_out_dil": [1000.0] * n,
            "total_assets": [500.0] * n,
            "total_current_assets": [200.0] * n,
            "total_current_liabilities": [100.0] * n,
            "long_term_debt": [50.0] * n,
            "cash_and_cash_equivalents": [30.0] * n,
            "free_cash_flow": [15.0] * n,
        }
        # Defaults for columns that may be overridden
        base.setdefault("ebit", [30.0] * n)
        base.setdefault("interest_expense", [5.0] * n)
        base.setdefault("operating_cash_flow", [25.0] * n)
        base.setdefault("total_debt", [100.0] * n)

        base.update(financials_data)
        return CompanyData(
            symbol="TEST",
            company_name="Test Co",
            sector="Test",
            exchange="TEST",
            financials=pd.DataFrame(base),
            latest_price=10.0,
            market_cap=10000.0,
            shares_outstanding=1000.0,
            price_history=pd.DataFrame({"date": [], "close": []}),
        )

    def test_zero_interest_expense_returns_none(self) -> None:
        """Zero interest expense produces None for interest_coverage."""
        company = self._make_company(
            {
                "ebit": [30.0, 30.0, 30.0, 30.0],
                "interest_expense": [0.0, 0.0, 0.0, 0.0],
            }
        )
        result = compute_safety(company)
        assert result.interest_coverage is None

    def test_zero_total_debt_returns_none(self) -> None:
        """Zero total debt produces None for ocf_to_debt."""
        company = self._make_company(
            {
                "operating_cash_flow": [25.0, 25.0, 25.0, 25.0],
                "total_debt": [0.0, 0.0, 0.0, 0.0],
            }
        )
        result = compute_safety(company)
        assert result.ocf_to_debt is None

    def test_nan_interest_expense_returns_none(self) -> None:
        """NaN interest expense sums to zero, producing None."""
        company = self._make_company(
            {
                "ebit": [30.0, 30.0, 30.0, 30.0],
                "interest_expense": [None, None, None, None],
            }
        )
        result = compute_safety(company)
        assert result.interest_coverage is None

    def test_nan_total_debt_returns_none(self) -> None:
        """NaN total debt produces None for ocf_to_debt."""
        company = self._make_company(
            {
                "operating_cash_flow": [25.0, 25.0, 25.0, 25.0],
                "total_debt": [None, None, None, None],
            }
        )
        result = compute_safety(company)
        assert result.ocf_to_debt is None

    def test_negative_interest_expense_returns_none(self) -> None:
        """Negative interest expense (net interest income) produces None."""
        company = self._make_company(
            {
                "ebit": [30.0, 30.0, 30.0, 30.0],
                "interest_expense": [-5.0, -5.0, -5.0, -5.0],
            }
        )
        result = compute_safety(company)
        assert result.interest_coverage is None

    def test_negative_ebit_still_computes(self) -> None:
        """Negative EBIT produces a negative interest coverage (valid)."""
        company = self._make_company(
            {
                "ebit": [-10.0, -10.0, -10.0, -10.0],
                "interest_expense": [5.0, 5.0, 5.0, 5.0],
            }
        )
        result = compute_safety(company)
        assert result.interest_coverage is not None
        assert result.interest_coverage < 0

    def test_negative_ocf_still_computes(self) -> None:
        """Negative OCF produces a negative ocf_to_debt (valid)."""
        company = self._make_company(
            {
                "operating_cash_flow": [-10.0, -10.0, -10.0, -10.0],
                "total_debt": [100.0, 100.0, 100.0, 100.0],
            }
        )
        result = compute_safety(company)
        assert result.ocf_to_debt is not None
        assert result.ocf_to_debt < 0

    def test_fewer_than_four_quarters(self) -> None:
        """With only 2 quarters, uses what's available."""
        company = self._make_company(
            {
                "ebit": [30.0, 30.0],
                "interest_expense": [5.0, 5.0],
                "operating_cash_flow": [25.0, 25.0],
                "total_debt": [100.0, 100.0],
            }
        )
        result = compute_safety(company)
        # 60 / 10 = 6.0
        assert result.interest_coverage is not None
        assert abs(result.interest_coverage - 6.0) < 1e-9
        # 50 / 100 = 0.5
        assert result.ocf_to_debt is not None
        assert abs(result.ocf_to_debt - 0.5) < 1e-9

    def test_exact_values_four_quarters(self) -> None:
        """Verify exact TTM arithmetic with known values."""
        company = self._make_company(
            {
                "ebit": [10.0, 20.0, 30.0, 40.0],
                "interest_expense": [2.0, 3.0, 4.0, 1.0],
                "operating_cash_flow": [15.0, 25.0, 35.0, 45.0],
                "total_debt": [50.0, 60.0, 70.0, 200.0],
            }
        )
        result = compute_safety(company)
        # TTM EBIT = 100, TTM interest = 10 → coverage = 10.0
        assert result.interest_coverage is not None
        assert abs(result.interest_coverage - 10.0) < 1e-9
        # TTM OCF = 120, latest debt = 200 → ratio = 0.6
        assert result.ocf_to_debt is not None
        assert abs(result.ocf_to_debt - 0.6) < 1e-9

    def test_partial_nan_in_components(self) -> None:
        """NaN in some quarters is treated as zero in the sum."""
        company = self._make_company(
            {
                "ebit": [10.0, None, 30.0, 40.0],
                "interest_expense": [5.0, None, 5.0, 5.0],
                "operating_cash_flow": [25.0, None, 25.0, 25.0],
                "total_debt": [100.0, 100.0, 100.0, 100.0],
            }
        )
        result = compute_safety(company)
        # TTM EBIT = 80, TTM interest = 15 → coverage = 80/15
        assert result.interest_coverage is not None
        assert abs(result.interest_coverage - (80.0 / 15.0)) < 1e-9
        # TTM OCF = 75, latest debt = 100 → ratio = 0.75
        assert result.ocf_to_debt is not None
        assert abs(result.ocf_to_debt - 0.75) < 1e-9
