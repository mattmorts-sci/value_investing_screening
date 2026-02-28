"""Tests for screening, CSV export, and CLI modules."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.config import PipelineConfig, SortOption
from pipeline.data.models import CompanyData
from pipeline.metrics.quality import QualityMetrics
from pipeline.metrics.safety import SafetyMetrics
from pipeline.metrics.sentiment import SentimentMetrics
from pipeline.metrics.trends import TrendMetrics
from pipeline.metrics.valuation import ValuationMetrics
from pipeline.output.csv_export import COLUMNS, _format_value, export_csv, round_iv
from pipeline.screening import (
    CompanyAnalysis,
    apply_filters,
    compute_mos,
    screen_companies,
)
from pipeline.simulation import PathData, SimulationOutput


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_company(
    symbol: str = "TEST",
    market_cap: float = 1_000_000_000.0,
    latest_price: float = 50.0,
    n_quarters: int = 12,
) -> CompanyData:
    """Create a minimal CompanyData for testing."""
    financials = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n_quarters, freq="QS"),
            "fiscal_year": [(2020 + i // 4) for i in range(n_quarters)],
            "period": [f"Q{(i % 4) + 1}" for i in range(n_quarters)],
            "revenue": [100.0] * n_quarters,
            "gross_profit": [50.0] * n_quarters,
            "operating_income": [30.0] * n_quarters,
            "ebit": [30.0] * n_quarters,
            "net_income": [20.0] * n_quarters,
            "income_before_tax": [25.0] * n_quarters,
            "income_tax_expense": [5.0] * n_quarters,
            "interest_expense": [2.0] * n_quarters,
            "weighted_average_shs_out_dil": [1000.0] * n_quarters,
            "total_assets": [500.0] * n_quarters,
            "total_current_assets": [200.0] * n_quarters,
            "total_current_liabilities": [100.0] * n_quarters,
            "total_debt": [150.0] * n_quarters,
            "long_term_debt": [100.0] * n_quarters,
            "cash_and_cash_equivalents": [50.0] * n_quarters,
            "operating_cash_flow": [25.0] * n_quarters,
            "free_cash_flow": [20.0] * n_quarters,
        }
    )
    return CompanyData(
        symbol=symbol,
        company_name=f"{symbol} Corp",
        sector="Technology",
        exchange="ASX",
        financials=financials,
        latest_price=latest_price,
        market_cap=market_cap,
        shares_outstanding=market_cap / latest_price if latest_price != 0 else 1.0,
        price_history=pd.DataFrame({"date": [], "close": []}),
    )


def _make_valuation(
    fcf_ev: float | None = 0.08,
    ebit_ev: float | None = 0.06,
    composite_yield: float | None = 0.072,
    ttm_fcf: float = 80.0,
) -> ValuationMetrics:
    return ValuationMetrics(
        enterprise_value=1_000_000.0,
        ttm_fcf=ttm_fcf,
        ttm_ebit=60.0,
        fcf_ev=fcf_ev,
        ebit_ev=ebit_ev,
        composite_yield=composite_yield,
    )


def _make_trends(
    conversion_r_squared: float | None = 0.85,
) -> TrendMetrics:
    return TrendMetrics(
        revenue_cagr=0.10,
        revenue_yoy_growth_std=0.05,
        revenue_qoq_growth_mean=0.02,
        revenue_qoq_growth_var=0.001,
        margin_intercept=0.15,
        margin_slope=0.001,
        margin_r_squared=0.80,
        conversion_intercept=0.70 if conversion_r_squared is not None else None,
        conversion_slope=0.005 if conversion_r_squared is not None else None,
        conversion_r_squared=conversion_r_squared,
        conversion_median=0.65 if conversion_r_squared is None else None,
        conversion_is_fallback=conversion_r_squared is None,
        roic_latest=0.15,
        roic_slope=0.002,
        roic_detrended_std=0.03,
        roic_minimum=0.08,
    )


def _make_safety(
    interest_coverage: float | None = 10.0,
    ocf_to_debt: float | None = 0.5,
) -> SafetyMetrics:
    return SafetyMetrics(
        interest_coverage=interest_coverage,
        ocf_to_debt=ocf_to_debt,
    )


def _make_quality() -> QualityMetrics:
    return QualityMetrics(
        f_roa_positive=True,
        f_ocf_positive=True,
        f_roa_improving=True,
        f_accruals_negative=True,
        f_leverage_decreasing=False,
        f_current_ratio_improving=True,
        f_no_dilution=True,
        f_gross_margin_improving=True,
        f_asset_turnover_improving=False,
        f_score=7,
        gross_profitability=0.30,
        accruals_ratio=-0.05,
    )


def _make_sentiment() -> SentimentMetrics:
    return SentimentMetrics(return_6m=0.10, return_12m=0.20)


def _make_simulation(
    iv_p25: float = 40.0,
    iv_p50: float = 55.0,
    iv_p75: float = 70.0,
) -> SimulationOutput:
    return SimulationOutput(
        iv_p10=30.0,
        iv_p25=iv_p25,
        iv_p50=iv_p50,
        iv_p75=iv_p75,
        iv_p90=85.0,
        iv_spread=iv_p75 - iv_p25,
        implied_cagr_p25=0.03,
        implied_cagr_p50=0.06,
        implied_cagr_p75=0.09,
        sample_paths=[
            PathData(
                quarterly_revenue=np.array([100.0, 102.0]),
                quarterly_fcf=np.array([20.0, 20.4]),
            )
        ],
        revenue_bands=None,
        fcf_bands=None,
    )


def _make_analysis(
    symbol: str = "TEST",
    market_cap: float = 1_000_000_000.0,
    latest_price: float = 50.0,
    n_quarters: int = 12,
    fcf_ev: float | None = 0.08,
    ebit_ev: float | None = 0.06,
    composite_yield: float | None = 0.072,
    ttm_fcf: float = 80.0,
    interest_coverage: float | None = 10.0,
    ocf_to_debt: float | None = 0.5,
    conversion_r_squared: float | None = 0.85,
    iv_p25: float = 40.0,
    iv_p50: float = 55.0,
    iv_p75: float = 70.0,
    trends: TrendMetrics | None = ...,  # type: ignore[assignment]
    simulation: SimulationOutput | None = ...,  # type: ignore[assignment]
) -> CompanyAnalysis:
    # Use sentinel ... to distinguish "not provided" from explicit None
    if trends is ...:
        trends = _make_trends(conversion_r_squared)
    if simulation is ...:
        simulation = _make_simulation(iv_p25, iv_p50, iv_p75)

    return CompanyAnalysis(
        company=_make_company(symbol, market_cap, latest_price, n_quarters),
        valuation=_make_valuation(fcf_ev, ebit_ev, composite_yield, ttm_fcf),
        trends=trends,
        safety=_make_safety(interest_coverage, ocf_to_debt),
        quality=_make_quality(),
        sentiment=_make_sentiment(),
        simulation=simulation,
    )


# ---------------------------------------------------------------------------
# MoS tests
# ---------------------------------------------------------------------------


class TestComputeMos:
    def test_positive_mos(self) -> None:
        assert compute_mos(100.0, 80.0) == pytest.approx(0.20)

    def test_negative_mos(self) -> None:
        assert compute_mos(80.0, 100.0) == pytest.approx(-0.25)

    def test_zero_iv_returns_none(self) -> None:
        assert compute_mos(0.0, 50.0) is None

    def test_negative_iv_returns_none(self) -> None:
        assert compute_mos(-10.0, 50.0) is None

    def test_price_equals_iv(self) -> None:
        assert compute_mos(50.0, 50.0) == pytest.approx(0.0)

    def test_zero_price(self) -> None:
        # Zero price with positive IV gives MoS of 1.0 (100% upside)
        assert compute_mos(100.0, 0.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------


class TestApplyFilters:
    def test_no_filters_enabled(self) -> None:
        analysis = _make_analysis()
        config = PipelineConfig(exchanges=["ASX"])
        config.filter_exclude_negative_ttm_fcf = False
        filtered, reasons = apply_filters(analysis, config)
        assert not filtered
        assert reasons == ""

    def test_market_cap_filter(self) -> None:
        analysis = _make_analysis(market_cap=500_000.0)
        config = PipelineConfig(
            exchanges=["ASX"],
            filter_min_market_cap=1_000_000.0,
            filter_exclude_negative_ttm_fcf=False,
        )
        filtered, reasons = apply_filters(analysis, config)
        assert filtered
        assert reasons == "MC"

    def test_interest_coverage_filter(self) -> None:
        analysis = _make_analysis(interest_coverage=1.2)
        config = PipelineConfig(
            exchanges=["ASX"],
            filter_min_interest_coverage=2.0,
            filter_exclude_negative_ttm_fcf=False,
        )
        filtered, reasons = apply_filters(analysis, config)
        assert filtered
        assert reasons == "IC"

    def test_interest_coverage_none_triggers_filter(self) -> None:
        analysis = _make_analysis(interest_coverage=None)
        config = PipelineConfig(
            exchanges=["ASX"],
            filter_min_interest_coverage=2.0,
            filter_exclude_negative_ttm_fcf=False,
        )
        filtered, reasons = apply_filters(analysis, config)
        assert filtered
        assert reasons == "IC"

    def test_ocf_to_debt_filter(self) -> None:
        analysis = _make_analysis(ocf_to_debt=0.1)
        config = PipelineConfig(
            exchanges=["ASX"],
            filter_min_ocf_to_debt=0.2,
            filter_exclude_negative_ttm_fcf=False,
        )
        filtered, reasons = apply_filters(analysis, config)
        assert filtered
        assert reasons == "OD"

    def test_conversion_r2_filter(self) -> None:
        analysis = _make_analysis(conversion_r_squared=0.3)
        config = PipelineConfig(
            exchanges=["ASX"],
            filter_min_fcf_conversion_r2=0.5,
            filter_exclude_negative_ttm_fcf=False,
        )
        filtered, reasons = apply_filters(analysis, config)
        assert filtered
        assert reasons == "CR"

    def test_conversion_r2_none_triggers_filter(self) -> None:
        analysis = _make_analysis(conversion_r_squared=None)
        config = PipelineConfig(
            exchanges=["ASX"],
            filter_min_fcf_conversion_r2=0.5,
            filter_exclude_negative_ttm_fcf=False,
        )
        filtered, reasons = apply_filters(analysis, config)
        assert filtered
        assert reasons == "CR"

    def test_negative_fcf_filter(self) -> None:
        analysis = _make_analysis(ttm_fcf=-10.0)
        config = PipelineConfig(
            exchanges=["ASX"],
            filter_exclude_negative_ttm_fcf=True,
        )
        filtered, reasons = apply_filters(analysis, config)
        assert filtered
        assert reasons == "FCF"

    def test_negative_fcf_filter_disabled(self) -> None:
        analysis = _make_analysis(ttm_fcf=-10.0)
        config = PipelineConfig(
            exchanges=["ASX"],
            filter_exclude_negative_ttm_fcf=False,
        )
        filtered, reasons = apply_filters(analysis, config)
        assert not filtered

    def test_multiple_filters(self) -> None:
        analysis = _make_analysis(
            market_cap=100.0,
            interest_coverage=0.5,
            ttm_fcf=-5.0,
        )
        config = PipelineConfig(
            exchanges=["ASX"],
            filter_min_market_cap=1000.0,
            filter_min_interest_coverage=2.0,
            filter_exclude_negative_ttm_fcf=True,
        )
        filtered, reasons = apply_filters(analysis, config)
        assert filtered
        reason_set = set(reasons.split(","))
        assert reason_set == {"MC", "IC", "FCF"}

    def test_passes_all_filters(self) -> None:
        analysis = _make_analysis(
            market_cap=2_000_000.0,
            interest_coverage=5.0,
            ocf_to_debt=0.5,
            conversion_r_squared=0.9,
            ttm_fcf=100.0,
        )
        config = PipelineConfig(
            exchanges=["ASX"],
            filter_min_market_cap=1_000_000.0,
            filter_min_interest_coverage=2.0,
            filter_min_ocf_to_debt=0.2,
            filter_min_fcf_conversion_r2=0.5,
            filter_exclude_negative_ttm_fcf=True,
        )
        filtered, reasons = apply_filters(analysis, config)
        assert not filtered
        assert reasons == ""


# ---------------------------------------------------------------------------
# Data sufficiency tests
# ---------------------------------------------------------------------------


class TestDataSufficiency:
    def test_insufficient_quarters_filtered_as_data(self) -> None:
        """Company with < min_quarters gets DATA filter reason."""
        analysis = _make_analysis(n_quarters=8)
        config = PipelineConfig(
            exchanges=["ASX"],
            min_quarters=12,
            filter_exclude_negative_ttm_fcf=False,
        )
        filtered, reasons = apply_filters(analysis, config)
        assert filtered
        assert reasons == "DATA"

    def test_none_trends_filtered_as_data(self) -> None:
        """Company with None trends gets DATA filter reason."""
        analysis = _make_analysis(trends=None)
        config = PipelineConfig(
            exchanges=["ASX"], filter_exclude_negative_ttm_fcf=False
        )
        filtered, reasons = apply_filters(analysis, config)
        assert filtered
        assert reasons == "DATA"

    def test_none_simulation_filtered_as_data(self) -> None:
        """Company with None simulation gets DATA filter reason."""
        analysis = _make_analysis(simulation=None)
        config = PipelineConfig(
            exchanges=["ASX"], filter_exclude_negative_ttm_fcf=False
        )
        filtered, reasons = apply_filters(analysis, config)
        assert filtered
        assert reasons == "DATA"

    def test_data_filtered_skips_stage2_filters(self) -> None:
        """DATA companies don't get additional Stage 2 reasons."""
        analysis = _make_analysis(
            n_quarters=4,
            market_cap=100.0,  # Would trigger MC filter
            trends=None,
        )
        config = PipelineConfig(
            exchanges=["ASX"],
            min_quarters=12,
            filter_min_market_cap=1_000_000.0,
            filter_exclude_negative_ttm_fcf=True,
        )
        filtered, reasons = apply_filters(analysis, config)
        assert filtered
        assert reasons == "DATA"  # Only DATA, not MC

    def test_data_insufficient_company_in_csv(self, tmp_path: Path) -> None:
        """Data-insufficient companies appear in CSV as filtered."""
        sufficient = _make_analysis(symbol="GOOD", n_quarters=12)
        insufficient = _make_analysis(symbol="SHORT", n_quarters=4, trends=None, simulation=None)
        config = PipelineConfig(
            exchanges=["ASX"],
            min_quarters=12,
            filter_exclude_negative_ttm_fcf=False,
        )
        results = screen_companies([sufficient, insufficient], config)

        output = tmp_path / "test.csv"
        export_csv(results, output)

        with open(output) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        short_row = next(r for r in rows if r["symbol"] == "SHORT")
        assert short_row["filtered"] == "TRUE"
        assert short_row["filter_reasons"] == "DATA"
        # Trend/simulation columns should be empty
        assert short_row["revenue_cagr"] == ""
        assert short_row["iv_p50"] == ""
        assert short_row["mos_p50"] == ""


# ---------------------------------------------------------------------------
# Screen companies tests
# ---------------------------------------------------------------------------


class TestScreenCompanies:
    def test_sort_by_composite_descending(self) -> None:
        a1 = _make_analysis(symbol="LOW", composite_yield=0.05)
        a2 = _make_analysis(symbol="HIGH", composite_yield=0.10)
        config = PipelineConfig(
            exchanges=["ASX"],
            sort_by=SortOption.COMPOSITE,
            filter_exclude_negative_ttm_fcf=False,
        )
        results = screen_companies([a1, a2], config)
        assert results[0].analysis.company.symbol == "HIGH"
        assert results[1].analysis.company.symbol == "LOW"

    def test_sort_by_fcf_ev(self) -> None:
        a1 = _make_analysis(symbol="LOW", fcf_ev=0.03)
        a2 = _make_analysis(symbol="HIGH", fcf_ev=0.12)
        config = PipelineConfig(
            exchanges=["ASX"],
            sort_by=SortOption.FCF_EV,
            filter_exclude_negative_ttm_fcf=False,
        )
        results = screen_companies([a1, a2], config)
        assert results[0].analysis.company.symbol == "HIGH"

    def test_sort_by_ebit_ev(self) -> None:
        a1 = _make_analysis(symbol="LOW", ebit_ev=0.02)
        a2 = _make_analysis(symbol="HIGH", ebit_ev=0.10)
        config = PipelineConfig(
            exchanges=["ASX"],
            sort_by=SortOption.EBIT_EV,
            filter_exclude_negative_ttm_fcf=False,
        )
        results = screen_companies([a1, a2], config)
        assert results[0].analysis.company.symbol == "HIGH"

    def test_filtered_companies_sort_after_unfiltered(self) -> None:
        a_filtered = _make_analysis(
            symbol="FILTERED", composite_yield=0.20, market_cap=100.0
        )
        a_unfiltered = _make_analysis(
            symbol="PASS", composite_yield=0.05, market_cap=5_000_000.0
        )
        config = PipelineConfig(
            exchanges=["ASX"],
            filter_min_market_cap=1_000_000.0,
            filter_exclude_negative_ttm_fcf=False,
        )
        results = screen_companies([a_filtered, a_unfiltered], config)
        assert results[0].analysis.company.symbol == "PASS"
        assert results[1].analysis.company.symbol == "FILTERED"
        assert not results[0].filtered
        assert results[1].filtered

    def test_none_yield_sorts_last(self) -> None:
        a_none = _make_analysis(
            symbol="NONE", composite_yield=None, fcf_ev=None, ebit_ev=None
        )
        a_valid = _make_analysis(symbol="VALID", composite_yield=0.05)
        config = PipelineConfig(
            exchanges=["ASX"],
            filter_exclude_negative_ttm_fcf=False,
        )
        results = screen_companies([a_none, a_valid], config)
        assert results[0].analysis.company.symbol == "VALID"
        assert results[1].analysis.company.symbol == "NONE"

    def test_mos_computed_correctly(self) -> None:
        analysis = _make_analysis(latest_price=50.0, iv_p50=100.0)
        config = PipelineConfig(
            exchanges=["ASX"],
            filter_exclude_negative_ttm_fcf=False,
        )
        results = screen_companies([analysis], config)
        assert results[0].mos_p50 == pytest.approx(0.50)

    def test_mos_none_when_no_simulation(self) -> None:
        analysis = _make_analysis(simulation=None)
        config = PipelineConfig(
            exchanges=["ASX"],
            filter_exclude_negative_ttm_fcf=False,
        )
        results = screen_companies([analysis], config)
        assert results[0].mos_p25 is None
        assert results[0].mos_p50 is None
        assert results[0].mos_p75 is None

    def test_filtered_companies_retained_in_output(self) -> None:
        analyses = [
            _make_analysis(symbol="PASS", market_cap=2_000_000.0),
            _make_analysis(symbol="FAIL", market_cap=100.0),
        ]
        config = PipelineConfig(
            exchanges=["ASX"],
            filter_min_market_cap=1_000_000.0,
            filter_exclude_negative_ttm_fcf=False,
        )
        results = screen_companies(analyses, config)
        assert len(results) == 2
        symbols = [r.analysis.company.symbol for r in results]
        assert "PASS" in symbols
        assert "FAIL" in symbols

    def test_all_companies_filtered(self) -> None:
        """All companies filtered still appear in output, sorted by yield."""
        a1 = _make_analysis(symbol="A", market_cap=100.0, composite_yield=0.10)
        a2 = _make_analysis(symbol="B", market_cap=200.0, composite_yield=0.05)
        config = PipelineConfig(
            exchanges=["ASX"],
            filter_min_market_cap=1_000_000.0,
            filter_exclude_negative_ttm_fcf=False,
        )
        results = screen_companies([a1, a2], config)
        assert len(results) == 2
        assert all(r.filtered for r in results)
        # Still sorted by yield descending
        assert results[0].analysis.company.symbol == "A"
        assert results[1].analysis.company.symbol == "B"

    def test_empty_input(self) -> None:
        config = PipelineConfig(exchanges=["ASX"])
        results = screen_companies([], config)
        assert results == []


# ---------------------------------------------------------------------------
# IV rounding tests
# ---------------------------------------------------------------------------


class TestRoundIv:
    def test_round_to_dollar_above_10(self) -> None:
        assert round_iv(45.67, 50.0) == 46.0

    def test_round_to_dollar_at_10(self) -> None:
        assert round_iv(12.45, 10.0) == 12.0

    def test_round_to_ten_cents_below_10(self) -> None:
        assert round_iv(7.84, 5.0) == 7.8

    def test_round_to_ten_cents_small_price(self) -> None:
        assert round_iv(3.17, 2.50) == 3.2

    def test_negative_iv_rounded(self) -> None:
        assert round_iv(-5.67, 50.0) == -6.0


# ---------------------------------------------------------------------------
# _format_value tests
# ---------------------------------------------------------------------------


class TestFormatValue:
    def test_none_returns_empty(self) -> None:
        assert _format_value(None) == ""

    def test_true_returns_TRUE(self) -> None:
        assert _format_value(True) == "TRUE"

    def test_false_returns_FALSE(self) -> None:
        assert _format_value(False) == "FALSE"

    def test_nan_returns_empty(self) -> None:
        assert _format_value(float("nan")) == ""

    def test_inf_returns_empty(self) -> None:
        assert _format_value(float("inf")) == ""

    def test_negative_inf_returns_empty(self) -> None:
        assert _format_value(float("-inf")) == ""

    def test_normal_float(self) -> None:
        assert _format_value(3.14) == "3.14"

    def test_int(self) -> None:
        assert _format_value(42) == "42"

    def test_string(self) -> None:
        assert _format_value("hello") == "hello"

    def test_numpy_float64_nan(self) -> None:
        assert _format_value(np.float64("nan")) == ""

    def test_numpy_float64_inf(self) -> None:
        assert _format_value(np.float64("inf")) == ""

    def test_numpy_float64_normal(self) -> None:
        result = _format_value(np.float64(3.14))
        assert float(result) == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# CSV export tests
# ---------------------------------------------------------------------------


class TestCsvExport:
    def test_column_completeness(self, tmp_path: Path) -> None:
        """All expected columns present in output CSV."""
        analysis = _make_analysis()
        config = PipelineConfig(
            exchanges=["ASX"], filter_exclude_negative_ttm_fcf=False
        )
        results = screen_companies([analysis], config)

        output = tmp_path / "test.csv"
        export_csv(results, output)

        with open(output) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            assert list(reader.fieldnames) == COLUMNS

    def test_row_count(self, tmp_path: Path) -> None:
        analyses = [
            _make_analysis(symbol="A"),
            _make_analysis(symbol="B"),
        ]
        config = PipelineConfig(
            exchanges=["ASX"], filter_exclude_negative_ttm_fcf=False
        )
        results = screen_companies(analyses, config)

        output = tmp_path / "test.csv"
        export_csv(results, output)

        with open(output) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2

    def test_iv_rounding_in_csv(self, tmp_path: Path) -> None:
        """IV values are rounded per the rounding rules."""
        analysis = _make_analysis(
            latest_price=50.0, iv_p25=42.37, iv_p50=58.84, iv_p75=73.16
        )
        config = PipelineConfig(
            exchanges=["ASX"], filter_exclude_negative_ttm_fcf=False
        )
        results = screen_companies([analysis], config)

        output = tmp_path / "test.csv"
        export_csv(results, output)

        with open(output) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["iv_p25"] == "42"
        assert row["iv_p50"] == "59"
        assert row["iv_p75"] == "73"

    def test_iv_rounding_low_price_in_csv(self, tmp_path: Path) -> None:
        """IV values rounded to 10 cents for cheap stocks."""
        analysis = _make_analysis(
            latest_price=5.0, iv_p25=4.37, iv_p50=6.84, iv_p75=8.16
        )
        config = PipelineConfig(
            exchanges=["ASX"], filter_exclude_negative_ttm_fcf=False
        )
        results = screen_companies([analysis], config)

        output = tmp_path / "test.csv"
        export_csv(results, output)

        with open(output) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["iv_p25"] == "4.4"
        assert row["iv_p50"] == "6.8"
        assert row["iv_p75"] == "8.2"

    def test_filtered_flag_in_csv(self, tmp_path: Path) -> None:
        analyses = [
            _make_analysis(symbol="PASS", market_cap=2_000_000.0),
            _make_analysis(symbol="FAIL", market_cap=100.0),
        ]
        config = PipelineConfig(
            exchanges=["ASX"],
            filter_min_market_cap=1_000_000.0,
            filter_exclude_negative_ttm_fcf=False,
        )
        results = screen_companies(analyses, config)

        output = tmp_path / "test.csv"
        export_csv(results, output)

        with open(output) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        pass_row = next(r for r in rows if r["symbol"] == "PASS")
        fail_row = next(r for r in rows if r["symbol"] == "FAIL")
        assert pass_row["filtered"] == "FALSE"
        assert fail_row["filtered"] == "TRUE"
        assert fail_row["filter_reasons"] == "MC"

    def test_none_values_exported_as_empty(self, tmp_path: Path) -> None:
        analysis = _make_analysis(interest_coverage=None, ocf_to_debt=None)
        config = PipelineConfig(
            exchanges=["ASX"], filter_exclude_negative_ttm_fcf=False
        )
        results = screen_companies([analysis], config)

        output = tmp_path / "test.csv"
        export_csv(results, output)

        with open(output) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["interest_coverage"] == ""
        assert row["ocf_to_debt"] == ""

    def test_boolean_f_score_signals(self, tmp_path: Path) -> None:
        analysis = _make_analysis()
        config = PipelineConfig(
            exchanges=["ASX"], filter_exclude_negative_ttm_fcf=False
        )
        results = screen_companies([analysis], config)

        output = tmp_path / "test.csv"
        export_csv(results, output)

        with open(output) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["f_roa_positive"] == "TRUE"
        assert row["f_leverage_decreasing"] == "FALSE"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        analysis = _make_analysis()
        config = PipelineConfig(
            exchanges=["ASX"], filter_exclude_negative_ttm_fcf=False
        )
        results = screen_companies([analysis], config)

        output = tmp_path / "deep" / "nested" / "dir" / "test.csv"
        export_csv(results, output)
        assert output.exists()
