"""Tests for HTML detail report generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.data.models import CompanyData
from pipeline.metrics.quality import QualityMetrics
from pipeline.metrics.safety import SafetyMetrics
from pipeline.metrics.sentiment import SentimentMetrics
from pipeline.metrics.trends import TrendMetrics
from pipeline.metrics.valuation import ValuationMetrics
from pipeline.output.detail_report import generate_detail_html
from pipeline.screening import CompanyAnalysis, ScreeningResult
from pipeline.simulation import (
    PathData,
    PercentileBands,
    SimulationInput,
    SimulationOutput,
)

NUM_PROJ_QUARTERS = 8


def _make_financials(n_quarters: int = 12) -> pd.DataFrame:
    """Build a minimal financials DataFrame."""
    return pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n_quarters, freq="QS"),
        "revenue": [100.0 + i * 5.0 for i in range(n_quarters)],
        "gross_profit": [60.0 + i * 3.0 for i in range(n_quarters)],
        "operating_income": [15.0 + i * 1.0 for i in range(n_quarters)],
        "ebit": [15.0 + i * 1.0 for i in range(n_quarters)],
        "net_income": [10.0 + i * 0.8 for i in range(n_quarters)],
        "income_before_tax": [13.0 + i * 0.9 for i in range(n_quarters)],
        "income_tax_expense": [3.0 + i * 0.1 for i in range(n_quarters)],
        "interest_expense": [2.0] * n_quarters,
        "weighted_average_shs_out_dil": [10.0] * n_quarters,
        "total_assets": [500.0] * n_quarters,
        "total_current_assets": [200.0] * n_quarters,
        "total_current_liabilities": [100.0] * n_quarters,
        "total_debt": [150.0] * n_quarters,
        "long_term_debt": [100.0] * n_quarters,
        "cash_and_cash_equivalents": [50.0] * n_quarters,
        "operating_cash_flow": [20.0 + i * 1.0 for i in range(n_quarters)],
        "free_cash_flow": [12.0 + i * 0.8 for i in range(n_quarters)],
    })


def _make_company(symbol: str = "TEST") -> CompanyData:
    return CompanyData(
        symbol=symbol,
        company_name="Test Corp",
        sector="Technology",
        exchange="ASX",
        financials=_make_financials(),
        latest_price=50.0,
        market_cap=5e8,
        shares_outstanding=10.0,
        price_history=pd.DataFrame(),
    )


def _make_trends() -> TrendMetrics:
    return TrendMetrics(
        revenue_cagr=0.12,
        revenue_yoy_growth_std=0.05,
        revenue_qoq_growth_mean=0.03,
        revenue_qoq_growth_var=0.001,
        margin_intercept=0.15,
        margin_slope=0.002,
        margin_r_squared=0.85,
        conversion_intercept=0.7,
        conversion_slope=0.001,
        conversion_r_squared=0.60,
        conversion_median=None,
        conversion_is_fallback=False,
        roic_latest=0.18,
        roic_slope=0.005,
        roic_detrended_std=0.02,
        roic_minimum=0.10,
    )


def _make_quality(f_score: int = 7) -> QualityMetrics:
    components = [True] * f_score + [False] * (9 - f_score)
    return QualityMetrics(
        f_roa_positive=components[0],
        f_ocf_positive=components[1],
        f_roa_improving=components[2],
        f_accruals_negative=components[3],
        f_leverage_decreasing=components[4],
        f_current_ratio_improving=components[5],
        f_no_dilution=components[6],
        f_gross_margin_improving=components[7],
        f_asset_turnover_improving=components[8],
        f_score=f_score,
        gross_profitability=0.12,
        accruals_ratio=-0.02,
    )


def _make_simulation() -> SimulationOutput:
    rng = np.random.default_rng(42)
    paths = []
    for _ in range(2):
        rev = 100.0 * np.cumprod(1 + rng.normal(0.02, 0.05, NUM_PROJ_QUARTERS))
        fcf = rev * rng.uniform(0.05, 0.15, NUM_PROJ_QUARTERS)
        paths.append(PathData(quarterly_revenue=rev, quarterly_fcf=fcf))

    base_rev = 100.0 * np.cumprod(1 + np.full(NUM_PROJ_QUARTERS, 0.02))
    base_fcf = 10.0 * np.cumprod(1 + np.full(NUM_PROJ_QUARTERS, 0.02))

    return SimulationOutput(
        iv_p10=30.0,
        iv_p25=40.0,
        iv_p50=60.0,
        iv_p75=80.0,
        iv_p90=100.0,
        iv_spread=40.0,
        implied_cagr_p25=-0.02,
        implied_cagr_p50=0.02,
        implied_cagr_p75=0.05,
        sample_paths=paths,
        revenue_bands=PercentileBands(
            p10=base_rev * 0.8, p25=base_rev * 0.9, p50=base_rev,
            p75=base_rev * 1.1, p90=base_rev * 1.2,
        ),
        fcf_bands=PercentileBands(
            p10=base_fcf * 0.5, p25=base_fcf * 0.7, p50=base_fcf,
            p75=base_fcf * 1.3, p90=base_fcf * 1.5,
        ),
    )


def _make_sim_input() -> SimulationInput:
    return SimulationInput(
        revenue_qoq_growth_mean=0.03,
        revenue_qoq_growth_var=0.001,
        margin_intercept=0.15,
        margin_slope=0.002,
        conversion_intercept=0.7,
        conversion_slope=0.001,
        conversion_median=None,
        conversion_is_fallback=False,
        starting_revenue=155.0,
        shares_outstanding=10.0,
        num_historical_quarters=12,
    )


_MISSING = object()


def _make_result(
    f_score: int = 7,
    simulation: SimulationOutput | None = _MISSING,  # type: ignore[assignment]
    trends: TrendMetrics | None = _MISSING,  # type: ignore[assignment]
    interest_coverage: float | None = 8.0,
    ocf_to_debt: float | None = 0.5,
) -> ScreeningResult:
    if simulation is _MISSING:
        simulation = _make_simulation()
    if trends is _MISSING:
        trends = _make_trends()
    return ScreeningResult(
        analysis=CompanyAnalysis(
            company=_make_company(),
            valuation=ValuationMetrics(
                enterprise_value=6e8,
                ttm_fcf=4e7,
                ttm_ebit=6e7,
                fcf_ev=0.067,
                ebit_ev=0.10,
                composite_yield=0.08,
            ),
            trends=trends,
            safety=SafetyMetrics(
                interest_coverage=interest_coverage,
                ocf_to_debt=ocf_to_debt,
            ),
            quality=_make_quality(f_score),
            sentiment=SentimentMetrics(return_6m=0.05, return_12m=0.12),
            simulation=simulation,
        ),
        mos_p25=-0.25,
        mos_p50=0.17,
        mos_p75=0.375,
        filtered=False,
        filter_reasons="",
    )


# --- HTML structure tests ---


class TestHtmlStructure:
    """Verify all expected blocks are present in the output."""

    def test_is_valid_html_document(self) -> None:
        html_str = generate_detail_html(_make_result(), sim_input=None)
        assert html_str.startswith("<!DOCTYPE html>")
        assert "</html>" in html_str

    def test_contains_symbol_and_name(self) -> None:
        html_str = generate_detail_html(_make_result(), sim_input=None)
        assert "TEST" in html_str
        assert "Test Corp" in html_str

    def test_contains_header_metrics(self) -> None:
        html_str = generate_detail_html(_make_result(), sim_input=None)
        assert "Price" in html_str
        assert "Market Cap" in html_str
        assert "FCF/EV" in html_str
        assert "EBIT/EV" in html_str
        assert "MoS (Median)" in html_str
        assert "F-Score" in html_str

    def test_contains_quality_section(self) -> None:
        html_str = generate_detail_html(_make_result(), sim_input=None)
        assert "<h2>Quality</h2>" in html_str
        assert "ROA Positive" in html_str
        assert "Gross Profitability" in html_str

    def test_contains_trends_section(self) -> None:
        html_str = generate_detail_html(_make_result(), sim_input=None)
        assert "<h2>Trends</h2>" in html_str
        assert "Revenue CAGR" in html_str
        assert "Margin Slope" in html_str
        assert "ROIC" in html_str

    def test_contains_safety_section(self) -> None:
        html_str = generate_detail_html(_make_result(), sim_input=None)
        assert "<h2>Safety</h2>" in html_str
        assert "Interest Coverage" in html_str
        assert "OCF / Total Debt" in html_str

    def test_contains_sentiment(self) -> None:
        html_str = generate_detail_html(_make_result(), sim_input=None)
        assert "6-Month Return" in html_str
        assert "12-Month Return" in html_str


# --- Chart embedding tests ---


class TestChartEmbedding:
    """Verify charts are embedded when simulation data exists."""

    def test_contains_chart_images_with_simulation(self) -> None:
        result = _make_result()
        html_str = generate_detail_html(result, sim_input=None)
        assert "data:image/png;base64," in html_str

    def test_no_projection_charts_without_simulation(self) -> None:
        result = _make_result(simulation=SimulationOutput(
            iv_p10=0, iv_p25=0, iv_p50=0, iv_p75=0, iv_p90=0,
            iv_spread=0, implied_cagr_p25=0, implied_cagr_p50=0,
            implied_cagr_p75=0, sample_paths=[],
            revenue_bands=None, fcf_bands=None,
        ))
        html_str = generate_detail_html(result, sim_input=None)
        # Should still have revenue growth and margin charts
        assert 'alt="Revenue Growth"' in html_str
        assert 'alt="Operating Margin"' in html_str
        # But not projection charts
        assert 'alt="Revenue Projection"' not in html_str
        assert 'alt="FCF Projection"' not in html_str

    def test_no_charts_without_simulation_or_trends(self) -> None:
        result = _make_result(simulation=None, trends=None)
        # Reduce financials to < 5 quarters so revenue growth chart is skipped
        result.analysis.company.financials = _make_financials(n_quarters=3)
        html_str = generate_detail_html(result, sim_input=None)
        assert "<h2>Charts</h2>" not in html_str


# --- F-Score visual encoding ---


class TestFScoreEncoding:
    """F-Score colour coding: strong (7-9), moderate (4-6), weak (0-3)."""

    def test_strong_fscore(self) -> None:
        html_str = generate_detail_html(_make_result(f_score=8), sim_input=None)
        assert "fscore-strong" in html_str

    def test_moderate_fscore(self) -> None:
        html_str = generate_detail_html(_make_result(f_score=5), sim_input=None)
        assert "fscore-moderate" in html_str

    def test_weak_fscore(self) -> None:
        html_str = generate_detail_html(_make_result(f_score=2), sim_input=None)
        assert "fscore-weak" in html_str

    def test_fscore_checkmarks(self) -> None:
        html_str = generate_detail_html(_make_result(f_score=7), sim_input=None)
        assert "\u2713" in html_str  # checkmark
        assert "\u2717" in html_str  # cross (f_score=7 means 2 false)


# --- Safety visual encoding ---


class TestSafetyEncoding:
    """Safety amber highlighting below threshold."""

    def test_safety_amber_below_threshold(self) -> None:
        html_str = generate_detail_html(
            _make_result(interest_coverage=1.2, ocf_to_debt=1.0),
            sim_input=None,
        )
        assert 'class="safety-amber"' in html_str

    def test_no_amber_above_threshold(self) -> None:
        html_str = generate_detail_html(
            _make_result(interest_coverage=5.0, ocf_to_debt=2.0),
            sim_input=None,
        )
        assert 'class="safety-amber"' not in html_str

    def test_no_amber_when_none(self) -> None:
        """None safety values show dash, not amber."""
        html_str = generate_detail_html(
            _make_result(interest_coverage=None, ocf_to_debt=None),
            sim_input=None,
        )
        assert 'class="safety-amber"' not in html_str
        assert "\u2014" in html_str  # em dash for N/A


# --- Trends handling ---


class TestTrendsHandling:
    """Trends section handles missing data."""

    def test_no_trends_shows_insufficient(self) -> None:
        html_str = generate_detail_html(
            _make_result(trends=None), sim_input=None,
        )
        assert "Insufficient data" in html_str

    def test_annualised_slopes(self) -> None:
        """Slopes are displayed annualised (x4)."""
        trends = _make_trends()
        trends.margin_slope = 0.01  # per quarter -> 4.0% annualised
        result = _make_result(trends=trends)
        html_str = generate_detail_html(result, sim_input=None)
        assert "4.0%" in html_str


# --- XSS safety ---


class TestXssSafety:
    """Ensure user-controlled strings are escaped."""

    def test_symbol_escaped(self) -> None:
        result = _make_result()
        result.analysis.company.symbol = "<script>alert(1)</script>"
        html_str = generate_detail_html(result, sim_input=None)
        assert "<script>" not in html_str
        assert "&lt;script&gt;" in html_str
