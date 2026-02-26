"""Tests for pipeline.charts modules.

Covers market_overview (6), comparative (5), company_detail (7),
tables (2), and edge cases (3). All chart functions must return
matplotlib.figure.Figure objects.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from pipeline.charts.company_detail import (
    financial_health_dashboard,
    growth_fan,
    growth_projection,
    historical_growth,
    risk_spider,
    scenario_comparison,
    valuation_matrix,
)
from pipeline.charts.comparative import (
    acquirers_multiple_analysis,
    comprehensive_rankings_table,
    projected_growth_stability,
    ranking_comparison_table,
    valuation_upside_comparison,
)
from pipeline.charts.market_overview import (
    factor_contributions_bar,
    factor_heatmap,
    growth_comparison_historical,
    growth_comparison_projected,
    growth_value_scatter,
    risk_adjusted_opportunity,
)
from pipeline.charts.tables import (
    filter_summary,
    watchlist_summary,
)
from pipeline.config.settings import AnalysisConfig
from pipeline.data.contracts import (
    AnalysisResults,
    FilterLog,
    IntrinsicValue,
    Projection,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_projection(
    entity_id: int,
    metric: str = "fcf",
    scenario: str = "base",
    cagr: float = 0.08,
    current_value: float = 100.0,
) -> Projection:
    """Create a Projection with 20 quarterly values."""
    return Projection(
        entity_id=entity_id,
        metric=metric,
        period_years=5,
        scenario=scenario,
        quarterly_growth_rates=[0.02] * 20,
        quarterly_values=[current_value + i * 5 for i in range(20)],
        annual_cagr=cagr,
        current_value=current_value,
    )


def _make_intrinsic_value(
    scenario: str = "base",
    iv_per_share: float = 10.0,
    growth_rate: float = 0.08,
) -> IntrinsicValue:
    """Create an IntrinsicValue for one scenario."""
    return IntrinsicValue(
        scenario=scenario,
        period_years=5,
        projected_annual_cash_flows=[100.0] * 5,
        terminal_value=1000.0,
        present_value=800.0,
        iv_per_share=iv_per_share,
        growth_rate=growth_rate,
        discount_rate=0.10,
        terminal_growth_rate=0.01,
        margin_of_safety=0.50,
    )


def _make_time_series() -> pd.DataFrame:
    """Time series for 2 companies, 4 periods each."""
    rows = []
    for eid, symbol in [(1, "AAA"), (2, "BBB")]:
        for pidx in range(4):
            rows.append({
                "entity_id": eid,
                "period_idx": pidx,
                "symbol": symbol,
                "fcf": 100.0 + pidx * 10 + (eid - 1) * 50,
                "revenue": 1000.0 + pidx * 100 + (eid - 1) * 500,
                "fcf_growth": 0.05 + pidx * 0.02,
                "revenue_growth": 0.04 + pidx * 0.01,
                "operating_income": 500.0 + pidx * 50,
                "shares_diluted": 10.0,
                "lt_debt": 50.0,
                "cash": 200.0,
                "adj_close": 5.0 + pidx * 0.5,
            })
    return pd.DataFrame(rows)


def _make_companies() -> pd.DataFrame:
    """Per-company DataFrame indexed by entity_id."""
    return pd.DataFrame({
        "symbol": ["AAA", "BBB"],
        "company_name": ["Co A", "Co B"],
        "fcf": [130.0, 230.0],
        "revenue": [1300.0, 2300.0],
        "operating_income": [650.0, 1150.0],
        "shares_diluted": [10.0, 20.0],
        "lt_debt": [50.0, 100.0],
        "cash": [200.0, 400.0],
        "adj_close": [6.5, 9.5],
        "market_cap": [65.0, 190.0],
        "enterprise_value": [-85.0, -110.0],
        "debt_cash_ratio": [0.25, 0.25],
        "fcf_per_share": [13.0, 11.5],
        "acquirers_multiple": [-0.131, -0.096],
        "fcf_to_market_cap": [2.0, 1.21],
        "period_idx": [3, 3],
    }, index=pd.Index([1, 2], name="entity_id"))


def _make_ranking_base(n: int = 2) -> dict[str, Any]:
    """Column values shared by all four ranking DataFrames."""
    symbols = ["AAA", "BBB"][:n]
    return {
        "symbol": symbols,
        "current_price": [6.5, 9.5][:n],
        "fcf_growth_annual": [0.10, 0.08][:n],
        "revenue_growth_annual": [0.08, 0.06][:n],
        "combined_growth": [0.09, 0.07][:n],
        "composite_iv_ratio": [2.0, 1.7][:n],
        "pessimistic_iv_ratio": [1.2, 1.0][:n],
        "base_iv_ratio": [2.0, 1.7][:n],
        "optimistic_iv_ratio": [3.0, 2.5][:n],
        "scenario_spread": [0.75, 0.70][:n],
        "downside_exposure": [0.60, 0.50][:n],
        "terminal_dependency": [0.55, 0.50][:n],
        "fcf_reliability": [0.90, 0.85][:n],
        "downside_exposure_score": [0.30, 0.25][:n],
        "scenario_spread_score": [0.57, 0.59][:n],
        "terminal_dependency_score": [0.45, 0.50][:n],
        "fcf_reliability_score": [0.90, 0.85][:n],
        "composite_safety": [0.70, 0.64][:n],
        "total_expected_return": [0.15, 0.12][:n],
        "risk_adjusted_score": [1.0, 0.9][:n],
        "growth_stability": [0.92, 0.90][:n],
        "growth_divergence": [0.02, 0.03][:n],
        "divergence_flag": [False, False][:n],
        "fcf": [130.0, 230.0][:n],
        "market_cap": [65.0, 190.0][:n],
        "debt_cash_ratio": [0.25, 0.25][:n],
        "dc_penalty": [0.04, 0.04][:n],
        "mc_penalty": [0.10, 0.20][:n],
        "growth_penalty": [0.05, 0.08][:n],
        "total_penalty": [0.19, 0.32][:n],
    }


def _make_growth_rankings() -> pd.DataFrame:
    base = _make_ranking_base()
    base.update({
        "fcf_growth_rank": [1, 2],
        "revenue_growth_rank": [1, 2],
        "combined_growth_rank": [1, 2],
        "stability_rank": [1, 2],
    })
    return pd.DataFrame(base, index=pd.Index([1, 2], name="entity_id"))


def _make_value_rankings() -> pd.DataFrame:
    base = _make_ranking_base()
    base.update({
        "value_rank": [1, 2],
    })
    return pd.DataFrame(base, index=pd.Index([1, 2], name="entity_id"))


def _make_weighted_rankings() -> pd.DataFrame:
    base = _make_ranking_base()
    base.update({
        "weighted_rank": [1, 2],
    })
    return pd.DataFrame(base, index=pd.Index([1, 2], name="entity_id"))


def _make_combined_rankings() -> pd.DataFrame:
    base = _make_ranking_base()
    base.update({
        "risk_adjusted_rank": [1, 2],
        "opportunity_rank": [1, 2],
        "opportunity_score": [100, 50],
        "growth_score": [90.0, 85.0],
        "value_score": [80.0, 75.0],
        "weighted_score": [70.0, 65.0],
        "stability_score": [85.0, 80.0],
        "divergence_penalty": [0, 0],
    })
    return pd.DataFrame(base, index=pd.Index([1, 2], name="entity_id"))


def _make_projections() -> dict[int, Any]:
    """Projections for 2 companies, primary period 5, fcf + revenue."""
    result: dict[int, Any] = {}
    for eid in (1, 2):
        result[eid] = {
            5: {
                "fcf": {
                    "base": _make_projection(eid, "fcf", "base", 0.08),
                    "pessimistic": _make_projection(eid, "fcf", "pessimistic", 0.04),
                    "optimistic": _make_projection(eid, "fcf", "optimistic", 0.12),
                },
                "revenue": {
                    "base": _make_projection(eid, "revenue", "base", 0.06, 1000.0),
                    "pessimistic": _make_projection(
                        eid, "revenue", "pessimistic", 0.03, 1000.0,
                    ),
                    "optimistic": _make_projection(
                        eid, "revenue", "optimistic", 0.10, 1000.0,
                    ),
                },
            },
        }
    return result


def _make_intrinsic_values() -> dict[int, Any]:
    """Intrinsic values for 2 companies, primary period 5."""
    result: dict[int, Any] = {}
    for eid in (1, 2):
        result[eid] = {
            5: {
                "base": _make_intrinsic_value("base", 10.0, 0.08),
                "pessimistic": _make_intrinsic_value("pessimistic", 7.0, 0.04),
                "optimistic": _make_intrinsic_value("optimistic", 14.0, 0.12),
            },
        }
    return result


def _make_factor_contributions() -> pd.DataFrame:
    return pd.DataFrame({
        "dc_pct": [21.1, 12.5],
        "mc_pct": [52.6, 62.5],
        "growth_pct": [26.3, 25.0],
    }, index=pd.Index([1, 2], name="entity_id"))


def _make_factor_dominance() -> pd.DataFrame:
    return pd.DataFrame({
        "primary_factor": ["mc_penalty", "dc_penalty"],
        "company_count": [1, 1],
        "pct_of_total": [50.0, 50.0],
        "avg_contribution": [52.6, 21.1],
        "avg_total_penalty": [0.32, 0.19],
    })


def _make_quadrant_analysis() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["AAA", "BBB"],
        "combined_growth": [0.09, 0.07],
        "composite_iv_ratio": [2.0, 1.7],
        "high_growth": [True, False],
        "high_value": [True, True],
        "quadrant": [1, 3],
    }, index=pd.Index([1, 2], name="entity_id"))


def _make_filter_log() -> FilterLog:
    flog = FilterLog()
    flog.removed["negative_fcf"] = ["CCC", "DDD"]
    flog.removed["low_market_cap"] = ["EEE"]
    flog.reasons["CCC"] = "Negative FCF"
    flog.reasons["DDD"] = "Negative FCF"
    flog.reasons["EEE"] = "Market cap below threshold"
    return flog


def _make_results() -> AnalysisResults:
    """Build a complete AnalysisResults with synthetic data."""
    return AnalysisResults(
        time_series=_make_time_series(),
        companies=_make_companies(),
        projections=_make_projections(),
        intrinsic_values=_make_intrinsic_values(),
        growth_rankings=_make_growth_rankings(),
        value_rankings=_make_value_rankings(),
        weighted_rankings=_make_weighted_rankings(),
        combined_rankings=_make_combined_rankings(),
        factor_contributions=_make_factor_contributions(),
        factor_dominance=_make_factor_dominance(),
        quadrant_analysis=_make_quadrant_analysis(),
        watchlist=["AAA", "BBB"],
        filter_log=_make_filter_log(),
        live_prices={"AAA": 6.0, "BBB": 9.0},
        config=AnalysisConfig(),
    )


@pytest.fixture()
def results() -> AnalysisResults:
    return _make_results()


@pytest.fixture(autouse=True)
def _close_figures() -> Generator[None, None, None]:
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ===================================================================
# Market overview tests
# ===================================================================


class TestGrowthValueScatter:
    def test_growth_value_scatter_returns_figure(
        self, results: AnalysisResults,
    ) -> None:
        fig = growth_value_scatter(results)
        assert isinstance(fig, Figure)


class TestGrowthComparisonHistorical:
    def test_growth_comparison_historical_returns_figure(
        self, results: AnalysisResults,
    ) -> None:
        fig = growth_comparison_historical(results)
        assert isinstance(fig, Figure)


class TestGrowthComparisonProjected:
    def test_growth_comparison_projected_returns_figure(
        self, results: AnalysisResults,
    ) -> None:
        fig = growth_comparison_projected(results)
        assert isinstance(fig, Figure)


class TestRiskAdjustedOpportunity:
    def test_risk_adjusted_opportunity_returns_figure(
        self, results: AnalysisResults,
    ) -> None:
        fig = risk_adjusted_opportunity(results)
        assert isinstance(fig, Figure)


class TestFactorContributionsBar:
    def test_factor_contributions_bar_returns_figure(
        self, results: AnalysisResults,
    ) -> None:
        fig = factor_contributions_bar(results)
        assert isinstance(fig, Figure)


class TestFactorHeatmap:
    def test_factor_heatmap_returns_figure(
        self, results: AnalysisResults,
    ) -> None:
        fig = factor_heatmap(results)
        assert isinstance(fig, Figure)


# ===================================================================
# Comparative tests
# ===================================================================


class TestProjectedGrowthStability:
    def test_projected_growth_stability_returns_figure(
        self, results: AnalysisResults,
    ) -> None:
        fig = projected_growth_stability(results, ["AAA", "BBB"])
        assert isinstance(fig, Figure)


class TestAcquirersMultipleAnalysis:
    def test_acquirers_multiple_analysis_returns_figure(
        self, results: AnalysisResults,
    ) -> None:
        fig = acquirers_multiple_analysis(results, ["AAA", "BBB"])
        assert isinstance(fig, Figure)


class TestValuationUpsideComparison:
    def test_valuation_upside_comparison_returns_figure(
        self, results: AnalysisResults,
    ) -> None:
        fig = valuation_upside_comparison(results, ["AAA", "BBB"])
        assert isinstance(fig, Figure)


class TestRankingComparisonTable:
    def test_ranking_comparison_table_returns_figure(
        self, results: AnalysisResults,
    ) -> None:
        fig = ranking_comparison_table(results, ["AAA", "BBB"])
        assert isinstance(fig, Figure)


class TestComprehensiveRankingsTable:
    def test_comprehensive_rankings_table_returns_figure(
        self, results: AnalysisResults,
    ) -> None:
        fig = comprehensive_rankings_table(results, ["AAA", "BBB"])
        assert isinstance(fig, Figure)


# ===================================================================
# Company detail tests
# ===================================================================


class TestHistoricalGrowth:
    def test_historical_growth_returns_figure(
        self, results: AnalysisResults,
    ) -> None:
        fig = historical_growth(results, 1)
        assert isinstance(fig, Figure)


class TestGrowthProjection:
    def test_growth_projection_returns_figure(
        self, results: AnalysisResults,
    ) -> None:
        fig = growth_projection(results, 1)
        assert isinstance(fig, Figure)


class TestValuationMatrix:
    def test_valuation_matrix_returns_figure(
        self, results: AnalysisResults,
    ) -> None:
        fig = valuation_matrix(results, 1)
        assert isinstance(fig, Figure)


class TestGrowthFan:
    def test_growth_fan_returns_figure(
        self, results: AnalysisResults,
    ) -> None:
        fig = growth_fan(results, 1)
        assert isinstance(fig, Figure)


class TestFinancialHealthDashboard:
    def test_financial_health_dashboard_returns_figure(
        self, results: AnalysisResults,
    ) -> None:
        fig = financial_health_dashboard(results, 1)
        assert isinstance(fig, Figure)


class TestRiskSpider:
    def test_risk_spider_returns_figure(
        self, results: AnalysisResults,
    ) -> None:
        fig = risk_spider(results, 1)
        assert isinstance(fig, Figure)

    def test_risk_spider_missing_entity(
        self, results: AnalysisResults,
    ) -> None:
        fig = risk_spider(results, 999)
        assert isinstance(fig, Figure)


class TestScenarioComparison:
    def test_scenario_comparison_returns_figure(
        self, results: AnalysisResults,
    ) -> None:
        fig = scenario_comparison(results, 1)
        assert isinstance(fig, Figure)


# ===================================================================
# Table tests
# ===================================================================


class TestWatchlistSummary:
    def test_watchlist_summary_returns_figure(
        self, results: AnalysisResults,
    ) -> None:
        fig = watchlist_summary(results)
        assert isinstance(fig, Figure)


class TestFilterSummary:
    def test_filter_summary_returns_figure(
        self, results: AnalysisResults,
    ) -> None:
        fig = filter_summary(results)
        assert isinstance(fig, Figure)


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:

    def test_empty_watchlist(self) -> None:
        """watchlist_summary handles an empty watchlist gracefully."""
        r = _make_results()
        r.watchlist = []
        fig = watchlist_summary(r)
        assert isinstance(fig, Figure)

    def test_empty_rankings(self) -> None:
        """Market overview charts handle empty combined_rankings."""
        r = _make_results()
        empty_cr = r.combined_rankings.iloc[:0].copy()
        r.combined_rankings = empty_cr
        r.growth_rankings = r.growth_rankings.iloc[:0].copy()
        r.value_rankings = r.value_rankings.iloc[:0].copy()
        r.weighted_rankings = r.weighted_rankings.iloc[:0].copy()
        r.factor_contributions = r.factor_contributions.iloc[:0].copy()

        # All market overview charts should return a figure, not raise
        fig = growth_value_scatter(r)
        assert isinstance(fig, Figure)

        fig = growth_comparison_historical(r)
        assert isinstance(fig, Figure)

        fig = growth_comparison_projected(r)
        assert isinstance(fig, Figure)

        fig = risk_adjusted_opportunity(r)
        assert isinstance(fig, Figure)

        fig = factor_contributions_bar(r)
        assert isinstance(fig, Figure)

        fig = factor_heatmap(r)
        assert isinstance(fig, Figure)

    def test_company_not_in_results(self) -> None:
        """Comparative charts handle symbols not present in data."""
        r = _make_results()
        # Pass a symbol that does not exist
        fig = projected_growth_stability(r, ["NONEXISTENT"])
        assert isinstance(fig, Figure)

        fig = acquirers_multiple_analysis(r, ["NONEXISTENT"])
        assert isinstance(fig, Figure)

        fig = valuation_upside_comparison(r, ["NONEXISTENT"])
        assert isinstance(fig, Figure)

        fig = ranking_comparison_table(r, ["NONEXISTENT"])
        assert isinstance(fig, Figure)

        fig = comprehensive_rankings_table(r, ["NONEXISTENT"])
        assert isinstance(fig, Figure)

    def test_nan_market_cap(self) -> None:
        """Charts handle NaN values in market_cap column."""
        r = _make_results()
        r.combined_rankings.loc[1, "market_cap"] = float("nan")
        fig = growth_value_scatter(r)
        assert isinstance(fig, Figure)

        fig = risk_adjusted_opportunity(r)
        assert isinstance(fig, Figure)

    def test_nan_factor_contributions(self) -> None:
        """Factor contribution charts handle NaN values."""
        r = _make_results()
        r.factor_contributions.loc[1, "dc_pct"] = float("nan")
        fig = factor_contributions_bar(r)
        assert isinstance(fig, Figure)

        fig = factor_heatmap(r)
        assert isinstance(fig, Figure)
