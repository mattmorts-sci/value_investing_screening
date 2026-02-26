"""Tests for pipeline.runner."""

from __future__ import annotations

import contextlib
from typing import Any
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pipeline.__main__ import _build_parser, main
from pipeline.config.settings import AnalysisConfig
from pipeline.data.contracts import (
    AnalysisResults,
    FilterLog,
    IntrinsicValue,
    Projection,
    RawFinancialData,
)
from pipeline.runner import _export_csv, _update_live_metrics, run_analysis

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_raw_data() -> RawFinancialData:
    """Minimal RawFinancialData for testing."""
    df = pd.DataFrame({
        "entity_id": [1, 1, 2, 2],
        "period_idx": [0, 1, 0, 1],
        "symbol": ["AAA", "AAA", "BBB", "BBB"],
        "company_name": ["Co A", "Co A", "Co B", "Co B"],
        "exchange": ["ASX", "ASX", "ASX", "ASX"],
        "country": ["AU", "AU", "AU", "AU"],
        "fcf": [100.0, 120.0, 200.0, 220.0],
        "revenue": [1000.0, 1100.0, 2000.0, 2200.0],
        "operating_income": [500.0, 550.0, 1000.0, 1100.0],
        "shares_diluted": [10.0, 10.0, 20.0, 20.0],
        "lt_debt": [50.0, 50.0, 100.0, 100.0],
        "cash": [200.0, 200.0, 400.0, 400.0],
        "adj_close": [5.0, 6.0, 8.0, 9.0],
        "fcf_growth": [0.1, 0.2, 0.05, 0.1],
        "revenue_growth": [0.08, 0.1, 0.06, 0.1],
        "fiscal_year": [2023, 2024, 2023, 2024],
        "period": ["Q1", "Q2", "Q1", "Q2"],
        "date": ["2023-03-31", "2024-03-31", "2023-03-31", "2024-03-31"],
    })
    return RawFinancialData(
        data=df,
        query_metadata={"market": "AU"},
        row_count=4,
        company_count=2,
        period_range=(2023, 2024),
        dropped_companies_path=Path("dropped.csv"),
    )


def _make_companies() -> pd.DataFrame:
    """Per-company DataFrame indexed by entity_id."""
    return pd.DataFrame({
        "symbol": ["AAA", "BBB"],
        "company_name": ["Co A", "Co B"],
        "exchange": ["ASX", "ASX"],
        "country": ["AU", "AU"],
        "fcf": [120.0, 220.0],
        "revenue": [1100.0, 2200.0],
        "operating_income": [550.0, 1100.0],
        "shares_diluted": [10.0, 20.0],
        "lt_debt": [50.0, 100.0],
        "cash": [200.0, 400.0],
        "adj_close": [6.0, 9.0],
        "market_cap": [60.0, 180.0],
        "enterprise_value": [-90.0, -120.0],
        "debt_cash_ratio": [0.25, 0.25],
        "fcf_per_share": [12.0, 11.0],
        "acquirers_multiple": [-0.164, -0.109],
        "fcf_to_market_cap": [2.0, 1.222],
        "period_idx": [1, 1],
    }, index=pd.Index([1, 2], name="entity_id"))


def _make_growth_stats() -> pd.DataFrame:
    """Growth stats DataFrame indexed by entity_id."""
    return pd.DataFrame({
        "fcf_growth_mean": [0.15, 0.075],
        "fcf_growth_var": [0.01, 0.005],
        "fcf_growth_std": [0.1, 0.07],
        "revenue_growth_mean": [0.09, 0.08],
        "revenue_growth_var": [0.005, 0.004],
        "revenue_growth_std": [0.07, 0.06],
        "fcf_cagr": [0.2, 0.1],
        "revenue_cagr": [0.1, 0.1],
        "combined_growth_mean": [0.132, 0.0765],
        "growth_stability": [0.92, 0.94],
        "fcf_reliability": [0.9, 0.85],
    }, index=pd.Index([1, 2], name="entity_id"))


def _make_projection(entity_id: int) -> Projection:
    return Projection(
        entity_id=entity_id,
        metric="fcf",
        period_years=5,
        scenario="base",
        quarterly_growth_rates=[0.02] * 20,
        quarterly_values=[100.0 + i * 5 for i in range(20)],
        annual_cagr=0.08,
        current_value=100.0,
    )


def _make_intrinsic_value() -> IntrinsicValue:
    return IntrinsicValue(
        scenario="base",
        period_years=5,
        projected_annual_cash_flows=[100.0] * 5,
        terminal_value=1000.0,
        present_value=800.0,
        iv_per_share=10.0,
        growth_rate=0.08,
        discount_rate=0.10,
        terminal_growth_rate=0.01,
        margin_of_safety=0.50,
    )


def _make_weighted_scores() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["AAA", "BBB"],
        "dc_penalty": [0.04, 0.04],
        "mc_penalty": [0.1, 0.2],
        "growth_penalty": [0.05, 0.08],
        "total_penalty": [0.19, 0.32],
        "weighted_rank": [1, 2],
    }, index=pd.Index([1, 2], name="entity_id"))


def _make_ranking_df(symbols: list[str]) -> pd.DataFrame:
    """Minimal ranking DataFrame."""
    n = len(symbols)
    return pd.DataFrame({
        "symbol": symbols,
        "current_price": [10.0] * n,
        "fcf_growth_annual": [0.1] * n,
        "revenue_growth_annual": [0.08] * n,
        "combined_growth": [0.09] * n,
        "composite_iv_ratio": [2.0 - i * 0.3 for i in range(n)],
        "pessimistic_iv_ratio": [1.2] * n,
        "base_iv_ratio": [2.0] * n,
        "optimistic_iv_ratio": [3.0] * n,
        "scenario_spread": [0.9] * n,
        "downside_exposure": [1.2] * n,
        "terminal_dependency": [0.4] * n,
        "fcf_reliability": [0.9] * n,
        "composite_safety": [0.6] * n,
        "total_expected_return": [0.15] * n,
        "risk_adjusted_score": [1.0 - i * 0.1 for i in range(n)],
        "growth_stability": [0.9] * n,
        "growth_divergence": [0.02] * n,
        "divergence_flag": [False] * n,
        "fcf": [100.0] * n,
        "market_cap": [1000.0] * n,
        "debt_cash_ratio": [0.5] * n,
        "dc_penalty": [0.1] * n,
        "mc_penalty": [0.05] * n,
        "growth_penalty": [0.03] * n,
        "total_penalty": [0.18] * n,
        "risk_adjusted_rank": list(range(1, n + 1)),
        "opportunity_rank": list(range(1, n + 1)),
        "opportunity_score": [100 - i * 50 for i in range(n)],
        "growth_score": [90.0] * n,
        "value_score": [80.0] * n,
        "weighted_score": [70.0] * n,
        "stability_score": [85.0] * n,
        "divergence_penalty": [0] * n,
    }, index=pd.Index(list(range(1, n + 1)), name="entity_id"))


def _make_factor_contributions() -> pd.DataFrame:
    return pd.DataFrame({
        "dc_pct": [21.1, 12.5],
        "mc_pct": [52.6, 62.5],
        "growth_pct": [26.3, 25.0],
    }, index=pd.Index([1, 2], name="entity_id"))


# ---------------------------------------------------------------------------
# _update_live_metrics
# ---------------------------------------------------------------------------

class TestUpdateLiveMetrics:

    def test_updates_price_and_market_cap(self) -> None:
        companies = _make_companies()
        live_prices = {"AAA": 10.0}

        result = _update_live_metrics(companies, live_prices)

        assert result.at[1, "adj_close"] == 10.0
        assert result.at[1, "market_cap"] == 100.0  # 10 * 10 shares

    def test_unchanged_without_live_price(self) -> None:
        companies = _make_companies()
        live_prices = {"AAA": 10.0}

        result = _update_live_metrics(companies, live_prices)

        # BBB not in live_prices — unchanged
        assert result.at[2, "adj_close"] == 9.0
        assert result.at[2, "market_cap"] == 180.0

    def test_updates_enterprise_value(self) -> None:
        companies = _make_companies()
        live_prices = {"AAA": 10.0}

        result = _update_live_metrics(companies, live_prices)

        # EV = market_cap + lt_debt - cash = 100 + 50 - 200 = -50
        assert result.at[1, "enterprise_value"] == -50.0

    def test_updates_acquirers_multiple(self) -> None:
        companies = _make_companies()
        live_prices = {"AAA": 10.0}

        result = _update_live_metrics(companies, live_prices)

        # acquirers_multiple = EV / OI = -50 / 550
        expected = -50.0 / 550.0
        assert abs(float(result.at[1, "acquirers_multiple"]) - expected) < 1e-6  # type: ignore[arg-type]

    def test_does_not_mutate_input(self) -> None:
        companies = _make_companies()
        original_price = companies.at[1, "adj_close"]
        live_prices = {"AAA": 99.0}

        _update_live_metrics(companies, live_prices)

        assert companies.at[1, "adj_close"] == original_price

    def test_empty_live_prices(self) -> None:
        companies = _make_companies()
        result = _update_live_metrics(companies, {})

        pd.testing.assert_frame_equal(result, companies)


# ---------------------------------------------------------------------------
# _export_csv
# ---------------------------------------------------------------------------

class TestExportCsv:

    def test_creates_csv_files(self, tmp_path: Path) -> None:
        rankings = _make_ranking_df(["AAA", "BBB"])
        results = AnalysisResults(
            time_series=pd.DataFrame(),
            companies=_make_companies(),
            projections={},
            intrinsic_values={},
            growth_rankings=rankings,
            value_rankings=rankings,
            weighted_rankings=rankings,
            combined_rankings=rankings,
            factor_contributions=_make_factor_contributions(),
            factor_dominance=pd.DataFrame(),
            quadrant_analysis=pd.DataFrame(),
            watchlist=["AAA"],
            filter_log=FilterLog(),
            live_prices={"AAA": 10.0},
            config=AnalysisConfig(),
        )

        _export_csv(results, tmp_path)

        expected_files = {
            "growth_rankings.csv",
            "value_rankings.csv",
            "weighted_rankings.csv",
            "combined_rankings.csv",
            "factor_contributions.csv",
            "watchlist.csv",
        }
        actual_files = {f.name for f in tmp_path.iterdir()}
        assert expected_files == actual_files

    def test_watchlist_csv_content(self, tmp_path: Path) -> None:
        rankings = _make_ranking_df(["AAA"])
        results = AnalysisResults(
            time_series=pd.DataFrame(),
            companies=_make_companies(),
            projections={},
            intrinsic_values={},
            growth_rankings=rankings,
            value_rankings=rankings,
            weighted_rankings=rankings,
            combined_rankings=rankings,
            factor_contributions=_make_factor_contributions(),
            factor_dominance=pd.DataFrame(),
            quadrant_analysis=pd.DataFrame(),
            watchlist=["AAA", "BBB"],
            filter_log=FilterLog(),
            live_prices={},
            config=AnalysisConfig(),
        )

        _export_csv(results, tmp_path)

        watchlist_df = pd.read_csv(tmp_path / "watchlist.csv")
        assert list(watchlist_df["symbol"]) == ["AAA", "BBB"]

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "nested" / "output"
        rankings = _make_ranking_df(["AAA"])
        results = AnalysisResults(
            time_series=pd.DataFrame(),
            companies=_make_companies(),
            projections={},
            intrinsic_values={},
            growth_rankings=rankings,
            value_rankings=rankings,
            weighted_rankings=rankings,
            combined_rankings=rankings,
            factor_contributions=_make_factor_contributions(),
            factor_dominance=pd.DataFrame(),
            quadrant_analysis=pd.DataFrame(),
            watchlist=[],
            filter_log=FilterLog(),
            live_prices={},
            config=AnalysisConfig(),
        )

        _export_csv(results, output_dir)

        assert output_dir.exists()


# ---------------------------------------------------------------------------
# run_analysis — mocked pipeline
# ---------------------------------------------------------------------------

class TestRunAnalysis:
    """Test the 12-step pipeline orchestration with all dependencies mocked."""

    def _setup_mocks(self) -> dict[str, Any]:
        """Create mock return values for all pipeline steps."""
        raw = _make_raw_data()
        companies = _make_companies()
        growth_stats = _make_growth_stats()
        projections = {
            1: {5: {"fcf": {"base": _make_projection(1)}}},
            2: {5: {"fcf": {"base": _make_projection(2)}}},
        }
        ivs = {
            1: {5: {"base": _make_intrinsic_value()}},
            2: {5: {"base": _make_intrinsic_value()}},
        }
        weighted_scores = _make_weighted_scores()
        rankings = _make_ranking_df(["AAA", "BBB"])
        contributions = _make_factor_contributions()
        dominance = pd.DataFrame({
            "primary_factor": ["mc_penalty", "dc_penalty"],
            "company_count": [1, 1],
            "pct_of_total": [50.0, 50.0],
            "avg_contribution": [52.6, 21.1],
            "avg_total_penalty": [0.32, 0.19],
        })
        quadrants = pd.DataFrame({
            "symbol": ["AAA", "BBB"],
            "combined_growth": [0.1, 0.08],
            "composite_iv_ratio": [2.0, 1.5],
            "high_growth": [True, False],
            "high_value": [True, True],
            "quadrant": [1, 3],
        }, index=pd.Index([1, 2], name="entity_id"))

        mock_provider = MagicMock()
        mock_provider.get_prices.return_value = {"AAA": 10.0, "BBB": 12.0}

        return {
            "raw": raw,
            "companies": companies,
            "growth_stats": growth_stats,
            "projections": projections,
            "ivs": ivs,
            "weighted_scores": weighted_scores,
            "rankings": rankings,
            "contributions": contributions,
            "dominance": dominance,
            "quadrants": quadrants,
            "provider": mock_provider,
        }

    @patch("pipeline.runner._export_csv")
    @patch("pipeline.runner.select_watchlist")
    @patch("pipeline.runner.create_quadrant_analysis")
    @patch("pipeline.runner.analyze_factor_dominance")
    @patch("pipeline.runner.calculate_factor_contributions")
    @patch("pipeline.runner.rank_companies")
    @patch("pipeline.runner.calculate_weighted_scores")
    @patch("pipeline.runner.auto_select_provider")
    @patch("pipeline.runner.calculate_all_dcf")
    @patch("pipeline.runner.project_all")
    @patch("pipeline.runner.apply_filters")
    @patch("pipeline.runner.compute_growth_statistics")
    @patch("pipeline.runner.compute_derived_metrics")
    @patch("pipeline.runner.load_raw_data")
    def test_returns_analysis_results(
        self,
        mock_load: MagicMock,
        mock_derived: MagicMock,
        mock_growth: MagicMock,
        mock_filter: MagicMock,
        mock_project: MagicMock,
        mock_dcf: MagicMock,
        mock_provider_factory: MagicMock,
        mock_weighted: MagicMock,
        mock_rank: MagicMock,
        mock_factor: MagicMock,
        mock_dominance: MagicMock,
        mock_quadrant: MagicMock,
        mock_watchlist: MagicMock,
        mock_export: MagicMock,
    ) -> None:
        mocks = self._setup_mocks()

        mock_load.return_value = mocks["raw"]
        mock_derived.return_value = mocks["companies"]
        mock_growth.return_value = mocks["growth_stats"]
        mock_filter.return_value = (mocks["companies"], FilterLog())
        mock_project.return_value = mocks["projections"]
        mock_dcf.return_value = mocks["ivs"]
        mock_provider_factory.return_value = mocks["provider"]
        mock_weighted.return_value = mocks["weighted_scores"]
        mock_rank.return_value = (
            mocks["rankings"], mocks["rankings"],
            mocks["rankings"], mocks["rankings"],
        )
        mock_factor.return_value = mocks["contributions"]
        mock_dominance.return_value = mocks["dominance"]
        mock_quadrant.return_value = mocks["quadrants"]
        mock_watchlist.return_value = ["AAA"]

        config = AnalysisConfig()
        result = run_analysis(config)

        assert isinstance(result, AnalysisResults)

    @patch("pipeline.runner._export_csv")
    @patch("pipeline.runner.select_watchlist")
    @patch("pipeline.runner.create_quadrant_analysis")
    @patch("pipeline.runner.analyze_factor_dominance")
    @patch("pipeline.runner.calculate_factor_contributions")
    @patch("pipeline.runner.rank_companies")
    @patch("pipeline.runner.calculate_weighted_scores")
    @patch("pipeline.runner.auto_select_provider")
    @patch("pipeline.runner.calculate_all_dcf")
    @patch("pipeline.runner.project_all")
    @patch("pipeline.runner.apply_filters")
    @patch("pipeline.runner.compute_growth_statistics")
    @patch("pipeline.runner.compute_derived_metrics")
    @patch("pipeline.runner.load_raw_data")
    def test_all_steps_called_in_order(
        self,
        mock_load: MagicMock,
        mock_derived: MagicMock,
        mock_growth: MagicMock,
        mock_filter: MagicMock,
        mock_project: MagicMock,
        mock_dcf: MagicMock,
        mock_provider_factory: MagicMock,
        mock_weighted: MagicMock,
        mock_rank: MagicMock,
        mock_factor: MagicMock,
        mock_dominance: MagicMock,
        mock_quadrant: MagicMock,
        mock_watchlist: MagicMock,
        mock_export: MagicMock,
    ) -> None:
        mocks = self._setup_mocks()

        mock_load.return_value = mocks["raw"]
        mock_derived.return_value = mocks["companies"]
        mock_growth.return_value = mocks["growth_stats"]
        mock_filter.return_value = (mocks["companies"], FilterLog())
        mock_project.return_value = mocks["projections"]
        mock_dcf.return_value = mocks["ivs"]
        mock_provider_factory.return_value = mocks["provider"]
        mock_weighted.return_value = mocks["weighted_scores"]
        mock_rank.return_value = (
            mocks["rankings"], mocks["rankings"],
            mocks["rankings"], mocks["rankings"],
        )
        mock_factor.return_value = mocks["contributions"]
        mock_dominance.return_value = mocks["dominance"]
        mock_quadrant.return_value = mocks["quadrants"]
        mock_watchlist.return_value = ["AAA"]

        config = AnalysisConfig()
        run_analysis(config)

        mock_load.assert_called_once()
        mock_derived.assert_called_once()
        mock_growth.assert_called_once()
        mock_filter.assert_called_once()
        mock_project.assert_called_once()
        mock_dcf.assert_called_once()
        mock_provider_factory.assert_called_once()
        mocks["provider"].get_prices.assert_called_once()
        mock_weighted.assert_called_once()
        mock_rank.assert_called_once()
        mock_factor.assert_called_once()
        mock_dominance.assert_called_once()
        mock_quadrant.assert_called_once()
        mock_watchlist.assert_called_once()
        mock_export.assert_called_once()

    @patch("pipeline.runner._export_csv")
    @patch("pipeline.runner.select_watchlist")
    @patch("pipeline.runner.create_quadrant_analysis")
    @patch("pipeline.runner.analyze_factor_dominance")
    @patch("pipeline.runner.calculate_factor_contributions")
    @patch("pipeline.runner.rank_companies")
    @patch("pipeline.runner.calculate_weighted_scores")
    @patch("pipeline.runner.auto_select_provider")
    @patch("pipeline.runner.calculate_all_dcf")
    @patch("pipeline.runner.project_all")
    @patch("pipeline.runner.apply_filters")
    @patch("pipeline.runner.compute_growth_statistics")
    @patch("pipeline.runner.compute_derived_metrics")
    @patch("pipeline.runner.load_raw_data")
    def test_results_fields_populated(
        self,
        mock_load: MagicMock,
        mock_derived: MagicMock,
        mock_growth: MagicMock,
        mock_filter: MagicMock,
        mock_project: MagicMock,
        mock_dcf: MagicMock,
        mock_provider_factory: MagicMock,
        mock_weighted: MagicMock,
        mock_rank: MagicMock,
        mock_factor: MagicMock,
        mock_dominance: MagicMock,
        mock_quadrant: MagicMock,
        mock_watchlist: MagicMock,
        mock_export: MagicMock,
    ) -> None:
        mocks = self._setup_mocks()

        mock_load.return_value = mocks["raw"]
        mock_derived.return_value = mocks["companies"]
        mock_growth.return_value = mocks["growth_stats"]
        mock_filter.return_value = (mocks["companies"], FilterLog())
        mock_project.return_value = mocks["projections"]
        mock_dcf.return_value = mocks["ivs"]
        mock_provider_factory.return_value = mocks["provider"]
        mock_weighted.return_value = mocks["weighted_scores"]
        mock_rank.return_value = (
            mocks["rankings"], mocks["rankings"],
            mocks["rankings"], mocks["rankings"],
        )
        mock_factor.return_value = mocks["contributions"]
        mock_dominance.return_value = mocks["dominance"]
        mock_quadrant.return_value = mocks["quadrants"]
        mock_watchlist.return_value = ["AAA"]

        config = AnalysisConfig()
        result = run_analysis(config)

        assert not result.time_series.empty
        assert not result.companies.empty
        assert len(result.projections) == 2
        assert len(result.intrinsic_values) == 2
        assert not result.growth_rankings.empty
        assert not result.value_rankings.empty
        assert not result.weighted_rankings.empty
        assert not result.combined_rankings.empty
        assert not result.factor_contributions.empty
        assert not result.factor_dominance.empty
        assert not result.quadrant_analysis.empty
        assert result.watchlist == ["AAA"]
        assert isinstance(result.filter_log, FilterLog)
        assert len(result.live_prices) == 2
        assert result.config is config

    def _run_with_mocks(
        self,
        **overrides: Any,
    ) -> AnalysisResults:
        """Run pipeline with all dependencies mocked, applying overrides."""
        mocks = self._setup_mocks()

        patches: dict[str, Any] = {
            "pipeline.runner.load_raw_data": mocks["raw"],
            "pipeline.runner.compute_derived_metrics": mocks["companies"],
            "pipeline.runner.compute_growth_statistics": mocks["growth_stats"],
            "pipeline.runner.apply_filters": (mocks["companies"], FilterLog()),
            "pipeline.runner.project_all": mocks["projections"],
            "pipeline.runner.calculate_all_dcf": mocks["ivs"],
            "pipeline.runner.calculate_weighted_scores": mocks["weighted_scores"],
            "pipeline.runner.calculate_factor_contributions": mocks["contributions"],
            "pipeline.runner.analyze_factor_dominance": mocks["dominance"],
            "pipeline.runner.create_quadrant_analysis": mocks["quadrants"],
            "pipeline.runner.select_watchlist": ["AAA"],
            "pipeline.runner.auto_select_provider": mocks["provider"],
            "pipeline.runner._export_csv": None,
        }

        # Ranking returns a tuple.
        patches["pipeline.runner.rank_companies"] = (
            mocks["rankings"], mocks["rankings"],
            mocks["rankings"], mocks["rankings"],
        )

        # Apply overrides.
        for key, value in overrides.items():
            patches[f"pipeline.runner.{key}"] = value

        with contextlib.ExitStack() as stack:
            for target, return_val in patches.items():
                m = stack.enter_context(patch(target))
                m.return_value = return_val
            return run_analysis(AnalysisConfig())

    def test_invalid_primary_period_raises(self) -> None:
        config = AnalysisConfig(
            projection_periods=(5, 10),
            primary_period=3,
        )
        with pytest.raises(ValueError, match="primary_period"):
            run_analysis(config)

    def test_all_companies_filtered_out(self) -> None:
        """Pipeline completes with empty results when all companies filtered."""
        empty_companies = pd.DataFrame(
            columns=_make_companies().columns,
        )
        empty_companies.index.name = "entity_id"

        result = self._run_with_mocks(
            apply_filters=(empty_companies, FilterLog()),
        )

        assert isinstance(result, AnalysisResults)

    def test_no_live_prices_available(self) -> None:
        """Pipeline completes when provider returns no prices."""
        provider = MagicMock()
        provider.get_prices.return_value = {}

        result = self._run_with_mocks(
            auto_select_provider=provider,
        )

        assert isinstance(result, AnalysisResults)
        assert result.live_prices == {}

    def test_partial_live_prices(self) -> None:
        """Pipeline completes when only some companies have live prices."""
        provider = MagicMock()
        provider.get_prices.return_value = {"AAA": 10.0}  # BBB missing

        result = self._run_with_mocks(
            auto_select_provider=provider,
        )

        assert isinstance(result, AnalysisResults)
        assert len(result.live_prices) == 1

    def test_empty_watchlist(self) -> None:
        """Pipeline completes with empty watchlist."""
        result = self._run_with_mocks(
            select_watchlist=[],
        )

        assert result.watchlist == []


# ---------------------------------------------------------------------------
# __main__.py
# ---------------------------------------------------------------------------

class TestMain:

    def test_build_parser_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])

        assert args.market == "AU"
        assert args.mode == "shortlist"
        assert args.owned == []
        assert args.log_level == "INFO"

    def test_build_parser_all_flags(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([
            "--market", "US",
            "--mode", "owned",
            "--owned", "AAPL", "MSFT",
            "--log-level", "DEBUG",
            "--output-dir", "test_output",
        ])

        assert args.market == "US"
        assert args.mode == "owned"
        assert args.owned == ["AAPL", "MSFT"]
        assert args.log_level == "DEBUG"
        assert args.output_dir == "test_output"

    @patch("pipeline.__main__.run_analysis")
    def test_main_success(self, mock_run: MagicMock) -> None:
        rankings = _make_ranking_df(["AAA"])
        mock_run.return_value = AnalysisResults(
            time_series=pd.DataFrame(),
            companies=_make_companies(),
            projections={},
            intrinsic_values={},
            growth_rankings=rankings,
            value_rankings=rankings,
            weighted_rankings=rankings,
            combined_rankings=rankings,
            factor_contributions=_make_factor_contributions(),
            factor_dominance=pd.DataFrame(),
            quadrant_analysis=pd.DataFrame(),
            watchlist=["AAA"],
            filter_log=FilterLog(),
            live_prices={},
            config=AnalysisConfig(),
        )

        exit_code = main(["--market", "AU"])

        assert exit_code == 0
        mock_run.assert_called_once()

    @patch("pipeline.__main__.run_analysis")
    def test_main_pipeline_failure(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = RuntimeError("DB not found")

        exit_code = main(["--market", "AU"])

        assert exit_code == 1

    def test_main_invalid_config(self) -> None:
        # owned mode without --owned symbols
        exit_code = main(["--mode", "owned"])

        assert exit_code == 1
