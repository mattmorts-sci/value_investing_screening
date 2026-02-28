"""Tests for CLI entry point (main.py)."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pipeline.data.models import CompanyData
from pipeline.main import _parse_args, run_detail
from pipeline.metrics.quality import QualityMetrics
from pipeline.metrics.safety import SafetyMetrics
from pipeline.metrics.sentiment import SentimentMetrics
from pipeline.metrics.trends import TrendMetrics
from pipeline.metrics.valuation import ValuationMetrics
from pipeline.screening import CompanyAnalysis
from pipeline.simulation import (
    PathData,
    PercentileBands,
    SimulationOutput,
)

NUM_PROJ_QUARTERS = 8


# --- Test fixtures ---


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


def _make_analysis(symbol: str = "TEST") -> CompanyAnalysis:
    rng = np.random.default_rng(42)
    paths = []
    for _ in range(2):
        rev = 100.0 * np.cumprod(1 + rng.normal(0.02, 0.05, NUM_PROJ_QUARTERS))
        fcf = rev * rng.uniform(0.05, 0.15, NUM_PROJ_QUARTERS)
        paths.append(PathData(quarterly_revenue=rev, quarterly_fcf=fcf))

    base_rev = 100.0 * np.cumprod(1 + np.full(NUM_PROJ_QUARTERS, 0.02))
    base_fcf = 10.0 * np.cumprod(1 + np.full(NUM_PROJ_QUARTERS, 0.02))

    return CompanyAnalysis(
        company=_make_company(symbol),
        valuation=ValuationMetrics(
            enterprise_value=6e8,
            ttm_fcf=4e7,
            ttm_ebit=6e7,
            fcf_ev=0.067,
            ebit_ev=0.10,
            composite_yield=0.08,
        ),
        trends=TrendMetrics(
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
        ),
        safety=SafetyMetrics(interest_coverage=8.0, ocf_to_debt=0.5),
        quality=QualityMetrics(
            f_roa_positive=True,
            f_ocf_positive=True,
            f_roa_improving=True,
            f_accruals_negative=True,
            f_leverage_decreasing=True,
            f_current_ratio_improving=True,
            f_no_dilution=True,
            f_gross_margin_improving=False,
            f_asset_turnover_improving=False,
            f_score=7,
            gross_profitability=0.12,
            accruals_ratio=-0.02,
        ),
        sentiment=SentimentMetrics(return_6m=0.05, return_12m=0.12),
        simulation=SimulationOutput(
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
        ),
    )


# --- CLI parsing tests ---


class TestDetailParser:
    """Tests for detail subcommand argument parsing."""

    def test_parses_single_ticker(self) -> None:
        args = _parse_args(["detail", "AAPL"])
        assert args.command == "detail"
        assert args.tickers == ["AAPL"]

    def test_parses_multiple_tickers(self) -> None:
        args = _parse_args(["detail", "AAPL", "MSFT", "CBA.AX"])
        assert args.tickers == ["AAPL", "MSFT", "CBA.AX"]

    def test_default_output_dir(self) -> None:
        args = _parse_args(["detail", "AAPL"])
        assert args.output_dir == Path("output/detail")

    def test_custom_output_dir(self) -> None:
        args = _parse_args(["detail", "AAPL", "--output-dir", "/tmp/reports"])
        assert args.output_dir == Path("/tmp/reports")

    def test_verbose_flag(self) -> None:
        args = _parse_args(["detail", "AAPL", "-v"])
        assert args.verbose is True

    def test_db_path(self) -> None:
        args = _parse_args(["detail", "AAPL", "--db-path", "/tmp/fmp.db"])
        assert args.db_path == Path("/tmp/fmp.db")

    def test_no_tickers_raises(self) -> None:
        with pytest.raises(SystemExit):
            _parse_args(["detail"])


# --- run_detail tests ---


class TestRunDetail:
    """Tests for run_detail with mocked data layer."""

    def test_generates_html_file(self, tmp_path: Path) -> None:
        """Happy path: one ticker produces one HTML file."""
        args = argparse.Namespace(
            tickers=["TEST"],
            output_dir=tmp_path,
            db_path=None,
            verbose=False,
        )

        company = _make_company("TEST")
        analysis = _make_analysis("TEST")

        with (
            patch("pipeline.main.lookup_entity_ids", return_value={"TEST": 1}),
            patch("pipeline.main.load_company", return_value=company),
            patch("pipeline.main._process_company", return_value=analysis),
        ):
            run_detail(args)

        output_file = tmp_path / "TEST.html"
        assert output_file.exists()
        content = output_file.read_text()
        assert "<!DOCTYPE html>" in content
        assert "TEST" in content

    def test_skips_unknown_ticker(self, tmp_path: Path) -> None:
        """Unknown tickers are skipped, no file produced."""
        args = argparse.Namespace(
            tickers=["UNKNOWN"],
            output_dir=tmp_path,
            db_path=None,
            verbose=False,
        )

        with patch("pipeline.main.lookup_entity_ids", return_value={}):
            run_detail(args)

        assert list(tmp_path.iterdir()) == []

    def test_skips_failed_load(self, tmp_path: Path) -> None:
        """Ticker found in DB but load_company returns None."""
        args = argparse.Namespace(
            tickers=["FAIL"],
            output_dir=tmp_path,
            db_path=None,
            verbose=False,
        )

        with (
            patch("pipeline.main.lookup_entity_ids", return_value={"FAIL": 1}),
            patch("pipeline.main.load_company", return_value=None),
        ):
            run_detail(args)

        assert list(tmp_path.iterdir()) == []

    def test_multiple_tickers(self, tmp_path: Path) -> None:
        """Multiple tickers each produce their own HTML file."""
        args = argparse.Namespace(
            tickers=["AAA", "BBB"],
            output_dir=tmp_path,
            db_path=None,
            verbose=False,
        )

        with (
            patch(
                "pipeline.main.lookup_entity_ids",
                return_value={"AAA": 1, "BBB": 2},
            ),
            patch(
                "pipeline.main.load_company",
                side_effect=[_make_company("AAA"), _make_company("BBB")],
            ),
            patch(
                "pipeline.main._process_company",
                side_effect=[_make_analysis("AAA"), _make_analysis("BBB")],
            ),
        ):
            run_detail(args)

        assert (tmp_path / "AAA.html").exists()
        assert (tmp_path / "BBB.html").exists()

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        """Output directory is created if it doesn't exist."""
        nested_dir = tmp_path / "sub" / "dir"
        args = argparse.Namespace(
            tickers=["TEST"],
            output_dir=nested_dir,
            db_path=None,
            verbose=False,
        )

        with (
            patch("pipeline.main.lookup_entity_ids", return_value={"TEST": 1}),
            patch("pipeline.main.load_company", return_value=_make_company()),
            patch(
                "pipeline.main._process_company",
                return_value=_make_analysis(),
            ),
        ):
            run_detail(args)

        assert nested_dir.exists()
        assert (nested_dir / "TEST.html").exists()

    def test_no_simulation_still_produces_report(self, tmp_path: Path) -> None:
        """Company without simulation still generates a valid report."""
        args = argparse.Namespace(
            tickers=["TEST"],
            output_dir=tmp_path,
            db_path=None,
            verbose=False,
        )

        analysis = _make_analysis()
        analysis.simulation = None
        analysis.trends = None

        with (
            patch("pipeline.main.lookup_entity_ids", return_value={"TEST": 1}),
            patch("pipeline.main.load_company", return_value=_make_company()),
            patch("pipeline.main._process_company", return_value=analysis),
        ):
            run_detail(args)

        output_file = tmp_path / "TEST.html"
        assert output_file.exists()
        content = output_file.read_text()
        assert "<!DOCTYPE html>" in content
