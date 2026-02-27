"""Tests for pipeline.reports.pdf.

Covers PDF generation (mocked WeasyPrint), figure-to-base64 conversion,
and template context building.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import matplotlib

matplotlib.use("Agg")

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from pipeline.config.settings import AnalysisConfig
from pipeline.data.contracts import (
    AnalysisResults,
    FilterLog,
    IntrinsicValue,
    Projection,
)
from pipeline.reports.pdf import _build_context, _fig_to_base64, generate_pdf

# ---------------------------------------------------------------------------
# Fixtures (mirrors test_charts.py synthetic data)
# ---------------------------------------------------------------------------


def _make_projection(
    entity_id: int,
    metric: str = "fcf",
    scenario: str = "base",
    cagr: float = 0.08,
    current_value: float = 100.0,
) -> Projection:
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
        "filing_acquirers_multiple": [-0.131, -0.096],
        "fcf_to_market_cap": [2.0, 1.21],
        "period_idx": [3, 3],
    }, index=pd.Index([1, 2], name="entity_id"))


def _make_ranking_base() -> dict[str, Any]:
    return {
        "symbol": ["AAA", "BBB"],
        "current_price": [6.5, 9.5],
        "fcf_growth_annual": [0.10, 0.08],
        "revenue_growth_annual": [0.08, 0.06],
        "combined_growth": [0.09, 0.07],
        "composite_iv_ratio": [2.0, 1.7],
        "pessimistic_iv_ratio": [1.2, 1.0],
        "base_iv_ratio": [2.0, 1.7],
        "optimistic_iv_ratio": [3.0, 2.5],
        "scenario_spread": [0.75, 0.70],
        "downside_exposure": [0.60, 0.50],
        "terminal_dependency": [0.55, 0.50],
        "fcf_reliability": [0.90, 0.85],
        "downside_exposure_score": [0.30, 0.25],
        "scenario_spread_score": [0.57, 0.59],
        "terminal_dependency_score": [0.45, 0.50],
        "fcf_reliability_score": [0.90, 0.85],
        "composite_safety": [0.70, 0.64],
        "total_expected_return": [0.15, 0.12],
        "risk_adjusted_score": [1.0, 0.9],
        "growth_stability": [0.92, 0.90],
        "growth_divergence": [0.02, 0.03],
        "divergence_flag": [False, False],
        "fcf": [130.0, 230.0],
        "market_cap": [65.0, 190.0],
        "debt_cash_ratio": [0.25, 0.25],
        "dc_penalty": [0.04, 0.04],
        "mc_penalty": [0.10, 0.20],
        "growth_penalty": [0.05, 0.08],
        "total_penalty": [0.19, 0.32],
    }


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


def _make_projections() -> dict[int, Any]:
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


def _make_filter_log() -> FilterLog:
    flog = FilterLog()
    flog.removed["negative_fcf"] = ["CCC", "DDD"]
    flog.removed["low_market_cap"] = ["EEE"]
    return flog


def _make_results() -> AnalysisResults:
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
        factor_dominance=pd.DataFrame({
            "primary_factor": ["mc_penalty", "dc_penalty"],
            "company_count": [1, 1],
            "pct_of_total": [50.0, 50.0],
            "avg_contribution": [52.6, 21.1],
            "avg_total_penalty": [0.32, 0.19],
        }),
        quadrant_analysis=pd.DataFrame({
            "symbol": ["AAA", "BBB"],
            "combined_growth": [0.09, 0.07],
            "composite_iv_ratio": [2.0, 1.7],
            "high_growth": [True, False],
            "high_value": [True, True],
            "quadrant": [1, 3],
        }, index=pd.Index([1, 2], name="entity_id")),
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
# Tests
# ===================================================================


class TestFigToBase64:

    def test_fig_to_base64_returns_string(self) -> None:
        """_fig_to_base64 returns a valid base64-encoded PNG string."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        result = _fig_to_base64(fig)

        assert isinstance(result, str)
        assert len(result) > 0
        # Verify it is valid base64
        decoded = base64.b64decode(result)
        # PNG magic bytes
        assert decoded[:4] == b"\x89PNG"


class TestBuildContext:

    def test_build_context_has_required_keys(
        self, results: AnalysisResults,
    ) -> None:
        """_build_context returns a dict with all keys needed by the template."""
        symbol_to_entity = {
            str(results.companies.loc[eid, "symbol"]): eid
            for eid in results.companies.index
        }
        context = _build_context(results, ["AAA"], symbol_to_entity)

        required_keys = {
            "title",
            "market",
            "analysis_date",
            "primary_period",
            "discount_rate",
            "margin_of_safety",
            "terminal_growth_rate",
            "company_count",
            "filtered_count",
            "priced_count",
            "watchlist_size",
            "detailed_count",
            "filter_chart",
            "market_charts",
            "comparative_charts",
            "detailed_companies",
            "watchlist_chart",
            "rankings_chart",
        }
        assert required_keys.issubset(context.keys())

    def test_build_context_market_charts_list(
        self, results: AnalysisResults,
    ) -> None:
        """market_charts is a non-empty list of base64 strings."""
        symbol_to_entity = {
            str(results.companies.loc[eid, "symbol"]): eid
            for eid in results.companies.index
        }
        context = _build_context(results, ["AAA"], symbol_to_entity)

        assert isinstance(context["market_charts"], list)
        assert len(context["market_charts"]) > 0
        # Each entry should be a base64 string
        for chart in context["market_charts"]:
            assert isinstance(chart, str)
            assert len(chart) > 0

    def test_build_context_detailed_companies(
        self, results: AnalysisResults,
    ) -> None:
        """detailed_companies has correct structure for requested symbols."""
        symbol_to_entity = {
            str(results.companies.loc[eid, "symbol"]): eid
            for eid in results.companies.index
        }
        context = _build_context(results, ["AAA"], symbol_to_entity)

        assert isinstance(context["detailed_companies"], list)
        assert len(context["detailed_companies"]) == 1
        page = context["detailed_companies"][0]
        assert page["symbol"] == "AAA"
        assert page["name"] == "Co A"
        assert isinstance(page["charts"], list)
        assert len(page["charts"]) > 0


class TestGeneratePdf:

    @patch("pipeline.reports.pdf.weasyprint")
    @patch("pipeline.reports.pdf.jinja2")
    def test_generate_pdf_creates_file(
        self,
        mock_jinja2: MagicMock,
        mock_weasyprint: MagicMock,
        results: AnalysisResults,
        tmp_path: Path,
    ) -> None:
        """generate_pdf writes a PDF file (WeasyPrint mocked)."""
        # Mock Jinja2 template rendering
        mock_env = MagicMock()
        mock_jinja2.Environment.return_value = mock_env
        mock_template = MagicMock()
        mock_env.get_template.return_value = mock_template
        mock_template.render.return_value = "<html>test</html>"

        # Mock WeasyPrint PDF generation
        mock_pdf_bytes = b"%PDF-1.4 mock pdf content"
        mock_html_instance = MagicMock()
        mock_weasyprint.HTML.return_value = mock_html_instance
        mock_html_instance.write_pdf.return_value = mock_pdf_bytes

        output_path = tmp_path / "report.pdf"
        result_path = generate_pdf(results, output_path, detailed_symbols=["AAA"])

        assert result_path == output_path
        assert output_path.exists()
        assert output_path.read_bytes() == mock_pdf_bytes
        mock_weasyprint.HTML.assert_called_once()
        mock_html_instance.write_pdf.assert_called_once()

    @patch("pipeline.reports.pdf.weasyprint")
    @patch("pipeline.reports.pdf.jinja2")
    def test_generate_pdf_creates_parent_directory(
        self,
        mock_jinja2: MagicMock,
        mock_weasyprint: MagicMock,
        results: AnalysisResults,
        tmp_path: Path,
    ) -> None:
        """generate_pdf creates parent directories if they don't exist."""
        mock_env = MagicMock()
        mock_jinja2.Environment.return_value = mock_env
        mock_template = MagicMock()
        mock_env.get_template.return_value = mock_template
        mock_template.render.return_value = "<html>test</html>"

        mock_html_instance = MagicMock()
        mock_weasyprint.HTML.return_value = mock_html_instance
        mock_html_instance.write_pdf.return_value = b"%PDF-1.4"

        output_path = tmp_path / "nested" / "dir" / "report.pdf"
        generate_pdf(results, output_path, detailed_symbols=["AAA"])

        assert output_path.parent.exists()

    @patch("pipeline.reports.pdf.weasyprint")
    @patch("pipeline.reports.pdf.jinja2")
    def test_generate_pdf_defaults_detailed_symbols(
        self,
        mock_jinja2: MagicMock,
        mock_weasyprint: MagicMock,
        results: AnalysisResults,
        tmp_path: Path,
    ) -> None:
        """generate_pdf uses watchlist when detailed_symbols is None."""
        mock_env = MagicMock()
        mock_jinja2.Environment.return_value = mock_env
        mock_template = MagicMock()
        mock_env.get_template.return_value = mock_template
        mock_template.render.return_value = "<html>test</html>"

        mock_html_instance = MagicMock()
        mock_weasyprint.HTML.return_value = mock_html_instance
        mock_html_instance.write_pdf.return_value = b"%PDF-1.4"

        output_path = tmp_path / "report.pdf"
        generate_pdf(results, output_path)  # detailed_symbols defaults to None

        # Should complete without error
        assert output_path.exists()
