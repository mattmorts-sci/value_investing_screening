"""Tests for pipeline.analysis.factor_analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from pipeline.analysis.factor_analysis import (
    analyze_factor_dominance,
    calculate_factor_contributions,
    create_quadrant_analysis,
)
from pipeline.config.settings import AnalysisConfig
from pipeline.data.contracts import IntrinsicValue

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_weighted_scores(rows: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    if rows is None:
        rows = [
            {"entity_id": 1, "dc_penalty": 2.0, "mc_penalty": 1.0,
             "growth_penalty": 1.0, "total_penalty": 4.0},
            {"entity_id": 2, "dc_penalty": 0.5, "mc_penalty": 3.0,
             "growth_penalty": 0.5, "total_penalty": 4.0},
            {"entity_id": 3, "dc_penalty": 0.5, "mc_penalty": 0.5,
             "growth_penalty": 3.0, "total_penalty": 4.0},
        ]
    return pd.DataFrame(rows).set_index("entity_id")


def _make_iv(iv_per_share: float = 50.0) -> IntrinsicValue:
    return IntrinsicValue(
        scenario="base", period_years=5,
        projected_annual_cash_flows=[100.0] * 5,
        terminal_value=1000.0, present_value=1500.0,
        iv_per_share=iv_per_share, growth_rate=0.10,
        discount_rate=0.10, terminal_growth_rate=0.01,
        margin_of_safety=0.50,
    )


# ---------------------------------------------------------------------------
# calculate_factor_contributions
# ---------------------------------------------------------------------------

class TestFactorContributions:

    def test_contributions_sum_to_100(self) -> None:
        ws = _make_weighted_scores()
        result = calculate_factor_contributions(ws)
        for eid in result.index:
            total = (
                result.loc[eid, "dc_pct"]
                + result.loc[eid, "mc_pct"]
                + result.loc[eid, "growth_pct"]
            )
            assert total == pytest.approx(100.0)

    def test_correct_percentages(self) -> None:
        ws = _make_weighted_scores([
            {"entity_id": 1, "dc_penalty": 3.0, "mc_penalty": 1.0,
             "growth_penalty": 1.0, "total_penalty": 5.0},
        ])
        result = calculate_factor_contributions(ws)
        assert result.loc[1, "dc_pct"] == pytest.approx(60.0)
        assert result.loc[1, "mc_pct"] == pytest.approx(20.0)
        assert result.loc[1, "growth_pct"] == pytest.approx(20.0)

    def test_zero_total_penalty(self) -> None:
        ws = _make_weighted_scores([
            {"entity_id": 1, "dc_penalty": 0.0, "mc_penalty": 0.0,
             "growth_penalty": 0.0, "total_penalty": 0.0},
        ])
        result = calculate_factor_contributions(ws)
        assert result.loc[1, "dc_pct"] == pytest.approx(0.0)
        assert result.loc[1, "mc_pct"] == pytest.approx(0.0)
        assert result.loc[1, "growth_pct"] == pytest.approx(0.0)

    def test_empty_input(self) -> None:
        ws = pd.DataFrame(
            columns=["dc_penalty", "mc_penalty", "growth_penalty", "total_penalty"],
        )
        ws.index.name = "entity_id"
        result = calculate_factor_contributions(ws)
        assert result.empty

    def test_missing_columns_raises(self) -> None:
        ws = pd.DataFrame({"dc_penalty": [1.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            calculate_factor_contributions(ws)


# ---------------------------------------------------------------------------
# analyze_factor_dominance
# ---------------------------------------------------------------------------

class TestFactorDominance:

    def test_identifies_dominant_factors(self) -> None:
        ws = _make_weighted_scores()
        contributions = calculate_factor_contributions(ws)
        result = analyze_factor_dominance(contributions, ws)

        factors = set(result["primary_factor"])
        assert factors == {"dc", "mc", "growth"}

    def test_company_counts(self) -> None:
        ws = _make_weighted_scores()
        contributions = calculate_factor_contributions(ws)
        result = analyze_factor_dominance(contributions, ws)

        total = result["company_count"].sum()
        assert total == 3

    def test_pct_of_total_sums_to_100(self) -> None:
        ws = _make_weighted_scores()
        contributions = calculate_factor_contributions(ws)
        result = analyze_factor_dominance(contributions, ws)

        assert result["pct_of_total"].sum() == pytest.approx(100.0)

    def test_avg_total_penalty_computed(self) -> None:
        ws = _make_weighted_scores([
            {"entity_id": 1, "dc_penalty": 5.0, "mc_penalty": 1.0,
             "growth_penalty": 1.0, "total_penalty": 7.0},
            {"entity_id": 2, "dc_penalty": 4.0, "mc_penalty": 1.0,
             "growth_penalty": 1.0, "total_penalty": 6.0},
            {"entity_id": 3, "dc_penalty": 0.5, "mc_penalty": 0.5,
             "growth_penalty": 3.0, "total_penalty": 4.0},
        ])
        contributions = calculate_factor_contributions(ws)
        result = analyze_factor_dominance(contributions, ws)

        assert "avg_total_penalty" in result.columns
        # DC dominates entities 1 and 2 (total_penalty 7.0 and 6.0)
        dc_row = result[result["primary_factor"] == "dc"]
        assert dc_row.iloc[0]["avg_total_penalty"] == pytest.approx(6.5)

    def test_empty_contributions(self) -> None:
        contributions = pd.DataFrame(
            columns=["dc_pct", "mc_pct", "growth_pct"],
        )
        contributions.index.name = "entity_id"
        ws = pd.DataFrame(columns=["total_penalty"])
        ws.index.name = "entity_id"
        result = analyze_factor_dominance(contributions, ws)
        assert result.empty
        assert "avg_total_penalty" in result.columns


# ---------------------------------------------------------------------------
# create_quadrant_analysis
# ---------------------------------------------------------------------------

class TestQuadrantAnalysis:

    def _make_companies(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"entity_id": 1, "symbol": "HG_HV"},  # high growth, high value
            {"entity_id": 2, "symbol": "HG_LV"},  # high growth, low value
            {"entity_id": 3, "symbol": "LG_HV"},  # low growth, high value
            {"entity_id": 4, "symbol": "LG_LV"},  # low growth, low value
        ]).set_index("entity_id")

    def _make_growth_stats(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"entity_id": 1, "combined_growth_mean": 0.15},
            {"entity_id": 2, "combined_growth_mean": 0.15},
            {"entity_id": 3, "combined_growth_mean": 0.05},
            {"entity_id": 4, "combined_growth_mean": 0.05},
        ]).set_index("entity_id")

    def _make_ivs(self) -> dict[int, Any]:
        return {
            1: {5: {"base": _make_iv(iv_per_share=20.0)}},   # ratio = 2.0
            2: {5: {"base": _make_iv(iv_per_share=5.0)}},    # ratio = 0.5
            3: {5: {"base": _make_iv(iv_per_share=20.0)}},   # ratio = 2.0
            4: {5: {"base": _make_iv(iv_per_share=5.0)}},    # ratio = 0.5
        }

    def test_quadrant_assignments(self) -> None:
        config = AnalysisConfig()  # min_acceptable_growth=0.10, min_iv_to_price_ratio=1.0
        prices = {"HG_HV": 10.0, "HG_LV": 10.0, "LG_HV": 10.0, "LG_LV": 10.0}

        result = create_quadrant_analysis(
            self._make_companies(), self._make_ivs(), prices,
            self._make_growth_stats(), config,
        )

        assert result.loc[1, "quadrant"] == 1  # high growth + high value
        assert result.loc[2, "quadrant"] == 2  # high growth + low value
        assert result.loc[3, "quadrant"] == 3  # low growth + high value
        assert result.loc[4, "quadrant"] == 4  # low growth + low value

    def test_company_without_price_excluded(self) -> None:
        config = AnalysisConfig()
        prices = {"HG_HV": 10.0}  # Only one company has price

        result = create_quadrant_analysis(
            self._make_companies(), self._make_ivs(), prices,
            self._make_growth_stats(), config,
        )

        assert len(result) == 1
        assert 1 in result.index

    def test_empty_input(self) -> None:
        companies = pd.DataFrame(columns=["symbol"])
        companies.index.name = "entity_id"
        stats = pd.DataFrame(columns=["combined_growth_mean"])
        stats.index.name = "entity_id"
        config = AnalysisConfig()
        result = create_quadrant_analysis(companies, {}, {}, stats, config)
        assert result.empty

    def test_threshold_boundary(self) -> None:
        """Growth exactly at threshold counts as high."""
        companies = pd.DataFrame([
            {"entity_id": 1, "symbol": "EDGE"},
        ]).set_index("entity_id")
        stats = pd.DataFrame([
            {"entity_id": 1, "combined_growth_mean": 0.10},
        ]).set_index("entity_id")
        ivs: dict[int, Any] = {
            1: {5: {"base": _make_iv(iv_per_share=10.0)}},
        }
        config = AnalysisConfig()  # min_acceptable_growth=0.10
        result = create_quadrant_analysis(
            companies, ivs, {"EDGE": 10.0}, stats, config,
        )
        assert result.loc[1, "high_growth"] == True  # noqa: E712
