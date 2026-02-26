"""Tests for pipeline.analysis.weighted_scoring."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest

from pipeline.analysis.weighted_scoring import (
    _dc_penalty,
    _growth_penalty,
    _mc_penalty,
    calculate_weighted_scores,
)
from pipeline.config.settings import AnalysisConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_companies(rows: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    if rows is None:
        rows = [
            {"entity_id": 1, "symbol": "AAA", "debt_cash_ratio": 1.0,
             "market_cap": 50e6},
            {"entity_id": 2, "symbol": "BBB", "debt_cash_ratio": 2.0,
             "market_cap": 100e6},
        ]
    return pd.DataFrame(rows).set_index("entity_id")


def _make_growth_stats(rows: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    if rows is None:
        rows = [
            {"entity_id": 1, "fcf_growth_mean": 0.15, "fcf_growth_std": 0.05,
             "revenue_growth_mean": 0.12, "revenue_growth_std": 0.04,
             "combined_growth_mean": 0.14},
            {"entity_id": 2, "fcf_growth_mean": 0.05, "fcf_growth_std": 0.10,
             "revenue_growth_mean": 0.08, "revenue_growth_std": 0.08,
             "combined_growth_mean": 0.06},
        ]
    return pd.DataFrame(rows).set_index("entity_id")


# ---------------------------------------------------------------------------
# DC penalty
# ---------------------------------------------------------------------------

class TestDcPenalty:

    def test_squared_ratio(self) -> None:
        assert _dc_penalty(2.0, 1.0) == pytest.approx(4.0)

    def test_weight_applied(self) -> None:
        assert _dc_penalty(2.0, 0.7) == pytest.approx(4.0 * 0.7)

    def test_zero_ratio(self) -> None:
        assert _dc_penalty(0.0, 0.7) == pytest.approx(0.0)

    def test_inf_ratio(self) -> None:
        assert math.isinf(_dc_penalty(float("inf"), 0.7))

    def test_negative_ratio_uses_abs(self) -> None:
        """abs() retained from legacy — negative ratio treated as positive."""
        assert _dc_penalty(-2.0, 1.0) == pytest.approx(4.0)

    def test_zero_weight_disables_penalty(self) -> None:
        """dc_weight=0 should return 0.0 even with inf ratio."""
        assert _dc_penalty(float("inf"), 0.0) == pytest.approx(0.0)
        assert _dc_penalty(5.0, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MC penalty
# ---------------------------------------------------------------------------

class TestMcPenalty:

    def test_log_scaling(self) -> None:
        # market_cap = 10x min -> log10(10) = 1.0
        assert _mc_penalty(200e6, 20e6, 1.0) == pytest.approx(1.0)

    def test_weight_applied(self) -> None:
        assert _mc_penalty(200e6, 20e6, 0.4) == pytest.approx(0.4)

    def test_below_min_is_zero(self) -> None:
        assert _mc_penalty(10e6, 20e6, 0.4) == pytest.approx(0.0)

    def test_at_min_is_zero(self) -> None:
        assert _mc_penalty(20e6, 20e6, 0.4) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Growth penalty
# ---------------------------------------------------------------------------

class TestGrowthPenalty:

    def test_above_threshold_no_rate_penalty(self) -> None:
        config = AnalysisConfig()
        result = _growth_penalty(
            avg_growth=0.15,
            fcf_growth_std=0.0, revenue_growth_std=0.0,
            fcf_growth_mean=0.15, revenue_growth_mean=0.15,
            config=config,
        )
        # No rate penalty, no stability penalty, no divergence penalty
        assert result == pytest.approx(0.0)

    def test_below_threshold_rate_penalty(self) -> None:
        config = AnalysisConfig()  # min_acceptable_growth = 0.10
        result = _growth_penalty(
            avg_growth=0.05,
            fcf_growth_std=0.0, revenue_growth_std=0.0,
            fcf_growth_mean=0.05, revenue_growth_mean=0.05,
            config=config,
        )
        expected_rate = (0.10 - 0.05) * 1.0 * 0.5  # growth_weight * rate_subweight
        assert result == pytest.approx(expected_rate)

    def test_stability_component(self) -> None:
        config = AnalysisConfig()
        result = _growth_penalty(
            avg_growth=0.20,  # above threshold -> no rate penalty
            fcf_growth_std=0.10, revenue_growth_std=0.06,
            fcf_growth_mean=0.20, revenue_growth_mean=0.20,  # no divergence
            config=config,
        )
        avg_std = (0.10 + 0.06) / 2
        expected = avg_std * 1.0 * 0.3  # growth_weight * stability_subweight
        assert result == pytest.approx(expected)

    def test_divergence_component(self) -> None:
        config = AnalysisConfig()
        result = _growth_penalty(
            avg_growth=0.20,
            fcf_growth_std=0.0, revenue_growth_std=0.0,
            fcf_growth_mean=0.25, revenue_growth_mean=0.10,
            config=config,
        )
        expected = abs(0.25 - 0.10) * 1.0 * 0.2  # growth_weight * divergence_subweight
        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# calculate_weighted_scores
# ---------------------------------------------------------------------------

class TestCalculateWeightedScores:

    def test_returns_all_columns(self) -> None:
        companies = _make_companies()
        stats = _make_growth_stats()
        config = AnalysisConfig()
        result = calculate_weighted_scores(companies, stats, config)

        expected_cols = {
            "symbol", "dc_penalty", "mc_penalty", "growth_penalty",
            "total_penalty", "weighted_rank",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_lower_penalty_gets_better_rank(self) -> None:
        """Company with lower total penalty should have lower rank (= better)."""
        companies = _make_companies([
            {"entity_id": 1, "symbol": "LOW", "debt_cash_ratio": 0.5,
             "market_cap": 30e6},
            {"entity_id": 2, "symbol": "HIGH", "debt_cash_ratio": 3.0,
             "market_cap": 500e6},
        ])
        stats = _make_growth_stats([
            {"entity_id": 1, "fcf_growth_mean": 0.20, "fcf_growth_std": 0.02,
             "revenue_growth_mean": 0.18, "revenue_growth_std": 0.02,
             "combined_growth_mean": 0.19},
            {"entity_id": 2, "fcf_growth_mean": 0.03, "fcf_growth_std": 0.15,
             "revenue_growth_mean": 0.04, "revenue_growth_std": 0.12,
             "combined_growth_mean": 0.03},
        ])
        config = AnalysisConfig()
        result = calculate_weighted_scores(companies, stats, config)

        assert float(result.loc[1, "total_penalty"]) < float(result.loc[2, "total_penalty"])  # type: ignore[arg-type]
        assert float(result.loc[1, "weighted_rank"]) < float(result.loc[2, "weighted_rank"])  # type: ignore[arg-type]

    def test_penalty_changes_with_weight_config(self) -> None:
        """Changing dc_weight should change DC penalty."""
        companies = _make_companies()
        stats = _make_growth_stats()

        config_low = AnalysisConfig(dc_weight=0.1)
        config_high = AnalysisConfig(dc_weight=2.0)

        result_low = calculate_weighted_scores(companies, stats, config_low)
        result_high = calculate_weighted_scores(companies, stats, config_high)

        # DC penalty should be higher with higher weight
        assert float(result_high.loc[1, "dc_penalty"]) > float(result_low.loc[1, "dc_penalty"])  # type: ignore[arg-type]

    def test_inf_dc_ratio_gets_inf_penalty(self) -> None:
        companies = _make_companies([
            {"entity_id": 1, "symbol": "NOCASH", "debt_cash_ratio": float("inf"),
             "market_cap": 50e6},
        ])
        stats = _make_growth_stats([
            {"entity_id": 1, "fcf_growth_mean": 0.10, "fcf_growth_std": 0.05,
             "revenue_growth_mean": 0.10, "revenue_growth_std": 0.05,
             "combined_growth_mean": 0.10},
        ])
        config = AnalysisConfig()
        result = calculate_weighted_scores(companies, stats, config)

        assert math.isinf(float(result.loc[1, "dc_penalty"]))  # type: ignore[arg-type]
        assert math.isinf(float(result.loc[1, "total_penalty"]))  # type: ignore[arg-type]

    def test_total_is_sum_of_components(self) -> None:
        companies = _make_companies()
        stats = _make_growth_stats()
        config = AnalysisConfig()
        result = calculate_weighted_scores(companies, stats, config)

        for eid in result.index:
            expected_total = (
                result.loc[eid, "dc_penalty"]
                + result.loc[eid, "mc_penalty"]
                + result.loc[eid, "growth_penalty"]
            )
            assert result.loc[eid, "total_penalty"] == pytest.approx(expected_total)

    def test_empty_companies(self) -> None:
        companies = pd.DataFrame(
            columns=["symbol", "debt_cash_ratio", "market_cap"],
        )
        companies.index.name = "entity_id"
        stats = pd.DataFrame(
            columns=[
                "fcf_growth_mean", "fcf_growth_std",
                "revenue_growth_mean", "revenue_growth_std",
                "combined_growth_mean",
            ],
        )
        stats.index.name = "entity_id"
        config = AnalysisConfig()
        result = calculate_weighted_scores(companies, stats, config)

        assert result.empty

    def test_missing_growth_stats_skips_company(self) -> None:
        companies = _make_companies([
            {"entity_id": 1, "symbol": "AAA", "debt_cash_ratio": 1.0,
             "market_cap": 50e6},
            {"entity_id": 2, "symbol": "BBB", "debt_cash_ratio": 1.0,
             "market_cap": 50e6},
        ])
        stats = _make_growth_stats([
            {"entity_id": 1, "fcf_growth_mean": 0.10, "fcf_growth_std": 0.05,
             "revenue_growth_mean": 0.10, "revenue_growth_std": 0.05,
             "combined_growth_mean": 0.10},
        ])
        config = AnalysisConfig()
        result = calculate_weighted_scores(companies, stats, config)

        assert 1 in result.index
        assert 2 not in result.index

    def test_missing_columns_raises(self) -> None:
        companies = pd.DataFrame(
            {"symbol": ["A"]}, index=pd.Index([1], name="entity_id"),
        )
        stats = _make_growth_stats()
        with pytest.raises(ValueError, match="Missing required columns"):
            calculate_weighted_scores(companies, stats, AnalysisConfig())
