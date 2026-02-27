"""Tests for pipeline.analysis.ranking."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from pipeline.analysis.ranking import (
    _compute_iv_ratios,
    _compute_terminal_dependency,
    rank_companies,
)
from pipeline.config.settings import AnalysisConfig
from pipeline.data.contracts import IntrinsicValue, Projection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_config(**overrides: Any) -> AnalysisConfig:
    kwargs: dict[str, Any] = {
        "projection_periods": (5,),
        "primary_period": 5,
    }
    kwargs.update(overrides)
    return AnalysisConfig(**kwargs)


def _make_projection(
    annual_cagr: float = 0.10,
    period_years: int = 5,
    scenario: str = "base",
    metric: str = "fcf",
) -> Projection:
    n = period_years * 4
    return Projection(
        entity_id=1,
        metric=metric,
        period_years=period_years,
        scenario=scenario,
        quarterly_growth_rates=[0.02] * n,
        quarterly_values=[100.0 * 1.02**i for i in range(1, n + 1)],
        annual_cagr=annual_cagr,
        current_value=100.0,
    )


def _make_iv(
    iv_per_share: float = 50.0,
    scenario: str = "base",
    period_years: int = 5,
    present_value: float = 1500.0,
    terminal_value: float = 1000.0,
    discount_rate: float = 0.10,
) -> IntrinsicValue:
    return IntrinsicValue(
        scenario=scenario,
        period_years=period_years,
        projected_annual_cash_flows=[100.0] * period_years,
        terminal_value=terminal_value,
        present_value=present_value,
        iv_per_share=iv_per_share,
        growth_rate=0.10,
        discount_rate=discount_rate,
        terminal_growth_rate=0.01,
        margin_of_safety=0.50,
    )


def _make_companies(n: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "entity_id": i,
            "symbol": f"CO{i}",
            "fcf": 100.0 * i,
            "market_cap": 50e6 * i,
            "debt_cash_ratio": 0.5 * i,
        })
    return pd.DataFrame(rows).set_index("entity_id")


def _make_growth_stats(n: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "entity_id": i,
            "fcf_growth_mean": 0.20 - 0.05 * i,
            "fcf_growth_std": 0.03 * i,
            "revenue_growth_mean": 0.18 - 0.04 * i,
            "revenue_growth_std": 0.02 * i,
            "combined_growth_mean": 0.19 - 0.04 * i,
            "growth_stability": 0.9 - 0.1 * i,
            "fcf_reliability": 0.9,
        })
    return pd.DataFrame(rows).set_index("entity_id")


def _make_projections(n: int = 3) -> dict[int, Any]:
    result = {}
    for i in range(1, n + 1):
        cagr = 0.20 - 0.05 * i
        result[i] = {
            5: {
                "fcf": {
                    "base": _make_projection(annual_cagr=cagr),
                    "optimistic": _make_projection(
                        annual_cagr=cagr + 0.05, scenario="optimistic",
                    ),
                    "pessimistic": _make_projection(
                        annual_cagr=cagr - 0.05, scenario="pessimistic",
                    ),
                },
                "revenue": {
                    "base": _make_projection(
                        annual_cagr=cagr - 0.02, metric="revenue",
                    ),
                },
            },
        }
    return result


def _make_ivs(n: int = 3) -> dict[int, Any]:
    """Create IVs where all companies pass the safety gate.

    Pessimistic IV must be >= current_price (which is 10*i in _make_live_prices).
    So pessimistic_iv_per_share must be >= 10*i for entity i.
    """
    result = {}
    for i in range(1, n + 1):
        price = 10.0 * i  # matches _make_live_prices
        iv_base = price * 2.0  # base IV = 2x price
        result[i] = {
            5: {
                "base": _make_iv(iv_per_share=iv_base),
                "optimistic": _make_iv(
                    iv_per_share=iv_base * 1.5, scenario="optimistic",
                ),
                "pessimistic": _make_iv(
                    iv_per_share=price * 1.2, scenario="pessimistic",
                ),
            },
        }
    return result


def _make_weighted_scores(n: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "entity_id": i,
            "symbol": f"CO{i}",
            "dc_penalty": 0.2 * i,
            "mc_penalty": 0.1 * i,
            "growth_penalty": 0.05 * i,
            "total_penalty": 0.35 * i,
            "weighted_rank": i,
        })
    return pd.DataFrame(rows).set_index("entity_id")


def _make_live_prices(n: int = 3) -> dict[str, float]:
    return {f"CO{i}": 10.0 * i for i in range(1, n + 1)}


# ---------------------------------------------------------------------------
# _compute_iv_ratios
# ---------------------------------------------------------------------------

class TestComputeIvRatios:

    def test_computes_all_scenario_ratios(self) -> None:
        ivs: dict[int, Any] = {
            1: {
                5: {
                    "base": _make_iv(iv_per_share=20.0),
                    "optimistic": _make_iv(iv_per_share=30.0, scenario="optimistic"),
                    "pessimistic": _make_iv(iv_per_share=15.0, scenario="pessimistic"),
                },
            },
        }
        result = _compute_iv_ratios(1, 10.0, ivs, 5)
        assert result["base_iv_ratio"] == pytest.approx(2.0)
        assert result["optimistic_iv_ratio"] == pytest.approx(3.0)
        assert result["pessimistic_iv_ratio"] == pytest.approx(1.5)

    def test_composite_ratio_weighted(self) -> None:
        ivs: dict[int, Any] = {
            1: {
                5: {
                    "base": _make_iv(iv_per_share=20.0),
                    "optimistic": _make_iv(iv_per_share=30.0, scenario="optimistic"),
                    "pessimistic": _make_iv(iv_per_share=10.0, scenario="pessimistic"),
                },
            },
        }
        result = _compute_iv_ratios(1, 10.0, ivs, 5)
        # 0.25*1.0 + 0.50*2.0 + 0.25*3.0 = 0.25 + 1.0 + 0.75 = 2.0
        assert result["composite_iv_ratio"] == pytest.approx(2.0)

    def test_no_ivs_returns_zeros(self) -> None:
        result = _compute_iv_ratios(1, 10.0, {}, 5)
        assert result["composite_iv_ratio"] == 0.0
        assert result["base_iv_ratio"] == 0.0

    def test_zero_price_returns_zeros(self) -> None:
        ivs = {1: {5: {"base": _make_iv(iv_per_share=50.0)}}}
        result = _compute_iv_ratios(1, 0.0, ivs, 5)
        assert result["composite_iv_ratio"] == 0.0

    def test_negative_iv_treated_as_zero(self) -> None:
        ivs = {1: {5: {"base": _make_iv(iv_per_share=-10.0)}}}
        result = _compute_iv_ratios(1, 10.0, ivs, 5)
        assert result["base_iv_ratio"] == 0.0
        assert result["composite_iv_ratio"] == 0.0


# ---------------------------------------------------------------------------
# _compute_terminal_dependency
# ---------------------------------------------------------------------------

class TestComputeTerminalDependency:

    def test_normal_values(self) -> None:
        # terminal_value=1000, discount_rate=0.10, period=5
        # terminal_pv = 1000 / 1.1^5 = 620.92
        # present_value = 1500
        # dependency = 620.92 / 1500 = 0.4139
        ivs: dict[int, Any] = {
            1: {5: {"base": _make_iv(
                present_value=1500.0,
                terminal_value=1000.0,
                discount_rate=0.10,
            )}},
        }
        dep = _compute_terminal_dependency(1, ivs, 5)
        expected = (1000.0 / 1.1**5) / 1500.0
        assert dep == pytest.approx(expected, rel=1e-6)

    def test_no_ivs_returns_max_risk(self) -> None:
        assert _compute_terminal_dependency(1, {}, 5) == 1.0

    def test_no_base_scenario_returns_max_risk(self) -> None:
        ivs: dict[int, Any] = {1: {5: {"optimistic": _make_iv()}}}
        assert _compute_terminal_dependency(1, ivs, 5) == 1.0

    def test_negative_present_value_returns_max_risk(self) -> None:
        ivs: dict[int, Any] = {1: {5: {"base": _make_iv(present_value=-100.0)}}}
        assert _compute_terminal_dependency(1, ivs, 5) == 1.0


# ---------------------------------------------------------------------------
# rank_companies — structure
# ---------------------------------------------------------------------------

class TestRankCompaniesStructure:

    def test_returns_four_dataframes(self) -> None:
        config = _default_config()
        result = rank_companies(
            _make_companies(), _make_growth_stats(), _make_projections(),
            _make_ivs(), _make_weighted_scores(), _make_live_prices(),
            config,
        )
        assert len(result) == 4
        for df in result:
            assert isinstance(df, pd.DataFrame)

    def test_all_companies_with_prices_included(self) -> None:
        config = _default_config()
        growth, value, weighted, combined = rank_companies(
            _make_companies(), _make_growth_stats(), _make_projections(),
            _make_ivs(), _make_weighted_scores(), _make_live_prices(),
            config,
        )
        for df in (growth, value, weighted, combined):
            assert len(df) == 3

    def test_company_without_price_excluded(self) -> None:
        config = _default_config()
        prices = {"CO1": 10.0, "CO2": 20.0}  # CO3 missing
        growth, _, _, _ = rank_companies(
            _make_companies(), _make_growth_stats(), _make_projections(),
            _make_ivs(), _make_weighted_scores(), prices,
            config,
        )
        assert 3 not in growth.index

    def test_output_has_new_columns(self) -> None:
        config = _default_config()
        _, _, _, combined = rank_companies(
            _make_companies(), _make_growth_stats(), _make_projections(),
            _make_ivs(), _make_weighted_scores(), _make_live_prices(),
            config,
        )
        expected_cols = {
            "composite_iv_ratio", "pessimistic_iv_ratio",
            "base_iv_ratio", "optimistic_iv_ratio",
            "scenario_spread", "downside_exposure",
            "terminal_dependency", "fcf_reliability",
            "downside_exposure_score", "scenario_spread_score",
            "terminal_dependency_score", "fcf_reliability_score",
            "composite_safety", "risk_adjusted_score",
        }
        assert expected_cols.issubset(set(combined.columns))

    def test_old_columns_removed(self) -> None:
        config = _default_config()
        _, _, _, combined = rank_companies(
            _make_companies(), _make_growth_stats(), _make_projections(),
            _make_ivs(), _make_weighted_scores(), _make_live_prices(),
            config,
        )
        removed_cols = {
            "best_iv_ratio", "best_iv_scenario",
            "volatility_risk", "combined_risk", "avg_volatility",
        }
        assert removed_cols.isdisjoint(set(combined.columns))


# ---------------------------------------------------------------------------
# rank_companies — safety gate
# ---------------------------------------------------------------------------

class TestSafetyGate:

    def test_excludes_company_below_pessimistic_gate(self) -> None:
        """Company with pessimistic IV < current price is excluded."""
        config = _default_config()
        ivs = _make_ivs(3)
        # Set CO2's pessimistic IV below its price (20.0)
        ivs[2][5]["pessimistic"] = _make_iv(
            iv_per_share=5.0, scenario="pessimistic",
        )
        _, _, _, combined = rank_companies(
            _make_companies(), _make_growth_stats(), _make_projections(),
            ivs, _make_weighted_scores(), _make_live_prices(),
            config,
        )
        assert 2 not in combined.index

    def test_company_without_pessimistic_iv_excluded(self) -> None:
        """Company with no pessimistic scenario is excluded."""
        config = _default_config()
        ivs = _make_ivs(3)
        del ivs[2][5]["pessimistic"]
        _, _, _, combined = rank_companies(
            _make_companies(), _make_growth_stats(), _make_projections(),
            ivs, _make_weighted_scores(), _make_live_prices(),
            config,
        )
        assert 2 not in combined.index

    def test_company_at_exactly_one_passes(self) -> None:
        """Pessimistic IV ratio exactly 1.0 does NOT pass (< 1.0 test)."""
        config = _default_config()
        ivs = _make_ivs(1)
        # Set pessimistic IV exactly = price (10.0)
        ivs[1][5]["pessimistic"] = _make_iv(
            iv_per_share=10.0, scenario="pessimistic",
        )
        _, _, _, combined = rank_companies(
            _make_companies(1), _make_growth_stats(1), _make_projections(1),
            ivs, _make_weighted_scores(1), _make_live_prices(1),
            config,
        )
        # ratio = 10/10 = 1.0, not < 1.0, so passes
        assert 1 in combined.index


# ---------------------------------------------------------------------------
# rank_companies — sort orders
# ---------------------------------------------------------------------------

class TestRankCompaniesSortOrders:

    def test_growth_sorted_by_combined_growth_desc(self) -> None:
        config = _default_config()
        growth, _, _, _ = rank_companies(
            _make_companies(), _make_growth_stats(), _make_projections(),
            _make_ivs(), _make_weighted_scores(), _make_live_prices(),
            config,
        )
        values = growth["combined_growth"].tolist()
        assert values == sorted(values, reverse=True)

    def test_value_sorted_by_composite_iv_ratio_desc(self) -> None:
        config = _default_config()
        _, value, _, _ = rank_companies(
            _make_companies(), _make_growth_stats(), _make_projections(),
            _make_ivs(), _make_weighted_scores(), _make_live_prices(),
            config,
        )
        values = value["composite_iv_ratio"].tolist()
        assert values == sorted(values, reverse=True)

    def test_weighted_sorted_by_total_penalty_asc(self) -> None:
        config = _default_config()
        _, _, weighted, _ = rank_companies(
            _make_companies(), _make_growth_stats(), _make_projections(),
            _make_ivs(), _make_weighted_scores(), _make_live_prices(),
            config,
        )
        values = weighted["total_penalty"].tolist()
        assert values == sorted(values)

    def test_combined_sorted_by_risk_adjusted_rank_asc(self) -> None:
        config = _default_config()
        _, _, _, combined = rank_companies(
            _make_companies(), _make_growth_stats(), _make_projections(),
            _make_ivs(), _make_weighted_scores(), _make_live_prices(),
            config,
        )
        values = combined["risk_adjusted_rank"].tolist()
        assert values == sorted(values)


# ---------------------------------------------------------------------------
# rank_companies — ranking properties
# ---------------------------------------------------------------------------

class TestRankingProperties:

    def test_risk_adjusted_rank_monotone(self) -> None:
        """Higher risk_adjusted_score should get lower (= better) rank."""
        config = _default_config()
        _, _, _, combined = rank_companies(
            _make_companies(), _make_growth_stats(), _make_projections(),
            _make_ivs(), _make_weighted_scores(), _make_live_prices(),
            config,
        )
        by_score = combined.sort_values("risk_adjusted_score", ascending=False)
        ranks = by_score["risk_adjusted_rank"].tolist()
        assert ranks == sorted(ranks)

    def test_opportunity_rank_equals_risk_adjusted_rank(self) -> None:
        config = _default_config()
        _, _, _, combined = rank_companies(
            _make_companies(), _make_growth_stats(), _make_projections(),
            _make_ivs(), _make_weighted_scores(), _make_live_prices(),
            config,
        )
        assert (
            combined["opportunity_rank"] == combined["risk_adjusted_rank"]
        ).all()

    def test_divergence_penalty_binary(self) -> None:
        config = _default_config()
        _, _, _, combined = rank_companies(
            _make_companies(), _make_growth_stats(), _make_projections(),
            _make_ivs(), _make_weighted_scores(), _make_live_prices(),
            config,
        )
        assert set(combined["divergence_penalty"].unique()).issubset({0, 10})

    def test_composite_safety_bounded(self) -> None:
        """Composite safety should be between 0 and 1."""
        config = _default_config()
        _, _, _, combined = rank_companies(
            _make_companies(), _make_growth_stats(), _make_projections(),
            _make_ivs(), _make_weighted_scores(), _make_live_prices(),
            config,
        )
        assert all(0 <= s <= 1 for s in combined["composite_safety"])

    def test_score_columns_bounded_zero_one(self) -> None:
        """All *_score columns should be in [0, 1]."""
        config = _default_config()
        _, _, _, combined = rank_companies(
            _make_companies(), _make_growth_stats(), _make_projections(),
            _make_ivs(), _make_weighted_scores(), _make_live_prices(),
            config,
        )
        for col in (
            "downside_exposure_score", "scenario_spread_score",
            "terminal_dependency_score", "fcf_reliability_score",
        ):
            assert all(0 <= v <= 1 for v in combined[col]), f"{col} out of range"

    def test_risk_adjusted_score_is_return_times_safety(self) -> None:
        """risk_adjusted_score = total_expected_return * composite_safety."""
        config = _default_config()
        _, _, _, combined = rank_companies(
            _make_companies(), _make_growth_stats(), _make_projections(),
            _make_ivs(), _make_weighted_scores(), _make_live_prices(),
            config,
        )
        for eid in combined.index:
            expected = (
                combined.loc[eid, "total_expected_return"]
                * combined.loc[eid, "composite_safety"]
            )
            assert combined.loc[eid, "risk_adjusted_score"] == pytest.approx(
                expected, rel=1e-9,
            )


# ---------------------------------------------------------------------------
# rank_companies — edge cases
# ---------------------------------------------------------------------------

class TestRankingEdgeCases:

    def test_empty_companies(self) -> None:
        config = _default_config()
        companies = pd.DataFrame(
            columns=["symbol", "fcf", "market_cap", "debt_cash_ratio"],
        )
        companies.index.name = "entity_id"
        stats = pd.DataFrame(
            columns=[
                "fcf_growth_mean", "fcf_growth_std",
                "revenue_growth_mean", "revenue_growth_std",
                "combined_growth_mean", "growth_stability",
                "fcf_reliability",
            ],
        )
        stats.index.name = "entity_id"

        growth, value, weighted, combined = rank_companies(
            companies, stats, {}, {}, pd.DataFrame(), {}, config,
        )
        for df in (growth, value, weighted, combined):
            assert df.empty

    def test_no_live_prices_empty_result(self) -> None:
        config = _default_config()
        growth, _, _, _ = rank_companies(
            _make_companies(), _make_growth_stats(), _make_projections(),
            _make_ivs(), _make_weighted_scores(), {},
            config,
        )
        assert growth.empty

    def test_missing_columns_raises(self) -> None:
        companies = pd.DataFrame(
            {"symbol": ["A"]}, index=pd.Index([1], name="entity_id"),
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            rank_companies(
                companies, _make_growth_stats(), {}, {}, pd.DataFrame(),
                {}, _default_config(),
            )

    def test_missing_fcf_reliability_raises(self) -> None:
        stats = _make_growth_stats()
        stats = stats.drop(columns=["fcf_reliability"])
        with pytest.raises(ValueError, match="Missing required columns"):
            rank_companies(
                _make_companies(), stats, {}, {}, pd.DataFrame(),
                {}, _default_config(),
            )
