"""Tests for pipeline.analysis.growth_projection."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest

from pipeline.analysis.growth_projection import (
    _compute_annual_cagr,
    _compute_fade_lambda,
    _fade_growth_rates,
    _project_negative_fcf,
    _project_values,
    _quarterly_rate,
    project_all,
    project_growth,
)
from pipeline.config.settings import AnalysisConfig
from pipeline.data.contracts import Projection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_config(**overrides: Any) -> AnalysisConfig:
    """AnalysisConfig with sensible test defaults."""
    kwargs: dict[str, Any] = {
        "projection_periods": (5,),
        "primary_period": 5,
        "equilibrium_growth_rate": 0.03,
        "base_fade_half_life_years": 2.5,
        "scenario_band_width": 1.0,
        "negative_fcf_improvement_cap": 0.15,
        "min_market_cap": 20_000_000,
        "quarters_per_year": 4,
    }
    kwargs.update(overrides)
    return AnalysisConfig(**kwargs)


def _default_stats(
    mean: float = 0.05,
    std: float = 0.02,
    latest_value: float = 100.0,
) -> dict[str, float]:
    return {"mean": mean, "std": std, "latest_value": latest_value}


# ---------------------------------------------------------------------------
# _quarterly_rate
# ---------------------------------------------------------------------------

class TestQuarterlyRate:

    def test_zero(self) -> None:
        assert _quarterly_rate(0.0) == 0.0

    def test_positive(self) -> None:
        # (1 + 0.10)^0.25 - 1 ≈ 0.02411
        result = _quarterly_rate(0.10)
        assert result == pytest.approx(0.02411, abs=1e-4)

    def test_compounds_back(self) -> None:
        annual = 0.08
        q = _quarterly_rate(annual)
        assert (1 + q) ** 4 - 1 == pytest.approx(annual, abs=1e-10)

    def test_extreme_negative_clamped(self) -> None:
        """Annual rate below -100% should be clamped, not crash."""
        result = _quarterly_rate(-1.5)
        # Clamped to -0.99 → (0.01)^0.25 - 1
        assert result == pytest.approx((0.01) ** 0.25 - 1, rel=1e-6)


# ---------------------------------------------------------------------------
# _compute_fade_lambda
# ---------------------------------------------------------------------------

class TestComputeFadeLambda:

    def test_base_case(self) -> None:
        # lambda = ln(2) / (2.5 * 4) = ln(2) / 10
        result = _compute_fade_lambda(2.5, 20_000_000, 20_000_000)
        expected = math.log(2) / 10
        # size_ratio = 1, so size_adj = 0
        assert result == pytest.approx(expected, rel=1e-6)

    def test_larger_company_fades_faster(self) -> None:
        small = _compute_fade_lambda(2.5, 20_000_000, 20_000_000)
        large = _compute_fade_lambda(2.5, 200_000_000, 20_000_000)
        assert large > small

    def test_below_min_market_cap(self) -> None:
        # size_ratio < 1 → size_adj = 0
        result = _compute_fade_lambda(2.5, 10_000_000, 20_000_000)
        base = math.log(2) / 10
        assert result == pytest.approx(base, rel=1e-6)

    def test_zero_market_cap(self) -> None:
        result = _compute_fade_lambda(2.5, 0, 20_000_000)
        base = math.log(2) / 10
        assert result == pytest.approx(base, rel=1e-6)


# ---------------------------------------------------------------------------
# _fade_growth_rates
# ---------------------------------------------------------------------------

class TestFadeGrowthRates:

    def test_converges_to_equilibrium(self) -> None:
        g_eq_q = _quarterly_rate(0.03)
        fade_lambda = math.log(2) / 10
        rates = _fade_growth_rates(0.10, g_eq_q, fade_lambda, 200)
        # After 200 quarters (50 years), should be very close to g_eq_q
        assert rates[-1] == pytest.approx(g_eq_q, abs=1e-6)

    def test_starts_near_g_0(self) -> None:
        g_0 = 0.08
        g_eq_q = _quarterly_rate(0.03)
        fade_lambda = math.log(2) / 10
        rates = _fade_growth_rates(g_0, g_eq_q, fade_lambda, 20)
        # First rate should be close to g_0 (only 1 quarter of decay)
        assert rates[0] == pytest.approx(
            g_eq_q + (g_0 - g_eq_q) * math.exp(-fade_lambda), rel=1e-6,
        )

    def test_monotonically_decreasing_when_g0_above_eq(self) -> None:
        g_eq_q = _quarterly_rate(0.03)
        rates = _fade_growth_rates(0.10, g_eq_q, 0.1, 20)
        for i in range(1, len(rates)):
            assert rates[i] < rates[i - 1]

    def test_monotonically_increasing_when_g0_below_eq(self) -> None:
        g_eq_q = _quarterly_rate(0.03)
        rates = _fade_growth_rates(-0.02, g_eq_q, 0.1, 20)
        for i in range(1, len(rates)):
            assert rates[i] > rates[i - 1]

    def test_correct_count(self) -> None:
        rates = _fade_growth_rates(0.05, 0.01, 0.1, 40)
        assert len(rates) == 40


# ---------------------------------------------------------------------------
# _project_values
# ---------------------------------------------------------------------------

class TestProjectValues:

    def test_constant_growth(self) -> None:
        values = _project_values(100.0, [0.10, 0.10, 0.10])
        assert values[0] == pytest.approx(110.0)
        assert values[1] == pytest.approx(121.0)
        assert values[2] == pytest.approx(133.1)

    def test_zero_growth(self) -> None:
        values = _project_values(100.0, [0.0, 0.0])
        assert all(v == pytest.approx(100.0) for v in values)

    def test_zero_start(self) -> None:
        values = _project_values(0.0, [0.10, 0.10])
        assert all(v == pytest.approx(0.0) for v in values)


# ---------------------------------------------------------------------------
# _project_negative_fcf
# ---------------------------------------------------------------------------

class TestProjectNegativeFcf:

    def test_moves_toward_zero(self) -> None:
        _rates, values = _project_negative_fcf(
            current_value=-100.0,
            revenue_growth_mean=0.05,
            n_quarters=20,
            improvement_cap=0.15,
            g_eq_q=_quarterly_rate(0.03),
            fade_lambda=math.log(2) / 10,
        )
        # All values should have smaller absolute magnitude than start
        for v in values:
            assert abs(v) < 100.0

    def test_monotonic_improvement_in_negative_phase(self) -> None:
        _rates, values = _project_negative_fcf(
            current_value=-100.0,
            revenue_growth_mean=0.05,
            n_quarters=10,
            improvement_cap=0.15,
            g_eq_q=_quarterly_rate(0.03),
            fade_lambda=math.log(2) / 10,
        )
        # Filter to just the negative values
        neg_values = [v for v in values if v < 0]
        for i in range(1, len(neg_values)):
            assert neg_values[i] > neg_values[i - 1]  # Less negative

    def test_eventually_crosses_zero(self) -> None:
        """With enough quarters, negative FCF should transition to positive."""
        _rates, values = _project_negative_fcf(
            current_value=-100.0,
            revenue_growth_mean=0.10,
            n_quarters=60,
            improvement_cap=0.15,
            g_eq_q=_quarterly_rate(0.03),
            fade_lambda=math.log(2) / 10,
        )
        assert any(v > 0 for v in values)

    def test_uses_revenue_growth_for_improvement_rate(self) -> None:
        """Higher revenue growth = faster improvement."""
        _, values_slow = _project_negative_fcf(
            current_value=-100.0,
            revenue_growth_mean=0.02,
            n_quarters=20,
            improvement_cap=0.15,
            g_eq_q=_quarterly_rate(0.03),
            fade_lambda=math.log(2) / 10,
        )
        _, values_fast = _project_negative_fcf(
            current_value=-100.0,
            revenue_growth_mean=0.10,
            n_quarters=20,
            improvement_cap=0.15,
            g_eq_q=_quarterly_rate(0.03),
            fade_lambda=math.log(2) / 10,
        )
        # Fast improvement should get closer to zero
        assert abs(values_fast[-1]) < abs(values_slow[-1])

    def test_declining_revenue_uses_slower_default(self) -> None:
        _, values = _project_negative_fcf(
            current_value=-100.0,
            revenue_growth_mean=-0.05,
            n_quarters=20,
            improvement_cap=0.15,
            g_eq_q=_quarterly_rate(0.03),
            fade_lambda=math.log(2) / 10,
        )
        # Should still improve (declining revenue → slower rate, not zero)
        assert abs(values[-1]) < 100.0

    def test_improvement_capped(self) -> None:
        """Revenue growth above cap should be capped."""
        _, values_capped = _project_negative_fcf(
            current_value=-100.0,
            revenue_growth_mean=0.50,
            n_quarters=10,
            improvement_cap=0.15,
            g_eq_q=_quarterly_rate(0.03),
            fade_lambda=math.log(2) / 10,
        )
        _, values_at_cap = _project_negative_fcf(
            current_value=-100.0,
            revenue_growth_mean=0.15,
            n_quarters=10,
            improvement_cap=0.15,
            g_eq_q=_quarterly_rate(0.03),
            fade_lambda=math.log(2) / 10,
        )
        # Both should produce the same result (capped at 0.15)
        for a, b in zip(values_capped, values_at_cap, strict=True):
            assert a == pytest.approx(b, rel=1e-6)

    def test_correct_length(self) -> None:
        rates, values = _project_negative_fcf(
            current_value=-50.0,
            revenue_growth_mean=0.05,
            n_quarters=20,
            improvement_cap=0.15,
            g_eq_q=_quarterly_rate(0.03),
            fade_lambda=math.log(2) / 10,
        )
        assert len(rates) == 20
        assert len(values) == 20


# ---------------------------------------------------------------------------
# _compute_annual_cagr
# ---------------------------------------------------------------------------

class TestComputeAnnualCagr:

    def test_both_positive(self) -> None:
        # 100 -> 200 over 5 years = ~14.87% annual
        result = _compute_annual_cagr(100.0, 200.0, 5)
        assert result == pytest.approx(0.1487, abs=1e-3)

    def test_negative_to_positive(self) -> None:
        assert _compute_annual_cagr(-100.0, 50.0, 5) == 1.0

    def test_both_negative_improving(self) -> None:
        result = _compute_annual_cagr(-100.0, -50.0, 5)
        assert result == pytest.approx(0.1)  # 50% improvement / 5 years

    def test_both_negative_worsening(self) -> None:
        assert _compute_annual_cagr(-50.0, -100.0, 5) == -0.5

    def test_positive_to_negative(self) -> None:
        assert _compute_annual_cagr(100.0, -50.0, 5) == -0.9

    def test_zero_period(self) -> None:
        assert _compute_annual_cagr(100.0, 200.0, 0) == 0.0

    def test_zero_values(self) -> None:
        assert _compute_annual_cagr(0.0, 0.0, 5) == 0.0


# ---------------------------------------------------------------------------
# project_growth (integration)
# ---------------------------------------------------------------------------

class TestProjectGrowth:

    def test_returns_all_periods_and_scenarios(self) -> None:
        config = _default_config(projection_periods=(5, 10))
        result = project_growth(
            entity_id=1,
            fcf_stats=_default_stats(),
            revenue_stats=_default_stats(),
            market_cap=50_000_000,
            config=config,
        )
        assert set(result.keys()) == {5, 10}
        for period in (5, 10):
            assert set(result[period].keys()) == {"fcf", "revenue"}
            for metric in ("fcf", "revenue"):
                assert set(result[period][metric].keys()) == {
                    "base", "optimistic", "pessimistic",
                }

    def test_projections_are_projection_instances(self) -> None:
        config = _default_config()
        result = project_growth(
            entity_id=42,
            fcf_stats=_default_stats(),
            revenue_stats=_default_stats(),
            market_cap=50_000_000,
            config=config,
        )
        proj = result[5]["fcf"]["base"]
        assert isinstance(proj, Projection)
        assert proj.entity_id == 42
        assert proj.metric == "fcf"
        assert proj.period_years == 5
        assert proj.scenario == "base"

    def test_quarterly_values_length(self) -> None:
        config = _default_config(projection_periods=(5,))
        result = project_growth(
            entity_id=1,
            fcf_stats=_default_stats(),
            revenue_stats=_default_stats(),
            market_cap=50_000_000,
            config=config,
        )
        proj = result[5]["fcf"]["base"]
        assert len(proj.quarterly_growth_rates) == 20  # 5 years * 4
        assert len(proj.quarterly_values) == 20

    def test_scenarios_bracket_base(self) -> None:
        """Pessimistic < base < optimistic for final values."""
        config = _default_config()
        result = project_growth(
            entity_id=1,
            fcf_stats=_default_stats(mean=0.05, std=0.02),
            revenue_stats=_default_stats(mean=0.03, std=0.01),
            market_cap=50_000_000,
            config=config,
        )
        fcf = result[5]["fcf"]
        pess = fcf["pessimistic"].quarterly_values[-1]
        base = fcf["base"].quarterly_values[-1]
        opt = fcf["optimistic"].quarterly_values[-1]
        assert pess < base < opt

    def test_negative_fcf_handled(self) -> None:
        config = _default_config()
        result = project_growth(
            entity_id=1,
            fcf_stats=_default_stats(mean=-0.05, std=0.02, latest_value=-100.0),
            revenue_stats=_default_stats(mean=0.05, std=0.01),
            market_cap=50_000_000,
            config=config,
        )
        proj = result[5]["fcf"]["base"]
        # Should move toward zero
        assert abs(proj.quarterly_values[-1]) < abs(proj.current_value)

    def test_negative_fcf_scenarios_differ(self) -> None:
        config = _default_config()
        result = project_growth(
            entity_id=1,
            fcf_stats=_default_stats(mean=-0.05, std=0.02, latest_value=-100.0),
            revenue_stats=_default_stats(mean=0.05, std=0.03),
            market_cap=50_000_000,
            config=config,
        )
        fcf = result[5]["fcf"]
        # Optimistic should improve faster (closer to zero or positive)
        assert abs(fcf["optimistic"].quarterly_values[-1]) < abs(
            fcf["pessimistic"].quarterly_values[-1]
        )

    def test_fade_converges_to_equilibrium(self) -> None:
        """Over a very long horizon, growth rates approach equilibrium."""
        config = _default_config(projection_periods=(50,))
        result = project_growth(
            entity_id=1,
            fcf_stats=_default_stats(mean=0.10, std=0.02),
            revenue_stats=_default_stats(mean=0.05, std=0.01),
            market_cap=50_000_000,
            config=config,
        )
        g_eq_q = _quarterly_rate(0.03)
        final_rate = result[50]["fcf"]["base"].quarterly_growth_rates[-1]
        assert final_rate == pytest.approx(g_eq_q, abs=1e-4)

    def test_current_value_preserved(self) -> None:
        config = _default_config()
        result = project_growth(
            entity_id=1,
            fcf_stats=_default_stats(latest_value=500.0),
            revenue_stats=_default_stats(latest_value=2000.0),
            market_cap=50_000_000,
            config=config,
        )
        assert result[5]["fcf"]["base"].current_value == 500.0
        assert result[5]["revenue"]["base"].current_value == 2000.0


# ---------------------------------------------------------------------------
# project_all
# ---------------------------------------------------------------------------

class TestProjectAll:

    def test_projects_all_companies(self) -> None:
        config = _default_config()
        companies = pd.DataFrame(
            {
                "fcf": [100.0, 200.0],
                "revenue": [1000.0, 2000.0],
                "market_cap": [50e6, 100e6],
            },
            index=pd.Index([10, 20], name="entity_id"),
        )
        growth_stats = pd.DataFrame(
            {
                "fcf_growth_mean": [0.05, 0.03],
                "fcf_growth_std": [0.02, 0.01],
                "revenue_growth_mean": [0.04, 0.02],
                "revenue_growth_std": [0.01, 0.005],
            },
            index=pd.Index([10, 20], name="entity_id"),
        )
        result = project_all(companies, growth_stats, config)
        assert set(result.keys()) == {10, 20}
        for eid in (10, 20):
            assert 5 in result[eid]

    def test_nan_growth_stats_treated_as_zero(self) -> None:
        """NaN growth stats should produce valid (non-NaN) projections."""
        config = _default_config()
        companies = pd.DataFrame(
            {
                "fcf": [100.0],
                "revenue": [1000.0],
                "market_cap": [50e6],
            },
            index=pd.Index([10], name="entity_id"),
        )
        growth_stats = pd.DataFrame(
            {
                "fcf_growth_mean": [float("nan")],
                "fcf_growth_std": [float("nan")],
                "revenue_growth_mean": [float("nan")],
                "revenue_growth_std": [float("nan")],
            },
            index=pd.Index([10], name="entity_id"),
        )
        result = project_all(companies, growth_stats, config)
        proj = result[10][5]["fcf"]["base"]
        # All values should be finite (no NaN propagation)
        assert all(math.isfinite(v) for v in proj.quarterly_values)
        assert all(math.isfinite(r) for r in proj.quarterly_growth_rates)

    def test_empty_companies(self) -> None:
        config = _default_config()
        companies = pd.DataFrame(
            columns=["fcf", "revenue", "market_cap"],
        )
        companies.index.name = "entity_id"
        growth_stats = pd.DataFrame(
            columns=[
                "fcf_growth_mean", "fcf_growth_std",
                "revenue_growth_mean", "revenue_growth_std",
            ],
        )
        growth_stats.index.name = "entity_id"
        result = project_all(companies, growth_stats, config)
        assert result == {}
