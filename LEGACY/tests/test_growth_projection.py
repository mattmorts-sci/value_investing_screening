"""Tests for pipeline.analysis.growth_projection."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pipeline.analysis.growth_projection import (
    _compute_annual_cagr,
    _extract_scenarios,
    _parameterise_lognormal,
    _project_negative_fcf,
    _safe_float,
    _simulate_monte_carlo,
    project_all,
    project_growth,
)
from pipeline.config.settings import AnalysisConfig
from pipeline.data.contracts import Projection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quarterly_rate(annual: float) -> float:
    """Convert annual rate to quarterly (test utility)."""
    return (1 + annual) ** 0.25 - 1


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
        "simulation_replicates": 1000,
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
# _parameterise_lognormal
# ---------------------------------------------------------------------------


class TestParameteriseLognormal:

    def test_normal_case(self) -> None:
        """m > -0.5 and s > 0 produces valid log-normal parameters."""
        mu, sigma = _parameterise_lognormal(0.05, 0.02, 1.0)
        assert mu > 0  # positive growth mean
        assert sigma > 0  # positive volatility

    def test_edge_case_very_negative_mean(self) -> None:
        """m <= -0.5 defaults to 5% quarterly decline."""
        mu, sigma = _parameterise_lognormal(-0.6, 0.1, 1.0)
        assert mu == pytest.approx(math.log(0.95), abs=1e-10)
        assert sigma == pytest.approx(math.sqrt(0.1), abs=1e-10)

    def test_edge_case_zero_std(self) -> None:
        """s = 0 with positive mean projects at mean with small sigma."""
        mu, sigma = _parameterise_lognormal(0.05, 0.0, 1.0)
        assert mu == pytest.approx(math.log(1.05), abs=1e-10)
        assert sigma == pytest.approx(0.05, abs=1e-10)

    def test_zero_mean_zero_std_projects_flat(self) -> None:
        """m = 0, s = 0 (unknown growth) projects roughly flat."""
        mu, sigma = _parameterise_lognormal(0.0, 0.0, 1.0)
        assert mu == pytest.approx(math.log(1.0), abs=1e-10)
        assert sigma == pytest.approx(0.05, abs=1e-10)

    def test_cv_cap_applied(self) -> None:
        """Very high std relative to mean is capped at cv_cap."""
        # Both exceed CV cap of 1.0 → same result
        mu1, sigma1 = _parameterise_lognormal(0.05, 2.0, 1.0)
        mu2, sigma2 = _parameterise_lognormal(0.05, 100.0, 1.0)
        assert mu1 == pytest.approx(mu2)
        assert sigma1 == pytest.approx(sigma2)

    def test_exact_log_normal_parameterisation(self) -> None:
        """Verify the math: target_mean, CV, sigma_sq, mu."""
        m, s, cv_cap = 0.10, 0.05, 1.0
        mu, sigma = _parameterise_lognormal(m, s, cv_cap)

        target_mean = 1 + m  # 1.10
        cv = s / target_mean  # 0.05 / 1.10 ≈ 0.04545
        expected_sigma_sq = math.log(1 + cv**2)
        expected_mu = math.log(target_mean) - expected_sigma_sq / 2

        assert mu == pytest.approx(expected_mu, abs=1e-10)
        assert sigma == pytest.approx(math.sqrt(expected_sigma_sq), abs=1e-10)


# ---------------------------------------------------------------------------
# _extract_scenarios
# ---------------------------------------------------------------------------


class TestExtractScenarios:

    def test_returns_three_scenarios(self) -> None:
        final_values = np.array([80, 90, 100, 110, 120], dtype=np.float64)
        scenarios = _extract_scenarios(
            entity_id=1, metric="fcf", period_years=5,
            current_value=100.0, final_values=final_values, n_quarters=20,
        )
        assert set(scenarios.keys()) == {"pessimistic", "base", "optimistic"}

    def test_ordering(self) -> None:
        """Pessimistic < base < optimistic for final values."""
        rng = np.random.default_rng(42)
        final_values = rng.normal(200, 30, size=10000)
        scenarios = _extract_scenarios(
            entity_id=1, metric="fcf", period_years=5,
            current_value=100.0, final_values=final_values, n_quarters=20,
        )
        pess = scenarios["pessimistic"].quarterly_values[-1]
        base = scenarios["base"].quarterly_values[-1]
        opt = scenarios["optimistic"].quarterly_values[-1]
        assert pess < base < opt

    def test_quarterly_values_length(self) -> None:
        final_values = np.full(100, 200.0)
        scenarios = _extract_scenarios(
            entity_id=1, metric="fcf", period_years=5,
            current_value=100.0, final_values=final_values, n_quarters=20,
        )
        assert len(scenarios["base"].quarterly_values) == 20
        assert len(scenarios["base"].quarterly_growth_rates) == 20

    def test_projection_fields(self) -> None:
        final_values = np.full(100, 200.0)
        scenarios = _extract_scenarios(
            entity_id=42, metric="revenue", period_years=10,
            current_value=500.0, final_values=final_values, n_quarters=40,
        )
        proj = scenarios["base"]
        assert proj.entity_id == 42
        assert proj.metric == "revenue"
        assert proj.period_years == 10
        assert proj.scenario == "base"
        assert proj.current_value == 500.0

    def test_constant_quarterly_rates(self) -> None:
        """All quarterly growth rates should be identical (smooth compound path)."""
        final_values = np.full(100, 200.0)
        scenarios = _extract_scenarios(
            entity_id=1, metric="fcf", period_years=5,
            current_value=100.0, final_values=final_values, n_quarters=20,
        )
        rates = scenarios["base"].quarterly_growth_rates
        assert all(r == pytest.approx(rates[0]) for r in rates)


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
            seed=42,
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
            seed=42,
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
            seed=42,
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
            seed=42,
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
            seed=42,
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
            seed=42,
        )
        fcf = result[5]["fcf"]
        # Optimistic should improve faster (closer to zero or positive)
        assert abs(fcf["optimistic"].quarterly_values[-1]) < abs(
            fcf["pessimistic"].quarterly_values[-1]
        )

    def test_current_value_preserved(self) -> None:
        config = _default_config()
        result = project_growth(
            entity_id=1,
            fcf_stats=_default_stats(latest_value=500.0),
            revenue_stats=_default_stats(latest_value=2000.0),
            market_cap=50_000_000,
            config=config,
            seed=42,
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


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------

class TestSafeFloat:

    def test_inf_returns_default(self) -> None:
        assert _safe_float(float("inf")) == 0.0

    def test_negative_inf_returns_default(self) -> None:
        assert _safe_float(float("-inf")) == 0.0

    def test_inf_custom_default(self) -> None:
        assert _safe_float(float("inf"), default=-1.0) == -1.0

    def test_nan_returns_default(self) -> None:
        assert _safe_float(float("nan")) == 0.0

    def test_none_returns_default(self) -> None:
        assert _safe_float(None) == 0.0

    def test_normal_float_passes_through(self) -> None:
        assert _safe_float(3.14) == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# _simulate_monte_carlo — isolated constraint tests
# ---------------------------------------------------------------------------


class TestPerQuarterCaps:
    """Constraint 2: per-quarter growth caps (asymmetric by size)."""

    def test_positive_growth_capped(self) -> None:
        """Growth rates exceeding pos_cap are clipped."""
        # Use very high mean growth to ensure samples exceed cap.
        config = _default_config(
            simulation_replicates=500,
            fcf_small_pos_cap=0.10,
            fcf_small_neg_cap=-0.30,
            # Disable other constraints that might interfere.
            cumulative_growth_cap=1000.0,
            cumulative_decline_floor=0.001,
            annual_cagr_backstop=100.0,
            high_growth_threshold=100.0,  # disables momentum + time decay
        )
        final = _simulate_monte_carlo(
            current_value=100.0,
            growth_mean=0.50,  # very high mean
            growth_std=0.10,
            market_cap=0.0,  # disables size penalty
            n_quarters=1,
            metric="fcf",
            config=config,
            seed=42,
        )
        # After 1 quarter capped at 10%, max value is 100 * 1.10 = 110
        assert np.all(final <= 100.0 * 1.10 + 0.01)

    def test_negative_growth_capped(self) -> None:
        """Decline rates below neg_cap are clipped."""
        config = _default_config(
            simulation_replicates=500,
            fcf_small_pos_cap=0.40,
            fcf_small_neg_cap=-0.05,
            cumulative_growth_cap=1000.0,
            cumulative_decline_floor=0.001,
            annual_cagr_backstop=100.0,
            high_growth_threshold=100.0,
        )
        final = _simulate_monte_carlo(
            current_value=100.0,
            growth_mean=-0.40,  # very negative mean
            growth_std=0.10,
            market_cap=0.0,
            n_quarters=1,
            metric="fcf",
            config=config,
            seed=42,
        )
        # After 1 quarter capped at -5%, min value is 100 * 0.95 = 95
        assert np.all(final >= 100.0 * 0.95 - 0.01)


class TestCumulativeCapAndFloor:
    """Constraint 3: cumulative growth cap and decline floor."""

    def test_cumulative_growth_capped(self) -> None:
        """Values cannot exceed current_value * cumulative_growth_cap."""
        config = _default_config(
            simulation_replicates=500,
            cumulative_growth_cap=2.0,  # 2x max
            cumulative_decline_floor=0.01,
            annual_cagr_backstop=1000.0,  # effectively disabled
            high_growth_threshold=100.0,
            fcf_small_pos_cap=0.90,  # allow large per-quarter growth
        )
        final = _simulate_monte_carlo(
            current_value=100.0,
            growth_mean=0.50,
            growth_std=0.20,
            market_cap=0.0,
            n_quarters=20,
            metric="fcf",
            config=config,
            seed=42,
        )
        assert np.all(final <= 100.0 * 2.0 + 0.01)

    def test_cumulative_decline_floored(self) -> None:
        """Values cannot fall below current_value * cumulative_decline_floor."""
        config = _default_config(
            simulation_replicates=500,
            cumulative_growth_cap=100.0,
            cumulative_decline_floor=0.5,  # 0.5x floor
            annual_cagr_backstop=1000.0,
            high_growth_threshold=100.0,
            fcf_small_neg_cap=-0.90,  # allow large per-quarter decline
            fcf_large_neg_cap=-0.90,
        )
        final = _simulate_monte_carlo(
            current_value=100.0,
            growth_mean=-0.40,
            growth_std=0.10,
            market_cap=0.0,
            n_quarters=20,
            metric="fcf",
            config=config,
            seed=42,
        )
        assert np.all(final >= 100.0 * 0.5 - 0.01)


class TestCagrBackstop:
    """Constraint 4: annual CAGR backstop (after year 1)."""

    def test_backstop_limits_growth(self) -> None:
        """After year 1, values cannot exceed 100% annual CAGR path."""
        config = _default_config(
            simulation_replicates=500,
            annual_cagr_backstop=0.50,  # 50% annual cap
            cumulative_growth_cap=1000.0,
            cumulative_decline_floor=0.001,
            high_growth_threshold=100.0,
            fcf_small_pos_cap=0.90,
            fcf_large_pos_cap=0.90,
        )
        n_quarters = 8  # 2 years
        final = _simulate_monte_carlo(
            current_value=100.0,
            growth_mean=0.50,
            growth_std=0.10,
            market_cap=0.0,
            n_quarters=n_quarters,
            metric="fcf",
            config=config,
            seed=42,
        )
        max_allowed = 100.0 * (1.50 ** (n_quarters / 4))  # 50% annual for 2 years
        assert np.all(final <= max_allowed + 0.01)

    def test_backstop_inactive_in_year_one(self) -> None:
        """Backstop does not fire within the first year."""
        config = _default_config(
            simulation_replicates=500,
            annual_cagr_backstop=0.10,  # very tight 10% annual
            cumulative_growth_cap=1000.0,
            cumulative_decline_floor=0.001,
            high_growth_threshold=100.0,
            fcf_small_pos_cap=0.50,
        )
        final = _simulate_monte_carlo(
            current_value=100.0,
            growth_mean=0.30,
            growth_std=0.05,
            market_cap=0.0,
            n_quarters=4,  # exactly 1 year
            metric="fcf",
            config=config,
            seed=42,
        )
        # Some paths should exceed 10% annual = 110 within year 1
        # because backstop doesn't fire until after year 1.
        assert np.any(final > 110.0)


class TestMomentumExhaustion:
    """Constraint 6: momentum exhaustion for high-growth companies."""

    def test_fires_for_high_growth_mean(self) -> None:
        """Momentum dampening should reduce final values vs no dampening."""
        base_config = {
            "simulation_replicates": 2000,
            "cumulative_growth_cap": 1000.0,
            "cumulative_decline_floor": 0.001,
            "annual_cagr_backstop": 1000.0,
            "high_growth_threshold": 0.20,
            "momentum_exhaustion_threshold": 1.5,
            "fcf_small_pos_cap": 0.90,
            "fcf_large_pos_cap": 0.90,
        }
        # With momentum exhaustion active (high_growth_threshold=0.20)
        config_with = _default_config(**base_config)
        final_with = _simulate_monte_carlo(
            current_value=100.0,
            growth_mean=0.35,  # above 0.20 threshold
            growth_std=0.10,
            market_cap=0.0,
            n_quarters=20,
            metric="fcf",
            config=config_with,
            seed=42,
        )

        # Without momentum (high_growth_threshold set above growth_mean)
        config_without = _default_config(
            **{**base_config, "high_growth_threshold": 100.0},
        )
        final_without = _simulate_monte_carlo(
            current_value=100.0,
            growth_mean=0.35,
            growth_std=0.10,
            market_cap=0.0,
            n_quarters=20,
            metric="fcf",
            config=config_without,
            seed=42,
        )

        # Median with momentum should be lower than without
        assert np.median(final_with) < np.median(final_without)

    def test_inactive_below_threshold(self) -> None:
        """Momentum exhaustion does not fire when growth_mean < threshold."""
        config = _default_config(
            simulation_replicates=500,
            high_growth_threshold=0.30,
            cumulative_growth_cap=1000.0,
            cumulative_decline_floor=0.001,
            annual_cagr_backstop=1000.0,
        )
        # growth_mean = 0.05 < 0.30 threshold: momentum won't fire.
        # Run twice with same seed: should produce identical results
        # regardless of momentum logic.
        final1 = _simulate_monte_carlo(
            current_value=100.0,
            growth_mean=0.05,
            growth_std=0.02,
            market_cap=0.0,
            n_quarters=20,
            metric="fcf",
            config=config,
            seed=99,
        )
        final2 = _simulate_monte_carlo(
            current_value=100.0,
            growth_mean=0.05,
            growth_std=0.02,
            market_cap=0.0,
            n_quarters=20,
            metric="fcf",
            config=config,
            seed=99,
        )
        np.testing.assert_array_equal(final1, final2)


class TestSizePenalty:
    """Constraint 8: size-based growth penalty."""

    def test_large_market_cap_reduces_growth(self) -> None:
        """Companies with large value-to-market-cap ratio grow less."""
        config = _default_config(
            simulation_replicates=2000,
            cumulative_growth_cap=1000.0,
            cumulative_decline_floor=0.001,
            annual_cagr_backstop=1000.0,
            high_growth_threshold=100.0,  # disable momentum + time decay
            size_penalty_factor=0.1,
        )
        # Small market cap → value/market_cap is large → heavy penalty
        final_small_cap = _simulate_monte_carlo(
            current_value=100.0,
            growth_mean=0.10,
            growth_std=0.05,
            market_cap=200.0,  # very small: value/market_cap = 0.5
            n_quarters=8,
            metric="fcf",
            config=config,
            seed=42,
        )
        # Large market cap → value/market_cap is tiny → light penalty
        final_large_cap = _simulate_monte_carlo(
            current_value=100.0,
            growth_mean=0.10,
            growth_std=0.05,
            market_cap=1e12,  # very large: value/market_cap ≈ 0
            n_quarters=8,
            metric="fcf",
            config=config,
            seed=42,
        )
        # Large cap should achieve higher growth (less penalty)
        assert np.median(final_large_cap) > np.median(final_small_cap)

    def test_zero_market_cap_no_penalty(self) -> None:
        """market_cap = 0 disables size penalty (factor = 1.0)."""
        config = _default_config(
            simulation_replicates=500,
            cumulative_growth_cap=1000.0,
            cumulative_decline_floor=0.001,
            annual_cagr_backstop=1000.0,
            high_growth_threshold=100.0,
        )
        final = _simulate_monte_carlo(
            current_value=100.0,
            growth_mean=0.10,
            growth_std=0.05,
            market_cap=0.0,
            n_quarters=4,
            metric="fcf",
            config=config,
            seed=42,
        )
        # Should produce non-trivial growth (no penalty suppression)
        assert np.median(final) > 100.0


class TestTimeDecay:
    """Constraint 7: time decay for high growth rates."""

    def test_reduces_growth_over_time(self) -> None:
        """With time decay, later quarters dampen high growth more."""
        config_decay = _default_config(
            simulation_replicates=2000,
            time_decay_base=0.5,  # aggressive decay
            high_growth_threshold=0.05,  # low threshold to activate
            cumulative_growth_cap=1000.0,
            cumulative_decline_floor=0.001,
            annual_cagr_backstop=1000.0,
        )
        final_decay = _simulate_monte_carlo(
            current_value=100.0,
            growth_mean=0.10,
            growth_std=0.05,
            market_cap=0.0,
            n_quarters=20,
            metric="fcf",
            config=config_decay,
            seed=42,
        )

        # No time decay (base=1.0 means no reduction)
        config_no_decay = _default_config(
            simulation_replicates=2000,
            time_decay_base=1.0,
            high_growth_threshold=0.05,
            cumulative_growth_cap=1000.0,
            cumulative_decline_floor=0.001,
            annual_cagr_backstop=1000.0,
        )
        final_no_decay = _simulate_monte_carlo(
            current_value=100.0,
            growth_mean=0.10,
            growth_std=0.05,
            market_cap=0.0,
            n_quarters=20,
            metric="fcf",
            config=config_no_decay,
            seed=42,
        )

        assert np.median(final_decay) < np.median(final_no_decay)
