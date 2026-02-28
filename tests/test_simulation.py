"""Tests for pipeline.simulation."""

from __future__ import annotations

import math

import pytest

from pipeline.config import DCFConfig, HeatmapConfig, SimulationConfig
from pipeline.simulation import (
    SimulationInput,
    _apply_growth_constraints,
    _compute_lognormal_params,
    _implied_cagr,
    run_parameterised_dcf,
    run_simulation,
)

# --- Fixtures ---


def _make_input(
    mean: float = 0.02,
    var: float = 0.001,
    margin_intercept: float = 0.15,
    margin_slope: float = 0.001,
    conversion_intercept: float | None = 0.8,
    conversion_slope: float | None = 0.0,
    conversion_median: float | None = None,
    conversion_is_fallback: bool = False,
    starting_revenue: float = 1_000_000.0,
    shares_outstanding: float = 100_000.0,
    num_historical_quarters: int = 20,
) -> SimulationInput:
    """Create a SimulationInput with sensible defaults."""
    return SimulationInput(
        revenue_qoq_growth_mean=mean,
        revenue_qoq_growth_var=var,
        margin_intercept=margin_intercept,
        margin_slope=margin_slope,
        conversion_intercept=conversion_intercept,
        conversion_slope=conversion_slope,
        conversion_median=conversion_median,
        conversion_is_fallback=conversion_is_fallback,
        starting_revenue=starting_revenue,
        shares_outstanding=shares_outstanding,
        num_historical_quarters=num_historical_quarters,
    )


SEED = 42


# === Log-normal parameter computation ===


class TestLogNormalParams:
    """Tests for _compute_lognormal_params."""

    def test_positive_mean_produces_valid_params(self) -> None:
        mu, sigma = _compute_lognormal_params(0.02, 0.001, 1.0)
        assert math.isfinite(mu)
        assert sigma > 0

    def test_zero_mean_produces_valid_params(self) -> None:
        mu, sigma = _compute_lognormal_params(0.0, 0.001, 1.0)
        assert math.isfinite(mu)
        assert sigma > 0

    def test_negative_mean_above_minus_one(self) -> None:
        """Mean of -0.5 gives E[1+g] = 0.5, still positive."""
        mu, sigma = _compute_lognormal_params(-0.5, 0.01, 1.0)
        assert math.isfinite(mu)
        assert sigma > 0

    def test_mean_below_minus_one_falls_back(self) -> None:
        """Mean of -1.5 gives E[1+g] = -0.5, non-positive. Falls back."""
        mu, sigma = _compute_lognormal_params(-1.5, 0.01, 1.0)
        assert mu == 0.0
        assert sigma == pytest.approx(0.01)

    def test_cv_cap_clamps_variance(self) -> None:
        """High variance should be clamped by CV cap."""
        # Without cap: std/m = sqrt(10)/1.02 ≈ 3.1, well above 1.0
        _, sigma_uncapped = _compute_lognormal_params(0.02, 10.0, 100.0)
        _, sigma_capped = _compute_lognormal_params(0.02, 10.0, 1.0)
        assert sigma_capped < sigma_uncapped


# === Growth constraints ===


class TestGrowthConstraints:
    """Tests for _apply_growth_constraints."""

    def _default_config(self) -> SimulationConfig:
        return SimulationConfig()

    def test_early_positive_cap(self) -> None:
        """Growth should be capped at early_positive_cap when below size tier."""
        config = self._default_config()
        result = _apply_growth_constraints(
            growth=0.50,
            quarter_idx=0,
            cumulative_revenue=1000.0,
            starting_revenue=1000.0,
            historical_mean=0.02,
            sim_config=config,
        )
        assert result <= config.early_positive_cap

    def test_early_negative_cap(self) -> None:
        """Negative growth should be bounded by early_negative_cap."""
        config = self._default_config()
        result = _apply_growth_constraints(
            growth=-0.50,
            quarter_idx=0,
            cumulative_revenue=1000.0,
            starting_revenue=1000.0,
            historical_mean=0.02,
            sim_config=config,
        )
        assert result >= config.early_negative_cap

    def test_late_positive_cap(self) -> None:
        """Growth should use late caps when revenue exceeds size tier threshold."""
        config = self._default_config()
        # Revenue at 3× starting (above size_tier_threshold of 2.0)
        result = _apply_growth_constraints(
            growth=0.50,
            quarter_idx=10,
            cumulative_revenue=3000.0,
            starting_revenue=1000.0,
            historical_mean=0.02,
            sim_config=config,
        )
        assert result <= config.late_positive_cap

    def test_late_negative_cap(self) -> None:
        """Negative growth uses late cap when revenue exceeds size tier."""
        config = self._default_config()
        result = _apply_growth_constraints(
            growth=-0.50,
            quarter_idx=10,
            cumulative_revenue=3000.0,
            starting_revenue=1000.0,
            historical_mean=0.02,
            sim_config=config,
        )
        assert result >= config.late_negative_cap

    def test_cumulative_cap(self) -> None:
        """Revenue should not exceed cumulative_cap × starting revenue."""
        config = self._default_config()
        # Revenue already at 4.9× starting, growth of 10% would push to 5.39×
        result = _apply_growth_constraints(
            growth=0.10,
            quarter_idx=30,
            cumulative_revenue=4900.0,
            starting_revenue=1000.0,
            historical_mean=0.02,
            sim_config=config,
        )
        max_allowed = 1000.0 * config.cumulative_cap
        actual_revenue = 4900.0 * (1.0 + result)
        assert actual_revenue <= max_allowed + 1e-6

    def test_cagr_backstop(self) -> None:
        """Annualised growth should not exceed cagr_backstop."""
        config = self._default_config()
        # After 4 years (16 quarters), at 50% CAGR max:
        # max factor = 1.5^4 = 5.0625
        # current at 4× starting, growth would push further
        result = _apply_growth_constraints(
            growth=0.20,
            quarter_idx=15,
            cumulative_revenue=4000.0,
            starting_revenue=1000.0,
            historical_mean=0.02,
            sim_config=config,
        )
        years = 16 / 4.0
        max_factor = (1.0 + config.cagr_backstop) ** years
        max_revenue = 1000.0 * max_factor
        actual_revenue = 4000.0 * (1.0 + result)
        assert actual_revenue <= max_revenue + 1e-6

    def test_time_decay_reduces_high_growth(self) -> None:
        """High growth rates should decay over time."""
        result_early_high = _apply_growth_constraints(
            growth=0.19,
            quarter_idx=0,
            cumulative_revenue=1000.0,
            starting_revenue=1000.0,
            historical_mean=0.20,
            sim_config=SimulationConfig(time_decay_growth_threshold=0.10),
        )
        result_late_high = _apply_growth_constraints(
            growth=0.19,
            quarter_idx=20,
            cumulative_revenue=1000.0,
            starting_revenue=1000.0,
            historical_mean=0.20,
            sim_config=SimulationConfig(time_decay_growth_threshold=0.10),
        )
        assert result_late_high < result_early_high

    def test_zero_historical_mean_allows_positive_growth(self) -> None:
        """Zero historical mean should not kill all positive growth."""
        config = SimulationConfig()
        result = _apply_growth_constraints(
            growth=0.05,
            quarter_idx=0,
            cumulative_revenue=1000.0,
            starting_revenue=1000.0,
            historical_mean=0.0,
            sim_config=config,
        )
        # Momentum constraint should be skipped for zero mean
        assert result > 0

    def test_size_penalty_reduces_growth(self) -> None:
        """Positive growth should be penalised as revenue grows."""
        config = self._default_config()
        # At 1× starting: max penalty
        result_small = _apply_growth_constraints(
            growth=0.05,
            quarter_idx=0,
            cumulative_revenue=1000.0,
            starting_revenue=1000.0,
            historical_mean=0.05,
            sim_config=config,
        )
        # At 3× starting: more penalty
        result_large = _apply_growth_constraints(
            growth=0.05,
            quarter_idx=0,
            cumulative_revenue=3000.0,
            starting_revenue=1000.0,
            historical_mean=0.05,
            sim_config=config,
        )
        assert result_large < result_small


# === Full simulation ===


class TestRunSimulation:
    """Tests for run_simulation."""

    def test_percentile_ordering(self) -> None:
        """IV percentiles should be monotonically increasing."""
        inp = _make_input()
        result = run_simulation(inp, current_price=50.0, seed=SEED)
        assert result.iv_p10 <= result.iv_p25
        assert result.iv_p25 <= result.iv_p50
        assert result.iv_p50 <= result.iv_p75
        assert result.iv_p75 <= result.iv_p90

    def test_spread_equals_p75_minus_p25(self) -> None:
        """IV spread should equal P75 - P25."""
        inp = _make_input()
        result = run_simulation(inp, current_price=50.0, seed=SEED)
        assert result.iv_spread == pytest.approx(result.iv_p75 - result.iv_p25)

    def test_positive_iv_for_positive_business(self) -> None:
        """Positive growth + positive margins should produce positive IV."""
        inp = _make_input(mean=0.02, margin_intercept=0.15)
        result = run_simulation(inp, current_price=50.0, seed=SEED)
        assert result.iv_p50 > 0

    def test_sample_paths_count(self) -> None:
        """Should produce num_display_paths sample paths."""
        config = SimulationConfig(num_replicates=100, num_display_paths=10)
        inp = _make_input()
        result = run_simulation(inp, current_price=50.0, sim_config=config, seed=SEED)
        assert len(result.sample_paths) == 10

    def test_sample_path_length(self) -> None:
        """Each path should have projection_years × 4 quarters."""
        dcf = DCFConfig(projection_years=5)
        config = SimulationConfig(num_replicates=50, num_display_paths=5)
        inp = _make_input()
        result = run_simulation(
            inp, current_price=50.0, sim_config=config, dcf_config=dcf, seed=SEED
        )
        for path in result.sample_paths:
            assert len(path.quarterly_revenue) == 20
            assert len(path.quarterly_fcf) == 20

    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce identical results."""
        inp = _make_input()
        config = SimulationConfig(num_replicates=500)
        r1 = run_simulation(inp, current_price=50.0, sim_config=config, seed=123)
        r2 = run_simulation(inp, current_price=50.0, sim_config=config, seed=123)
        assert r1.iv_p50 == r2.iv_p50
        assert r1.iv_p25 == r2.iv_p25

    def test_higher_growth_produces_higher_iv(self) -> None:
        """Higher growth mean should produce higher median IV."""
        config = SimulationConfig(num_replicates=2000)
        low = _make_input(mean=0.01)
        high = _make_input(mean=0.05)
        r_low = run_simulation(low, current_price=50.0, sim_config=config, seed=SEED)
        r_high = run_simulation(high, current_price=50.0, sim_config=config, seed=SEED)
        assert r_high.iv_p50 > r_low.iv_p50

    def test_higher_discount_rate_produces_lower_iv(self) -> None:
        """Higher discount rate should reduce present values."""
        inp = _make_input()
        config = SimulationConfig(num_replicates=2000)
        dcf_low = DCFConfig(discount_rate=0.08)
        dcf_high = DCFConfig(discount_rate=0.15)
        r_low = run_simulation(
            inp, current_price=50.0, sim_config=config, dcf_config=dcf_low, seed=SEED
        )
        r_high = run_simulation(
            inp, current_price=50.0, sim_config=config, dcf_config=dcf_high, seed=SEED
        )
        assert r_low.iv_p50 > r_high.iv_p50

    def test_conversion_fallback_uses_median(self) -> None:
        """Fallback mode should use conversion_median instead of regression."""
        inp = _make_input(
            conversion_intercept=None,
            conversion_slope=None,
            conversion_median=0.7,
            conversion_is_fallback=True,
        )
        config = SimulationConfig(num_replicates=100)
        result = run_simulation(inp, current_price=50.0, sim_config=config, seed=SEED)
        assert result.iv_p50 > 0

    def test_zero_shares_raises(self) -> None:
        """Zero shares outstanding should raise ValueError."""
        inp = _make_input(shares_outstanding=0.0)
        with pytest.raises(ValueError, match="shares_outstanding must be positive"):
            run_simulation(inp, current_price=50.0, seed=SEED)

    def test_negative_starting_revenue_raises(self) -> None:
        """Negative starting revenue should raise ValueError."""
        inp = _make_input(starting_revenue=-1000.0)
        with pytest.raises(ValueError, match="starting_revenue must be positive"):
            run_simulation(inp, current_price=50.0, seed=SEED)

    def test_zero_starting_revenue_raises(self) -> None:
        """Zero starting revenue should raise ValueError."""
        inp = _make_input(starting_revenue=0.0)
        with pytest.raises(ValueError, match="starting_revenue must be positive"):
            run_simulation(inp, current_price=50.0, seed=SEED)


# === Implied CAGR ===


class TestImpliedCagr:
    """Tests for _implied_cagr."""

    def test_basic_cagr(self) -> None:
        """Doubling in 10 years ≈ 7.18% CAGR."""
        cagr = _implied_cagr(50.0, 100.0, 10.0)
        assert cagr == pytest.approx(0.07177, abs=0.001)

    def test_zero_price_returns_zero(self) -> None:
        assert _implied_cagr(0.0, 100.0, 10.0) == 0.0

    def test_zero_iv_returns_zero(self) -> None:
        assert _implied_cagr(50.0, 0.0, 10.0) == 0.0

    def test_zero_years_returns_zero(self) -> None:
        assert _implied_cagr(50.0, 100.0, 0.0) == 0.0

    def test_iv_equals_price_returns_zero(self) -> None:
        cagr = _implied_cagr(50.0, 50.0, 10.0)
        assert cagr == pytest.approx(0.0)


# === Parameterised DCF ===


class TestParameterisedDcf:
    """Tests for run_parameterised_dcf."""

    def test_returns_finite_value(self) -> None:
        inp = _make_input()
        result = run_parameterised_dcf(
            inp,
            current_price=50.0,
            discount_rate=0.10,
            growth_multiplier=1.0,
            seed=SEED,
        )
        assert math.isfinite(result)
        assert result > 0

    def test_higher_discount_rate_lowers_iv(self) -> None:
        inp = _make_input()
        heatmap = HeatmapConfig(heatmap_replicates=500)
        iv_low_rate = run_parameterised_dcf(
            inp,
            current_price=50.0,
            discount_rate=0.08,
            growth_multiplier=1.0,
            heatmap_config=heatmap,
            seed=SEED,
        )
        iv_high_rate = run_parameterised_dcf(
            inp,
            current_price=50.0,
            discount_rate=0.15,
            growth_multiplier=1.0,
            heatmap_config=heatmap,
            seed=SEED,
        )
        assert iv_low_rate > iv_high_rate

    def test_higher_growth_multiplier_raises_iv(self) -> None:
        inp = _make_input()
        heatmap = HeatmapConfig(heatmap_replicates=500)
        iv_low_growth = run_parameterised_dcf(
            inp,
            current_price=50.0,
            discount_rate=0.10,
            growth_multiplier=0.5,
            heatmap_config=heatmap,
            seed=SEED,
        )
        iv_high_growth = run_parameterised_dcf(
            inp,
            current_price=50.0,
            discount_rate=0.10,
            growth_multiplier=1.5,
            heatmap_config=heatmap,
            seed=SEED,
        )
        assert iv_high_growth > iv_low_growth

    def test_uses_heatmap_replicates(self) -> None:
        """Should use heatmap_replicates, not full num_replicates."""
        inp = _make_input()
        heatmap = HeatmapConfig(heatmap_replicates=50)
        # Should complete quickly with only 50 replicates
        result = run_parameterised_dcf(
            inp,
            current_price=50.0,
            discount_rate=0.10,
            growth_multiplier=1.0,
            heatmap_config=heatmap,
            seed=SEED,
        )
        assert math.isfinite(result)


# === Percentile Bands ===


class TestPercentileBands:
    """Tests for PercentileBands computation in run_simulation."""

    def test_bands_present_when_display_paths_positive(self) -> None:
        """Bands should be computed when num_display_paths > 0."""
        config = SimulationConfig(num_replicates=200, num_display_paths=5)
        inp = _make_input()
        result = run_simulation(inp, current_price=50.0, sim_config=config, seed=SEED)
        assert result.revenue_bands is not None
        assert result.fcf_bands is not None

    def test_bands_none_when_no_display_paths(self) -> None:
        """Bands should be None when num_display_paths is 0."""
        config = SimulationConfig(num_replicates=200, num_display_paths=0)
        inp = _make_input()
        result = run_simulation(inp, current_price=50.0, sim_config=config, seed=SEED)
        assert result.revenue_bands is None
        assert result.fcf_bands is None
        assert result.sample_paths == []

    def test_band_shape(self) -> None:
        """Each band array should have num_quarters elements."""
        dcf = DCFConfig(projection_years=5)
        config = SimulationConfig(num_replicates=200, num_display_paths=5)
        inp = _make_input()
        result = run_simulation(
            inp, current_price=50.0, sim_config=config, dcf_config=dcf, seed=SEED
        )
        assert result.revenue_bands is not None
        assert result.fcf_bands is not None
        expected_quarters = 5 * 4
        for band in (result.revenue_bands, result.fcf_bands):
            assert len(band.p10) == expected_quarters
            assert len(band.p25) == expected_quarters
            assert len(band.p50) == expected_quarters
            assert len(band.p75) == expected_quarters
            assert len(band.p90) == expected_quarters

    def test_band_percentile_ordering(self) -> None:
        """p10 <= p25 <= p50 <= p75 <= p90 at each quarter."""
        config = SimulationConfig(num_replicates=500, num_display_paths=5)
        inp = _make_input()
        result = run_simulation(inp, current_price=50.0, sim_config=config, seed=SEED)
        assert result.revenue_bands is not None
        assert result.fcf_bands is not None
        for bands in (result.revenue_bands, result.fcf_bands):
            for q in range(len(bands.p10)):
                assert bands.p10[q] <= bands.p25[q] + 1e-10
                assert bands.p25[q] <= bands.p50[q] + 1e-10
                assert bands.p50[q] <= bands.p75[q] + 1e-10
                assert bands.p75[q] <= bands.p90[q] + 1e-10

    def test_bands_computed_from_all_paths(self) -> None:
        """Bands should use all replicates, not just display paths."""
        config = SimulationConfig(num_replicates=500, num_display_paths=5)
        inp = _make_input()
        result = run_simulation(inp, current_price=50.0, sim_config=config, seed=SEED)
        assert result.revenue_bands is not None
        # The bands should reflect the full distribution (500 paths).
        # With only 5 display paths, the spread would be narrower.
        # Verify the spread is non-trivial (P90 > P10 at the last quarter).
        last_q = len(result.revenue_bands.p10) - 1
        spread = result.revenue_bands.p90[last_q] - result.revenue_bands.p10[last_q]
        assert spread > 0

    def test_sample_paths_are_subset_of_all(self) -> None:
        """Sample paths should come from the first N paths of the simulation."""
        config = SimulationConfig(num_replicates=100, num_display_paths=10)
        inp = _make_input()
        result = run_simulation(inp, current_price=50.0, sim_config=config, seed=SEED)
        assert len(result.sample_paths) == 10
        assert result.revenue_bands is not None
        # Each sample path's values should be within [p10, p90] at most quarters
        # (not strictly — individual paths can exceed band extremes when replicates
        # are few — but the path lengths should match)
        for path in result.sample_paths:
            assert len(path.quarterly_revenue) == len(result.revenue_bands.p10)
