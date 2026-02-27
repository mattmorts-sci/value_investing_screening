"""Monte Carlo growth projection with constraint system.

Positive-FCF companies: Monte Carlo simulation with 8 constraint
layers, extracting P25/P50/P75 scenarios.
Negative-FCF companies: improvement-toward-zero model using
revenue-growth-derived improvement rates.
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from pipeline.config.settings import AnalysisConfig
from pipeline.data.contracts import Projection

logger = logging.getLogger(__name__)


def _safe_float(value: object, default: float = 0.0) -> float:
    """Convert to float, replacing NaN/None/inf with default."""
    if value is None:
        return default
    f: float = float(value)  # type: ignore[arg-type]
    if np.isnan(f) or np.isinf(f):
        return default
    return f


def _parameterise_lognormal(
    m: float,
    s: float,
    cv_cap: float,
) -> tuple[float, float]:
    """Convert growth mean/std to log-normal mu, sigma parameters.

    For the growth multiplier Y = 1 + growth_rate:
        target_mean = 1 + m
        CV = min(s / target_mean, cv_cap)
        sigma^2 = ln(1 + CV^2)
        mu = ln(target_mean) - sigma^2 / 2

    Edge cases:
        - m <= -0.5: deeply negative mean, defaults to 5% quarterly
          decline (mu = ln(0.95), sigma = sqrt(0.1)).
        - s <= 0 and m > -0.5: unknown volatility (e.g. insufficient
          data). Uses mu = ln(1 + m) with small fixed sigma = 0.05.
          When m = 0 this projects roughly flat rather than declining.

    Args:
        m: Mean quarterly growth rate.
        s: Std of quarterly growth rate.
        cv_cap: Maximum coefficient of variation.

    Returns:
        (mu, sigma) for log-normal sampling.
    """
    if m > -0.5 and s > 0:
        target_mean = 1 + m
        cv = min(s / target_mean, cv_cap)
        sigma_sq = math.log(1 + cv**2)
        mu = math.log(target_mean) - sigma_sq / 2
    elif m <= -0.5:
        # Deeply negative mean: 5% quarterly decline.
        mu = math.log(0.95)
        sigma_sq = 0.1
    else:
        # Unknown volatility (s <= 0, m > -0.5): project at mean with
        # minimal noise. When m = 0 this is roughly flat.
        mu = math.log(max(1 + m, 0.01))
        sigma_sq = 0.05**2
    return mu, math.sqrt(sigma_sq)


def _simulate_monte_carlo(
    current_value: float,
    growth_mean: float,
    growth_std: float,
    market_cap: float,
    n_quarters: int,
    metric: str,
    config: AnalysisConfig,
    seed: int | None = None,
) -> np.ndarray:
    """Run vectorised Monte Carlo simulation for one metric.

    Simulates config.simulation_replicates paths of n_quarters steps.
    Each step: sample from log-normal, apply 7 constraint layers.

    Args:
        current_value: Starting value (must be positive).
        growth_mean: Mean quarterly growth rate from growth_stats.
        growth_std: Std of quarterly growth rate from growth_stats.
        market_cap: Company market capitalisation.
        n_quarters: Number of quarters to simulate.
        metric: "fcf" or "revenue" (selects per-quarter caps).
        config: Analysis configuration.
        seed: Optional RNG seed for reproducibility.

    Returns:
        1D array of shape (simulation_replicates,) with final values.
    """
    n = config.simulation_replicates

    if current_value == 0:
        return np.zeros(n, dtype=np.float64)

    mu, sigma = _parameterise_lognormal(growth_mean, growth_std, config.cv_cap)

    # Select per-quarter caps by metric
    if metric == "fcf":
        small_pos = config.fcf_small_pos_cap
        small_neg = config.fcf_small_neg_cap
        large_pos = config.fcf_large_pos_cap
        large_neg = config.fcf_large_neg_cap
    else:
        small_pos = config.revenue_small_pos_cap
        small_neg = config.revenue_small_neg_cap
        large_pos = config.revenue_large_pos_cap
        large_neg = config.revenue_large_neg_cap

    values = np.full(n, current_value, dtype=np.float64)
    recent_growth = np.zeros((n, 4), dtype=np.float64)
    rng = np.random.default_rng(seed)

    for t in range(n_quarters):
        # --- Constraint 6: Momentum exhaustion ---
        adjusted_mu = np.full(n, mu)
        if t >= 4 and growth_mean > config.high_growth_threshold:
            avg_recent = recent_growth.mean(axis=1)

            high_mask = (
                avg_recent > growth_mean * config.momentum_exhaustion_threshold
            )
            if high_mask.any():
                excess = avg_recent[high_mask] / growth_mean
                adj = -np.log(
                    1 + (excess - config.momentum_exhaustion_threshold) * 0.2,
                )
                adjusted_mu[high_mask] += np.maximum(adj, -0.3)

            low_mask = avg_recent < growth_mean * 0.5
            if low_mask.any():
                shortfall = avg_recent[low_mask] / growth_mean
                adj = np.log(1 + (0.5 - shortfall) * 0.1)
                adjusted_mu[low_mask] += np.minimum(adj, 0.2)

        # --- Constraint 1: Log-normal sampling ---
        growth_rate = np.exp(rng.normal(adjusted_mu, sigma)) - 1

        # --- Constraint 2: Per-quarter caps (asymmetric by size) ---
        size_evolution = np.abs(values / current_value)
        large = size_evolution > 2.0
        pos_cap = np.where(large, large_pos, small_pos)
        neg_cap = np.where(large, large_neg, small_neg)
        growth_rate = np.clip(growth_rate, neg_cap, pos_cap)

        # --- Constraint 8: Size-based growth penalty ---
        if market_cap > 0:
            value_to_market = np.abs(values) / market_cap
            size_factor = np.where(
                value_to_market < 0.001,
                0.8,
                np.where(
                    value_to_market < 0.01,
                    np.clip(0.03 / value_to_market, 0.5, 0.9),
                    np.clip(
                        config.size_penalty_factor / value_to_market,
                        0.4,
                        1.0,
                    ),
                ),
            )
        else:
            size_factor = np.ones(n)

        # --- Constraint 7: Time decay for high growth ---
        if t > 0:
            years_elapsed = t / config.quarters_per_year
            time_factor = np.where(
                growth_rate > config.high_growth_threshold,
                config.time_decay_base**years_elapsed,
                1.0,
            )
        else:
            time_factor = np.ones(n)

        # Apply all per-step constraints
        constrained_growth = growth_rate * size_factor * time_factor

        # Update momentum window (shift left, append new)
        recent_growth[:, :-1] = recent_growth[:, 1:]
        recent_growth[:, -1] = constrained_growth

        # Apply growth
        values *= 1 + constrained_growth

        # --- Constraint 3: Cumulative cap (10x) and floor (0.1x) ---
        ratio = values / current_value
        exceeded = ratio > config.cumulative_growth_cap
        if exceeded.any():
            values[exceeded] = current_value * config.cumulative_growth_cap
        declined = ratio < config.cumulative_decline_floor
        if declined.any():
            values[declined] = current_value * config.cumulative_decline_floor

        # --- Constraint 4: 100% annual CAGR backstop (after year 1) ---
        quarters_elapsed = t + 1
        if quarters_elapsed > config.quarters_per_year:
            max_value = current_value * (
                (1 + config.annual_cagr_backstop)
                ** (quarters_elapsed / config.quarters_per_year)
            )
            backstop_exceeded = values > max_value
            if backstop_exceeded.any():
                values[backstop_exceeded] = max_value

    return values


def _extract_scenarios(
    entity_id: int,
    metric: str,
    period_years: int,
    current_value: float,
    final_values: np.ndarray,
    n_quarters: int,
) -> dict[str, Projection]:
    """Extract P25/P50/P75 scenarios from simulated final values.

    Back-calculates annual CAGR from each percentile's final value,
    then projects a smooth compound path for the DCF.

    Args:
        entity_id: Company entity ID.
        metric: "fcf" or "revenue".
        period_years: Projection horizon in years.
        current_value: Starting value.
        final_values: 1D array of simulated final values.
        n_quarters: Number of quarters in the projection.

    Returns:
        {scenario_name: Projection} for pessimistic/base/optimistic.
    """
    percentile_map = {
        "pessimistic": 25,
        "base": 50,
        "optimistic": 75,
    }

    scenarios: dict[str, Projection] = {}

    for scenario_name, pct in percentile_map.items():
        final_val = float(np.percentile(final_values, pct))
        annual_cagr = _compute_annual_cagr(current_value, final_val, period_years)

        # Smooth compound path: constant quarterly rate implied by CAGR
        if annual_cagr > -1.0:
            quarterly_rate = (1 + annual_cagr) ** 0.25 - 1
        else:
            quarterly_rate = 0.0

        quarterly_values: list[float] = []
        quarterly_rates: list[float] = []
        v = current_value
        for _ in range(n_quarters):
            v *= 1 + quarterly_rate
            quarterly_values.append(v)
            quarterly_rates.append(quarterly_rate)

        scenarios[scenario_name] = Projection(
            entity_id=entity_id,
            metric=metric,
            period_years=period_years,
            scenario=scenario_name,
            quarterly_growth_rates=quarterly_rates,
            quarterly_values=quarterly_values,
            annual_cagr=annual_cagr,
            current_value=current_value,
        )

    return scenarios


def _project_negative_fcf(
    current_value: float,
    revenue_growth_mean: float,
    n_quarters: int,
    improvement_cap: float,
    g_eq_q: float,
    fade_lambda: float,
) -> tuple[list[float], list[float]]:
    """Project negative FCF toward zero, then switch to conservative fade.

    Negative FCF cannot use normal growth rates (percentage growth on a
    negative base is meaningless). Instead, the model applies a constant
    fractional improvement toward zero each quarter.

    Once FCF is within 1% of its original magnitude, it snaps to a
    small positive value and switches to a conservative fade toward
    equilibrium.

    Args:
        current_value: Starting FCF (must be negative).
        revenue_growth_mean: Quarterly mean revenue growth (determines
            improvement speed; higher revenue growth = faster FCF recovery).
        n_quarters: Projection horizon in quarters.
        improvement_cap: Maximum improvement rate per quarter.
        g_eq_q: Quarterly equilibrium growth rate (for post-crossing fade).
        fade_lambda: Decay constant (for post-crossing fade).

    Returns:
        (quarterly_growth_rates, quarterly_values) tuple.
    """
    # Derive improvement rate from revenue growth
    if revenue_growth_mean > 0:
        improvement_rate = min(improvement_cap, revenue_growth_mean)
    else:
        improvement_rate = improvement_cap * 0.3
    improvement_rate = max(improvement_rate, 0.01)

    cross_threshold = abs(current_value) * 0.01

    rates: list[float] = []
    values: list[float] = []
    v = current_value
    in_negative_phase = True
    cross_quarter = 0

    for t in range(1, n_quarters + 1):
        if in_negative_phase:
            v = v * (1 - improvement_rate)
            g = -improvement_rate

            if abs(v) < cross_threshold:
                v = cross_threshold
                in_negative_phase = False
                cross_quarter = t
        else:
            conservative_g0 = g_eq_q * 0.5
            quarters_since = t - cross_quarter
            g = g_eq_q + (conservative_g0 - g_eq_q) * math.exp(
                -fade_lambda * quarters_since
            )
            v = v * (1 + g)

        rates.append(g)
        values.append(v)

    return rates, values


def _compute_annual_cagr(
    current_value: float,
    final_value: float,
    period_years: int,
) -> float:
    """Compute implied annual CAGR from current to final value.

    Handles sign changes and negative values with capped returns.
    """
    if period_years <= 0:
        return 0.0

    # Both positive: standard CAGR
    if current_value > 0 and final_value > 0:
        return float((final_value / current_value) ** (1 / period_years) - 1)

    # Negative to positive: cap at 100% annual
    if current_value < 0 and final_value > 0:
        return 1.0

    # Both negative
    if current_value < 0 and final_value < 0:
        if abs(final_value) < abs(current_value):
            improvement = 1 - (abs(final_value) / abs(current_value))
            return improvement / period_years
        return -0.5

    # Positive to negative
    if current_value > 0 and final_value < 0:
        return -0.9

    # Zero cases
    return 0.0


def project_growth(
    entity_id: int,
    fcf_stats: dict[str, float],
    revenue_stats: dict[str, float],
    market_cap: float,
    config: AnalysisConfig,
    seed: int | None = None,
) -> dict[int, dict[str, dict[str, Projection]]]:
    """Project growth for one company.

    Positive FCF: Monte Carlo simulation with constraint system,
    extracting P25/P50/P75 scenarios.
    Negative FCF: improvement-toward-zero model for FCF; Monte Carlo
    for revenue.

    Args:
        entity_id: Company entity ID.
        fcf_stats: Keys: mean (quarterly), std (quarterly), latest_value.
        revenue_stats: Keys: mean (quarterly), std (quarterly), latest_value.
        market_cap: Company market capitalisation.
        config: Analysis configuration.
        seed: Optional RNG seed for reproducibility.

    Returns:
        {period_years: {metric: {scenario: Projection}}}
    """
    fcf_is_negative = fcf_stats["latest_value"] < 0

    # Pre-compute fade parameters for negative-FCF improvement model
    if fcf_is_negative:
        g_eq_q = (1 + config.equilibrium_growth_rate) ** 0.25 - 1
        base_lambda = math.log(2) / (
            config.base_fade_half_life_years * config.quarters_per_year
        )
        if market_cap > 0 and config.min_market_cap > 0:
            size_ratio = market_cap / config.min_market_cap
            size_adj = math.log10(size_ratio) * 0.1 if size_ratio > 1 else 0.0
        else:
            size_adj = 0.0
        fade_lambda = base_lambda * (1 + size_adj)

    result: dict[int, dict[str, dict[str, Projection]]] = {}

    for period_years in config.projection_periods:
        n_quarters = period_years * config.quarters_per_year
        period_result: dict[str, dict[str, Projection]] = {}

        for metric, stats in [("fcf", fcf_stats), ("revenue", revenue_stats)]:
            current_value = stats["latest_value"]

            if metric == "fcf" and fcf_is_negative:
                # Negative FCF: improvement model with 3 scenarios
                k = config.scenario_band_width
                scenarios: dict[str, Projection] = {}

                for scenario_name, rev_growth in [
                    ("base", revenue_stats["mean"]),
                    (
                        "optimistic",
                        revenue_stats["mean"] + k * revenue_stats["std"],
                    ),
                    (
                        "pessimistic",
                        revenue_stats["mean"] - k * revenue_stats["std"],
                    ),
                ]:
                    rates, values = _project_negative_fcf(
                        current_value=current_value,
                        revenue_growth_mean=rev_growth,
                        n_quarters=n_quarters,
                        improvement_cap=config.negative_fcf_improvement_cap,
                        g_eq_q=g_eq_q,
                        fade_lambda=fade_lambda,
                    )
                    final_value = values[-1] if values else current_value
                    annual_cagr = _compute_annual_cagr(
                        current_value, final_value, period_years,
                    )
                    scenarios[scenario_name] = Projection(
                        entity_id=entity_id,
                        metric=metric,
                        period_years=period_years,
                        scenario=scenario_name,
                        quarterly_growth_rates=rates,
                        quarterly_values=values,
                        annual_cagr=annual_cagr,
                        current_value=current_value,
                    )
            else:
                # Positive FCF or revenue: Monte Carlo simulation
                final_values = _simulate_monte_carlo(
                    current_value=current_value,
                    growth_mean=stats["mean"],
                    growth_std=stats["std"],
                    market_cap=market_cap,
                    n_quarters=n_quarters,
                    metric=metric,
                    config=config,
                    seed=seed,
                )
                scenarios = _extract_scenarios(
                    entity_id=entity_id,
                    metric=metric,
                    period_years=period_years,
                    current_value=current_value,
                    final_values=final_values,
                    n_quarters=n_quarters,
                )

            period_result[metric] = scenarios
        result[period_years] = period_result

    return result


def project_all(
    companies: pd.DataFrame,
    growth_stats: pd.DataFrame,
    config: AnalysisConfig,
) -> dict[int, Any]:
    """Project growth for all companies.

    Args:
        companies: Per-company DataFrame indexed by entity_id
            (from derived_metrics). Requires columns: fcf, revenue,
            market_cap.
        growth_stats: Per-company growth stats indexed by entity_id
            (from growth_stats). Requires columns: fcf_growth_mean,
            fcf_growth_std, revenue_growth_mean, revenue_growth_std.
        config: Analysis configuration.

    Returns:
        {entity_id: {period: {metric: {scenario: Projection}}}}
    """
    projections: dict[int, Any] = {}

    for entity_id in companies.index:
        row = companies.loc[entity_id]
        stats_row = growth_stats.loc[entity_id]

        fcf_stats = {
            "mean": _safe_float(stats_row.get("fcf_growth_mean", 0.0)),
            "std": _safe_float(stats_row.get("fcf_growth_std", 0.0)),
            "latest_value": float(row["fcf"]),
        }
        rev_stats = {
            "mean": _safe_float(stats_row.get("revenue_growth_mean", 0.0)),
            "std": _safe_float(stats_row.get("revenue_growth_std", 0.0)),
            "latest_value": float(row["revenue"]),
        }
        market_cap = float(row.get("market_cap", 0.0))

        projections[entity_id] = project_growth(
            entity_id=entity_id,
            fcf_stats=fcf_stats,
            revenue_stats=rev_stats,
            market_cap=market_cap,
            config=config,
        )

    return projections
