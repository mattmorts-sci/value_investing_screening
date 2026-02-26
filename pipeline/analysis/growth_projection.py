"""Fade-to-equilibrium growth projection model.

Deterministic model replacing the legacy Monte Carlo simulation.
Projects growth rates for FCF and revenue using exponential fade
toward a long-run equilibrium rate.

Model: g(t) = g_eq + (g_0 - g_eq) * exp(-lambda * t)

Three scenarios per metric per period:
- base: fade from mean historical growth
- optimistic: fade from mean + k * sigma
- pessimistic: fade from mean - k * sigma

Negative FCF uses a separate improvement-toward-zero model
with revenue-growth-derived improvement rates.
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
    """Convert to float, replacing NaN/None with default."""
    if value is None:
        return default
    f: float = float(value)  # type: ignore[arg-type]
    if np.isnan(f):
        return default
    return f


def _quarterly_rate(annual_rate: float) -> float:
    """Convert an annual rate to quarterly via compound interest.

    Clamps to [-0.99, +inf) to prevent negative base in fractional
    exponentiation (which would raise ValueError).
    """
    clamped = max(annual_rate, -0.99)
    return float((1 + clamped) ** 0.25 - 1)


def _compute_fade_lambda(
    half_life_years: float,
    market_cap: float,
    min_market_cap: float,
) -> float:
    """Compute the exponential decay constant for the fade model.

    Args:
        half_life_years: Base half-life in years (time for growth to
            move halfway to equilibrium).
        market_cap: Company market capitalisation.
        min_market_cap: Minimum market cap from config (scaling reference).

    Returns:
        Lambda (per-quarter decay constant). Higher = faster fade.
    """
    base_lambda = math.log(2) / (half_life_years * 4)

    # Larger companies fade faster (less room to grow)
    if market_cap > 0 and min_market_cap > 0:
        size_ratio = market_cap / min_market_cap
        if size_ratio > 1:
            size_adj = math.log10(size_ratio) * 0.1
        else:
            size_adj = 0.0
    else:
        size_adj = 0.0

    return base_lambda * (1 + size_adj)


def _fade_growth_rates(
    g_0: float,
    g_eq_q: float,
    fade_lambda: float,
    n_quarters: int,
) -> list[float]:
    """Generate quarterly growth rates using fade-to-equilibrium.

    g(t) = g_eq + (g_0 - g_eq) * exp(-lambda * t)

    Args:
        g_0: Starting quarterly growth rate.
        g_eq_q: Equilibrium quarterly growth rate.
        fade_lambda: Per-quarter decay constant.
        n_quarters: Number of quarters to project.

    Returns:
        List of quarterly growth rates.
    """
    return [
        g_eq_q + (g_0 - g_eq_q) * math.exp(-fade_lambda * t)
        for t in range(1, n_quarters + 1)
    ]


def _project_values(
    current_value: float,
    growth_rates: list[float],
) -> list[float]:
    """Project values forward using per-quarter growth rates.

    Value(t) = Value(t-1) * (1 + g(t))
    """
    values: list[float] = []
    v = current_value
    for g in growth_rates:
        v = v * (1 + g)
        values.append(v)
    return values


def _project_negative_fcf(
    current_value: float,
    revenue_growth_mean: float,
    n_quarters: int,
    improvement_cap: float,
    g_eq_q: float,
    fade_lambda: float,
) -> tuple[list[float], list[float]]:
    """Project negative FCF toward zero, then switch to standard fade.

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
) -> dict[int, dict[str, dict[str, Projection]]]:
    """Fade-to-equilibrium growth projection for one company.

    Args:
        entity_id: Company entity ID.
        fcf_stats: Keys: mean (quarterly), std (quarterly), latest_value.
        revenue_stats: Keys: mean (quarterly), std (quarterly), latest_value.
        market_cap: Company market capitalisation (for size adjustment).
        config: Analysis configuration.

    Returns:
        {period_years: {metric: {scenario: Projection}}}
    """
    g_eq_q = _quarterly_rate(config.equilibrium_growth_rate)
    fade_lambda = _compute_fade_lambda(
        config.base_fade_half_life_years,
        market_cap,
        config.min_market_cap,
    )
    k = config.scenario_band_width

    result: dict[int, dict[str, dict[str, Projection]]] = {}

    for period_years in config.projection_periods:
        n_quarters = period_years * config.quarters_per_year
        period_result: dict[str, dict[str, Projection]] = {}

        for metric, stats in [("fcf", fcf_stats), ("revenue", revenue_stats)]:
            g_0 = stats["mean"]
            sigma = stats["std"]
            current_value = stats["latest_value"]

            scenarios: dict[str, Projection] = {}

            for scenario_name, g_start in [
                ("base", g_0),
                ("optimistic", g_0 + k * sigma),
                ("pessimistic", g_0 - k * sigma),
            ]:
                if metric == "fcf" and current_value < 0:
                    # Negative FCF: improvement path
                    # Scenario affects revenue growth assumption
                    if scenario_name == "base":
                        rev_growth = revenue_stats["mean"]
                    elif scenario_name == "optimistic":
                        rev_growth = (
                            revenue_stats["mean"]
                            + k * revenue_stats["std"]
                        )
                    else:
                        rev_growth = (
                            revenue_stats["mean"]
                            - k * revenue_stats["std"]
                        )

                    rates, values = _project_negative_fcf(
                        current_value=current_value,
                        revenue_growth_mean=rev_growth,
                        n_quarters=n_quarters,
                        improvement_cap=config.negative_fcf_improvement_cap,
                        g_eq_q=g_eq_q,
                        fade_lambda=fade_lambda,
                    )
                else:
                    rates = _fade_growth_rates(
                        g_start, g_eq_q, fade_lambda, n_quarters,
                    )
                    values = _project_values(current_value, rates)

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
