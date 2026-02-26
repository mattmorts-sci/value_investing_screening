"""Penalty-based weighted scoring.

Three penalty types (lower = better):
1. DC penalty — squared debt-to-cash ratio, penalises high leverage.
2. MC penalty — log10 size penalty, reflects diminishing growth at scale.
3. Growth penalty — three sub-components: rate, stability, divergence.

All weights configurable via AnalysisConfig. No normalisation constraint
on penalty weights (dc_weight, mc_weight, growth_weight). Growth
sub-weights must sum to 1.0 (validated in AnalysisConfig).
"""

from __future__ import annotations

import logging
import math

import pandas as pd

from pipeline.config.settings import AnalysisConfig

logger = logging.getLogger(__name__)


def _dc_penalty(debt_cash_ratio: float, weight: float) -> float:
    """Squared debt-to-cash penalty.

    abs() retained from legacy for defensive robustness.
    Returns inf if debt_cash_ratio is inf (zero cash) and weight > 0.
    Returns 0.0 if weight is 0 (penalty disabled).
    """
    if weight == 0:
        return 0.0
    if math.isinf(debt_cash_ratio):
        return float("inf")
    return (abs(debt_cash_ratio) ** 2) * weight


def _mc_penalty(
    market_cap: float,
    min_market_cap: float,
    weight: float,
) -> float:
    """Logarithmic size penalty.

    Larger companies get higher penalties — diminishing growth potential
    at scale. Zero penalty if market_cap <= min_market_cap.
    """
    if market_cap <= min_market_cap or min_market_cap <= 0:
        return 0.0
    return math.log10(market_cap / min_market_cap) * weight


def _growth_penalty(
    avg_growth: float,
    fcf_growth_std: float,
    revenue_growth_std: float,
    fcf_growth_mean: float,
    revenue_growth_mean: float,
    config: AnalysisConfig,
) -> float:
    """Three-component growth penalty.

    a. Rate penalty: triggered when avg_growth < min_acceptable_growth.
    b. Stability penalty: standard deviation of growth rates.
    c. Divergence penalty: absolute difference between FCF and revenue growth.
    """
    penalty = 0.0

    # Rate penalty
    if avg_growth < config.min_acceptable_growth:
        penalty += (
            (config.min_acceptable_growth - avg_growth)
            * config.growth_weight
            * config.growth_rate_subweight
        )

    # Stability penalty
    avg_std = (fcf_growth_std + revenue_growth_std) / 2
    penalty += avg_std * config.growth_weight * config.growth_stability_subweight

    # Divergence penalty
    divergence = abs(fcf_growth_mean - revenue_growth_mean)
    penalty += divergence * config.growth_weight * config.growth_divergence_subweight

    return penalty


def calculate_weighted_scores(
    companies: pd.DataFrame,
    growth_stats: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Calculate penalty-based weighted scores for all companies.

    Args:
        companies: Per-company DataFrame indexed by entity_id.
            Requires columns: symbol, debt_cash_ratio, market_cap.
        growth_stats: Per-company growth stats indexed by entity_id.
            Requires columns: fcf_growth_mean, fcf_growth_std,
            revenue_growth_mean, revenue_growth_std, combined_growth_mean.
        config: Analysis configuration.

    Returns:
        DataFrame indexed by entity_id with columns: symbol,
        dc_penalty, mc_penalty, growth_penalty, total_penalty,
        weighted_rank.

    Raises:
        ValueError: If required columns are missing.
    """
    required_companies = {"symbol", "debt_cash_ratio", "market_cap"}
    missing = required_companies - set(companies.columns)
    if missing:
        raise ValueError(f"Missing required columns in companies: {sorted(missing)}")

    required_stats = {
        "fcf_growth_mean", "fcf_growth_std",
        "revenue_growth_mean", "revenue_growth_std",
        "combined_growth_mean",
    }
    missing_stats = required_stats - set(growth_stats.columns)
    if missing_stats:
        raise ValueError(
            f"Missing required columns in growth_stats: {sorted(missing_stats)}"
        )

    rows: list[dict[str, object]] = []

    for entity_id in companies.index:
        company = companies.loc[entity_id]
        symbol = str(company["symbol"])

        if entity_id not in growth_stats.index:
            logger.warning(
                "Skipping entity %d (%s): no growth stats", entity_id, symbol,
            )
            continue

        stats = growth_stats.loc[entity_id]

        dc = _dc_penalty(float(company["debt_cash_ratio"]), config.dc_weight)
        mc = _mc_penalty(
            float(company["market_cap"]),
            config.min_market_cap,
            config.mc_weight,
        )
        growth = _growth_penalty(
            avg_growth=float(stats["combined_growth_mean"]),
            fcf_growth_std=float(stats["fcf_growth_std"]),
            revenue_growth_std=float(stats["revenue_growth_std"]),
            fcf_growth_mean=float(stats["fcf_growth_mean"]),
            revenue_growth_mean=float(stats["revenue_growth_mean"]),
            config=config,
        )

        total = dc + mc + growth

        rows.append({
            "entity_id": entity_id,
            "symbol": symbol,
            "dc_penalty": dc,
            "mc_penalty": mc,
            "growth_penalty": growth,
            "total_penalty": total,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame(
            columns=[
                "entity_id", "symbol", "dc_penalty", "mc_penalty",
                "growth_penalty", "total_penalty", "weighted_rank",
            ],
        )
        result = result.set_index("entity_id")
        return result

    result = result.set_index("entity_id")

    # Rank by total_penalty ascending (lower = better).
    # inf penalties sort last (worst rank).
    result["weighted_rank"] = result["total_penalty"].rank(
        method="min", ascending=True, na_option="bottom",
    ).astype(int)

    logger.info("Computed weighted scores for %d companies", len(result))
    return result
