"""Factor contribution analysis and quadrant classification.

Analyses penalty factor contributions (what percentage of each company's
total penalty comes from debt-cash, market-cap, and growth factors),
identifies which factor dominates, and assigns companies to growth-value
quadrants.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from pipeline.config.settings import AnalysisConfig

logger = logging.getLogger(__name__)


def calculate_factor_contributions(
    weighted_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate percentage contribution of each penalty factor.

    For each company: factor_pct = factor_penalty / total_penalty * 100.
    Companies with zero total penalty get 0% across all factors.
    Companies with inf total penalty get NaN contributions.

    Args:
        weighted_scores: Output of calculate_weighted_scores. Indexed
            by entity_id with: dc_penalty, mc_penalty, growth_penalty,
            total_penalty.

    Returns:
        DataFrame indexed by entity_id with: dc_pct, mc_pct, growth_pct.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {"dc_penalty", "mc_penalty", "growth_penalty", "total_penalty"}
    missing = required - set(weighted_scores.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = pd.DataFrame(index=weighted_scores.index)
    result.index.name = "entity_id"

    total = weighted_scores["total_penalty"]

    for factor, col in [
        ("dc_pct", "dc_penalty"),
        ("mc_pct", "mc_penalty"),
        ("growth_pct", "growth_penalty"),
    ]:
        result[factor] = np.where(
            total == 0,
            0.0,
            weighted_scores[col] / total * 100,
        )

    logger.info(
        "Computed factor contributions for %d companies", len(result),
    )
    return result


def analyze_factor_dominance(
    contributions: pd.DataFrame,
    weighted_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Identify which factor dominates each company's penalty.

    Args:
        contributions: Output of calculate_factor_contributions.
            Indexed by entity_id with: dc_pct, mc_pct, growth_pct.
        weighted_scores: Output of calculate_weighted_scores. Indexed
            by entity_id with: total_penalty.

    Returns:
        Summary DataFrame with columns: primary_factor, company_count,
        pct_of_total, avg_contribution, avg_total_penalty. One row per
        dominant factor.
    """
    if contributions.empty:
        return pd.DataFrame(
            columns=[
                "primary_factor", "company_count", "pct_of_total",
                "avg_contribution", "avg_total_penalty",
            ],
        )

    factor_cols = {"dc_pct": "dc", "mc_pct": "mc", "growth_pct": "growth"}

    # Determine dominant factor per company
    factor_df = contributions[list(factor_cols.keys())]
    dominant = factor_df.idxmax(axis=1).map(factor_cols)

    n = len(contributions)
    rows: list[dict[str, object]] = []

    for factor_name in sorted(factor_cols.values()):
        mask = dominant == factor_name
        count = int(mask.sum())
        if count == 0:
            continue

        pct_col = next(k for k, v in factor_cols.items() if v == factor_name)
        avg_contribution = float(contributions.loc[mask, pct_col].mean())

        # Average total penalty for companies dominated by this factor
        dominated_ids = contributions.index[mask]
        common_ids = dominated_ids.intersection(weighted_scores.index)
        if len(common_ids) > 0:
            avg_total_penalty = float(
                weighted_scores.loc[common_ids, "total_penalty"].mean(),
            )
        else:
            avg_total_penalty = 0.0

        rows.append({
            "primary_factor": factor_name,
            "company_count": count,
            "pct_of_total": count / n * 100 if n > 0 else 0.0,
            "avg_contribution": avg_contribution,
            "avg_total_penalty": avg_total_penalty,
        })

    result = pd.DataFrame(rows)
    logger.info("Factor dominance: %d factors identified", len(result))
    return result


def create_quadrant_analysis(
    companies: pd.DataFrame,
    intrinsic_values: dict[int, Any],
    live_prices: dict[str, float],
    growth_stats: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Assign companies to growth-value quadrants.

    Quadrant 1 (best): high growth + high value (IV/price >= threshold)
    Quadrant 2: high growth + low value
    Quadrant 3: low growth + high value
    Quadrant 4 (worst): low growth + low value

    Args:
        companies: Per-company DataFrame indexed by entity_id.
            Requires: symbol.
        intrinsic_values: {entity_id: {period: {scenario: IntrinsicValue}}}.
        live_prices: {symbol: current_price}.
        growth_stats: Per-company growth stats indexed by entity_id.
            Requires: combined_growth_mean.
        config: Analysis configuration (min_acceptable_growth,
            min_iv_to_price_ratio, primary_period).

    Returns:
        DataFrame indexed by entity_id with: symbol, combined_growth,
        composite_iv_ratio, high_growth, high_value, quadrant.
    """
    rows: list[dict[str, object]] = []

    for entity_id in companies.index:
        company = companies.loc[entity_id]
        symbol = str(company["symbol"])
        price = live_prices.get(symbol)

        if price is None or price <= 0:
            continue

        if entity_id not in growth_stats.index:
            continue

        combined_growth = float(growth_stats.loc[entity_id, "combined_growth_mean"])

        # Composite IV/price ratio at primary period (25/50/25 weighting).
        composite_ratio = 0.0
        entity_ivs = intrinsic_values.get(entity_id, {})
        period_ivs = entity_ivs.get(config.primary_period, {})

        scenario_ratios: dict[str, float] = {}
        for scenario_name in ("pessimistic", "base", "optimistic"):
            iv = period_ivs.get(scenario_name)
            if hasattr(iv, "iv_per_share") and iv.iv_per_share > 0:
                scenario_ratios[scenario_name] = iv.iv_per_share / price

        if "base" in scenario_ratios:
            composite_ratio = (
                0.25 * scenario_ratios.get("pessimistic", 0.0)
                + 0.50 * scenario_ratios["base"]
                + 0.25 * scenario_ratios.get("optimistic", 0.0)
            )

        high_growth = combined_growth >= config.min_acceptable_growth
        high_value = composite_ratio >= config.min_iv_to_price_ratio

        if high_growth and high_value:
            quadrant = 1
        elif high_growth and not high_value:
            quadrant = 2
        elif not high_growth and high_value:
            quadrant = 3
        else:
            quadrant = 4

        rows.append({
            "entity_id": entity_id,
            "symbol": symbol,
            "combined_growth": combined_growth,
            "composite_iv_ratio": composite_ratio,
            "high_growth": high_growth,
            "high_value": high_value,
            "quadrant": quadrant,
        })

    if not rows:
        return pd.DataFrame(
            columns=[
                "entity_id", "symbol", "combined_growth",
                "composite_iv_ratio",
                "high_growth", "high_value", "quadrant",
            ],
        ).set_index("entity_id")

    result = pd.DataFrame(rows).set_index("entity_id")
    logger.info(
        "Quadrant analysis: %d companies classified "
        "(Q1=%d, Q2=%d, Q3=%d, Q4=%d)",
        len(result),
        int((result["quadrant"] == 1).sum()),
        int((result["quadrant"] == 2).sum()),
        int((result["quadrant"] == 3).sum()),
        int((result["quadrant"] == 4).sum()),
    )
    return result
