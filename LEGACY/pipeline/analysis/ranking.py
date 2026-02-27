"""Company ranking using risk-adjusted scoring.

Combines composite IV ratio (weighted across scenarios) with
value-investing risk factors to produce risk-adjusted scores.
Produces 4 ranking DataFrames.

Companies without a live price are excluded from ranking (no IV/price
ratio possible). Companies where pessimistic IV < current price are
filtered out (safety gate).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from pipeline.config.settings import AnalysisConfig
from pipeline.data.contracts import IntrinsicValue

logger = logging.getLogger(__name__)


def _compute_iv_ratios(
    entity_id: int,
    current_price: float,
    intrinsic_values: dict[int, Any],
    primary_period: int,
) -> dict[str, float]:
    """Compute IV/price ratios at primary period for all three scenarios.

    Returns:
        Dict with keys: composite_iv_ratio, pessimistic_iv_ratio,
        base_iv_ratio, optimistic_iv_ratio. All 0.0 if data unavailable.
    """
    result = {
        "composite_iv_ratio": 0.0,
        "pessimistic_iv_ratio": 0.0,
        "base_iv_ratio": 0.0,
        "optimistic_iv_ratio": 0.0,
    }

    entity_ivs = intrinsic_values.get(entity_id)
    if entity_ivs is None or current_price <= 0:
        return result

    period_ivs = entity_ivs.get(primary_period)
    if period_ivs is None:
        return result

    ratios: dict[str, float] = {}
    for scenario_name in ("pessimistic", "base", "optimistic"):
        iv = period_ivs.get(scenario_name)
        if isinstance(iv, IntrinsicValue) and iv.iv_per_share > 0:
            ratios[scenario_name] = iv.iv_per_share / current_price
        else:
            ratios[scenario_name] = 0.0

    result["pessimistic_iv_ratio"] = ratios.get("pessimistic", 0.0)
    result["base_iv_ratio"] = ratios.get("base", 0.0)
    result["optimistic_iv_ratio"] = ratios.get("optimistic", 0.0)

    # Composite: 25% pessimistic + 50% base + 25% optimistic.
    if ratios.get("base", 0.0) > 0:
        result["composite_iv_ratio"] = (
            0.25 * ratios.get("pessimistic", 0.0)
            + 0.50 * ratios["base"]
            + 0.25 * ratios.get("optimistic", 0.0)
        )

    return result


def _compute_terminal_dependency(
    entity_id: int,
    intrinsic_values: dict[int, Any],
    primary_period: int,
) -> float:
    """Compute terminal value dependency from base-case DCF.

    Returns fraction of present value from terminal value (0 to 1).
    Returns 1.0 (max risk) if data unavailable.
    """
    entity_ivs = intrinsic_values.get(entity_id)
    if entity_ivs is None:
        return 1.0

    period_ivs = entity_ivs.get(primary_period)
    if period_ivs is None:
        return 1.0

    base_iv = period_ivs.get("base")
    if not isinstance(base_iv, IntrinsicValue):
        return 1.0

    if base_iv.present_value <= 0:
        return 1.0

    terminal_pv = base_iv.terminal_value / (
        (1 + base_iv.discount_rate) ** base_iv.period_years
    )
    dependency = terminal_pv / base_iv.present_value
    return max(0.0, min(1.0, dependency))


def _annualised_valuation_return(
    iv_ratio: float,
    period_years: int,
) -> float:
    """Annualised return implied by IV/price ratio over a period.

    valuation_return = iv_ratio ^ (1/period) - 1
    """
    if iv_ratio <= 0 or period_years <= 0:
        return 0.0
    return float(iv_ratio ** (1 / period_years) - 1)


def rank_companies(
    companies: pd.DataFrame,
    growth_stats: pd.DataFrame,
    projections: dict[int, Any],
    intrinsic_values: dict[int, Any],
    weighted_scores: pd.DataFrame,
    live_prices: dict[str, float],
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rank companies using risk-adjusted scoring.

    Only companies with a live price and pessimistic IV >= current price
    (safety gate) are included in ranking.

    Args:
        companies: Per-company DataFrame indexed by entity_id.
            Requires: symbol, fcf, market_cap, debt_cash_ratio.
        growth_stats: Per-company growth stats indexed by entity_id.
            Requires: fcf_growth_mean, fcf_growth_std,
            revenue_growth_mean, revenue_growth_std,
            combined_growth_mean, growth_stability, fcf_reliability.
        projections: {entity_id: {period: {metric: {scenario: Projection}}}}.
        intrinsic_values: {entity_id: {period: {scenario: IntrinsicValue}}}.
        weighted_scores: Output of calculate_weighted_scores. Indexed by
            entity_id with: dc_penalty, mc_penalty, growth_penalty,
            total_penalty, weighted_rank.
        live_prices: {symbol: current_price}.
        config: Analysis configuration.

    Returns:
        (growth_rankings, value_rankings, weighted_rankings,
         combined_rankings) — four DataFrames.

    Raises:
        ValueError: If required columns are missing.
    """
    required_companies = {"symbol", "fcf", "market_cap", "debt_cash_ratio"}
    missing = required_companies - set(companies.columns)
    if missing:
        raise ValueError(f"Missing required columns in companies: {sorted(missing)}")

    required_stats = {
        "fcf_growth_mean", "fcf_growth_std",
        "revenue_growth_mean", "revenue_growth_std",
        "combined_growth_mean", "growth_stability",
        "fcf_reliability",
    }
    missing_stats = required_stats - set(growth_stats.columns)
    if missing_stats:
        raise ValueError(
            f"Missing required columns in growth_stats: {sorted(missing_stats)}"
        )

    rows: list[dict[str, object]] = []
    safety_gate_excluded = 0

    for entity_id in companies.index:
        company = companies.loc[entity_id]
        symbol = str(company["symbol"])

        current_price = live_prices.get(symbol)
        if current_price is None or current_price <= 0:
            logger.warning(
                "Excluding entity %d (%s) from ranking: no valid live price",
                entity_id, symbol,
            )
            continue

        if entity_id not in growth_stats.index:
            logger.warning(
                "Excluding entity %d (%s) from ranking: no growth stats",
                entity_id, symbol,
            )
            continue

        stats = growth_stats.loc[entity_id]

        # --- IV ratios ---
        iv_ratios = _compute_iv_ratios(
            entity_id, current_price, intrinsic_values, config.primary_period,
        )

        # Safety gate: exclude companies where pessimistic IV < current price.
        if iv_ratios["pessimistic_iv_ratio"] < 1.0:
            safety_gate_excluded += 1
            logger.debug(
                "Safety gate: excluding entity %d (%s), "
                "pessimistic_iv_ratio=%.3f < 1.0",
                entity_id, symbol, iv_ratios["pessimistic_iv_ratio"],
            )
            continue

        # --- Expected return ---
        period_for_valuation = config.primary_period

        # Projected annual growth rates from fade model, base scenario.
        fcf_growth_annual = 0.0
        revenue_growth_annual = 0.0

        entity_proj = projections.get(entity_id, {})
        primary_proj = entity_proj.get(config.primary_period, {})
        fcf_proj = primary_proj.get("fcf", {})
        rev_proj = primary_proj.get("revenue", {})

        base_fcf_proj = fcf_proj.get("base")
        if base_fcf_proj is not None:
            fcf_growth_annual = base_fcf_proj.annual_cagr

        base_rev_proj = rev_proj.get("base")
        if base_rev_proj is not None:
            revenue_growth_annual = base_rev_proj.annual_cagr

        combined_growth = (
            fcf_growth_annual * config.fcf_growth_weight
            + revenue_growth_annual * config.revenue_growth_weight
        )

        valuation_return = _annualised_valuation_return(
            iv_ratios["composite_iv_ratio"], period_for_valuation,
        )
        total_expected_return = combined_growth + valuation_return

        # --- Risk factors (normalised to [0, 1] where 1 = safest) ---

        # 1. Scenario spread
        base_iv_ratio = iv_ratios["base_iv_ratio"]
        if base_iv_ratio > 0:
            scenario_spread_raw = (
                iv_ratios["optimistic_iv_ratio"]
                - iv_ratios["pessimistic_iv_ratio"]
            ) / base_iv_ratio
        else:
            scenario_spread_raw = float("inf")
        scenario_spread_score = (
            1.0 / (1.0 + scenario_spread_raw)
            if scenario_spread_raw >= 0
            else 0.0
        )

        # 2. Downside exposure
        downside_exposure_raw = iv_ratios["pessimistic_iv_ratio"]
        downside_exposure_score = min(downside_exposure_raw, 2.0) / 2.0

        # 3. Terminal value dependency
        terminal_dependency_raw = _compute_terminal_dependency(
            entity_id, intrinsic_values, config.primary_period,
        )
        terminal_dependency_score = 1.0 - terminal_dependency_raw

        # 4. FCF reliability
        fcf_reliability_raw = float(stats["fcf_reliability"])
        fcf_reliability_score = fcf_reliability_raw

        # Composite safety score
        composite_safety = (
            downside_exposure_score * config.downside_exposure_weight
            + scenario_spread_score * config.scenario_spread_weight
            + terminal_dependency_score * config.terminal_dependency_weight
            + fcf_reliability_score * config.fcf_reliability_weight
        )

        # Risk-adjusted score: multiplication so both higher return and
        # higher safety increase the score (linear influence).  Division
        # (Sharpe-like) would penalise low safety more aggressively but
        # also amplify noise when composite_safety is near zero.
        risk_adjusted_score = total_expected_return * composite_safety

        # Growth divergence
        growth_divergence = abs(fcf_growth_annual - revenue_growth_annual)
        divergence_flag = growth_divergence > config.growth_divergence_threshold

        # Growth stability from pre-computed stats
        growth_stability = float(stats["growth_stability"])

        # Weighted scoring columns (from weighted_scores)
        if entity_id in weighted_scores.index:
            ws = weighted_scores.loc[entity_id]
            dc_penalty = float(ws["dc_penalty"])
            mc_penalty = float(ws["mc_penalty"])
            growth_penalty = float(ws["growth_penalty"])
            total_penalty = float(ws["total_penalty"])
        else:
            dc_penalty = 0.0
            mc_penalty = 0.0
            growth_penalty = 0.0
            total_penalty = 0.0

        rows.append({
            "entity_id": entity_id,
            "symbol": symbol,
            "current_price": current_price,
            "fcf_growth_annual": fcf_growth_annual,
            "revenue_growth_annual": revenue_growth_annual,
            "combined_growth": combined_growth,
            "composite_iv_ratio": iv_ratios["composite_iv_ratio"],
            "pessimistic_iv_ratio": iv_ratios["pessimistic_iv_ratio"],
            "base_iv_ratio": iv_ratios["base_iv_ratio"],
            "optimistic_iv_ratio": iv_ratios["optimistic_iv_ratio"],
            "scenario_spread": scenario_spread_raw,
            "downside_exposure": downside_exposure_raw,
            "terminal_dependency": terminal_dependency_raw,
            "fcf_reliability": fcf_reliability_raw,
            "downside_exposure_score": downside_exposure_score,
            "scenario_spread_score": scenario_spread_score,
            "terminal_dependency_score": terminal_dependency_score,
            "fcf_reliability_score": fcf_reliability_score,
            "composite_safety": composite_safety,
            "total_expected_return": total_expected_return,
            "risk_adjusted_score": risk_adjusted_score,
            "growth_stability": growth_stability,
            "growth_divergence": growth_divergence,
            "divergence_flag": divergence_flag,
            "fcf": float(company["fcf"]),
            "market_cap": float(company["market_cap"]),
            "debt_cash_ratio": float(company["debt_cash_ratio"]),
            "dc_penalty": dc_penalty,
            "mc_penalty": mc_penalty,
            "growth_penalty": growth_penalty,
            "total_penalty": total_penalty,
        })

    if safety_gate_excluded > 0:
        logger.info(
            "Safety gate excluded %d companies (pessimistic IV < price)",
            safety_gate_excluded,
        )

    if not rows:
        empty_cols = [
            "entity_id", "symbol", "current_price",
            "fcf_growth_annual", "revenue_growth_annual", "combined_growth",
            "composite_iv_ratio", "pessimistic_iv_ratio",
            "base_iv_ratio", "optimistic_iv_ratio",
            "scenario_spread", "downside_exposure",
            "terminal_dependency", "fcf_reliability",
            "downside_exposure_score", "scenario_spread_score",
            "terminal_dependency_score", "fcf_reliability_score",
            "composite_safety",
            "total_expected_return", "risk_adjusted_score",
            "growth_stability", "growth_divergence", "divergence_flag",
            "fcf", "market_cap", "debt_cash_ratio",
            "dc_penalty", "mc_penalty", "growth_penalty", "total_penalty",
        ]
        empty = pd.DataFrame(columns=empty_cols).set_index("entity_id")
        return empty, empty.copy(), empty.copy(), empty.copy()

    base = pd.DataFrame(rows).set_index("entity_id")
    n = len(base)

    # --- Growth rankings: sorted by combined_growth desc ---
    growth = base.copy()
    growth["fcf_growth_rank"] = growth["fcf_growth_annual"].rank(
        ascending=False, method="min",
    ).astype(int)
    growth["revenue_growth_rank"] = growth["revenue_growth_annual"].rank(
        ascending=False, method="min",
    ).astype(int)
    growth["combined_growth_rank"] = growth["combined_growth"].rank(
        ascending=False, method="min",
    ).astype(int)
    growth["stability_rank"] = growth["growth_stability"].rank(
        ascending=False, method="min",
    ).astype(int)
    growth = growth.sort_values("combined_growth", ascending=False)

    # --- Value rankings: sorted by composite_iv_ratio desc ---
    value = base.copy()
    value["value_rank"] = value["composite_iv_ratio"].rank(
        ascending=False, method="min",
    ).astype(int)
    value = value.sort_values("composite_iv_ratio", ascending=False)

    # --- Weighted rankings: sorted by total_penalty asc ---
    weighted = base.copy()
    weighted["weighted_rank"] = weighted["total_penalty"].rank(
        ascending=True, method="min", na_option="bottom",
    ).astype(int)
    weighted = weighted.sort_values("total_penalty", ascending=True)

    # --- Combined rankings: sorted by risk_adjusted_rank asc ---
    combined = base.copy()
    combined["risk_adjusted_rank"] = combined["risk_adjusted_score"].rank(
        ascending=False, method="min",
    ).astype(int)
    combined["opportunity_rank"] = combined["risk_adjusted_rank"]
    combined["opportunity_score"] = 100 - (combined["risk_adjusted_rank"] / n * 100)

    # Normalised rank-based scores (0-100, higher = better)
    combined["growth_score"] = 100 - (
        combined["combined_growth"].rank(ascending=False, method="min") / n * 100
    )
    combined["value_score"] = 100 - (
        combined["composite_iv_ratio"].rank(ascending=False, method="min") / n * 100
    )
    combined["weighted_score"] = 100 - (
        combined["total_penalty"].rank(ascending=True, method="min") / n * 100
    )
    combined["stability_score"] = 100 - (
        combined["growth_stability"].rank(ascending=False, method="min") / n * 100
    )

    # Divergence penalty: binary 0 or 10
    combined["divergence_penalty"] = np.where(
        combined["divergence_flag"], 10, 0,
    )

    combined = combined.sort_values("risk_adjusted_rank", ascending=True)

    logger.info("Ranked %d companies across 4 DataFrames", n)
    return growth, value, weighted, combined
