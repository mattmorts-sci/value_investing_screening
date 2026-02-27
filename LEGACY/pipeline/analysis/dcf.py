"""DCF intrinsic value calculation.

FCF-only discounted cash flow with quarterly internal discounting.
Receives projected quarterly cash flows from the growth projection
model and discounts them to present value.

Terminal value uses the Gordon Growth Model on the final annual
cash flow. Margin of safety applied before per-share conversion.
"""

import logging
from typing import Any

import pandas as pd

from pipeline.config.settings import AnalysisConfig
from pipeline.data.contracts import IntrinsicValue, Projection

logger = logging.getLogger(__name__)


def calculate_dcf(
    base_fcf: float,
    shares: float,
    projection: Projection,
    config: AnalysisConfig,
) -> IntrinsicValue:
    """Calculate DCF intrinsic value from a growth projection.

    Uses the projected quarterly cash flow values from the Projection
    directly (the growth model already handles negative FCF transitions
    and fade-to-equilibrium dynamics).

    Args:
        base_fcf: Current free cash flow (for reference; projection
            already starts from this value).
        shares: Diluted shares outstanding.
        projection: FCF projection with quarterly values and growth rates.
        config: Analysis configuration (discount rate, terminal growth,
            margin of safety).

    Returns:
        IntrinsicValue with present value, IV per share, and supporting
        detail.
    """
    if not projection.quarterly_values:
        raise ValueError(
            f"Projection for entity {projection.entity_id} "
            f"({projection.scenario}, {projection.period_years}yr) "
            f"has empty quarterly_values."
        )

    quarterly_discount = (1 + config.discount_rate) ** 0.25 - 1
    quarterly_cash_flows = projection.quarterly_values

    # Discount projected quarterly cash flows to present value
    pv_cash_flows = 0.0
    for q, cf in enumerate(quarterly_cash_flows, 1):
        pv_cash_flows += cf / (1 + quarterly_discount) ** q

    # Terminal value: Gordon Growth Model on final annual cash flow
    final_quarterly_cf = quarterly_cash_flows[-1]
    final_annual_cf = final_quarterly_cf * config.quarters_per_year
    terminal_cf = final_annual_cf * (1 + config.terminal_growth_rate)
    terminal_value = terminal_cf / (
        config.discount_rate - config.terminal_growth_rate
    )

    # Discount terminal value using annual rate
    terminal_pv = terminal_value / (
        (1 + config.discount_rate) ** projection.period_years
    )

    present_value = pv_cash_flows + terminal_pv

    # Margin of safety and per-share conversion
    iv_per_share = present_value * (1 - config.margin_of_safety) / shares

    # Annual cash flows for reporting (sum each group of 4 quarters)
    annual_cash_flows: list[float] = []
    for y in range(projection.period_years):
        start = y * config.quarters_per_year
        end = start + config.quarters_per_year
        annual_cash_flows.append(sum(quarterly_cash_flows[start:end]))

    return IntrinsicValue(
        scenario=projection.scenario,
        period_years=projection.period_years,
        projected_annual_cash_flows=annual_cash_flows,
        terminal_value=terminal_value,
        present_value=present_value,
        iv_per_share=iv_per_share,
        growth_rate=projection.annual_cagr,
        discount_rate=config.discount_rate,
        terminal_growth_rate=config.terminal_growth_rate,
        margin_of_safety=config.margin_of_safety,
    )


def calculate_all_dcf(
    companies: pd.DataFrame,
    projections: dict[int, Any],
    config: AnalysisConfig,
) -> dict[int, Any]:
    """Calculate DCF for all companies, all periods, all scenarios.

    Only uses FCF projections (revenue projections are not valued).

    Args:
        companies: Per-company DataFrame indexed by entity_id.
            Requires columns: fcf, shares_diluted.
        projections: {entity_id: {period: {metric: {scenario: Projection}}}}.
        config: Analysis configuration.

    Returns:
        {entity_id: {period: {scenario: IntrinsicValue}}}
    """
    result: dict[int, Any] = {}

    for entity_id in companies.index:
        row = companies.loc[entity_id]
        base_fcf = float(row["fcf"])
        shares = float(row["shares_diluted"])

        if shares <= 0:
            logger.warning(
                "Skipping entity %d: shares_diluted = %.2f",
                entity_id,
                shares,
            )
            continue

        entity_proj = projections.get(entity_id)
        if entity_proj is None:
            logger.warning(
                "Skipping entity %d: no projections found", entity_id,
            )
            continue

        entity_ivs: dict[int, dict[str, IntrinsicValue]] = {}

        for period_years in config.projection_periods:
            period_proj = entity_proj.get(period_years)
            if period_proj is None:
                continue

            fcf_scenarios = period_proj.get("fcf")
            if fcf_scenarios is None:
                continue

            period_ivs: dict[str, IntrinsicValue] = {}
            for scenario_name, projection in fcf_scenarios.items():
                period_ivs[scenario_name] = calculate_dcf(
                    base_fcf, shares, projection, config,
                )

            entity_ivs[period_years] = period_ivs

        result[entity_id] = entity_ivs

    return result
