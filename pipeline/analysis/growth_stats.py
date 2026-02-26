"""Per-company growth statistics from time series data.

Computes mean, variance, standard deviation, and CAGR for FCF and
revenue growth using raw quarter-over-quarter growth rates.
Produces combined growth and stability metrics.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipeline.config.settings import AnalysisConfig

logger = logging.getLogger(__name__)

_MIN_DATA_POINTS = 3
_MIN_CAGR_QUARTERS = 4


def _calculate_cagr(values: pd.Series) -> float:
    """Calculate annualised CAGR from a quarterly time series of absolute values.

    Legacy algorithm:
        1. Filter out NaN and zero values, keeping their original indices.
        2. Require at least 2 valid points with >= 4 quarters between them.
        3. Return 0.0 on sign change between first and last valid values.
        4. Compute quarterly CAGR from first to last valid value.
        5. Annualise by compounding 4 quarters.
        6. Preserve sign for both-negative cases.

    Args:
        values: Ordered quarterly values (e.g. FCF or revenue per quarter).

    Returns:
        Annualised CAGR as a decimal (e.g. 0.15 for 15%). Returns 0.0 when
        CAGR cannot be computed.
    """
    # Filter out None, NaN, and zero, keeping position indices.
    valid = [
        (i, v)
        for i, (_, v) in enumerate(values.items())
        if not (pd.isna(v) or v == 0)
    ]

    if len(valid) < 2:
        return 0.0

    first_idx, first_val = valid[0]
    last_idx, last_val = valid[-1]

    periods_between = last_idx - first_idx
    if periods_between < _MIN_CAGR_QUARTERS:
        return 0.0

    # Sign change: can't compute meaningful CAGR.
    if first_val * last_val < 0:
        return 0.0

    try:
        quarterly_cagr = (
            (abs(last_val) / abs(first_val)) ** (1 / periods_between) - 1
        )
        annual_cagr = (1 + quarterly_cagr) ** 4 - 1

        # Both negative: positive quarterly_cagr means magnitude increased
        # (more negative), which is negative growth.
        if first_val < 0 and last_val < 0:
            if abs(last_val) > abs(first_val):
                annual_cagr = -annual_cagr

        return float(annual_cagr)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _compute_raw_qoq_growth(values: pd.Series) -> pd.Series:
    """Compute raw quarter-over-quarter growth rates.

    g = (Q_t - Q_{t-1}) / |Q_{t-1}|

    Absolute-value denominator handles sign transitions without
    producing NaN. With ~20 quarters of input this yields ~19
    independent observations with no autocorrelation.

    Args:
        values: Quarterly absolute values (FCF or revenue), ordered
            chronologically. Must have at least 2 entries to produce
            any growth observations.

    Returns:
        Series of QoQ growth rates. NaN where the previous quarter's
        value is zero.
    """
    prev = values.shift(1)
    growth = (values - prev) / prev.abs()
    return growth.replace([np.inf, -np.inf], np.nan)


def compute_growth_statistics(
    data: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Compute per-company growth statistics from time series.

    Growth rates are raw quarter-over-quarter changes:
    g = (Q_t - Q_{t-1}) / |Q_{t-1}|. Independent observations with no
    autocorrelation, used to parameterise the Monte Carlo simulation.

    Args:
        data: Multi-period DataFrame from loader. Must contain columns:
            entity_id, period_idx, fcf, revenue.
        config: Pipeline configuration (for fcf_growth_weight,
            revenue_growth_weight).

    Returns:
        One-row-per-company DataFrame indexed by entity_id with columns:
            fcf_growth_mean, fcf_growth_var, fcf_growth_std,
            revenue_growth_mean, revenue_growth_var, revenue_growth_std,
            fcf_cagr, revenue_cagr,
            combined_growth_mean, growth_stability, fcf_reliability.

    Raises:
        ValueError: If required columns are missing from the input.
    """
    required = {"entity_id", "period_idx", "fcf", "revenue"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Sort by entity and time for correct CAGR and QoQ ordering.
    sorted_data = data.sort_values(["entity_id", "period_idx"])

    rows: list[dict[str, object]] = []

    for entity_id, group in sorted_data.groupby("entity_id"):
        row: dict[str, object] = {"entity_id": entity_id}

        for metric, abs_col in [("fcf_growth", "fcf"), ("revenue_growth", "revenue")]:
            qoq_growth = _compute_raw_qoq_growth(group[abs_col])
            valid = qoq_growth.dropna()

            if len(valid) >= _MIN_DATA_POINTS:
                row[f"{metric}_mean"] = float(valid.mean())
                row[f"{metric}_var"] = float(valid.var(ddof=0))
                row[f"{metric}_std"] = float(valid.std(ddof=0))
            else:
                row[f"{metric}_mean"] = 0.0
                row[f"{metric}_var"] = 0.0
                row[f"{metric}_std"] = 0.0
                logger.warning(
                    "Entity %s has only %d valid QoQ %s growth points "
                    "(need %d); defaulting stats to 0.0",
                    entity_id,
                    len(valid),
                    abs_col,
                    _MIN_DATA_POINTS,
                )

        # CAGR from absolute values (not growth rates).
        row["fcf_cagr"] = _calculate_cagr(group.set_index("period_idx")["fcf"])
        row["revenue_cagr"] = _calculate_cagr(
            group.set_index("period_idx")["revenue"],
        )

        # Combined growth = weighted mean of FCF and revenue growth means.
        fcf_mean = float(row["fcf_growth_mean"])  # type: ignore[arg-type]
        rev_mean = float(row["revenue_growth_mean"])  # type: ignore[arg-type]
        row["combined_growth_mean"] = (
            fcf_mean * config.fcf_growth_weight
            + rev_mean * config.revenue_growth_weight
        )

        # Growth stability = 1 / (1 + avg_std), bounded [0, 1].
        fcf_std = float(row["fcf_growth_std"])  # type: ignore[arg-type]
        rev_std = float(row["revenue_growth_std"])  # type: ignore[arg-type]
        avg_std = (fcf_std + rev_std) / 2
        row["growth_stability"] = 1.0 / (1.0 + avg_std)

        # FCF reliability = proportion of quarters with positive FCF.
        fcf_values = group["fcf"].dropna()
        if len(fcf_values) > 0:
            row["fcf_reliability"] = float((fcf_values > 0).sum() / len(fcf_values))
        else:
            row["fcf_reliability"] = 0.0

        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.set_index("entity_id")

    logger.info("Computed growth statistics for %d companies", len(result))
    return result
