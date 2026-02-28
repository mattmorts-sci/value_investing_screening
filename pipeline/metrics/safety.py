"""Safety metrics: interest coverage and operating cash flow to debt."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import pandas as pd

from pipeline.data.models import CompanyData

logger = logging.getLogger(__name__)


@dataclass
class SafetyMetrics:
    """Safety metric outputs for debt coverage assessment.

    Attributes:
        interest_coverage: TTM EBIT / TTM interest expense.
            None if interest_expense is zero or missing (no debt to cover).
        ocf_to_debt: TTM operating cash flow / latest total debt.
            None if total_debt is zero or missing.
    """

    interest_coverage: float | None
    ocf_to_debt: float | None


def _safe_sum(series: pd.Series) -> float:
    """Sum a pandas Series, treating NaN values as zero.

    Args:
        series: Numeric series to sum.

    Returns:
        Sum of non-NaN values, or 0.0 if all values are NaN.
    """
    result = series.sum(skipna=True)
    if pd.isna(result) or not math.isfinite(result):
        return 0.0
    return float(result)


def _safe_float(value: object) -> float:
    """Extract a finite float from a scalar value, defaulting to 0.0.

    Args:
        value: Scalar value (may be NaN, None, or non-finite).

    Returns:
        Finite float, or 0.0 on any failure.
    """
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f):
        return 0.0
    return f


def compute_safety(company: CompanyData) -> SafetyMetrics:
    """Compute safety metrics for a single company.

    Uses trailing twelve months (last 4 quarters) for flow metrics
    (EBIT, interest expense, operating cash flow) and the latest
    quarter's balance sheet for stock metrics (total debt).

    If fewer than 4 quarters are available, uses what exists.

    Args:
        company: Fully populated CompanyData.

    Returns:
        SafetyMetrics with interest_coverage and ocf_to_debt.
    """
    financials = company.financials

    if financials.empty:
        logger.warning("%s: empty financials", company.symbol)
        return SafetyMetrics(interest_coverage=None, ocf_to_debt=None)

    ttm_rows = financials.tail(4)

    # Interest coverage: TTM EBIT / TTM interest expense.
    # Negative interest expense indicates net interest income — treat
    # as no interest cost to cover (return None).
    ttm_ebit = _safe_sum(ttm_rows["ebit"])
    ttm_interest = _safe_sum(ttm_rows["interest_expense"])

    if ttm_interest <= 0.0:
        interest_coverage = None
        logger.debug(
            "%s: interest expense is zero, negative, or missing (%.2f), "
            "interest coverage set to None",
            company.symbol, ttm_interest,
        )
    else:
        interest_coverage = ttm_ebit / ttm_interest

    # OCF to debt: TTM operating cash flow / latest total debt
    ttm_ocf = _safe_sum(ttm_rows["operating_cash_flow"])
    latest_total_debt = _safe_float(financials.iloc[-1]["total_debt"])

    if latest_total_debt == 0.0:
        ocf_to_debt = None
        logger.debug(
            "%s: total debt is zero or missing, OCF to debt set to None",
            company.symbol,
        )
    else:
        ocf_to_debt = ttm_ocf / latest_total_debt

    return SafetyMetrics(
        interest_coverage=interest_coverage,
        ocf_to_debt=ocf_to_debt,
    )
