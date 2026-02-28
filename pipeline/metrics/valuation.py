"""Valuation metrics: EV, FCF/EV, EBIT/EV, and weighted composite yield."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import pandas as pd

from pipeline.config import PipelineConfig
from pipeline.data.models import CompanyData

logger = logging.getLogger(__name__)


@dataclass
class ValuationMetrics:
    """Valuation output consumed by screening and ranking.

    Attributes:
        enterprise_value: Market cap + total debt - cash.
        ttm_fcf: Trailing twelve months free cash flow (sum of last 4 quarters).
        ttm_ebit: Trailing twelve months EBIT (sum of last 4 quarters).
        fcf_ev: FCF / EV yield. None if EV <= 0.
        ebit_ev: EBIT / EV yield. None if EV <= 0.
        composite_yield: Weighted combination of FCF/EV and EBIT/EV. None if
            either component is None.
    """

    enterprise_value: float
    ttm_fcf: float
    ttm_ebit: float
    fcf_ev: float | None
    ebit_ev: float | None
    composite_yield: float | None


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


def compute_valuation(
    company: CompanyData, config: PipelineConfig
) -> ValuationMetrics:
    """Compute valuation metrics for a single company.

    Calculates enterprise value from the latest quarter's balance sheet,
    trailing-twelve-month FCF and EBIT from up to four recent quarters,
    and yield ratios (FCF/EV, EBIT/EV) with a weighted composite.

    If fewer than 4 quarters of data are available, uses what exists.
    If EV <= 0, yield ratios are set to None.

    Args:
        company: Fully populated CompanyData.
        config: Pipeline configuration (provides yield weights).

    Returns:
        ValuationMetrics with all computed fields.
    """
    financials = company.financials

    if financials.empty:
        logger.warning("%s: empty financials", company.symbol)
        return ValuationMetrics(
            enterprise_value=company.market_cap,
            ttm_fcf=0.0,
            ttm_ebit=0.0,
            fcf_ev=None,
            ebit_ev=None,
            composite_yield=None,
        )

    # Latest quarter balance sheet values (last row, sorted by date)
    latest = financials.iloc[-1]
    total_debt = _safe_float(latest["total_debt"])
    cash = _safe_float(latest["cash_and_cash_equivalents"])

    # Enterprise value
    ev = company.market_cap + total_debt - cash

    # TTM sums from last 4 quarters (or fewer if unavailable)
    ttm_rows = financials.tail(4)
    ttm_fcf = _safe_sum(ttm_rows["free_cash_flow"])
    ttm_ebit = _safe_sum(ttm_rows["ebit"])

    # Yield ratios
    if ev > 0:
        fcf_ev = ttm_fcf / ev
        ebit_ev = ttm_ebit / ev
        composite = (
            fcf_ev * config.fcf_ev_weight + ebit_ev * config.ebit_ev_weight
        )
    else:
        logger.warning(
            "%s: EV is non-positive (%.2f), yield ratios set to None",
            company.symbol,
            ev,
        )
        fcf_ev = None
        ebit_ev = None
        composite = None

    return ValuationMetrics(
        enterprise_value=ev,
        ttm_fcf=ttm_fcf,
        ttm_ebit=ttm_ebit,
        fcf_ev=fcf_ev,
        ebit_ev=ebit_ev,
        composite_yield=composite,
    )
