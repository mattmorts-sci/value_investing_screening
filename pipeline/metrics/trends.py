"""Trend analysis metrics: growth statistics, regressions, and ROIC."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import linregress  # type: ignore[import-untyped]

from pipeline.config import TrendsConfig
from pipeline.data.models import CompanyData

logger = logging.getLogger(__name__)


@dataclass
class TrendMetrics:
    """Trend and regression outputs consumed by simulation and display.

    Attributes:
        revenue_cagr: Annualised compound growth rate of revenue.
        revenue_yoy_growth_std: Standard deviation of year-over-year
            quarterly revenue growth rates.
        revenue_qoq_growth_mean: Mean of quarter-over-quarter revenue
            growth rates (simulation input).
        revenue_qoq_growth_var: Variance of quarter-over-quarter revenue
            growth rates (simulation input).
        margin_intercept: Intercept of operating margin regression.
        margin_slope: Slope of operating margin regression (per quarter, decimal).
        margin_r_squared: R-squared of operating margin regression.
        conversion_intercept: Intercept of FCF conversion regression
            (None if fallback).
        conversion_slope: Slope of FCF conversion regression
            (None if fallback).
        conversion_r_squared: R-squared of FCF conversion regression
            (None if fallback).
        conversion_median: Median FCF conversion (set if fallback).
        conversion_is_fallback: True if fewer than threshold valid quarters.
        roic_latest: ROIC of the most recent valid quarter.
        roic_slope: Slope from ROIC regression.
        roic_detrended_std: Standard deviation of ROIC regression residuals.
        roic_minimum: Minimum ROIC across valid quarters.
    """

    # Revenue (display)
    revenue_cagr: float
    revenue_yoy_growth_std: float

    # Revenue (simulation input)
    revenue_qoq_growth_mean: float
    revenue_qoq_growth_var: float

    # Operating margin regression
    margin_intercept: float
    margin_slope: float
    margin_r_squared: float

    # FCF conversion regression
    conversion_intercept: float | None
    conversion_slope: float | None
    conversion_r_squared: float | None
    conversion_median: float | None
    conversion_is_fallback: bool

    # ROIC
    roic_latest: float
    roic_slope: float
    roic_detrended_std: float
    roic_minimum: float


def compute_trends(
    company: CompanyData, config: TrendsConfig
) -> TrendMetrics | None:
    """Compute trend metrics for a company.

    Calculates revenue growth statistics, operating margin regression,
    FCF conversion regression (with fallback to median), and ROIC metrics.

    Args:
        company: Company data with quarterly financials sorted by date.
        config: Trend analysis configuration.

    Returns:
        TrendMetrics if sufficient data, None otherwise.
    """
    fin = company.financials
    if fin.empty:
        logger.warning("%s: empty financials", company.symbol)
        return None

    # Revenue CAGR and growth statistics
    revenue_cagr = _compute_revenue_cagr(fin, company.symbol)
    if revenue_cagr is None:
        return None

    revenue_yoy_std = _compute_revenue_yoy_growth_std(fin, company.symbol)
    if revenue_yoy_std is None:
        return None

    qoq_mean, qoq_var = _compute_revenue_qoq_stats(fin, company.symbol)
    if qoq_mean is None or qoq_var is None:
        return None

    # Operating margin regression
    margin_result = _compute_margin_regression(fin, company.symbol)
    if margin_result is None:
        return None

    # FCF conversion regression (with fallback)
    conversion_result = _compute_conversion_regression(
        fin, config.min_quarters_fcf_conversion, company.symbol
    )
    if conversion_result is None:
        return None

    # ROIC metrics
    roic_result = _compute_roic_metrics(fin, company.symbol)
    if roic_result is None:
        return None

    return TrendMetrics(
        revenue_cagr=revenue_cagr,
        revenue_yoy_growth_std=revenue_yoy_std,
        revenue_qoq_growth_mean=qoq_mean,
        revenue_qoq_growth_var=qoq_var,
        margin_intercept=margin_result[0],
        margin_slope=margin_result[1],
        margin_r_squared=margin_result[2],
        conversion_intercept=conversion_result[0],
        conversion_slope=conversion_result[1],
        conversion_r_squared=conversion_result[2],
        conversion_median=conversion_result[3],
        conversion_is_fallback=conversion_result[4],
        roic_latest=roic_result[0],
        roic_slope=roic_result[1],
        roic_detrended_std=roic_result[2],
        roic_minimum=roic_result[3],
    )


def _compute_revenue_cagr(fin: pd.DataFrame, symbol: str) -> float | None:
    """Annualised compound growth rate from earliest to latest revenue.

    Uses TTM (trailing twelve months) revenue at start and end where
    four consecutive quarters are available, otherwise falls back to
    quarterly values.
    """
    revenues = fin[["date", "revenue"]].copy()
    revenues = revenues.dropna(subset=["revenue"])
    revenues = revenues[revenues["revenue"] > 0]

    if len(revenues) < 2:
        logger.warning("%s: insufficient positive revenue quarters for CAGR", symbol)
        return None

    # Attempt TTM: sum of four consecutive quarters
    if len(revenues) >= 4:
        start_ttm = revenues["revenue"].iloc[:4].sum()
        end_ttm = revenues["revenue"].iloc[-4:].sum()
    else:
        start_ttm = revenues["revenue"].iloc[0]
        end_ttm = revenues["revenue"].iloc[-1]

    if start_ttm <= 0 or end_ttm <= 0:
        logger.warning("%s: non-positive TTM revenue for CAGR", symbol)
        return None

    # Number of years between first and last data point (calendar dates)
    first_date = pd.to_datetime(revenues["date"].iloc[0])
    last_date = pd.to_datetime(revenues["date"].iloc[-1])
    years = (last_date - first_date).days / 365.25

    if years <= 0:
        logger.warning("%s: zero time span for CAGR", symbol)
        return None

    cagr = (end_ttm / start_ttm) ** (1.0 / years) - 1.0

    if not math.isfinite(cagr):
        logger.warning("%s: non-finite CAGR", symbol)
        return None

    return cagr


def _compute_revenue_yoy_growth_std(
    fin: pd.DataFrame, symbol: str
) -> float | None:
    """Standard deviation of year-over-year quarterly revenue growth rates.

    Compares each quarter to the same quarter one year prior (4 quarters back).
    """
    revenues = fin["revenue"].values
    n = len(revenues)

    yoy_rates: list[float] = []
    for i in range(4, n):
        prev = revenues[i - 4]
        curr = revenues[i]
        if prev is not None and curr is not None:
            prev_f = float(prev)
            curr_f = float(curr)
            if prev_f > 0 and math.isfinite(curr_f):
                rate = (curr_f - prev_f) / prev_f
                if math.isfinite(rate):
                    yoy_rates.append(rate)

    if len(yoy_rates) < 2:
        logger.warning(
            "%s: insufficient YoY growth observations (%d)", symbol, len(yoy_rates)
        )
        return None

    std = float(np.std(yoy_rates, ddof=1))
    if not math.isfinite(std):
        logger.warning("%s: non-finite YoY growth std", symbol)
        return None

    return std


def _compute_revenue_qoq_stats(
    fin: pd.DataFrame, symbol: str
) -> tuple[float | None, float | None]:
    """Mean and variance of quarter-over-quarter revenue growth rates."""
    revenues = fin["revenue"].values
    n = len(revenues)

    qoq_rates: list[float] = []
    for i in range(1, n):
        prev = revenues[i - 1]
        curr = revenues[i]
        if prev is not None and curr is not None:
            prev_f = float(prev)
            curr_f = float(curr)
            if prev_f > 0 and math.isfinite(curr_f):
                rate = (curr_f - prev_f) / prev_f
                if math.isfinite(rate):
                    qoq_rates.append(rate)

    if len(qoq_rates) < 2:
        logger.warning(
            "%s: insufficient QoQ growth observations (%d)", symbol, len(qoq_rates)
        )
        return None, None

    arr = np.array(qoq_rates)
    mean = float(np.mean(arr))
    var = float(np.var(arr, ddof=1))

    if not (math.isfinite(mean) and math.isfinite(var)):
        logger.warning("%s: non-finite QoQ growth stats", symbol)
        return None, None

    return mean, var


def _compute_margin_regression(
    fin: pd.DataFrame, symbol: str
) -> tuple[float, float, float] | None:
    """Linear regression of operating margin over quarter index.

    margin = operating_income / revenue for each quarter.

    Returns:
        (intercept, slope, r_squared) or None if insufficient data.
    """
    margins: list[float] = []
    indices: list[int] = []

    for i in range(len(fin)):
        rev = fin["revenue"].iloc[i]
        oi = fin["operating_income"].iloc[i]
        if rev is not None and oi is not None:
            rev_f = float(rev)
            oi_f = float(oi)
            if rev_f != 0 and math.isfinite(oi_f) and math.isfinite(rev_f):
                margin = oi_f / rev_f
                if math.isfinite(margin):
                    margins.append(margin)
                    indices.append(i)

    if len(margins) < 2:
        logger.warning(
            "%s: insufficient margin observations (%d)", symbol, len(margins)
        )
        return None

    x = np.array(indices, dtype=np.float64)
    y = np.array(margins, dtype=np.float64)

    result = linregress(x, y)
    intercept = float(result.intercept)
    slope = float(result.slope)
    r_sq = float(result.rvalue ** 2)

    if not (math.isfinite(intercept) and math.isfinite(slope) and math.isfinite(r_sq)):
        logger.warning("%s: non-finite margin regression", symbol)
        return None

    return intercept, slope, r_sq


def _compute_conversion_regression(
    fin: pd.DataFrame, min_quarters: int, symbol: str
) -> tuple[float | None, float | None, float | None, float | None, bool] | None:
    """FCF conversion regression or median fallback.

    conversion = free_cash_flow / operating_income (excluding quarters
    where operating_income <= 0).

    Returns:
        (intercept, slope, r_squared, median, is_fallback) or None if
        no valid quarters at all.
    """
    conversions: list[float] = []
    indices: list[int] = []

    for i in range(len(fin)):
        oi = fin["operating_income"].iloc[i]
        fcf = fin["free_cash_flow"].iloc[i]
        if oi is not None and fcf is not None:
            oi_f = float(oi)
            fcf_f = float(fcf)
            if oi_f > 0 and math.isfinite(fcf_f):
                conv = fcf_f / oi_f
                if math.isfinite(conv):
                    conversions.append(conv)
                    indices.append(i)

    if len(conversions) == 0:
        logger.warning("%s: no valid FCF conversion quarters", symbol)
        return None

    # Fallback: fewer than threshold valid quarters
    if len(conversions) < min_quarters:
        median = float(np.median(conversions))
        if not math.isfinite(median):
            logger.warning("%s: non-finite conversion median", symbol)
            return None
        return None, None, None, median, True

    # Full regression
    x = np.array(indices, dtype=np.float64)
    y = np.array(conversions, dtype=np.float64)

    result = linregress(x, y)
    intercept = float(result.intercept)
    slope = float(result.slope)
    r_sq = float(result.rvalue ** 2)

    if not (math.isfinite(intercept) and math.isfinite(slope) and math.isfinite(r_sq)):
        logger.warning("%s: non-finite conversion regression", symbol)
        return None

    return intercept, slope, r_sq, None, False


def _compute_roic_metrics(
    fin: pd.DataFrame, symbol: str
) -> tuple[float, float, float, float] | None:
    """ROIC metrics: latest, slope, detrended std, minimum.

    ROIC = NOPAT / Invested Capital, where:
        NOPAT = EBIT × (1 − effective_tax_rate)
        effective_tax_rate = income_tax_expense / income_before_tax
        Invested Capital = total_assets − total_current_liabilities

    Excludes quarters where income_before_tax <= 0 (effective tax
    rate is undefined).

    Returns:
        (roic_latest, roic_slope, roic_detrended_std, roic_minimum) or
        None if insufficient valid quarters.
    """
    roic_values: list[float] = []
    indices: list[int] = []

    for i in range(len(fin)):
        ebit = fin["ebit"].iloc[i]
        ibt = fin["income_before_tax"].iloc[i]
        ite = fin["income_tax_expense"].iloc[i]
        ta = fin["total_assets"].iloc[i]
        tcl = fin["total_current_liabilities"].iloc[i]
        if (
            ebit is not None
            and ibt is not None
            and ite is not None
            and ta is not None
            and tcl is not None
        ):
            ebit_f = float(ebit)
            ibt_f = float(ibt)
            ite_f = float(ite)
            ta_f = float(ta)
            tcl_f = float(tcl)
            if ibt_f <= 0:
                continue
            if not (
                math.isfinite(ebit_f)
                and math.isfinite(ite_f)
                and math.isfinite(ta_f)
                and math.isfinite(tcl_f)
            ):
                continue
            effective_tax_rate = ite_f / ibt_f
            nopat = ebit_f * (1.0 - effective_tax_rate)
            invested_capital = ta_f - tcl_f
            if invested_capital <= 0:
                continue
            roic = nopat / invested_capital
            if math.isfinite(roic):
                roic_values.append(roic)
                indices.append(i)

    if len(roic_values) < 2:
        logger.warning(
            "%s: insufficient valid ROIC quarters (%d)", symbol, len(roic_values)
        )
        return None

    roic_latest = roic_values[-1]
    roic_minimum = min(roic_values)

    x = np.array(indices, dtype=np.float64)
    y = np.array(roic_values, dtype=np.float64)

    result = linregress(x, y)
    roic_slope = float(result.slope)

    # Detrended std: std of residuals
    predicted = result.intercept + result.slope * x
    residuals = y - predicted
    roic_detrended_std = float(np.std(residuals, ddof=1))

    if not (
        math.isfinite(roic_latest)
        and math.isfinite(roic_slope)
        and math.isfinite(roic_detrended_std)
        and math.isfinite(roic_minimum)
    ):
        logger.warning("%s: non-finite ROIC metrics", symbol)
        return None

    return roic_latest, roic_slope, roic_detrended_std, roic_minimum
