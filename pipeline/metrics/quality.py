"""Quality metrics: Piotroski F-Score, gross profitability, and accruals ratio."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import pandas as pd

from pipeline.data.models import CompanyData

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics output.

    Attributes:
        f_roa_positive: TTM net income / latest total assets > 0.
        f_ocf_positive: TTM operating cash flow > 0.
        f_roa_improving: Current TTM ROA > prior year TTM ROA.
        f_accruals_negative: TTM OCF > TTM net income (quality of earnings).
        f_leverage_decreasing: Current long-term debt / total assets < prior year.
        f_current_ratio_improving: Current ratio improved year-on-year.
        f_no_dilution: Diluted shares not increased year-on-year.
        f_gross_margin_improving: TTM gross margin improved year-on-year.
        f_asset_turnover_improving: TTM asset turnover improved year-on-year.
        f_score: Sum of all F-Score component booleans (0-9).
        gross_profitability: TTM gross profit / latest total assets (Novy-Marx).
            None if total assets is zero.
        accruals_ratio: (TTM net income - TTM OCF) / latest total assets.
            None if total assets is zero.
    """

    # F-Score components
    f_roa_positive: bool
    f_ocf_positive: bool
    f_roa_improving: bool
    f_accruals_negative: bool
    f_leverage_decreasing: bool
    f_current_ratio_improving: bool
    f_no_dilution: bool
    f_gross_margin_improving: bool
    f_asset_turnover_improving: bool

    # F-Score composite
    f_score: int

    # Other quality metrics
    gross_profitability: float | None
    accruals_ratio: float | None


def _safe_float(value: object) -> float:
    """Extract a finite float from a scalar, defaulting to 0.0.

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


def _safe_sum(series: pd.Series) -> float:
    """Sum a pandas Series, treating NaN as zero.

    Args:
        series: Numeric series to sum.

    Returns:
        Sum of non-NaN values, or 0.0 if all NaN or non-finite.
    """
    result = series.sum(skipna=True)
    if pd.isna(result) or not math.isfinite(result):
        return 0.0
    return float(result)


def _ttm_sum(financials: pd.DataFrame, column: str, start: int, count: int) -> float:
    """Sum a flow column over a range of rows (for TTM calculation).

    Args:
        financials: DataFrame sorted by date.
        column: Column name to sum.
        start: Starting row index (from end, 0 = last row).
        count: Number of rows to sum.

    Returns:
        Sum of the specified rows, NaN treated as zero.
    """
    end_idx = len(financials) - start
    begin_idx = end_idx - count
    if begin_idx < 0:
        begin_idx = 0
    return _safe_sum(financials[column].iloc[begin_idx:end_idx])


def compute_quality(company: CompanyData) -> QualityMetrics:
    """Compute Piotroski F-Score and quality metrics for a single company.

    Uses trailing-twelve-month (TTM) figures constructed from the last 4
    quarters for flow items and the latest quarter for stock items.
    Year-on-year comparisons use quarters 5-8 as the prior year.

    Signals default to False when insufficient data or zero denominators.

    Args:
        company: Fully populated CompanyData with quarterly financials
            sorted by date.

    Returns:
        QualityMetrics with all F-Score components, composite score,
        gross profitability, and accruals ratio.
    """
    fin = company.financials
    n = len(fin)

    if n == 0:
        logger.warning("%s: empty financials", company.symbol)
        return QualityMetrics(
            f_roa_positive=False,
            f_ocf_positive=False,
            f_roa_improving=False,
            f_accruals_negative=False,
            f_leverage_decreasing=False,
            f_current_ratio_improving=False,
            f_no_dilution=False,
            f_gross_margin_improving=False,
            f_asset_turnover_improving=False,
            f_score=0,
            gross_profitability=None,
            accruals_ratio=None,
        )

    has_current = n >= 4
    has_prior = n >= 8

    # --- Current year TTM (last 4 quarters) ---
    if has_current:
        ttm_net_income = _ttm_sum(fin, "net_income", 0, 4)
        ttm_ocf = _ttm_sum(fin, "operating_cash_flow", 0, 4)
        ttm_gross_profit = _ttm_sum(fin, "gross_profit", 0, 4)
        ttm_revenue = _ttm_sum(fin, "revenue", 0, 4)
    else:
        # Use whatever quarters exist
        ttm_net_income = _safe_sum(fin["net_income"])
        ttm_ocf = _safe_sum(fin["operating_cash_flow"])
        ttm_gross_profit = _safe_sum(fin["gross_profit"])
        ttm_revenue = _safe_sum(fin["revenue"])

    # Current balance sheet (latest quarter)
    current_total_assets = _safe_float(fin["total_assets"].iloc[-1])
    current_long_term_debt = _safe_float(fin["long_term_debt"].iloc[-1])
    current_total_current_assets = _safe_float(
        fin["total_current_assets"].iloc[-1]
    )
    current_total_current_liabilities = _safe_float(
        fin["total_current_liabilities"].iloc[-1]
    )
    current_shares = _safe_float(fin["weighted_average_shs_out_dil"].iloc[-1])

    # --- Prior year TTM (quarters 5-8 back) ---
    if has_prior:
        prior_ttm_net_income = _ttm_sum(fin, "net_income", 4, 4)
        prior_ttm_gross_profit = _ttm_sum(fin, "gross_profit", 4, 4)
        prior_ttm_revenue = _ttm_sum(fin, "revenue", 4, 4)

        # Prior balance sheet (4 quarters ago)
        prior_idx = n - 5  # 0-indexed: 4 quarters back from last
        prior_total_assets = _safe_float(fin["total_assets"].iloc[prior_idx])
        prior_long_term_debt = _safe_float(fin["long_term_debt"].iloc[prior_idx])
        prior_total_current_assets = _safe_float(
            fin["total_current_assets"].iloc[prior_idx]
        )
        prior_total_current_liabilities = _safe_float(
            fin["total_current_liabilities"].iloc[prior_idx]
        )
        prior_shares = _safe_float(
            fin["weighted_average_shs_out_dil"].iloc[prior_idx]
        )

    # === F-Score signals ===

    # 1. ROA positive: TTM net_income / latest total_assets > 0
    if current_total_assets > 0:
        current_roa = ttm_net_income / current_total_assets
        f_roa_positive = current_roa > 0
    else:
        current_roa = 0.0
        f_roa_positive = False

    # 2. OCF positive: TTM operating_cash_flow > 0
    f_ocf_positive = ttm_ocf > 0

    # 3. ROA improving: current TTM ROA > prior year TTM ROA
    if has_prior and current_total_assets > 0 and prior_total_assets > 0:
        prior_roa = prior_ttm_net_income / prior_total_assets
        f_roa_improving = current_roa > prior_roa
    else:
        f_roa_improving = False

    # 4. Accruals negative: TTM OCF > TTM net_income
    f_accruals_negative = ttm_ocf > ttm_net_income

    # 5. Leverage decreasing: current LTD/TA < prior LTD/TA
    if has_prior and current_total_assets > 0 and prior_total_assets > 0:
        current_leverage = current_long_term_debt / current_total_assets
        prior_leverage = prior_long_term_debt / prior_total_assets
        f_leverage_decreasing = current_leverage < prior_leverage
    else:
        f_leverage_decreasing = False

    # 6. Current ratio improving
    if has_prior:
        current_cr = (
            current_total_current_assets / current_total_current_liabilities
            if current_total_current_liabilities > 0
            else float("inf")
        )
        prior_cr = (
            prior_total_current_assets / prior_total_current_liabilities
            if prior_total_current_liabilities > 0
            else float("inf")
        )
        f_current_ratio_improving = current_cr > prior_cr
    else:
        f_current_ratio_improving = False

    # 7. No dilution: current shares <= prior shares
    if has_prior and prior_shares > 0:
        f_no_dilution = current_shares <= prior_shares
    else:
        f_no_dilution = False

    # 8. Gross margin improving: current TTM GM > prior TTM GM
    if has_prior and ttm_revenue > 0 and prior_ttm_revenue > 0:
        current_gm = ttm_gross_profit / ttm_revenue
        prior_gm = prior_ttm_gross_profit / prior_ttm_revenue
        f_gross_margin_improving = current_gm > prior_gm
    else:
        f_gross_margin_improving = False

    # 9. Asset turnover improving: current TTM rev/TA > prior TTM rev/TA
    if has_prior and current_total_assets > 0 and prior_total_assets > 0:
        current_at = ttm_revenue / current_total_assets
        prior_at = prior_ttm_revenue / prior_total_assets
        f_asset_turnover_improving = current_at > prior_at
    else:
        f_asset_turnover_improving = False

    # === Composite F-Score ===
    components = [
        f_roa_positive,
        f_ocf_positive,
        f_roa_improving,
        f_accruals_negative,
        f_leverage_decreasing,
        f_current_ratio_improving,
        f_no_dilution,
        f_gross_margin_improving,
        f_asset_turnover_improving,
    ]
    f_score = sum(components)

    # === Other quality metrics ===

    # Gross profitability (Novy-Marx): TTM gross_profit / latest total_assets
    if current_total_assets > 0:
        gross_profitability = ttm_gross_profit / current_total_assets
    else:
        gross_profitability = None

    # Accruals ratio: (TTM net_income - TTM OCF) / latest total_assets
    if current_total_assets > 0:
        accruals_ratio_val = (ttm_net_income - ttm_ocf) / current_total_assets
    else:
        accruals_ratio_val = None

    return QualityMetrics(
        f_roa_positive=f_roa_positive,
        f_ocf_positive=f_ocf_positive,
        f_roa_improving=f_roa_improving,
        f_accruals_negative=f_accruals_negative,
        f_leverage_decreasing=f_leverage_decreasing,
        f_current_ratio_improving=f_current_ratio_improving,
        f_no_dilution=f_no_dilution,
        f_gross_margin_improving=f_gross_margin_improving,
        f_asset_turnover_improving=f_asset_turnover_improving,
        f_score=f_score,
        gross_profitability=gross_profitability,
        accruals_ratio=accruals_ratio_val,
    )
