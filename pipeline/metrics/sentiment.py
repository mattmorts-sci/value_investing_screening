"""Sentiment metrics: 6-month and 12-month price returns."""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass

import pandas as pd

from pipeline.data.models import CompanyData

logger = logging.getLogger(__name__)

# Calendar-day offsets for return lookback periods
_DAYS_6M = 182
_DAYS_12M = 365


@dataclass
class SentimentMetrics:
    """Price momentum outputs.

    Attributes:
        return_6m: 6-month simple price return. None if insufficient data
            or zero reference price.
        return_12m: 12-month simple price return. None if insufficient data
            or zero reference price.
    """

    return_6m: float | None
    return_12m: float | None


def _find_closest_close(
    price_history: pd.DataFrame,
    target_date: datetime.date,
) -> float | None:
    """Find the close price on the date nearest to target_date.

    Args:
        price_history: DataFrame with 'date' (datetime64) and 'close' columns,
            sorted by date.
        target_date: The target calendar date.

    Returns:
        The close price on the nearest available date, or None if the
        DataFrame is empty.
    """
    if price_history.empty:
        return None

    deltas = (price_history["date"] - pd.Timestamp(target_date)).abs()
    idx = deltas.idxmin()
    return float(price_history.loc[idx, "close"])  # type: ignore[arg-type]


def _compute_return(
    price_history: pd.DataFrame,
    latest_date: datetime.date,
    latest_close: float,
    lookback_days: int,
) -> float | None:
    """Compute simple return over a lookback period.

    Returns None if the price history does not span the required period
    or the reference close is zero.

    Args:
        price_history: DataFrame with 'date' (datetime64) and 'close' columns.
        latest_date: The most recent date in the price history.
        latest_close: The close price on the latest date.
        lookback_days: Number of calendar days to look back.

    Returns:
        Simple return as a float, or None.
    """
    earliest_date = price_history["date"].min().date()
    target_date = latest_date - datetime.timedelta(days=lookback_days)

    # Data must span the lookback period
    if earliest_date > target_date:
        return None

    ref_close = _find_closest_close(price_history, target_date)
    if ref_close is None or ref_close == 0.0:
        return None

    return (latest_close - ref_close) / ref_close


def compute_sentiment(company: CompanyData) -> SentimentMetrics:
    """Compute sentiment (price momentum) metrics for a single company.

    Calculates 6-month and 12-month simple price returns from the daily
    price history. Uses the closest available trading date when the exact
    target date is not present in the data.

    Args:
        company: Fully populated CompanyData.

    Returns:
        SentimentMetrics with return_6m and return_12m.
    """
    price_history = company.price_history

    if price_history.empty or len(price_history) < 2:
        logger.debug("%s: insufficient price history for sentiment", company.symbol)
        return SentimentMetrics(return_6m=None, return_12m=None)

    # Ensure date column is datetime for arithmetic
    ph = price_history.copy()
    ph["date"] = pd.to_datetime(ph["date"])
    ph = ph.sort_values("date").reset_index(drop=True)

    latest_date = ph["date"].max().date()
    latest_close = float(ph.loc[ph["date"].idxmax(), "close"])  # type: ignore[arg-type]

    if latest_close == 0.0:
        logger.debug("%s: latest close is zero", company.symbol)
        return SentimentMetrics(return_6m=None, return_12m=None)

    return_6m = _compute_return(ph, latest_date, latest_close, _DAYS_6M)
    return_12m = _compute_return(ph, latest_date, latest_close, _DAYS_12M)

    logger.debug(
        "%s: return_6m=%s, return_12m=%s",
        company.symbol,
        f"{return_6m:.4f}" if return_6m is not None else "None",
        f"{return_12m:.4f}" if return_12m is not None else "None",
    )

    return SentimentMetrics(return_6m=return_6m, return_12m=return_12m)
