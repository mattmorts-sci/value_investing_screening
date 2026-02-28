"""Live price fetching with FMP API primary, yfinance fallback."""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)


def fetch_current_price(symbol: str) -> float | None:
    """Fetch current price for a symbol.

    Tries FMP API first (requires FMP_API_KEY env var),
    falls back to yfinance.

    Args:
        symbol: Stock ticker symbol.

    Returns:
        Current price, or None if unavailable from all sources.
    """
    api_key = os.environ.get("FMP_API_KEY")
    if api_key:
        price = _fetch_fmp_price(symbol, api_key)
        if price is not None:
            return price
        logger.warning("%s: FMP API failed, trying yfinance", symbol)
    else:
        logger.info("%s: no FMP_API_KEY, using yfinance", symbol)

    return _fetch_yfinance_price(symbol)


def _fetch_fmp_price(symbol: str, api_key: str) -> float | None:
    """Fetch price from FMP API."""
    url = "https://financialmodelingprep.com/stable/batch-quote"
    params = {"symbols": symbol, "apikey": api_key}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data and isinstance(data, list) and len(data) > 0:
            price = data[0].get("price")
            if price is not None:
                return float(price)
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.warning("%s: FMP API error: %s", symbol, e)

    return None


def _fetch_yfinance_price(symbol: str) -> float | None:
    """Fetch price from yfinance (fallback)."""
    try:
        import yfinance as yf  # type: ignore[import-untyped]  # lazy import

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.warning("%s: yfinance error: %s", symbol, e)

    return None
