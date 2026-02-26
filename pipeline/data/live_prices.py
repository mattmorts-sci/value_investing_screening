"""Live price fetching.

Provides a PriceProvider protocol with two implementations:
- FMPPriceProvider: Primary, using FMP /stable/batch-quote endpoint.
- YFinancePriceProvider: Fallback when FMP API key is unavailable.

Adapted from algorithmic-investing/prediction/live_data.py.
"""

import logging
import os
import time
from typing import Protocol

import requests

logger = logging.getLogger(__name__)

_FMP_BATCH_SIZE = 100
_FMP_MAX_RETRIES = 3
_FMP_BACKOFF_FACTOR = 1.0
_FMP_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
_FMP_REQUEST_TIMEOUT = 30
_FMP_BATCH_SLEEP = 0.3


class PriceProvider(Protocol):
    """Interface for fetching current market prices."""

    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        """Fetch current prices for the given ticker symbols.

        Args:
            symbols: List of ticker symbols (e.g. ["AAPL", "MSFT"]).

        Returns:
            symbol -> current price. Symbols with no price omitted.
        """
        ...


class FMPPriceProvider:
    """Fetch live prices from the FMP /stable/batch-quote endpoint.

    Batches symbols in chunks of 100 per request. Retries with
    exponential backoff on 429/5xx status codes.

    Args:
        api_key: FMP API key (from FMP_API_KEY environment variable).
        base_url: FMP API base URL.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://financialmodelingprep.com/stable",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url

    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        """Fetch current prices for all symbols via FMP batch-quote.

        Args:
            symbols: Ticker symbols to look up.

        Returns:
            symbol -> price mapping. Missing symbols omitted.
        """
        result: dict[str, float] = {}

        for i in range(0, len(symbols), _FMP_BATCH_SIZE):
            chunk = symbols[i : i + _FMP_BATCH_SIZE]
            chunk_prices = self._fetch_batch(chunk)
            result.update(chunk_prices)

            if i + _FMP_BATCH_SIZE < len(symbols):
                time.sleep(_FMP_BATCH_SLEEP)

        logger.info(
            "FMP: fetched prices for %d of %d symbols",
            len(result),
            len(symbols),
        )
        return result

    def _fetch_batch(self, symbols: list[str]) -> dict[str, float]:
        """Fetch prices for a single batch with retry logic.

        Args:
            symbols: Up to 100 ticker symbols.

        Returns:
            symbol -> price for symbols in this batch.
        """
        symbols_str = ",".join(symbols)
        url = f"{self._base_url}/batch-quote"
        params = {"symbols": symbols_str, "apikey": self._api_key}

        for attempt in range(_FMP_MAX_RETRIES):
            try:
                response = requests.get(
                    url,
                    params=params,
                    timeout=_FMP_REQUEST_TIMEOUT,
                )

                if response.status_code in _FMP_RETRY_STATUS_CODES:
                    sleep_time = _FMP_BACKOFF_FACTOR * (2**attempt)
                    logger.warning(
                        "FMP batch-quote returned %d, retrying in %.1fs "
                        "(attempt %d/%d)",
                        response.status_code,
                        sleep_time,
                        attempt + 1,
                        _FMP_MAX_RETRIES,
                    )
                    time.sleep(sleep_time)
                    continue

                response.raise_for_status()
                data = response.json()

                result: dict[str, float] = {}
                if isinstance(data, list):
                    for item in data:
                        symbol = item.get("symbol")
                        price = item.get("price")
                        if symbol and price is not None and price > 0:
                            result[symbol] = float(price)
                return result

            except requests.RequestException as e:
                if attempt < _FMP_MAX_RETRIES - 1:
                    sleep_time = _FMP_BACKOFF_FACTOR * (2**attempt)
                    logger.warning(
                        "FMP request failed: %s. Retrying in %.1fs "
                        "(attempt %d/%d)",
                        e,
                        sleep_time,
                        attempt + 1,
                        _FMP_MAX_RETRIES,
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        "FMP batch-quote failed after %d attempts: %s",
                        _FMP_MAX_RETRIES,
                        e,
                    )

        logger.error(
            "FMP batch-quote failed after %d attempts for %d symbols",
            _FMP_MAX_RETRIES,
            len(symbols),
        )
        return {}


class YFinancePriceProvider:
    """Fetch live prices via yfinance (fallback provider).

    yfinance is imported lazily to avoid requiring the optional
    dependency when FMP is available.
    """

    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        """Fetch current prices via yfinance.download.

        Args:
            symbols: Ticker symbols to look up.

        Returns:
            symbol -> price mapping. Missing symbols omitted.
        """
        try:
            import yfinance as yf  # noqa: PLC0415
        except ImportError:
            logger.error(
                "yfinance is not installed. Install it with: "
                "pip install yfinance"
            )
            return {}

        try:
            data = yf.download(
                tickers=symbols,
                period="1d",
                progress=False,
                auto_adjust=True,
            )
        except Exception:
            logger.error("yfinance download failed", exc_info=True)
            return {}

        result: dict[str, float] = {}

        if data.empty:
            logger.warning("yfinance returned empty data")
            return result

        if len(symbols) == 1:
            if "Close" in data.columns:
                price = float(data["Close"].iloc[-1])
                if price > 0:
                    result[symbols[0]] = price
        else:
            if "Close" in data.columns.get_level_values(0):
                close_data = data["Close"]
                for sym in symbols:
                    if sym in close_data.columns:
                        val = close_data[sym].iloc[-1]
                        if val is not None and float(val) > 0:
                            result[sym] = float(val)

        logger.info(
            "yfinance: fetched prices for %d of %d symbols",
            len(result),
            len(symbols),
        )
        return result


def auto_select_provider() -> PriceProvider:
    """Select price provider based on available credentials.

    Returns FMPPriceProvider if FMP_API_KEY is set in the environment,
    otherwise falls back to YFinancePriceProvider.

    Returns:
        A PriceProvider instance.
    """
    api_key = os.environ.get("FMP_API_KEY")
    if api_key:
        logger.info("Using FMPPriceProvider (FMP_API_KEY found)")
        return FMPPriceProvider(api_key=api_key)
    logger.warning("FMP_API_KEY not found, falling back to yfinance")
    return YFinancePriceProvider()
