"""Tests for pipeline.data.live_prices."""

from __future__ import annotations

import builtins
from typing import Any
from unittest.mock import MagicMock, patch

from pipeline.data.live_prices import (
    FMPPriceProvider,
    YFinancePriceProvider,
    auto_select_provider,
)

# ---------------------------------------------------------------------------
# FMPPriceProvider
# ---------------------------------------------------------------------------

class TestFMPPriceProvider:

    def test_successful_batch(self) -> None:
        provider = FMPPriceProvider(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"symbol": "AAPL", "price": 150.0},
            {"symbol": "MSFT", "price": 300.0},
        ]

        with patch("pipeline.data.live_prices.requests.get", return_value=mock_response):
            result = provider.get_prices(["AAPL", "MSFT"])

        assert result == {"AAPL": 150.0, "MSFT": 300.0}

    def test_skips_zero_price(self) -> None:
        provider = FMPPriceProvider(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"symbol": "AAPL", "price": 150.0},
            {"symbol": "DEAD", "price": 0},
        ]

        with patch("pipeline.data.live_prices.requests.get", return_value=mock_response):
            result = provider.get_prices(["AAPL", "DEAD"])

        assert "DEAD" not in result
        assert result == {"AAPL": 150.0}

    def test_skips_null_price(self) -> None:
        provider = FMPPriceProvider(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"symbol": "AAPL", "price": None},
        ]

        with patch("pipeline.data.live_prices.requests.get", return_value=mock_response):
            result = provider.get_prices(["AAPL"])

        assert result == {}

    def test_empty_symbols_list(self) -> None:
        provider = FMPPriceProvider(api_key="test-key")
        result = provider.get_prices([])
        assert result == {}

    def test_retries_on_server_error(self) -> None:
        provider = FMPPriceProvider(api_key="test-key")

        error_response = MagicMock()
        error_response.status_code = 500

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.json.return_value = [{"symbol": "AAPL", "price": 150.0}]

        with patch(
            "pipeline.data.live_prices.requests.get",
            side_effect=[error_response, ok_response],
        ), patch("pipeline.data.live_prices.time.sleep"):
            result = provider.get_prices(["AAPL"])

        assert result == {"AAPL": 150.0}

    def test_returns_empty_after_max_retries(self) -> None:
        provider = FMPPriceProvider(api_key="test-key")

        error_response = MagicMock()
        error_response.status_code = 500

        with patch(
            "pipeline.data.live_prices.requests.get",
            return_value=error_response,
        ), patch("pipeline.data.live_prices.time.sleep"):
            result = provider.get_prices(["AAPL"])

        assert result == {}

    def test_batches_large_request(self) -> None:
        """Symbols exceeding batch size should be split into chunks."""
        provider = FMPPriceProvider(api_key="test-key")

        def mock_get(*args: Any, **kwargs: Any) -> MagicMock:
            response = MagicMock()
            response.status_code = 200
            # Parse symbols from the request params
            params = kwargs.get("params", {})
            syms = params.get("symbols", "").split(",")
            response.json.return_value = [
                {"symbol": s, "price": 10.0} for s in syms if s
            ]
            return response

        symbols = [f"SYM{i}" for i in range(150)]

        with patch(
            "pipeline.data.live_prices.requests.get", side_effect=mock_get,
        ), patch("pipeline.data.live_prices.time.sleep"):
            result = provider.get_prices(symbols)

        assert len(result) == 150


# ---------------------------------------------------------------------------
# YFinancePriceProvider
# ---------------------------------------------------------------------------

class TestYFinancePriceProvider:

    def test_import_error_returns_empty(self) -> None:
        provider = YFinancePriceProvider()
        with patch.dict("sys.modules", {"yfinance": None}):
            # Force ImportError by making the import fail
            original_import = builtins.__import__

            def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "yfinance":
                    raise ImportError("no yfinance")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result = provider.get_prices(["AAPL"])
                assert result == {}


# ---------------------------------------------------------------------------
# auto_select_provider
# ---------------------------------------------------------------------------

class TestAutoSelectProvider:

    def test_selects_fmp_when_key_present(self) -> None:
        with patch.dict("os.environ", {"FMP_API_KEY": "test-key"}):
            provider = auto_select_provider()
        assert isinstance(provider, FMPPriceProvider)

    def test_selects_yfinance_when_no_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            provider = auto_select_provider()
        assert isinstance(provider, YFinancePriceProvider)
