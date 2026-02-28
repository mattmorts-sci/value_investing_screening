"""Tests for Phase 1: data loading."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest
import requests

from pipeline.config import PipelineConfig
from pipeline.data import load_company, load_universe, lookup_entity_ids
from pipeline.data import fmp as fmp_module
from pipeline.data import live_price as live_price_module
from pipeline.data.models import CompanyData

DB_PATH = "/home/mattm/projects/Pers/financial_db/data/fmp.db"


@pytest.fixture
def db_config() -> PipelineConfig:
    """Config with a valid exchange discovered from the database."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    row = conn.execute(
        """
        SELECT exchange FROM entity
        WHERE isEtf = 0 AND isAdr = 0 AND isFund = 0 AND isActivelyTrading = 1
        GROUP BY exchange
        ORDER BY COUNT(*) DESC
        LIMIT 1
        """
    ).fetchone()
    conn.close()
    if row is None:
        pytest.skip("No eligible exchanges in database")
    return PipelineConfig(exchanges=[row[0]])


@pytest.fixture
def sample_entity(db_config: PipelineConfig) -> tuple[int, str]:
    """A valid (entity_id, symbol) from the universe."""
    universe = load_universe(db_config)
    if not universe:
        pytest.skip("Universe is empty")
    return universe[0]


# -- load_universe tests --


class TestLoadUniverse:
    """Tests for entity universe loading and filtering."""

    def test_returns_nonempty(self, db_config: PipelineConfig) -> None:
        universe = load_universe(db_config)
        assert len(universe) > 0

    def test_returns_entity_id_symbol_tuples(self, db_config: PipelineConfig) -> None:
        universe = load_universe(db_config)
        entity_id, symbol = universe[0]
        assert isinstance(entity_id, int)
        assert isinstance(symbol, str)
        assert len(symbol) > 0

    def test_raises_on_empty_exchanges(self) -> None:
        config = PipelineConfig(exchanges=[])
        with pytest.raises(ValueError, match="No exchanges configured"):
            load_universe(config)

    def test_excludes_etfs(self, db_config: PipelineConfig) -> None:
        universe = load_universe(db_config)
        universe_symbols = {s for _, s in universe}

        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        etf_row = conn.execute(
            "SELECT currentSymbol FROM entity "
            "WHERE isEtf = 1 AND exchange IN ("
            + ", ".join("?" for _ in db_config.exchanges)
            + ") LIMIT 1",
            db_config.exchanges,
        ).fetchone()
        conn.close()

        if etf_row:
            assert etf_row[0] not in universe_symbols


# -- lookup_entity_ids tests --


class TestLookupEntityIds:
    """Tests for entity ID lookup from ticker symbols."""

    def test_returns_known_symbol(
        self, db_config: PipelineConfig, sample_entity: tuple[int, str]
    ) -> None:
        entity_id, symbol = sample_entity
        result = lookup_entity_ids([symbol], db_config)
        assert symbol in result
        assert result[symbol] == entity_id

    def test_returns_empty_for_unknown(self, db_config: PipelineConfig) -> None:
        result = lookup_entity_ids(["ZZZZZZ_NONEXISTENT"], db_config)
        assert result == {}

    def test_handles_mixed_known_unknown(
        self, db_config: PipelineConfig, sample_entity: tuple[int, str]
    ) -> None:
        _, symbol = sample_entity
        result = lookup_entity_ids([symbol, "ZZZZZZ_NONEXISTENT"], db_config)
        assert symbol in result
        assert "ZZZZZZ_NONEXISTENT" not in result

    def test_empty_input_returns_empty(self, db_config: PipelineConfig) -> None:
        result = lookup_entity_ids([], db_config)
        assert result == {}

    def test_multiple_valid_symbols(self, db_config: PipelineConfig) -> None:
        universe = load_universe(db_config)
        if len(universe) < 2:
            pytest.skip("Need at least 2 symbols")
        symbols = [s for _, s in universe[:2]]
        result = lookup_entity_ids(symbols, db_config)
        assert len(result) == 2
        for sym in symbols:
            assert sym in result


# -- fmp.load_company tests --


class TestFmpLoadCompany:
    """Tests for FMP database company loading."""

    def test_returns_company_data(
        self, db_config: PipelineConfig, sample_entity: tuple[int, str]
    ) -> None:
        entity_id, _ = sample_entity
        company = fmp_module.load_company(entity_id, db_config)
        assert company is not None
        assert isinstance(company, CompanyData)

    def test_symbol_matches(
        self, db_config: PipelineConfig, sample_entity: tuple[int, str]
    ) -> None:
        entity_id, symbol = sample_entity
        company = fmp_module.load_company(entity_id, db_config)
        assert company is not None
        assert company.symbol == symbol

    def test_financials_has_required_columns(
        self, db_config: PipelineConfig, sample_entity: tuple[int, str]
    ) -> None:
        entity_id, _ = sample_entity
        company = fmp_module.load_company(entity_id, db_config)
        assert company is not None

        expected = {
            "date",
            "fiscal_year",
            "period",
            "revenue",
            "gross_profit",
            "operating_income",
            "ebit",
            "net_income",
            "income_before_tax",
            "income_tax_expense",
            "interest_expense",
            "weighted_average_shs_out_dil",
            "total_assets",
            "total_current_assets",
            "total_current_liabilities",
            "total_debt",
            "long_term_debt",
            "cash_and_cash_equivalents",
            "operating_cash_flow",
            "free_cash_flow",
        }
        assert expected.issubset(set(company.financials.columns))

    def test_financials_sorted_by_date(
        self, db_config: PipelineConfig, sample_entity: tuple[int, str]
    ) -> None:
        entity_id, _ = sample_entity
        company = fmp_module.load_company(entity_id, db_config)
        assert company is not None
        dates = company.financials["date"].tolist()
        assert dates == sorted(dates)

    def test_financials_quarterly_only(
        self, db_config: PipelineConfig, sample_entity: tuple[int, str]
    ) -> None:
        entity_id, _ = sample_entity
        company = fmp_module.load_company(entity_id, db_config)
        assert company is not None
        periods = set(company.financials["period"].unique())
        assert periods.issubset({"Q1", "Q2", "Q3", "Q4"})

    def test_price_history_columns(
        self, db_config: PipelineConfig, sample_entity: tuple[int, str]
    ) -> None:
        entity_id, _ = sample_entity
        company = fmp_module.load_company(entity_id, db_config)
        assert company is not None
        assert "date" in company.price_history.columns
        assert "close" in company.price_history.columns

    def test_price_fields_populated(
        self, db_config: PipelineConfig, sample_entity: tuple[int, str]
    ) -> None:
        entity_id, _ = sample_entity
        company = fmp_module.load_company(entity_id, db_config)
        assert company is not None
        assert company.latest_price >= 0
        assert company.market_cap >= 0
        assert company.shares_outstanding >= 0

    def test_returns_none_for_invalid_entity(self, db_config: PipelineConfig) -> None:
        company = fmp_module.load_company(-1, db_config)
        assert company is None


# -- live_price tests --


class TestLivePrice:
    """Tests for live price fetching with mocked network calls."""

    def test_fmp_api_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FMP_API_KEY", "test_key")
        mock_response = MagicMock()
        mock_response.json.return_value = [{"price": 150.0}]
        mock_response.raise_for_status = MagicMock()

        with patch(
            "pipeline.data.live_price.requests.get", return_value=mock_response
        ):
            price = live_price_module.fetch_current_price("AAPL")
        assert price == 150.0

    def test_fmp_failure_falls_back_to_yfinance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FMP_API_KEY", "test_key")

        with (
            patch(
                "pipeline.data.live_price.requests.get",
                side_effect=requests.RequestException("fail"),
            ),
            patch(
                "pipeline.data.live_price._fetch_yfinance_price",
                return_value=149.0,
            ) as mock_yf,
        ):
            price = live_price_module.fetch_current_price("AAPL")
        assert price == 149.0
        mock_yf.assert_called_once_with("AAPL")

    def test_no_api_key_uses_yfinance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FMP_API_KEY", raising=False)

        with patch(
            "pipeline.data.live_price._fetch_yfinance_price",
            return_value=148.0,
        ):
            price = live_price_module.fetch_current_price("AAPL")
        assert price == 148.0

    def test_all_sources_fail_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FMP_API_KEY", raising=False)

        with patch(
            "pipeline.data.live_price._fetch_yfinance_price",
            return_value=None,
        ):
            price = live_price_module.fetch_current_price("INVALID")
        assert price is None


# -- Integration tests --


class TestLoadCompanyIntegration:
    """Integration: FMP database + mocked live price."""

    def test_live_price_updates_fields(
        self, db_config: PipelineConfig, sample_entity: tuple[int, str]
    ) -> None:
        entity_id, _ = sample_entity

        with patch("pipeline.data.fetch_current_price", return_value=999.99):
            company = load_company(entity_id, db_config)

        assert company is not None
        assert company.latest_price == 999.99
        if company.shares_outstanding > 0:
            expected_mcap = 999.99 * company.shares_outstanding
            assert abs(company.market_cap - expected_mcap) < 0.01

    def test_no_live_price_retains_db_values(
        self, db_config: PipelineConfig, sample_entity: tuple[int, str]
    ) -> None:
        entity_id, _ = sample_entity

        db_company = fmp_module.load_company(entity_id, db_config)
        assert db_company is not None

        with patch("pipeline.data.fetch_current_price", return_value=None):
            company = load_company(entity_id, db_config)

        assert company is not None
        assert company.latest_price == db_company.latest_price
        assert company.market_cap == db_company.market_cap
