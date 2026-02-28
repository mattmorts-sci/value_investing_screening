"""FMP database data loading (read-only, explicit column selects)."""

from __future__ import annotations

import logging
import sqlite3

import pandas as pd

from pipeline.config import PipelineConfig
from pipeline.data.models import CompanyData

logger = logging.getLogger(__name__)

# Column mappings: FMP camelCase → pipeline snake_case
_INCOME_COLUMNS = {
    "revenue": "revenue",
    "grossProfit": "gross_profit",
    "operatingIncome": "operating_income",
    "ebit": "ebit",
    "netIncome": "net_income",
    "incomeBeforeTax": "income_before_tax",
    "incomeTaxExpense": "income_tax_expense",
    "interestExpense": "interest_expense",
    "weightedAverageShsOutDil": "weighted_average_shs_out_dil",
}

_BALANCE_COLUMNS = {
    "totalAssets": "total_assets",
    "totalCurrentAssets": "total_current_assets",
    "totalCurrentLiabilities": "total_current_liabilities",
    "totalDebt": "total_debt",
    "longTermDebt": "long_term_debt",
    "cashAndCashEquivalents": "cash_and_cash_equivalents",
}

_CASHFLOW_COLUMNS = {
    "operatingCashFlow": "operating_cash_flow",
    "freeCashFlow": "free_cash_flow",
}


def _connect(config: PipelineConfig) -> sqlite3.Connection:
    """Open a read-only connection to the FMP database."""
    return sqlite3.connect(f"file:{config.db_path}?mode=ro", uri=True)


def load_universe(config: PipelineConfig) -> list[tuple[int, str]]:
    """Load eligible (entity_id, symbol) pairs from entity table.

    Filters: non-ETF, non-ADR, non-fund, actively trading,
    exchange in configured exchanges.

    Args:
        config: Pipeline configuration (must have non-empty exchanges).

    Returns:
        List of (entity_id, symbol) tuples, sorted by symbol.

    Raises:
        ValueError: If no exchanges configured.
    """
    if not config.exchanges:
        raise ValueError("No exchanges configured")

    placeholders = ", ".join("?" for _ in config.exchanges)
    query = f"""
        SELECT entityId, currentSymbol
        FROM entity
        WHERE isEtf = 0
          AND isAdr = 0
          AND isFund = 0
          AND isActivelyTrading = 1
          AND exchange IN ({placeholders})
        ORDER BY currentSymbol
    """

    conn = _connect(config)
    try:
        cursor = conn.execute(query, config.exchanges)
        return cursor.fetchall()
    finally:
        conn.close()


def lookup_entity_ids(
    symbols: list[str], config: PipelineConfig
) -> dict[str, int]:
    """Look up entity IDs for a list of ticker symbols.

    Queries the entity table for matching currentSymbol values. If
    multiple entities share the same symbol, prefers actively-trading
    entities, then the highest entity ID.

    Args:
        symbols: Ticker symbols to look up.
        config: Pipeline configuration (provides db_path).

    Returns:
        Dict mapping found symbols to their entity IDs. Symbols not
        found in the database are omitted.
    """
    if not symbols:
        return {}

    conn = _connect(config)
    try:
        placeholders = ", ".join("?" for _ in symbols)
        query = f"""
            SELECT currentSymbol, entityId
            FROM entity
            WHERE currentSymbol IN ({placeholders})
            ORDER BY isActivelyTrading DESC, entityId DESC
        """
        rows = conn.execute(query, symbols).fetchall()

        # First occurrence per symbol wins (actively trading, highest ID)
        result: dict[str, int] = {}
        for symbol, entity_id in rows:
            if symbol not in result:
                result[symbol] = entity_id
        return result
    finally:
        conn.close()


def load_company(entity_id: int, config: PipelineConfig) -> CompanyData | None:
    """Load all data for one company from FMP database.

    Loads entity metadata, quarterly financials (income statement +
    balance sheet + cash flow joined), price history, and company
    profile. Returns None if critical data is missing.

    Args:
        entity_id: FMP entity ID (from load_universe).
        config: Pipeline configuration.

    Returns:
        CompanyData with database values, or None if data is missing.
    """
    conn = _connect(config)
    conn.row_factory = sqlite3.Row
    try:
        # Entity metadata
        entity = conn.execute(
            "SELECT currentSymbol, companyName, sector, exchange "
            "FROM entity WHERE entityId = ?",
            (entity_id,),
        ).fetchone()
        if entity is None:
            logger.warning("Entity %d not found", entity_id)
            return None

        symbol = entity["currentSymbol"]

        # Quarterly financials: join income + balance + cashflow
        income_cols = ", ".join(f"i.{c}" for c in _INCOME_COLUMNS)
        balance_cols = ", ".join(f"b.{c}" for c in _BALANCE_COLUMNS)
        cashflow_cols = ", ".join(f"c.{c}" for c in _CASHFLOW_COLUMNS)

        financials_query = f"""
            SELECT i.date, i.fiscalYear, i.period,
                   {income_cols},
                   {balance_cols},
                   {cashflow_cols}
            FROM incomeStatement i
            JOIN balanceSheet b
                ON i.entityId = b.entityId
                AND i.date = b.date
                AND i.period = b.period
            JOIN cashFlow c
                ON i.entityId = c.entityId
                AND i.date = c.date
                AND i.period = c.period
            WHERE i.entityId = ?
              AND i.period IN ('Q1', 'Q2', 'Q3', 'Q4')
            ORDER BY i.date
        """
        financials_df = pd.read_sql_query(financials_query, conn, params=(entity_id,))

        if financials_df.empty:
            logger.warning("%s: no quarterly financials", symbol)
            return None

        # Rename columns to snake_case
        rename_map: dict[str, str] = {"fiscalYear": "fiscal_year"}
        rename_map.update(_INCOME_COLUMNS)
        rename_map.update(_BALANCE_COLUMNS)
        rename_map.update(_CASHFLOW_COLUMNS)
        financials_df = financials_df.rename(columns=rename_map)

        # Price history
        price_df = pd.read_sql_query(
            "SELECT date, close FROM eodPrice WHERE entityId = ? ORDER BY date",
            conn,
            params=(entity_id,),
        )
        if price_df.empty:
            logger.warning("%s: no price history", symbol)

        # Company profile (price, market cap)
        profile = conn.execute(
            "SELECT price, marketCap FROM companyProfile WHERE entityId = ?",
            (entity_id,),
        ).fetchone()
        if profile is None:
            logger.warning("%s: no company profile", symbol)
            return None

        latest_price = (
            float(profile["price"]) if profile["price"] is not None else 0.0
        )
        market_cap = (
            float(profile["marketCap"]) if profile["marketCap"] is not None else 0.0
        )

        # Shares outstanding from latest quarter's diluted shares
        latest_shares = financials_df["weighted_average_shs_out_dil"].iloc[-1]
        shares_outstanding = float(latest_shares) if pd.notna(latest_shares) else 0.0
        if shares_outstanding == 0:
            logger.warning("%s: zero shares outstanding", symbol)

        return CompanyData(
            symbol=symbol,
            company_name=entity["companyName"] or "",
            sector=entity["sector"] or "",
            exchange=entity["exchange"] or "",
            financials=financials_df,
            latest_price=latest_price,
            market_cap=market_cap,
            shares_outstanding=shares_outstanding,
            price_history=price_df,
        )
    finally:
        conn.close()
