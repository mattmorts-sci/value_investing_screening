"""Data models for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CompanyData:
    """Central data contract consumed by all metric modules.

    Attributes:
        symbol: Stock ticker symbol.
        company_name: Company name.
        sector: Business sector.
        exchange: Stock exchange.
        financials: Quarterly financials, one row per quarter, sorted by date.
            Columns: date, fiscal_year, period,
            revenue, gross_profit, operating_income, ebit, net_income,
            income_before_tax, income_tax_expense, interest_expense,
            weighted_average_shs_out_dil,
            total_assets, total_current_assets, total_current_liabilities,
            total_debt, long_term_debt, cash_and_cash_equivalents,
            operating_cash_flow, free_cash_flow.
        latest_price: Most recent stock price.
        market_cap: Market capitalisation.
        shares_outstanding: Diluted shares outstanding.
        price_history: Daily price history. Columns: date, close.
    """

    symbol: str
    company_name: str
    sector: str
    exchange: str
    financials: pd.DataFrame
    latest_price: float
    market_cap: float
    shares_outstanding: float
    price_history: pd.DataFrame
