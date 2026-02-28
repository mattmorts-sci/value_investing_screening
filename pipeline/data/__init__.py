"""Data loading orchestration."""

from __future__ import annotations

import logging

from pipeline.config import PipelineConfig
from pipeline.data.fmp import load_company as _load_from_fmp
from pipeline.data.fmp import load_universe
from pipeline.data.fmp import lookup_entity_ids
from pipeline.data.live_price import fetch_current_price
from pipeline.data.models import CompanyData

logger = logging.getLogger(__name__)

__all__ = ["CompanyData", "load_company", "load_universe", "lookup_entity_ids"]


def load_company(entity_id: int, config: PipelineConfig) -> CompanyData | None:
    """Load company data from FMP database, then update with live price.

    Loading sequence:
        1. Load all fundamentals and price history from FMP database.
        2. Fetch current price (FMP API, then yfinance fallback).
        3. Recalculate price-dependent fields (latest_price, market_cap).

    Companies without a live price retain database-derived values unchanged.

    Args:
        entity_id: FMP entity ID (from load_universe).
        config: Pipeline configuration.

    Returns:
        Populated CompanyData, or None if critical data is missing.
    """
    company = _load_from_fmp(entity_id, config)
    if company is None:
        return None

    live_price = fetch_current_price(company.symbol)
    if live_price is not None:
        company.latest_price = live_price
        if company.shares_outstanding > 0:
            company.market_cap = live_price * company.shares_outstanding
        logger.info("%s: live price $%.2f", company.symbol, live_price)
    else:
        logger.info(
            "%s: using database price $%.2f", company.symbol, company.latest_price
        )

    if company.latest_price <= 0:
        logger.warning(
            "%s: no valid price (%.2f), skipping",
            company.symbol, company.latest_price,
        )
        return None

    return company
