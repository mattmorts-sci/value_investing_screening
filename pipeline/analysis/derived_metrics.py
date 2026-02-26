"""Derived metrics from raw financial data.

Extracts the latest-period snapshot per company and computes derived
ratios (market_cap, enterprise_value, debt_cash_ratio, etc.).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_derived_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Extract latest-period snapshot per company and compute derived ratios.

    Takes the multi-period DataFrame from the loader (one row per entity
    per quarter) and produces a one-row-per-company DataFrame with the
    latest-period financials plus computed ratios.

    Args:
        data: Multi-period DataFrame from loader. Must contain columns:
            entity_id, period_idx, symbol, company_name, exchange, country,
            fcf, revenue, operating_income, shares_diluted, lt_debt, cash,
            adj_close.

    Returns:
        One-row-per-company DataFrame indexed by entity_id with columns:
            - Carried forward: symbol, company_name, exchange, country,
              fcf, revenue, operating_income, shares_diluted, lt_debt,
              cash, adj_close
            - Computed: market_cap, enterprise_value, debt_cash_ratio,
              fcf_per_share, acquirers_multiple, fcf_to_market_cap

    Raises:
        ValueError: If required columns are missing from the input.
    """
    required = {
        "entity_id", "period_idx", "symbol", "company_name", "exchange",
        "country", "fcf", "revenue", "operating_income", "shares_diluted",
        "lt_debt", "cash", "adj_close",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Latest period per company.
    idx = data.groupby("entity_id")["period_idx"].idxmax()
    latest = data.loc[idx].copy()

    # Computed metrics.
    latest["market_cap"] = latest["adj_close"] * latest["shares_diluted"]

    latest["enterprise_value"] = (
        latest["market_cap"] + latest["lt_debt"] - latest["cash"]
    )

    latest["debt_cash_ratio"] = np.where(
        latest["cash"] == 0,
        np.inf,
        latest["lt_debt"] / latest["cash"],
    )

    latest["fcf_per_share"] = latest["fcf"] / latest["shares_diluted"]

    latest["acquirers_multiple"] = np.where(
        latest["operating_income"] == 0,
        np.inf,
        latest["enterprise_value"] / latest["operating_income"],
    )

    latest["fcf_to_market_cap"] = np.where(
        latest["market_cap"] == 0,
        np.inf,
        latest["fcf"] / latest["market_cap"],
    )

    latest = latest.set_index("entity_id")

    logger.info(
        "Computed derived metrics for %d companies",
        len(latest),
    )
    return latest
