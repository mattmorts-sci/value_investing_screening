"""Toggleable financial health filters.

Applies individually configurable filters to the per-company DataFrame
and tracks removals in a FilterLog.
"""

from __future__ import annotations

import logging

import pandas as pd

from pipeline.config.settings import AnalysisConfig
from pipeline.data.contracts import FilterLog

logger = logging.getLogger(__name__)


def apply_filters(
    companies: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, FilterLog]:
    """Apply individually toggleable financial health filters.

    Filters (each controlled by ``config.enable_*``):
        1. Negative/zero FCF (``fcf <= 0``)
        2. Data consistency (``operating_income > revenue``, only when
           ``revenue > 0``)
        3. Minimum market cap (``< config.min_market_cap``)
        4. Maximum debt-to-cash ratio (``> config.max_debt_to_cash_ratio``)

    In owned mode, owned companies bypass active filters with failures
    tracked in the FilterLog.  Post-filter, the result is sorted by
    market_cap descending.

    Args:
        companies: Per-company DataFrame (indexed by entity_id) with at
            least: symbol, fcf, operating_income, revenue, market_cap,
            debt_cash_ratio.
        config: Pipeline configuration.

    Returns:
        Tuple of (filtered DataFrame, FilterLog).

    Raises:
        ValueError: If required columns are missing.
    """
    required = {"symbol", "fcf", "operating_income", "revenue", "market_cap", "debt_cash_ratio"}
    missing = required - set(companies.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    log = FilterLog()
    is_owned_mode = config.mode == "owned" and config.owned_companies
    owned_set = set(config.owned_companies) if is_owned_mode else set()

    # Initialise owned tracking.
    if is_owned_mode:
        for symbol in config.owned_companies:
            in_data = symbol in companies["symbol"].values
            log.owned_tracking[symbol] = {
                "in_initial_data": in_data,
                "filters_passed": [],
                "filters_failed": [],
            }
            if not in_data:
                logger.warning("Owned company %s not found in data", symbol)

    # Build a mask of companies to keep (start with all True).
    keep = pd.Series(True, index=companies.index)

    # Helper: record a symbol lookup for the DataFrame.
    symbols = companies["symbol"]

    def _apply_filter(
        name: str,
        enabled: bool,
        mask: pd.Series,
    ) -> None:
        """Apply a single filter, updating keep mask and log."""
        if not enabled:
            logger.info("Filter '%s' disabled, skipping", name)
            # Track as passed for owned companies present in data.
            if is_owned_mode:
                for sym in owned_set:
                    tracking = log.owned_tracking.get(sym)
                    if tracking and tracking["in_initial_data"]:
                        tracking["filters_passed"].append(name)
            return

        failed = mask
        failed_symbols = symbols[failed].tolist()

        # Track owned companies.
        if is_owned_mode:
            for sym in owned_set:
                if sym in log.owned_tracking:
                    if sym in failed_symbols:
                        log.owned_tracking[sym]["filters_failed"].append(name)
                    else:
                        log.owned_tracking[sym]["filters_passed"].append(name)

        # Record removals (excluding owned companies — they bypass).
        # reasons records the *first* filter that caught each symbol.
        for sym in failed_symbols:
            if sym not in owned_set:
                if name not in log.removed:
                    log.removed[name] = []
                log.removed[name].append(sym)
                if sym not in log.reasons:
                    log.reasons[sym] = name

        # Update the keep mask: remove failed companies that are not owned.
        owned_mask = symbols.isin(owned_set)
        nonlocal keep
        keep = keep & (~failed | owned_mask)

        removed_count = failed.sum() - (failed & owned_mask).sum()
        logger.info("Filter '%s': removed %d companies", name, removed_count)

    # 1. Negative/zero FCF.
    _apply_filter(
        "negative_fcf",
        config.enable_negative_fcf_filter,
        companies["fcf"] <= 0,
    )

    # 2. Data consistency: operating_income > revenue (when revenue > 0).
    _apply_filter(
        "data_consistency",
        config.enable_data_consistency_filter,
        (companies["revenue"] > 0) & (companies["operating_income"] > companies["revenue"]),
    )

    # 3. Minimum market cap.
    _apply_filter(
        "market_cap",
        config.enable_market_cap_filter,
        companies["market_cap"] < config.min_market_cap,
    )

    # 4. Maximum debt-to-cash ratio.
    _apply_filter(
        "debt_cash",
        config.enable_debt_cash_filter,
        companies["debt_cash_ratio"] > config.max_debt_to_cash_ratio,
    )

    # Record owned bypasses.
    if is_owned_mode:
        for sym in owned_set:
            tracking = log.owned_tracking.get(sym, {})
            if tracking.get("in_initial_data") and tracking.get("filters_failed"):
                log.owned_bypassed.append(sym)
                logger.info(
                    "Owned company %s bypassed filters: %s",
                    sym,
                    tracking["filters_failed"],
                )

    result = companies[keep].copy()
    result = result.sort_values("market_cap", ascending=False)

    logger.info(
        "Filtering complete: %d → %d companies",
        len(companies),
        len(result),
    )
    return result, log
