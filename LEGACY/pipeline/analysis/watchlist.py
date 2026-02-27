"""Two-step watchlist selection.

Step 1: Pre-filter by IV/price ratio (top N).
Step 2: From that set, select top M by opportunity rank.

In owned mode, owned companies are always included regardless of rank,
with remaining slots filled from the ranked list.
"""

from __future__ import annotations

import logging

import pandas as pd

from pipeline.config.settings import AnalysisConfig

logger = logging.getLogger(__name__)


def select_watchlist(
    rankings: pd.DataFrame,
    config: AnalysisConfig,
) -> list[str]:
    """Select watchlist using two-step approach.

    Args:
        rankings: Combined rankings DataFrame indexed by entity_id.
            Requires columns: symbol, composite_iv_ratio, opportunity_rank.
        config: Analysis configuration (iv_prefilter_count,
            target_watchlist_size, mode, owned_companies).

    Returns:
        Symbols sorted by opportunity_rank. May be fewer than
        target_watchlist_size if insufficient companies pass
        the pre-filter.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {"symbol", "composite_iv_ratio", "opportunity_rank"}
    missing = required - set(rankings.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if rankings.empty:
        logger.warning("Empty rankings — watchlist is empty")
        return []

    # Step 1: Top N by composite_iv_ratio
    prefiltered = rankings.nlargest(
        config.iv_prefilter_count, "composite_iv_ratio",
    )

    # Step 2: From pre-filtered, top M by opportunity_rank (ascending = best)
    selected = prefiltered.nsmallest(
        config.target_watchlist_size, "opportunity_rank",
    )

    if config.mode == "owned" and config.owned_companies:
        # Owned companies always included
        owned_set = set(config.owned_companies)
        owned_in_rankings = rankings[rankings["symbol"].isin(owned_set)]
        already_selected = set(selected["symbol"])

        # Add missing owned companies
        missing_owned = owned_in_rankings[
            ~owned_in_rankings["symbol"].isin(already_selected)
        ]
        if not missing_owned.empty:
            selected = pd.concat([selected, missing_owned])
            logger.info(
                "Added %d owned companies to watchlist",
                len(missing_owned),
            )

    # Sort by opportunity_rank and extract symbols
    selected = selected.sort_values("opportunity_rank")
    watchlist = selected["symbol"].tolist()

    logger.info(
        "Watchlist: %d companies selected (pre-filter=%d, target=%d)",
        len(watchlist),
        config.iv_prefilter_count,
        config.target_watchlist_size,
    )
    return watchlist
