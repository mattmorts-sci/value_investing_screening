"""Pipeline orchestrator.

Executes the 12-step analysis pipeline from config to AnalysisResults.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.analysis.dcf import calculate_all_dcf
from pipeline.analysis.derived_metrics import compute_derived_metrics
from pipeline.analysis.factor_analysis import (
    analyze_factor_dominance,
    calculate_factor_contributions,
    create_quadrant_analysis,
)
from pipeline.analysis.filtering import apply_filters
from pipeline.analysis.growth_projection import project_all
from pipeline.analysis.growth_stats import compute_growth_statistics
from pipeline.analysis.ranking import rank_companies
from pipeline.analysis.watchlist import select_watchlist
from pipeline.analysis.weighted_scoring import calculate_weighted_scores
from pipeline.config.settings import AnalysisConfig
from pipeline.data.contracts import AnalysisResults
from pipeline.data.live_prices import auto_select_provider
from pipeline.data.loader import load_raw_data

logger = logging.getLogger(__name__)


def _update_live_metrics(
    companies: pd.DataFrame,
    live_prices: dict[str, float],
) -> pd.DataFrame:
    """Update market_cap, enterprise_value, and acquirers_multiple from live prices.

    Companies without a live price retain their DB-derived values unchanged.

    Args:
        companies: Per-company DataFrame indexed by entity_id.
        live_prices: {symbol: current_price}.

    Returns:
        Updated DataFrame (copy).
    """
    df = companies.copy()

    for entity_id in df.index:
        symbol = str(df.at[entity_id, "symbol"])
        if symbol not in live_prices:
            continue

        price = live_prices[symbol]
        shares = float(df.at[entity_id, "shares_diluted"])  # type: ignore[arg-type]

        df.at[entity_id, "adj_close"] = price
        new_mc = price * shares
        df.at[entity_id, "market_cap"] = new_mc
        lt_debt = float(df.at[entity_id, "lt_debt"])  # type: ignore[arg-type]
        cash = float(df.at[entity_id, "cash"])  # type: ignore[arg-type]
        df.at[entity_id, "enterprise_value"] = new_mc + lt_debt - cash

        oi = float(df.at[entity_id, "operating_income"])  # type: ignore[arg-type]
        ev = float(df.at[entity_id, "enterprise_value"])  # type: ignore[arg-type]
        df.at[entity_id, "acquirers_multiple"] = (
            np.inf if oi == 0 else ev / oi
        )

        fcf = float(df.at[entity_id, "fcf"])  # type: ignore[arg-type]
        df.at[entity_id, "fcf_to_market_cap"] = (
            np.inf if new_mc == 0 else fcf / new_mc
        )

    updated_count = sum(
        1 for eid in df.index
        if str(df.at[eid, "symbol"]) in live_prices
    )
    logger.info(
        "Updated live metrics for %d / %d companies",
        updated_count,
        len(df),
    )
    return df


def _export_csv(
    results: AnalysisResults,
    output_dir: Path,
) -> None:
    """Export ranking DataFrames and watchlist to CSV files.

    Args:
        results: Complete pipeline output.
        output_dir: Directory for CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    exports = {
        "growth_rankings.csv": results.growth_rankings,
        "value_rankings.csv": results.value_rankings,
        "weighted_rankings.csv": results.weighted_rankings,
        "combined_rankings.csv": results.combined_rankings,
        "factor_contributions.csv": results.factor_contributions,
    }

    for filename, df in exports.items():
        path = output_dir / filename
        df.to_csv(path)
        logger.info("Exported %s (%d rows)", filename, len(df))

    # Watchlist as single-column CSV.
    watchlist_path = output_dir / "watchlist.csv"
    pd.DataFrame({"symbol": results.watchlist}).to_csv(
        watchlist_path, index=False,
    )
    logger.info("Exported watchlist.csv (%d symbols)", len(results.watchlist))


def run_analysis(config: AnalysisConfig) -> AnalysisResults:
    """Execute the 12-step analysis pipeline.

    Args:
        config: Analysis configuration. Validated on construction.

    Returns:
        AnalysisResults with all data for charts and reports.
    """
    logger.info(
        "Starting analysis: market=%s, mode=%s",
        config.market,
        config.mode,
    )

    # Step 1: Validate config.
    # AnalysisConfig.__post_init__ handles structural validation.
    # Additional runtime check: primary_period must be in projection_periods.
    if config.primary_period not in config.projection_periods:
        raise ValueError(
            f"primary_period ({config.primary_period}) must be in "
            f"projection_periods ({config.projection_periods})."
        )
    logger.info("Config validated")

    # Step 2: Load data.
    output_dir = config.output_directory / config.market.lower()
    raw = load_raw_data(config, output_dir=output_dir / "data_loading")
    logger.info(
        "Loaded %d rows, %d companies, periods %s",
        raw.row_count,
        raw.company_count,
        raw.period_range,
    )

    # Step 3: Compute derived metrics (latest-period snapshot).
    companies = compute_derived_metrics(raw.data)
    logger.info("Computed derived metrics for %d companies", len(companies))

    # Step 4: Compute growth statistics.
    growth_stats = compute_growth_statistics(raw.data, config)
    logger.info("Computed growth stats for %d companies", len(growth_stats))

    # Step 5: Merge stats into companies DataFrame, then filter.
    # Growth stats columns needed downstream (filtering doesn't need them,
    # but the merged DataFrame is used after filtering).
    companies = companies.join(growth_stats, how="left")
    filtered, filter_log = apply_filters(companies, config)
    logger.info(
        "Filtering: %d -> %d companies (%d removed)",
        len(companies),
        len(filtered),
        len(companies) - len(filtered),
    )

    # Use filtered set for all subsequent steps.
    companies = filtered
    # Growth stats for filtered companies only.
    growth_stats = growth_stats.loc[growth_stats.index.isin(companies.index)]

    # Step 6: Growth projection.
    projections = project_all(companies, growth_stats, config)
    logger.info("Projected growth for %d companies", len(projections))

    # Step 7: DCF intrinsic value.
    intrinsic_values = calculate_all_dcf(companies, projections, config)
    logger.info("Calculated DCF for %d companies", len(intrinsic_values))

    # Step 8: Fetch live prices.
    symbols = companies["symbol"].tolist()
    provider = auto_select_provider()
    live_prices = provider.get_prices(symbols)
    logger.info(
        "Fetched live prices: %d / %d symbols",
        len(live_prices),
        len(symbols),
    )

    # Preserve filing-period AM before live-price overwrite (used by AM chart).
    companies["filing_acquirers_multiple"] = companies["acquirers_multiple"]

    # Step 9: Update metrics with live prices.
    companies = _update_live_metrics(companies, live_prices)

    # Step 10: Weighted scoring + ranking.
    weighted_scores = calculate_weighted_scores(
        companies, growth_stats, config,
    )
    growth_rankings, value_rankings, weighted_rankings, combined_rankings = (
        rank_companies(
            companies,
            growth_stats,
            projections,
            intrinsic_values,
            weighted_scores,
            live_prices,
            config,
        )
    )
    logger.info(
        "Ranked %d companies across 4 DataFrames",
        len(combined_rankings),
    )

    # Step 11: Factor analysis.
    factor_contributions = calculate_factor_contributions(weighted_scores)
    factor_dominance = analyze_factor_dominance(
        factor_contributions, weighted_scores,
    )
    quadrant_df = create_quadrant_analysis(
        companies, intrinsic_values, live_prices, growth_stats, config,
    )
    logger.info(
        "Factor analysis: %d contributions, %d dominance rows, "
        "%d quadrant assignments",
        len(factor_contributions),
        len(factor_dominance),
        len(quadrant_df),
    )

    # Step 12: Select watchlist.
    watchlist = select_watchlist(combined_rankings, config)
    logger.info("Watchlist: %d companies selected", len(watchlist))

    # Assemble results.
    results = AnalysisResults(
        time_series=raw.data,
        companies=companies,
        projections=projections,
        intrinsic_values=intrinsic_values,
        growth_rankings=growth_rankings,
        value_rankings=value_rankings,
        weighted_rankings=weighted_rankings,
        combined_rankings=combined_rankings,
        factor_contributions=factor_contributions,
        factor_dominance=factor_dominance,
        quadrant_analysis=quadrant_df,
        watchlist=watchlist,
        filter_log=filter_log,
        live_prices=live_prices,
        config=config,
    )

    # Export CSVs.
    _export_csv(results, output_dir)

    logger.info(
        "Analysis complete: %d companies ranked, %d on watchlist",
        len(combined_rankings),
        len(watchlist),
    )
    return results
