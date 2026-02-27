"""Summary table rendering for reports and notebooks.

Provides helper functions for rendering DataFrames as formatted
matplotlib table figures. Used by both the notebook and PDF report.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from pipeline.data.contracts import AnalysisResults

logger = logging.getLogger(__name__)


def watchlist_summary(results: AnalysisResults) -> Figure:
    """Render watchlist companies as a formatted summary table.

    Shows key metrics for all watchlist symbols in a compact table:
    Symbol, Price, FCF Growth, Rev Growth, IV/Price, Stability,
    Opp. Rank, Market Cap.

    Args:
        results: Complete pipeline output.

    Returns:
        Matplotlib figure containing the rendered table.
    """
    rankings = results.combined_rankings
    watchlist_set = set(results.watchlist)

    rows: list[list[str]] = []
    for entity_id in rankings.index:
        row = rankings.loc[entity_id]
        symbol = str(row["symbol"])
        if symbol not in watchlist_set:
            continue

        fcf_gr = float(row["fcf_growth_annual"]) * 100
        rev_gr = float(row["revenue_growth_annual"]) * 100
        iv_ratio = float(row["composite_iv_ratio"])
        stability = float(row["growth_stability"])
        opp_rank = int(row["opportunity_rank"])
        price = float(row["current_price"])
        mkt_cap = float(row["market_cap"])

        rows.append([
            symbol,
            f"${price:.2f}",
            f"{fcf_gr:.1f}%",
            f"{rev_gr:.1f}%",
            f"{iv_ratio:.2f}x",
            f"{stability:.2f}",
            f"#{opp_rank}",
            _format_market_cap(mkt_cap),
        ])

    columns = [
        "Symbol", "Price", "FCF Gr.", "Rev Gr.",
        "IV/Price", "Stability", "Opp. Rk", "Mkt Cap",
    ]

    height = max(4, len(rows) * 0.4 + 1.5)
    fig, ax = plt.subplots(figsize=(12, height))
    ax.axis("off")

    if not rows:
        ax.text(
            0.5, 0.5, "No watchlist companies to display",
            ha="center", va="center", fontsize=14, transform=ax.transAxes,
        )
        return fig

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    header_colour = plt.cm.viridis(0.8)  # type: ignore[attr-defined]
    for col_idx in range(len(columns)):
        cell = table[(0, col_idx)]
        cell.set_facecolor(header_colour)
        cell.set_text_props(weight="bold", color="white")

    # Colour-code opportunity rank column
    for row_idx in range(1, len(rows) + 1):
        rank_val = int(rows[row_idx - 1][6].replace("#", ""))
        colour = _rank_colour(rank_val)
        table[(row_idx, 6)].set_facecolor(colour)

    ax.set_title(
        "Watchlist Summary",
        fontsize=16, fontweight="bold", pad=20,
    )

    logger.info("Rendered watchlist summary table with %d companies", len(rows))
    return fig


def filter_summary(results: AnalysisResults) -> Figure:
    """Render filter results as a summary table.

    Shows how many companies each filter removed and which owned
    companies bypassed filters.

    Args:
        results: Complete pipeline output.

    Returns:
        Matplotlib figure containing the filter summary table.
    """
    flog = results.filter_log

    rows: list[list[str]] = []
    for filter_name, removed_symbols in flog.removed.items():
        rows.append([filter_name, str(len(removed_symbols))])

    if flog.owned_bypassed:
        rows.append(["Owned bypassed", ", ".join(flog.owned_bypassed)])

    columns = ["Filter / Category", "Count / Detail"]

    fig, ax = plt.subplots(figsize=(8, max(3, len(rows) * 0.5 + 1.5)))
    ax.axis("off")

    if not rows:
        ax.text(
            0.5, 0.5, "No filter information available",
            ha="center", va="center", fontsize=14, transform=ax.transAxes,
        )
        return fig

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    header_colour = plt.cm.viridis(0.8)  # type: ignore[attr-defined]
    for col_idx in range(len(columns)):
        cell = table[(0, col_idx)]
        cell.set_facecolor(header_colour)
        cell.set_text_props(weight="bold", color="white")

    ax.set_title(
        "Filter Summary",
        fontsize=14, fontweight="bold", pad=20,
    )

    return fig


def _format_market_cap(value: float) -> str:
    """Format market cap with appropriate suffix."""
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"${value / 1e6:.0f}M"
    return f"${value:,.0f}"


def _rank_colour(rank: int) -> Any:
    """Map an opportunity rank to a viridis colour."""
    cmap = plt.colormaps["viridis"]
    if rank <= 5:
        return cmap(0.9)
    if rank <= 10:
        return cmap(0.7)
    if rank <= 20:
        return cmap(0.5)
    return cmap(0.3)
