"""Market-wide overview charts for the value investing screening pipeline.

Six chart functions producing market-level visualisations from
AnalysisResults. All return matplotlib Figure objects; the caller
owns the figure lifecycle.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from pipeline.data.contracts import AnalysisResults

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OUTLIER_GROWTH_THRESHOLD = 1.0  # 100% combined growth — skip as outlier
_MAX_COMPANIES = 20  # cap for bar charts
_VIRIDIS = "viridis"
_VIRIDIS_R = "viridis_r"

# Quadrant label positions (x-frac, y-frac relative to axes)
_SCATTER_FONTSIZE = 7
_ANNOTATION_FONTSIZE = 6
_TITLE_FONTSIZE = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _market_cap_sizes(
    market_caps: pd.Series,
    min_size: float = 30.0,
    max_size: float = 600.0,
) -> np.ndarray:
    """Scale market_cap to bubble sizes for scatter plots (log scale)."""
    caps = np.asarray(market_caps.fillna(0), dtype=float)
    caps = np.maximum(caps, 1.0)  # avoid log(0)
    log_caps = np.log10(caps)
    if log_caps.max() == log_caps.min():
        return np.full(len(caps), (min_size + max_size) / 2)
    normalised = (log_caps - log_caps.min()) / (log_caps.max() - log_caps.min())
    return normalised * (max_size - min_size) + min_size  # type: ignore[no-any-return]


def _size_legend(
    ax: Axes,
    market_caps: pd.Series,
    min_size: float = 30.0,
    max_size: float = 600.0,
) -> None:
    """Add a market-cap size legend to a scatter axes."""
    market_caps = market_caps.fillna(0)
    cap_min = market_caps.min()
    cap_max = market_caps.max()
    cap_mid = (cap_min + cap_max) / 2

    labels = []
    sizes = []
    for cap_val, label in [
        (cap_min, f"${cap_min / 1e6:.0f}M"),
        (cap_mid, f"${cap_mid / 1e6:.0f}M"),
        (cap_max, f"${cap_max / 1e6:.0f}M"),
    ]:
        if cap_max == cap_min:
            s = (min_size + max_size) / 2
        else:
            s = (cap_val - cap_min) / (cap_max - cap_min) * (max_size - min_size) + min_size
        sizes.append(s)
        labels.append(label)

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="grey",
            markersize=np.sqrt(s),
            linestyle="None",
            label=lbl,
        )
        for s, lbl in zip(sizes, labels, strict=False)
    ]
    ax.legend(
        handles=handles,
        title="Market Cap",
        loc="upper left",
        fontsize=_ANNOTATION_FONTSIZE,
        title_fontsize=_ANNOTATION_FONTSIZE,
        framealpha=0.7,
    )


def _outlier_textbox(
    ax: Axes,
    outlier_symbols: list[str],
    label: str = "Outliers excluded",
) -> None:
    """Annotate outlier exclusions as a text box on the axes."""
    if not outlier_symbols:
        return
    text = f"{label}:\n" + ", ".join(outlier_symbols)
    ax.text(
        0.02, 0.02, text,
        transform=ax.transAxes,
        fontsize=_ANNOTATION_FONTSIZE,
        verticalalignment="bottom",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.7},
    )


def _watchlist_entity_ids(results: AnalysisResults) -> list[int]:
    """Return entity_ids for watchlist symbols present in combined_rankings."""
    cr = results.combined_rankings
    if cr.empty:
        return []
    symbol_to_eid = (
        cr["symbol"].reset_index()
        .drop_duplicates(subset="symbol")
        .set_index("symbol")["entity_id"]
    )
    eids = []
    for sym in results.watchlist:
        if sym in symbol_to_eid.index:
            eids.append(int(symbol_to_eid[sym]))
    return eids


# ---------------------------------------------------------------------------
# 1. Growth vs Value scatter
# ---------------------------------------------------------------------------


def growth_value_scatter(results: AnalysisResults) -> Figure:
    """Scatter plot of growth rank vs IV/price ratio.

    Bubble size represents market capitalisation; colour represents
    opportunity rank (viridis_r — darker = better rank). Quadrant
    labels classify companies by growth/value characteristics.
    Companies with >100% combined growth are excluded as outliers.

    Args:
        results: Complete pipeline output.

    Returns:
        Matplotlib Figure with the scatter plot.
    """
    cr = results.combined_rankings.copy()
    if cr.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title("Growth vs Value — no data")
        return fig

    # Separate outliers
    outlier_mask = cr["combined_growth"].abs() > _OUTLIER_GROWTH_THRESHOLD
    outlier_symbols = cr.loc[outlier_mask, "symbol"].tolist()
    if outlier_symbols:
        logger.info(
            "growth_value_scatter: excluding %d outliers (>100%% growth): %s",
            len(outlier_symbols), ", ".join(outlier_symbols),
        )
    df = cr.loc[~outlier_mask].copy()

    if df.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title("Growth vs Value — all companies are outliers")
        _outlier_textbox(ax, outlier_symbols)
        return fig

    # Rank within the filtered set for x-axis
    df["growth_rank"] = df["combined_growth"].rank(ascending=False, method="min")

    fig, ax = plt.subplots(figsize=(12, 8))

    sizes = _market_cap_sizes(df["market_cap"])
    scatter = ax.scatter(
        df["growth_rank"],
        df["composite_iv_ratio"],
        s=sizes,
        c=df["opportunity_rank"],
        cmap=_VIRIDIS_R,
        alpha=0.75,
        edgecolors="grey",
        linewidths=0.5,
    )

    # Annotate watchlist tickers only
    watchlist_eids = set(_watchlist_entity_ids(results))
    for eid, row in df.iterrows():
        if eid not in watchlist_eids:
            continue
        ax.annotate(
            row["symbol"],
            (row["growth_rank"], row["composite_iv_ratio"]),
            fontsize=_ANNOTATION_FONTSIZE,
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )

    # Reference lines
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=0.8, label="IV = Price (fair value)")
    ax.axhline(y=2.0, color="green", linestyle="--", linewidth=0.8, label="IV = 2x Price")

    # Invert x-axis (lower rank = better, on right)
    ax.invert_xaxis()

    # Quadrant labels
    quadrants = [
        (0.95, 0.95, "Best Buys\n(High Growth / Good Value)"),
        (0.05, 0.95, "Deep Value\n(Low Growth / Good Value)"),
        (0.95, 0.05, "Too Expensive\n(High Growth / Poor Value)"),
        (0.05, 0.05, "Avoid\n(Low Growth / Poor Value)"),
    ]
    for x_frac, y_frac, label in quadrants:
        ax.text(
            x_frac, y_frac, label,
            transform=ax.transAxes,
            fontsize=_SCATTER_FONTSIZE,
            ha="right" if x_frac > 0.5 else "left",
            va="top" if y_frac > 0.5 else "bottom",
            alpha=0.4,
            fontstyle="italic",
        )

    # Colour bar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Opportunity Rank (lower = better)")

    # Size legend
    _size_legend(ax, df["market_cap"])

    _outlier_textbox(ax, outlier_symbols)

    ax.set_xlabel("Combined Growth Rank (lower = better, right side)")
    ax.set_ylabel("Composite IV / Price Ratio")
    ax.set_title("Growth vs Value Overview", fontsize=_TITLE_FONTSIZE)
    ax.legend(loc="lower left", fontsize=_ANNOTATION_FONTSIZE, framealpha=0.7)
    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# 2. Projected growth comparison (bar chart)
# ---------------------------------------------------------------------------


def growth_comparison_historical(results: AnalysisResults) -> Figure:
    """Grouped bar chart of historical FCF CAGR vs Revenue CAGR.

    Computes historical annualised growth from time_series data for
    watchlist companies (max 20). Outliers (>100% CAGR) are excluded
    and annotated. Divergent growth is marked with an asterisk.

    Args:
        results: Complete pipeline output.

    Returns:
        Matplotlib Figure with the grouped bar chart.
    """
    cr = results.combined_rankings
    ts = results.time_series
    if cr.empty:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_title("Historical Growth Comparison — no data")
        return fig

    # Filter to watchlist companies
    watchlist_eids = _watchlist_entity_ids(results)
    if not watchlist_eids:
        target_eids = cr.index.tolist()[:_MAX_COMPANIES]
    else:
        target_eids = [e for e in watchlist_eids if e in cr.index][:_MAX_COMPANIES]

    # Compute historical annualised growth from time_series
    rows: list[dict[str, object]] = []
    for eid in target_eids:
        symbol = str(cr.loc[eid, "symbol"])
        eid_ts = ts[ts["entity_id"] == eid].sort_values("period_idx")
        if len(eid_ts) < 2:
            continue

        # Annualise: mean of quarterly growth rates, compounded over 4 quarters
        fcf_gr = eid_ts["fcf_growth"].dropna()
        rev_gr = eid_ts["revenue_growth"].dropna()
        if fcf_gr.empty or rev_gr.empty:
            continue

        fcf_annual = ((1 + fcf_gr.mean()) ** 4 - 1) * 100
        rev_annual = ((1 + rev_gr.mean()) ** 4 - 1) * 100
        rows.append({
            "entity_id": eid,
            "symbol": symbol,
            "fcf_pct": fcf_annual,
            "rev_pct": rev_annual,
        })

    if not rows:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_title("Historical Growth Comparison — no valid data")
        return fig

    df = pd.DataFrame(rows)

    # Separate outliers
    outlier_mask = (df["fcf_pct"].abs() > 100) | (df["rev_pct"].abs() > 100)
    outlier_symbols = df.loc[outlier_mask, "symbol"].tolist()
    if outlier_symbols:
        logger.info(
            "growth_comparison_historical: excluding %d outliers: %s",
            len(outlier_symbols), ", ".join(outlier_symbols),
        )
    df = df.loc[~outlier_mask].copy()

    if df.empty:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_title("Historical Growth Comparison — all companies are outliers")
        _outlier_textbox(ax, outlier_symbols)
        return fig

    # Mark divergence
    threshold_pct = results.config.growth_divergence_threshold * 100
    df["divergent"] = (df["fcf_pct"] - df["rev_pct"]).abs() > threshold_pct
    labels = [
        f"{row['symbol']}*" if row["divergent"] else str(row["symbol"])
        for _, row in df.iterrows()
    ]

    # Colours from viridis
    cmap = plt.get_cmap(_VIRIDIS)
    fcf_colour = cmap(0.3)
    rev_colour = cmap(0.7)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(df))
    bar_width = 0.35

    ax.bar(
        x - bar_width / 2, df["fcf_pct"], bar_width,
        label="FCF CAGR (historical)", color=fcf_colour, edgecolor="white",
    )
    ax.bar(
        x + bar_width / 2, df["rev_pct"], bar_width,
        label="Revenue CAGR (historical)", color=rev_colour, edgecolor="white",
    )

    # Reference line at min acceptable growth
    min_growth_pct = results.config.min_acceptable_growth * 100
    ax.axhline(
        y=min_growth_pct, color="red", linestyle="--", linewidth=0.8,
        label=f"Min acceptable ({min_growth_pct:.0f}%)",
    )

    ax.set_xlabel("Company")
    ax.set_ylabel("Annual Growth Rate (%)")
    ax.set_title("Historical Growth Comparison (FCF vs Revenue CAGR)", fontsize=_TITLE_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=_SCATTER_FONTSIZE)
    ax.legend(fontsize=_ANNOTATION_FONTSIZE, framealpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.3)

    _outlier_textbox(ax, outlier_symbols)

    if any(df["divergent"]):
        ax.text(
            0.98, 0.02, "* = growth divergence (|FCF - Rev| > threshold)",
            transform=ax.transAxes,
            fontsize=_ANNOTATION_FONTSIZE,
            ha="right",
            va="bottom",
            fontstyle="italic",
            alpha=0.6,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Projected growth with scenario range (error bars)
# ---------------------------------------------------------------------------


def growth_comparison_projected(results: AnalysisResults) -> Figure:
    """Grouped bar chart with error bars showing scenario range.

    Bar height is the base-scenario CAGR. Error bars extend from the
    pessimistic to optimistic scenario CAGR. Covers watchlist companies
    (max 20). Outliers are excluded and annotated.

    Args:
        results: Complete pipeline output.

    Returns:
        Matplotlib Figure with the grouped bar chart.
    """
    cr = results.combined_rankings
    projections = results.projections
    primary_period = results.config.primary_period

    if cr.empty:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_title("Projected Growth with Scenario Range — no data")
        return fig

    watchlist_eids = _watchlist_entity_ids(results)
    if not watchlist_eids:
        target_eids = cr.index.tolist()[:_MAX_COMPANIES]
    else:
        target_eids = [e for e in watchlist_eids if e in cr.index][:_MAX_COMPANIES]

    # Extract scenario data per company
    rows: list[dict[str, object]] = []
    outlier_symbols: list[str] = []

    for eid in target_eids:
        symbol = str(cr.loc[eid, "symbol"])
        entity_proj = projections.get(eid, {})
        period_proj = entity_proj.get(primary_period, {})  # type: ignore[call-overload]

        fcf_scenarios = period_proj.get("fcf", {})
        rev_scenarios = period_proj.get("revenue", {})

        # Base CAGR
        fcf_base = fcf_scenarios.get("base")
        rev_base = rev_scenarios.get("base")
        if fcf_base is None or rev_base is None:
            logger.warning(
                "growth_comparison_projected: skipping %s — missing base projection",
                symbol,
            )
            continue

        fcf_base_cagr = fcf_base.annual_cagr * 100
        rev_base_cagr = rev_base.annual_cagr * 100

        # Check outlier
        if abs(fcf_base_cagr) > 100 or abs(rev_base_cagr) > 100:
            outlier_symbols.append(symbol)
            continue

        # Pessimistic / optimistic
        fcf_pess = fcf_scenarios.get("pessimistic")
        fcf_opt = fcf_scenarios.get("optimistic")
        rev_pess = rev_scenarios.get("pessimistic")
        rev_opt = rev_scenarios.get("optimistic")

        fcf_pess_cagr = fcf_pess.annual_cagr * 100 if fcf_pess else fcf_base_cagr
        fcf_opt_cagr = fcf_opt.annual_cagr * 100 if fcf_opt else fcf_base_cagr
        rev_pess_cagr = rev_pess.annual_cagr * 100 if rev_pess else rev_base_cagr
        rev_opt_cagr = rev_opt.annual_cagr * 100 if rev_opt else rev_base_cagr

        # Divergence check
        threshold_pct = results.config.growth_divergence_threshold * 100
        divergent = abs(fcf_base_cagr - rev_base_cagr) > threshold_pct

        rows.append({
            "entity_id": eid,
            "symbol": symbol,
            "fcf_base": fcf_base_cagr,
            "fcf_low": fcf_pess_cagr,
            "fcf_high": fcf_opt_cagr,
            "rev_base": rev_base_cagr,
            "rev_low": rev_pess_cagr,
            "rev_high": rev_opt_cagr,
            "divergent": divergent,
        })

    if outlier_symbols:
        logger.info(
            "growth_comparison_projected: excluding %d outliers: %s",
            len(outlier_symbols), ", ".join(outlier_symbols),
        )

    if not rows:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_title("Projected Growth with Scenario Range — no valid data")
        _outlier_textbox(ax, outlier_symbols)
        return fig

    df = pd.DataFrame(rows)

    # Labels with divergence marker
    labels = [
        f"{row['symbol']}*" if row["divergent"] else str(row["symbol"])
        for _, row in df.iterrows()
    ]

    # Error bar deltas (asymmetric: [lower_error, upper_error])
    fcf_err_low = (df["fcf_base"] - df["fcf_low"]).clip(lower=0).values
    fcf_err_high = (df["fcf_high"] - df["fcf_base"]).clip(lower=0).values
    rev_err_low = (df["rev_base"] - df["rev_low"]).clip(lower=0).values
    rev_err_high = (df["rev_high"] - df["rev_base"]).clip(lower=0).values

    cmap = plt.get_cmap(_VIRIDIS)
    fcf_colour = cmap(0.3)
    rev_colour = cmap(0.7)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(df))
    bar_width = 0.35

    ax.bar(
        x - bar_width / 2, df["fcf_base"], bar_width,
        yerr=[fcf_err_low, fcf_err_high],
        label="FCF CAGR", color=fcf_colour, edgecolor="white",
        capsize=3, error_kw={"linewidth": 0.8},
    )
    ax.bar(
        x + bar_width / 2, df["rev_base"], bar_width,
        yerr=[rev_err_low, rev_err_high],
        label="Revenue CAGR", color=rev_colour, edgecolor="white",
        capsize=3, error_kw={"linewidth": 0.8},
    )

    # Reference line
    min_growth_pct = results.config.min_acceptable_growth * 100
    ax.axhline(
        y=min_growth_pct, color="red", linestyle="--", linewidth=0.8,
        label=f"Min acceptable ({min_growth_pct:.0f}%)",
    )
    ax.axhline(y=0, color="black", linewidth=0.3)

    ax.set_xlabel("Company")
    ax.set_ylabel("Annual Growth Rate (%)")
    ax.set_title(
        f"Projected Growth with Scenario Range ({primary_period}-Year)",
        fontsize=_TITLE_FONTSIZE,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=_SCATTER_FONTSIZE)
    ax.legend(fontsize=_ANNOTATION_FONTSIZE, framealpha=0.7)

    _outlier_textbox(ax, outlier_symbols)

    if any(df["divergent"]):
        ax.text(
            0.98, 0.02, "* = growth divergence",
            transform=ax.transAxes,
            fontsize=_ANNOTATION_FONTSIZE,
            ha="right",
            va="bottom",
            fontstyle="italic",
            alpha=0.6,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Risk-adjusted opportunity scatter
# ---------------------------------------------------------------------------


def risk_adjusted_opportunity(results: AnalysisResults) -> Figure:
    """Scatter plot of opportunity rank vs composite safety score.

    Bubble size represents market capitalisation; colour represents
    risk-adjusted rank (viridis_r — darker = better rank). Quadrant
    labels classify companies by opportunity/safety characteristics.
    Higher composite safety = safer company.

    Args:
        results: Complete pipeline output.

    Returns:
        Matplotlib Figure with the scatter plot.
    """
    cr = results.combined_rankings.copy()
    if cr.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title("Risk-Adjusted Opportunity — no data")
        return fig

    fig, ax = plt.subplots(figsize=(12, 8))

    sizes = _market_cap_sizes(cr["market_cap"])
    scatter = ax.scatter(
        cr["opportunity_rank"],
        cr["composite_safety"],
        s=sizes,
        c=cr["risk_adjusted_rank"],
        cmap=_VIRIDIS_R,
        alpha=0.75,
        edgecolors="grey",
        linewidths=0.5,
    )

    # Annotate tickers
    for _eid, row in cr.iterrows():
        ax.annotate(
            row["symbol"],
            (row["opportunity_rank"], row["composite_safety"]),
            fontsize=_ANNOTATION_FONTSIZE,
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )

    # Invert x-axis (lower rank = better, on right)
    ax.invert_xaxis()

    # Quadrant labels (higher safety = better, top of chart)
    quadrants = [
        (0.95, 0.95, "Ideal\n(High Opportunity / High Safety)"),
        (0.05, 0.95, "Low Opportunity / High Safety"),
        (0.95, 0.05, "High Opportunity / Low Safety"),
        (0.05, 0.05, "Avoid\n(Low Opportunity / Low Safety)"),
    ]
    for x_frac, y_frac, label in quadrants:
        ax.text(
            x_frac, y_frac, label,
            transform=ax.transAxes,
            fontsize=_SCATTER_FONTSIZE,
            ha="right" if x_frac > 0.5 else "left",
            va="top" if y_frac > 0.5 else "bottom",
            alpha=0.4,
            fontstyle="italic",
        )

    # Colour bar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Risk-Adjusted Rank (lower = better)")

    # Size legend
    _size_legend(ax, cr["market_cap"])

    ax.set_xlabel("Opportunity Rank (lower = better, right side)")
    ax.set_ylabel("Composite Safety Score")
    ax.set_title("Risk-Adjusted Opportunity Overview", fontsize=_TITLE_FONTSIZE)
    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# 5. Factor contributions stacked bar
# ---------------------------------------------------------------------------


def factor_contributions_bar(results: AnalysisResults) -> Figure:
    """Stacked horizontal bar chart of penalty factor contributions.

    Shows the relative contribution of debt/cash, market cap, and
    growth penalties for each company. Top 20 companies by total
    penalty from weighted_rankings.

    Args:
        results: Complete pipeline output.

    Returns:
        Matplotlib Figure with the stacked bar chart.
    """
    fc = results.factor_contributions
    wr = results.weighted_rankings

    if fc.empty or wr.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title("Factor Contributions — no data")
        return fig

    # Top 20 by total_penalty (ascending — lowest penalties first)
    top_eids = wr.head(_MAX_COMPANIES).index
    df = fc.loc[fc.index.isin(top_eids)].copy()

    if df.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title("Factor Contributions — no matching companies")
        return fig

    # Merge symbol from weighted_rankings
    df["symbol"] = wr.loc[df.index, "symbol"]

    # Preserve penalty ordering from weighted_rankings
    ordered_eids = [e for e in top_eids if e in df.index]
    df = df.loc[ordered_eids]

    # Reverse so highest penalty is at top of horizontal bar chart
    df = df.iloc[::-1]

    cmap = plt.get_cmap(_VIRIDIS)
    colours = [cmap(0.2), cmap(0.5), cmap(0.8)]

    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(df))

    # Stacked horizontal bars
    left = np.zeros(len(df))
    for col, colour, label in [
        ("dc_pct", colours[0], "Debt/Cash"),
        ("mc_pct", colours[1], "Market Cap"),
        ("growth_pct", colours[2], "Growth"),
    ]:
        values = np.asarray(df[col].fillna(0), dtype=float)
        ax.barh(y, values, left=left, color=colour, label=label, edgecolor="white")

        # Percentage labels on bars (only if >= 5%)
        for i, (val, bar_left) in enumerate(zip(values, left, strict=False)):
            if val >= 5.0:
                ax.text(
                    bar_left + val / 2, i, f"{val:.0f}%",
                    ha="center", va="center",
                    fontsize=_ANNOTATION_FONTSIZE,
                    color="white" if val > 15 else "black",
                    fontweight="bold",
                )
        left = left + values

    ax.set_yticks(y)
    ax.set_yticklabels(df["symbol"], fontsize=_SCATTER_FONTSIZE)
    ax.set_xlabel("Contribution (%)")
    ax.set_title(
        "Factor Contributions to Total Penalty (Top 20)",
        fontsize=_TITLE_FONTSIZE,
    )
    ax.legend(loc="lower right", fontsize=_ANNOTATION_FONTSIZE, framealpha=0.7)
    ax.set_xlim(0, 100)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Factor contribution heatmap
# ---------------------------------------------------------------------------


def factor_heatmap(results: AnalysisResults) -> Figure:
    """Heatmap of factor contribution percentages.

    Shows dc_pct, mc_pct, growth_pct for each company using
    matplotlib imshow. Top 20 companies by total_penalty from
    weighted_rankings.

    Args:
        results: Complete pipeline output.

    Returns:
        Matplotlib Figure with the heatmap.
    """
    fc = results.factor_contributions
    wr = results.weighted_rankings

    if fc.empty or wr.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title("Factor Contribution Heatmap — no data")
        return fig

    # Top 20 by total_penalty
    top_eids = wr.head(_MAX_COMPANIES).index
    df = fc.loc[fc.index.isin(top_eids)].copy()

    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title("Factor Contribution Heatmap — no matching companies")
        return fig

    # Merge symbol from weighted_rankings
    df["symbol"] = wr.loc[df.index, "symbol"]

    # Preserve penalty ordering
    ordered_eids = [e for e in top_eids if e in df.index]
    df = df.loc[ordered_eids]

    symbols = df["symbol"].tolist()
    factor_cols = ["dc_pct", "mc_pct", "growth_pct"]
    factor_labels = ["Debt/Cash %", "Market Cap %", "Growth %"]

    data = df[factor_cols].fillna(0).values.astype(float)

    fig, ax = plt.subplots(figsize=(10, max(6, len(symbols) * 0.4)))
    im = ax.imshow(data, cmap=_VIRIDIS, aspect="auto", vmin=0, vmax=100)

    # Annotate cells with percentage values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            # Use white text on dark cells, black on light
            text_colour = "white" if value > 50 else "black"
            ax.text(
                j, i, f"{value:.1f}%",
                ha="center", va="center",
                fontsize=_ANNOTATION_FONTSIZE,
                color=text_colour,
                fontweight="bold",
            )

    ax.set_xticks(np.arange(len(factor_labels)))
    ax.set_xticklabels(factor_labels, fontsize=_SCATTER_FONTSIZE)
    ax.set_yticks(np.arange(len(symbols)))
    ax.set_yticklabels(symbols, fontsize=_SCATTER_FONTSIZE)

    # Move x-axis labels to top
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    ax.set_title(
        "Factor Contribution Heatmap (Top 20 by Penalty)",
        fontsize=_TITLE_FONTSIZE,
        pad=20,
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label("Contribution (%)")

    fig.tight_layout()
    return fig
