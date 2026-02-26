"""Multi-company comparison charts.

Five chart functions for comparing companies across growth, valuation,
and ranking dimensions. All functions take AnalysisResults and a list
of symbols, returning a matplotlib Figure.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from pipeline.data.contracts import AnalysisResults

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_entity_ids(
    df: pd.DataFrame,
    symbols: list[str],
) -> pd.DataFrame:
    """Filter a ranking/companies DataFrame to rows matching *symbols*.

    Args:
        df: DataFrame indexed by entity_id with a ``symbol`` column.
        symbols: Symbols to include.

    Returns:
        Filtered DataFrame (may be empty if no symbols match).
    """
    mask = df["symbol"].isin(symbols)
    filtered = df.loc[mask]
    missing = set(symbols) - set(filtered["symbol"])
    if missing:
        logger.warning("Symbols not found in data: %s", sorted(missing))
    return filtered


def _format_large_number(value: float) -> str:
    """Format a large number with B/M suffix for display."""
    abs_val = abs(value)
    if abs_val >= 1e9:
        return f"${value / 1e9:.1f}B"
    if abs_val >= 1e6:
        return f"${value / 1e6:.0f}M"
    return f"${value:,.0f}"


# ---------------------------------------------------------------------------
# 1. Projected growth vs stability scatter
# ---------------------------------------------------------------------------


def projected_growth_stability(
    results: AnalysisResults,
    symbols: list[str],
) -> Figure:
    """Scatter plot of projected FCF growth vs growth stability.

    Bubble size proportional to market cap. Colour encodes a quality rank
    derived from normalised growth (40 %) and stability (60 %).
    Quadrant labels based on median splits.  Reference line at
    ``min_acceptable_growth * 100``.  Outliers (>100 % growth) are
    annotated in a text box rather than plotted.

    Args:
        results: Complete pipeline output.
        symbols: Company symbols to display.

    Returns:
        Matplotlib Figure.
    """
    cr = _resolve_entity_ids(results.combined_rankings, symbols)
    if cr.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.text(0.5, 0.5, "No matching companies", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        ax.set_title("Projected Growth vs Stability")
        return fig

    growth_pct = cr["fcf_growth_annual"].astype(float) * 100
    stability = cr["growth_stability"].astype(float)
    market_cap = cr["market_cap"].astype(float)
    syms = cr["symbol"].values

    # Separate outliers (>100 % growth)
    outlier_mask = growth_pct.abs() > 100
    plot_mask = ~outlier_mask
    outlier_syms = syms[np.asarray(outlier_mask)]
    outlier_growth = growth_pct[outlier_mask].values

    gp = growth_pct[plot_mask]
    st = stability[plot_mask]
    mc = market_cap[plot_mask]
    ps = syms[np.asarray(plot_mask)]

    if gp.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.text(0.5, 0.5, "All companies are outliers (>100% growth)",
                transform=ax.transAxes, ha="center", va="center", fontsize=14)
        ax.set_title("Projected Growth vs Stability")
        return fig

    # Quality rank: (normalised_growth * 0.4 + stability * 0.6) descending
    g_min, g_max = gp.min(), gp.max()
    g_range = g_max - g_min if g_max != g_min else 1.0
    norm_growth = (gp - g_min) / g_range

    s_min, s_max = st.min(), st.max()
    s_range = s_max - s_min if s_max != s_min else 1.0
    norm_stability = (st - s_min) / s_range

    quality = norm_growth * 0.4 + norm_stability * 0.6
    quality_rank = quality.rank(ascending=False, method="min")

    # Bubble sizes scaled between 50 and 600
    mc_min, mc_max = mc.min(), mc.max()
    mc_range = mc_max - mc_min if mc_max != mc_min else 1.0
    sizes = 50 + 550 * (mc - mc_min) / mc_range

    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.get_cmap("viridis_r")

    scatter = ax.scatter(
        gp, st, s=sizes, c=quality_rank, cmap=cmap,
        alpha=0.75, edgecolors="white", linewidths=0.5,
    )

    # Annotate points
    for sym, x, y in zip(ps, gp, st, strict=False):
        ax.annotate(sym, (x, y), fontsize=7, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points")

    # Quadrant labels based on median splits
    med_g = gp.median()
    med_s = st.median()
    ax.axvline(med_g, color="grey", linestyle=":", alpha=0.5)
    ax.axhline(med_s, color="grey", linestyle=":", alpha=0.5)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    label_kw: dict[str, Any] = {
        "fontsize": 8, "alpha": 0.4, "ha": "center", "va": "center",
        "style": "italic",
    }
    ax.text((med_g + x_max) / 2, (med_s + y_max) / 2,
            "Stable / Fast Growth\n(Ideal)", **label_kw)
    ax.text((x_min + med_g) / 2, (med_s + y_max) / 2,
            "Stable / Slow Growth", **label_kw)
    ax.text((med_g + x_max) / 2, (y_min + med_s) / 2,
            "Unstable / Fast Growth", **label_kw)
    ax.text((x_min + med_g) / 2, (y_min + med_s) / 2,
            "Unstable / Slow Growth\n(Avoid)", **label_kw)

    # Reference line at min_acceptable_growth
    mag_pct = results.config.min_acceptable_growth * 100
    ax.axvline(mag_pct, color="red", linestyle="--", alpha=0.6,
               label=f"Min acceptable growth ({mag_pct:.0f}%)")

    # Outlier text box
    if len(outlier_syms) > 0:
        lines = [f"{s}: {outlier_growth[i]:+.0f}%"
                 for i, s in enumerate(outlier_syms)]
        box_text = "Outliers (>100% growth):\n" + "\n".join(lines)
        ax.text(0.98, 0.02, box_text, transform=ax.transAxes, fontsize=7,
                va="bottom", ha="right",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow",
                          "alpha": 0.8})

    # Size legend for market cap
    mc_legend_vals = [mc_min, (mc_min + mc_max) / 2, mc_max]
    mc_legend_sizes = [50 + 550 * (v - mc_min) / mc_range
                       for v in mc_legend_vals]
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="grey", markersize=np.sqrt(s) / 2,
               label=_format_large_number(v))
        for v, s in zip(mc_legend_vals, mc_legend_sizes, strict=False)
    ]
    size_legend = ax.legend(handles=legend_handles, title="Market Cap",
                            loc="upper left", fontsize=7, title_fontsize=8)
    ax.add_artist(size_legend)

    ax.legend(loc="lower left", fontsize=8)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Quality Rank (lower = better)", fontsize=9)

    ax.set_xlabel("Projected FCF Growth (%)", fontsize=10)
    ax.set_ylabel("Growth Stability", fontsize=10)
    ax.set_title("Projected Growth vs Stability", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Acquirer's multiple analysis
# ---------------------------------------------------------------------------


def acquirers_multiple_analysis(
    results: AnalysisResults,
    symbols: list[str],
) -> Figure:
    """Violin plots of acquirer's multiple for selected companies.

    Each company's historical acquirer's multiple is computed from
    time_series (EV / operating_income per quarter). Current AM from
    live prices overlaid as red dots. A reference line at AM=10
    indicates the typical value threshold. Limited to 20 companies.

    Args:
        results: Complete pipeline output.
        symbols: Company symbols to display (max 20 used).

    Returns:
        Matplotlib Figure.
    """
    companies = _resolve_entity_ids(results.companies, symbols)
    if companies.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "No matching companies", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        ax.set_title("Acquirer's Multiple Analysis")
        return fig

    # Limit to 20
    if len(companies) > 20:
        logger.warning(
            "Limiting acquirers_multiple_analysis to first 20 of %d companies",
            len(companies),
        )
        companies = companies.iloc[:20]

    syms = companies["symbol"].values
    am_values = companies["acquirers_multiple"].astype(float).values
    n = len(syms)
    ts = results.time_series

    # Compute current AM from live prices if available
    current_am: list[float | None] = []
    for _, row in companies.iterrows():
        sym = str(row["symbol"])
        live_price = results.live_prices.get(sym)
        if (live_price is not None
                and live_price > 0
                and row["shares_diluted"] > 0
                and row["operating_income"] != 0):
            live_ev = (
                live_price * row["shares_diluted"]
                + row["lt_debt"]
                - row["cash"]
            )
            current_am.append(live_ev / row["operating_income"])
        else:
            current_am.append(None)

    # Historical AM from time_series (EV / operating_income per quarter)
    historical_am: list[np.ndarray] = []
    for eid in companies.index:
        eid_ts = ts[ts["entity_id"] == eid]
        if not eid_ts.empty:
            op_inc = eid_ts["operating_income"]
            valid = op_inc != 0
            if valid.any():
                ev = (
                    eid_ts.loc[valid, "adj_close"] * eid_ts.loc[valid, "shares_diluted"]
                    + eid_ts.loc[valid, "lt_debt"]
                    - eid_ts.loc[valid, "cash"]
                )
                am_hist = (ev / eid_ts.loc[valid, "operating_income"]).values
                # Filter out inf/NaN
                am_hist = am_hist[np.isfinite(am_hist)]
                if len(am_hist) >= 2:
                    historical_am.append(am_hist.astype(float))
                    continue
        # Fallback: single point estimate (no violin spread)
        historical_am.append(np.array([float(am_values[len(historical_am)])]))

    fig, ax = plt.subplots(figsize=(max(10, n * 0.7), 7))
    cmap = plt.get_cmap("viridis")
    colours = [cmap(i / max(n - 1, 1)) for i in range(n)]

    parts = ax.violinplot(historical_am, positions=range(n), showmeans=True,
                          showmedians=False)

    # Colour violins
    for i, body in enumerate(parts["bodies"]):  # type: ignore[arg-type, var-annotated]
        body.set_facecolor(colours[i])
        body.set_alpha(0.6)
    for partname in ("cmeans", "cbars", "cmins", "cmaxes"):
        if partname in parts:
            parts[partname].set_edgecolor("grey")

    # Overlay current AM as red dots with % change labels.
    # Values below -10 are capped at the floor and annotated with true value.
    am_floor = -10.0
    for i, cam in enumerate(current_am):
        if cam is not None:
            display_y = max(cam, am_floor)
            ax.scatter(i, display_y, color="red", s=60, zorder=5,
                       edgecolors="black", linewidths=0.5)
            if cam < am_floor:
                ax.annotate(f"AM: {cam:.1f}", (i, display_y),
                            fontsize=7, ha="center", va="top",
                            xytext=(0, -8), textcoords="offset points",
                            color="red", fontweight="bold")
            elif am_values[i] != 0:
                pct_change = (cam / am_values[i] - 1) * 100
                ax.annotate(f"{pct_change:+.0f}%", (i, display_y),
                            fontsize=7, ha="center", va="bottom",
                            xytext=(0, 6), textcoords="offset points",
                            color="red", fontweight="bold")

    # Reference line at AM = 10
    ax.axhline(10, color="green", linestyle="--", alpha=0.6,
               label="AM = 10 (threshold)")

    # Floor y-axis at -10
    ax.set_ylim(bottom=am_floor)

    ax.set_xticks(range(n))
    ax.set_xticklabels(syms, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Acquirer's Multiple (EV / Operating Income)", fontsize=10)
    ax.set_title("Acquirer's Multiple Analysis", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    # Custom legend entry for red dot
    red_dot = Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                     markersize=8, label="Current AM (live price)")
    ax.legend(handles=[ax.get_legend_handles_labels()[0][0], red_dot],
              labels=["AM = 10 (threshold)", "Current AM (live price)"],
              fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Valuation upside comparison
# ---------------------------------------------------------------------------


def valuation_upside_comparison(
    results: AnalysisResults,
    symbols: list[str],
) -> Figure:
    """Horizontal bar chart of valuation upside/downside by scenario.

    For each symbol, three horizontal bars show the percentage
    upside (positive) or downside (negative) relative to current price
    for pessimistic, base, and optimistic scenarios at the primary
    projection period.

    Args:
        results: Complete pipeline output.
        symbols: Company symbols to display.

    Returns:
        Matplotlib Figure.
    """
    cr = _resolve_entity_ids(results.combined_rankings, symbols)
    if cr.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No matching companies", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        ax.set_title("Valuation Upside / Downside Comparison")
        return fig

    period = results.config.primary_period
    scenarios = ["pessimistic", "base", "optimistic"]
    cmap = plt.get_cmap("viridis")
    scenario_colours = [cmap(0.15), cmap(0.5), cmap(0.85)]

    rows: list[dict[str, object]] = []
    for entity_id, row in cr.iterrows():
        sym = str(row["symbol"])
        price = float(row["current_price"])
        if price <= 0:
            continue

        eid = int(entity_id)  # type: ignore[call-overload]
        entity_ivs = results.intrinsic_values.get(eid, {})
        period_ivs = entity_ivs.get(period, {})  # type: ignore[call-overload]

        entry: dict[str, object] = {"symbol": sym}
        has_data = False
        for scenario in scenarios:
            iv = period_ivs.get(scenario)
            if iv is not None and iv.iv_per_share > 0:
                upside = (iv.iv_per_share / price - 1) * 100
                entry[scenario] = upside
                has_data = True
            else:
                entry[scenario] = None

        if has_data:
            rows.append(entry)

    if not rows:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No intrinsic value data for selected companies",
                transform=ax.transAxes, ha="center", va="center", fontsize=14)
        ax.set_title("Valuation Upside / Downside Comparison")
        return fig

    df = pd.DataFrame(rows)
    n = len(df)
    bar_height = 0.25
    y_positions = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, max(4, n * 0.6)))

    for i, scenario in enumerate(scenarios):
        vals = np.asarray(df[scenario].astype(float).fillna(0), dtype=float)
        ax.barh(
            y_positions + (i - 1) * bar_height,
            vals,
            height=bar_height,
            color=scenario_colours[i],
            label=scenario.capitalize(),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels
        for j, v in enumerate(vals):
            if v != 0:
                ha = "left" if v >= 0 else "right"
                offset = 2 if v >= 0 else -2
                ax.text(v + offset, y_positions[j] + (i - 1) * bar_height,
                        f"{v:+.0f}%", fontsize=7, va="center", ha=ha)

    # Shaded regions
    x_min, x_max = ax.get_xlim()
    ax.axvspan(x_min, 0, alpha=0.05, color="red", zorder=0)
    ax.axvspan(0, x_max, alpha=0.05, color="green", zorder=0)
    ax.text(x_min + 2, n - 0.3, "Overvalued", fontsize=8, alpha=0.4,
            color="red", style="italic")
    ax.text(x_max - 2, n - 0.3, "Undervalued", fontsize=8, alpha=0.4,
            color="green", style="italic", ha="right")

    # Zero line
    ax.axvline(0, color="black", linewidth=0.8, zorder=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["symbol"], fontsize=9)
    ax.set_xlabel("Upside / Downside (%)", fontsize=10)
    ax.set_title(
        f"Valuation Upside / Downside ({period}-Year Projection)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Ranking comparison table
# ---------------------------------------------------------------------------


def ranking_comparison_table(
    results: AnalysisResults,
    symbols: list[str],
) -> Figure:
    """Table summarising key rankings and metrics for selected companies.

    Columns: Symbol, Price, FCF Gr., Rev Gr., IV/Price, Stability,
    Growth Rk, Value Rk, Opp. Rk.

    Args:
        results: Complete pipeline output.
        symbols: Company symbols to display.

    Returns:
        Matplotlib Figure with ax.table().
    """
    cr = _resolve_entity_ids(results.combined_rankings, symbols)
    gr = _resolve_entity_ids(results.growth_rankings, symbols)
    vr = _resolve_entity_ids(results.value_rankings, symbols)

    if cr.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No matching companies", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        ax.set_title("Ranking Comparison")
        ax.axis("off")
        return fig

    # Build lookup dicts for growth and value ranks (keyed by entity_id)
    growth_rank_map: dict[int, int] = {}
    for eid in gr.index:
        growth_rank_map[int(eid)] = int(gr.loc[eid, "combined_growth_rank"])

    value_rank_map: dict[int, int] = {}
    for eid in vr.index:
        value_rank_map[int(eid)] = int(vr.loc[eid, "value_rank"])

    col_labels = [
        "Symbol", "Price", "FCF Gr.", "Rev Gr.", "IV/Price",
        "Stability", "Growth Rk", "Value Rk", "Opp. Rk",
    ]

    table_data: list[list[str]] = []
    opp_ranks: list[int] = []

    for entity_id, row in cr.iterrows():
        eid = int(entity_id)  # type: ignore[call-overload]
        table_data.append([
            str(row["symbol"]),
            f"${float(row['current_price']):.2f}",
            f"{float(row['fcf_growth_annual']) * 100:.1f}%",
            f"{float(row['revenue_growth_annual']) * 100:.1f}%",
            f"{float(row['composite_iv_ratio']):.2f}",
            f"{float(row['growth_stability']):.2f}",
            str(growth_rank_map.get(eid, "-")),
            str(value_rank_map.get(eid, "-")),
            str(int(row["opportunity_rank"])),
        ])
        opp_ranks.append(int(row["opportunity_rank"]))

    n_rows = len(table_data)
    fig_height = max(3, 0.5 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # Header styling (viridis)
    cmap = plt.get_cmap("viridis")
    header_colour = cmap(0.6)
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(header_colour)
        cell.set_text_props(color="white", fontweight="bold")

    # Colour-code opportunity rank column (last column, index 8)
    if opp_ranks:
        max_rank = max(opp_ranks) if opp_ranks else 1
        for i, rank in enumerate(opp_ranks):
            norm_val = 1 - (rank - 1) / max(max_rank - 1, 1)
            cell = table[i + 1, 8]
            cell.set_facecolor(cmap(norm_val))
            cell.set_text_props(
                color="white" if norm_val > 0.5 else "black",
            )

    ax.set_title("Ranking Comparison", fontsize=13, fontweight="bold",
                 pad=20)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Comprehensive rankings table
# ---------------------------------------------------------------------------


def comprehensive_rankings_table(
    results: AnalysisResults,
    symbols: list[str],
) -> Figure:
    """Large ranking table (up to 25 rows) sorted by risk-adjusted rank.

    Thirteen columns covering ranks, growth rates, FCF range,
    safety, valuation, and price/market cap. Colour-coded by
    risk_adjusted_rank. Low safety (<0.40) highlighted yellow,
    high growth (>20 %) highlighted green.

    Args:
        results: Complete pipeline output.
        symbols: Company symbols to display (max 25).

    Returns:
        Matplotlib Figure with ax.table().
    """
    cr = _resolve_entity_ids(results.combined_rankings, symbols)
    gr = _resolve_entity_ids(results.growth_rankings, symbols)
    vr = _resolve_entity_ids(results.value_rankings, symbols)

    if cr.empty:
        fig, ax = plt.subplots(figsize=(18, 4))
        ax.text(0.5, 0.5, "No matching companies", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        ax.set_title("Comprehensive Rankings")
        ax.axis("off")
        return fig

    # Sort by risk_adjusted_rank, limit 25
    cr = cr.sort_values("risk_adjusted_rank", ascending=True)
    if len(cr) > 25:
        logger.warning(
            "Limiting comprehensive_rankings_table to 25 of %d companies",
            len(cr),
        )
        cr = cr.iloc[:25]

    # Build rank lookup maps
    growth_rank_map: dict[int, int] = {}
    for eid in gr.index:
        growth_rank_map[int(eid)] = int(gr.loc[eid, "combined_growth_rank"])

    value_rank_map: dict[int, int] = {}
    for eid in vr.index:
        value_rank_map[int(eid)] = int(vr.loc[eid, "value_rank"])

    primary_period = results.config.primary_period

    col_labels = [
        "Symbol", "Risk-Adj Rk", "Opp. Rk", "Value Rk", "Growth Rk",
        "FCF CAGR", "FCF Range", "Rev CAGR", "Safety",
        "IV/Price", "Stability", "Price", "Mkt Cap",
    ]

    table_data: list[list[str]] = []
    ra_ranks: list[int] = []
    # Track cells needing highlight: (row_idx, col_idx, colour)
    highlights: list[tuple[int, int, str]] = []

    for row_idx, (entity_id, row) in enumerate(cr.iterrows()):
        eid = int(entity_id)  # type: ignore[call-overload]

        # FCF CAGR (base scenario)
        fcf_cagr = float(row["fcf_growth_annual"])
        rev_cagr = float(row["revenue_growth_annual"])

        # FCF range from projections (pessimistic - optimistic)
        entity_proj = results.projections.get(eid, {})
        period_proj = entity_proj.get(primary_period, {})  # type: ignore[call-overload]
        fcf_proj = period_proj.get("fcf", {})

        pess_cagr = None
        opt_cagr = None
        pess_proj = fcf_proj.get("pessimistic")
        if pess_proj is not None:
            pess_cagr = pess_proj.annual_cagr
        opt_proj = fcf_proj.get("optimistic")
        if opt_proj is not None:
            opt_cagr = opt_proj.annual_cagr

        if pess_cagr is not None and opt_cagr is not None:
            fcf_range_str = (
                f"{pess_cagr * 100:.0f}% - {opt_cagr * 100:.0f}%"
            )
        else:
            fcf_range_str = "-"

        safety = float(row["composite_safety"])

        ra_rank = int(row["risk_adjusted_rank"])
        opp_rank = int(row["opportunity_rank"])

        table_data.append([
            str(row["symbol"]),
            str(ra_rank),
            str(opp_rank),
            str(value_rank_map.get(eid, "-")),
            str(growth_rank_map.get(eid, "-")),
            f"{fcf_cagr * 100:.1f}%",
            fcf_range_str,
            f"{rev_cagr * 100:.1f}%",
            f"{safety:.2f}",
            f"{float(row['composite_iv_ratio']):.2f}",
            f"{float(row['growth_stability']):.2f}",
            f"${float(row['current_price']):.2f}",
            _format_large_number(float(row["market_cap"])),
        ])
        ra_ranks.append(ra_rank)

        # Highlight low safety (<0.40) — Safety col 8
        if safety < 0.40:
            highlights.append((row_idx, 8, "yellow"))

        # Highlight high growth (>20 %) — FCF CAGR col 5, Rev CAGR col 7
        if fcf_cagr * 100 > 20:
            highlights.append((row_idx, 5, "lightgreen"))
        if rev_cagr * 100 > 20:
            highlights.append((row_idx, 7, "lightgreen"))

    n_rows = len(table_data)
    fig_height = max(4, 0.45 * n_rows + 2)
    fig, ax = plt.subplots(figsize=(20, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    # Header styling (viridis)
    cmap = plt.get_cmap("viridis")
    header_colour = cmap(0.6)
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(header_colour)
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)

    # Row background colour by risk_adjusted_rank
    max_rank = max(ra_ranks) if ra_ranks else 1
    for i, rank in enumerate(ra_ranks):
        norm_val = 1 - (rank - 1) / max(max_rank - 1, 1)
        bg = cmap(norm_val)
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            cell.set_facecolor((*bg[:3], 0.15))

    # Apply conditional highlights (overwrites row colour for those cells)
    for row_idx, col_idx, colour in highlights:
        cell = table[row_idx + 1, col_idx]
        cell.set_facecolor(colour)
        cell.set_text_props(fontweight="bold")

    ax.set_title("Comprehensive Rankings", fontsize=13, fontweight="bold",
                 pad=20)
    fig.tight_layout()
    return fig
