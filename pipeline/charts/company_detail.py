"""Per-company detail charts for the value investing screening pipeline.

Seven public chart functions and four private sub-panel helpers.
All public functions take AnalysisResults and entity_id, returning a
matplotlib Figure. Uses viridis colourmap throughout.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from pipeline.data.contracts import AnalysisResults, IntrinsicValue, Projection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette (viridis samples)
# ---------------------------------------------------------------------------

_VIRIDIS = plt.colormaps["viridis"]
_C0 = _VIRIDIS(0.2)
_C1 = _VIRIDIS(0.5)
_C2 = _VIRIDIS(0.8)
_C_FILL = _VIRIDIS(0.4)


def _get_symbol(results: AnalysisResults, entity_id: int) -> str:
    """Look up the company symbol from the companies DataFrame."""
    if entity_id not in results.companies.index:
        return f"Entity {entity_id}"
    return str(results.companies.loc[entity_id, "symbol"])


def _get_primary_projections(
    results: AnalysisResults,
    entity_id: int,
    metric: str,
) -> dict[str, Projection]:
    """Return {scenario: Projection} for metric at primary_period."""
    period = results.config.primary_period
    entity_data = results.projections.get(entity_id, {})
    period_data = entity_data.get(period, {})  # type: ignore[call-overload]
    return period_data.get(metric, {})  # type: ignore[no-any-return]


def _get_primary_ivs(
    results: AnalysisResults,
    entity_id: int,
) -> dict[str, IntrinsicValue]:
    """Return {scenario: IntrinsicValue} for primary_period."""
    period = results.config.primary_period
    entity_data = results.intrinsic_values.get(entity_id, {})
    return entity_data.get(period, {})  # type: ignore[call-overload, no-any-return]


def _current_price(results: AnalysisResults, entity_id: int) -> float:
    """Get current live price for this entity."""
    symbol = _get_symbol(results, entity_id)
    price = results.live_prices.get(symbol, 0.0)
    return float(price)


# ===================================================================
# Public chart functions
# ===================================================================


def historical_growth(results: AnalysisResults, entity_id: int) -> Figure:
    """Bar chart of FCF and revenue growth rates over time.

    Quarterly growth rates from time_series, displayed as percentages.
    Includes an annotation box with annualised volatility.
    Values capped at +/-100% for display clarity.

    Args:
        results: Complete pipeline output.
        entity_id: Company identifier.

    Returns:
        Matplotlib Figure with the growth bar chart.
    """
    symbol = _get_symbol(results, entity_id)
    ts = results.time_series
    df = ts[ts["entity_id"] == entity_id].copy().sort_values("period_idx")

    periods = df["period_idx"].values
    fcf_growth = np.asarray(df["fcf_growth"], dtype=float) * 100
    rev_growth = np.asarray(df["revenue_growth"], dtype=float) * 100

    # Cap display at +/-100%
    fcf_display = np.clip(fcf_growth, -100, 100)
    rev_display = np.clip(rev_growth, -100, 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(periods))

    ax.bar(x - width / 2, fcf_display, width, label="FCF Growth", color=_C0)
    ax.bar(x + width / 2, rev_display, width, label="Revenue Growth", color=_C1)

    ax.set_xlabel("Year")
    ax.set_ylabel("Growth Rate (%)")
    ax.set_title(f"{symbol} — Historical Growth")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p / 4:.1f}" for p in periods], rotation=45, ha="right")
    ax.axhline(y=0, color="grey", linewidth=0.8, linestyle="-")
    ax.legend()

    # Annualised volatility annotation
    fcf_vol = float(np.nanstd(np.asarray(df["fcf_growth"], dtype=float))) * 2 * 100
    rev_vol = float(np.nanstd(np.asarray(df["revenue_growth"], dtype=float))) * 2 * 100
    annotation = (
        f"Annualised Vol.\n"
        f"FCF: {fcf_vol:.1f}%\n"
        f"Rev: {rev_vol:.1f}%"
    )
    ax.annotate(
        annotation,
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "wheat", "alpha": 0.8},
    )

    fig.tight_layout()
    return fig


def growth_projection(results: AnalysisResults, entity_id: int) -> Figure:
    """Line chart of FCF and revenue projected values over time.

    Plots base scenario trajectories with fill between pessimistic
    and optimistic bounds. Values displayed in $M.

    Args:
        results: Complete pipeline output.
        entity_id: Company identifier.

    Returns:
        Matplotlib Figure with the projection chart.
    """
    symbol = _get_symbol(results, entity_id)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{symbol} — Growth Projections ({results.config.primary_period}yr)")

    for idx, metric in enumerate(("fcf", "revenue")):
        ax: Axes = axes[idx]
        projs = _get_primary_projections(results, entity_id, metric)

        if not projs:
            ax.set_title(f"{metric.upper()} — no data")
            continue

        base = projs.get("base")
        pess = projs.get("pessimistic")
        opt = projs.get("optimistic")

        if base is None:
            ax.set_title(f"{metric.upper()} — no base projection")
            continue

        quarters = np.arange(len(base.quarterly_values))
        years = quarters / 4
        base_vals = np.array(base.quarterly_values) / 1e6

        colour = _C0 if idx == 0 else _C1
        ax.plot(
            years, base_vals,
            color=colour, linewidth=2,
            label=f"Base (CAGR {base.annual_cagr * 100:.1f}%)",
        )

        if pess is not None and opt is not None:
            pess_vals = np.array(pess.quarterly_values) / 1e6
            opt_vals = np.array(opt.quarterly_values) / 1e6
            ax.fill_between(
                years, pess_vals, opt_vals,
                alpha=0.25, color=colour,
                label=(
                    f"Range: {pess.annual_cagr * 100:.1f}% - "
                    f"{opt.annual_cagr * 100:.1f}%"
                ),
            )

        ax.set_xlabel("Year")
        ax.set_ylabel("Value ($M)")
        ax.set_title(metric.upper())
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def valuation_matrix(results: AnalysisResults, entity_id: int) -> Figure:
    """Bar chart of intrinsic value per share across scenarios.

    Shows pessimistic, base, and optimistic IV per share for the
    primary projection period. Current price shown as a red dashed
    line with upside/downside annotation for the base case.

    Args:
        results: Complete pipeline output.
        entity_id: Company identifier.

    Returns:
        Matplotlib Figure with the valuation bar chart.
    """
    symbol = _get_symbol(results, entity_id)
    ivs = _get_primary_ivs(results, entity_id)
    price = _current_price(results, entity_id)

    fig, ax = plt.subplots(figsize=(8, 6))

    scenarios = ["pessimistic", "base", "optimistic"]
    colours = [_C0, _C1, _C2]
    iv_values: list[float] = []

    for scenario in scenarios:
        iv = ivs.get(scenario)
        iv_values.append(iv.iv_per_share if iv is not None else 0.0)

    x = np.arange(len(scenarios))
    bars = ax.bar(x, iv_values, color=colours, width=0.5)

    # Annotate bar values
    for bar, val in zip(bars, iv_values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"${val:.2f}", ha="center", va="bottom", fontsize=10,
        )

    # Current price line
    if price > 0:
        ax.axhline(
            y=price, color="red", linewidth=1.5, linestyle="--",
            label=f"Current Price ${price:.2f}",
        )

        # Upside/downside for base
        base_iv = iv_values[1]
        if base_iv > 0:
            upside = (base_iv / price - 1) * 100
            direction = "Upside" if upside >= 0 else "Downside"
            ax.annotate(
                f"{direction}: {upside:+.1f}%",
                xy=(1, base_iv),
                xytext=(1.35, base_iv),
                fontsize=10, fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "grey"},
            )

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in scenarios])
    ax.set_ylabel("IV Per Share ($)")
    ax.set_title(
        f"{symbol} — Intrinsic Value by Scenario "
        f"({results.config.primary_period}yr)"
    )
    ax.legend()

    fig.tight_layout()
    return fig


def growth_fan(results: AnalysisResults, entity_id: int) -> Figure:
    """Fan chart showing FCF and revenue trajectories with confidence bands.

    Displays base trajectory as a solid line with filled region between
    pessimistic and optimistic scenarios. X-axis in years, Y-axis in $M.
    Marks divergence if FCF and revenue growth differ by more than the
    configured threshold.

    Args:
        results: Complete pipeline output.
        entity_id: Company identifier.

    Returns:
        Matplotlib Figure with the fan chart.
    """
    symbol = _get_symbol(results, entity_id)

    fig, ax = plt.subplots(figsize=(12, 7))

    metrics = ("fcf", "revenue")
    metric_colours = (_C0, _C1)
    metric_labels = ("FCF", "Revenue")

    base_cagrs: dict[str, float] = {}

    for metric, colour, label in zip(metrics, metric_colours, metric_labels, strict=False):
        projs = _get_primary_projections(results, entity_id, metric)
        if not projs:
            logger.info(
                "No %s projections for entity %d, skipping in fan chart",
                metric, entity_id,
            )
            continue

        base = projs.get("base")
        pess = projs.get("pessimistic")
        opt = projs.get("optimistic")

        if base is None:
            continue

        base_cagrs[metric] = base.annual_cagr
        quarters = np.arange(len(base.quarterly_values))
        years = quarters / 4
        base_vals = np.array(base.quarterly_values) / 1e6

        ax.plot(
            years, base_vals, color=colour, linewidth=2,
            label=f"{label} Base (CAGR {base.annual_cagr * 100:.1f}%)",
        )

        if pess is not None and opt is not None:
            pess_vals = np.array(pess.quarterly_values) / 1e6
            opt_vals = np.array(opt.quarterly_values) / 1e6
            ax.fill_between(
                years, pess_vals, opt_vals,
                alpha=0.2, color=colour,
            )

    # Mark divergence
    fcf_cagr = base_cagrs.get("fcf")
    rev_cagr = base_cagrs.get("revenue")
    if fcf_cagr is not None and rev_cagr is not None:
        divergence = abs(fcf_cagr - rev_cagr)
        if divergence > results.config.growth_divergence_threshold:
            ax.annotate(
                f"Divergence: {divergence * 100:.1f}%",
                xy=(0.02, 0.98),
                xycoords="axes fraction",
                fontsize=10, fontweight="bold", color="red",
                verticalalignment="top",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow"},
            )

    ax.set_xlabel("Years")
    ax.set_ylabel("Value ($M)")
    ax.set_title(
        f"{symbol} — Growth Fan Chart ({results.config.primary_period}yr)"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def financial_health_dashboard(
    results: AnalysisResults,
    entity_id: int,
) -> Figure:
    """Text-based gauge display of four financial health metrics.

    Displays debt/cash ratio, FCF yield, operating margin, and
    acquirer's multiple with colour-coded good/medium/bad assessment
    using viridis palette.

    Args:
        results: Complete pipeline output.
        entity_id: Company identifier.

    Returns:
        Matplotlib Figure with the health dashboard.
    """
    symbol = _get_symbol(results, entity_id)
    if entity_id not in results.companies.index:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axis("off")
        ax.set_title(f"{symbol} — Financial Health (no company data)")
        return fig

    company = results.companies.loc[entity_id]

    debt_cash = float(company["debt_cash_ratio"])
    fcf_yield = float(company["fcf_to_market_cap"]) * 100
    revenue_val = float(company["revenue"])
    op_income = float(company["operating_income"])
    op_margin = (op_income / revenue_val * 100) if revenue_val != 0 else 0.0
    acq_multiple = float(company["acquirers_multiple"])

    # Thresholds: (value, good_test, good_threshold, bad_threshold)
    metrics = [
        ("Debt/Cash Ratio", debt_cash, "below", 2.5, 5.0),
        ("FCF Yield", fcf_yield, "above", 10.0, 5.0),
        ("Operating Margin", op_margin, "above", 15.0, 5.0),
        ("Acquirer's Multiple", acq_multiple, "below", 15.0, 25.0),
    ]

    # Viridis colours for good/medium/bad
    colour_good = _VIRIDIS(0.8)
    colour_medium = _VIRIDIS(0.5)
    colour_bad = _VIRIDIS(0.15)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(metrics) + 1)

    ax.text(
        5, len(metrics) + 0.5,
        f"{symbol} — Financial Health",
        fontsize=14, fontweight="bold", ha="center", va="center",
    )

    for i, (name, value, direction, good_thresh, bad_thresh) in enumerate(metrics):
        y = len(metrics) - i - 0.3

        # Determine rating
        if direction == "below":
            if value <= good_thresh:
                rating, colour = "Good", colour_good
            elif value <= bad_thresh:
                rating, colour = "Medium", colour_medium
            else:
                rating, colour = "Bad", colour_bad
        else:  # above
            if value >= good_thresh:
                rating, colour = "Good", colour_good
            elif value >= bad_thresh:
                rating, colour = "Medium", colour_medium
            else:
                rating, colour = "Bad", colour_bad

        # Format value
        if name == "Debt/Cash Ratio":
            if np.isinf(value):
                val_str = "Inf"
            else:
                val_str = f"{value:.2f}x"
        elif name == "Acquirer's Multiple":
            val_str = f"{value:.1f}x"
        else:
            val_str = f"{value:.1f}%"

        # Metric name
        ax.text(0.5, y, name, fontsize=12, va="center", fontweight="bold")
        # Value
        ax.text(5.5, y, val_str, fontsize=12, va="center", ha="center")
        # Rating badge
        ax.text(
            8.0, y, rating, fontsize=12, va="center", ha="center",
            fontweight="bold", color="white",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": colour},
        )

    fig.tight_layout()
    return fig


def risk_spider(results: AnalysisResults, entity_id: int) -> Figure:
    """Radar chart of four normalised safety scores.

    Axes: downside exposure, scenario spread, terminal dependency,
    FCF reliability. All normalised to [0, 1] where 1 = safest.
    Composite safety score displayed at centre.

    Args:
        results: Complete pipeline output.
        entity_id: Company identifier.

    Returns:
        Matplotlib Figure with the radar chart.
    """
    symbol = _get_symbol(results, entity_id)
    cr = results.combined_rankings

    if entity_id not in cr.index:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"{symbol} — Risk Profile (no ranking data)")
        ax.axis("off")
        return fig

    row = cr.loc[entity_id]

    categories = [
        "Downside\nExposure",
        "Scenario\nSpread",
        "Terminal\nDependency",
        "FCF\nReliability",
    ]
    values = [
        float(row["downside_exposure_score"]),
        float(row["scenario_spread_score"]),
        float(row["terminal_dependency_score"]),
        float(row["fcf_reliability_score"]),
    ]

    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()

    # Close the polygon
    values_closed = [*values, values[0]]
    angles_closed = [*angles, angles[0]]

    fig, ax = plt.subplots(
        figsize=(8, 8), subplot_kw={"projection": "polar"},
    )

    ax.plot(angles_closed, values_closed, color=_C1, linewidth=2)
    ax.fill(angles_closed, values_closed, color=_C_FILL, alpha=0.3)

    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)

    # Annotate values at each point
    for angle, value in zip(angles, values, strict=False):
        ax.annotate(
            f"{value:.2f}",
            xy=(angle, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    # Composite safety score in centre
    composite = float(row["composite_safety"])
    ax.text(
        0, 0, f"Safety\n{composite:.2f}",
        ha="center", va="center",
        fontsize=12, fontweight="bold",
        transform=ax.transData,
    )

    ax.set_title(
        f"{symbol} — Risk Profile",
        fontsize=14, fontweight="bold",
        pad=20,
    )

    fig.tight_layout()
    return fig


def scenario_comparison(results: AnalysisResults, entity_id: int) -> Figure:
    """2x2 grid of four sub-charts for detailed scenario analysis.

    Sub-panels:
        - DCF waterfall breakdown
        - Sensitivity analysis heatmap
        - Growth trajectories across scenarios
        - Risk-return profile scatter

    Args:
        results: Complete pipeline output.
        entity_id: Company identifier.

    Returns:
        Matplotlib Figure with the 2x2 grid.
    """
    symbol = _get_symbol(results, entity_id)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{symbol} — Scenario Comparison", fontsize=14, fontweight="bold")

    _dcf_waterfall(axes[0, 0], entity_id, results)
    _sensitivity_analysis(axes[0, 1], entity_id, results)
    _growth_trajectories(axes[1, 0], entity_id, results)
    _risk_return_profile(axes[1, 1], entity_id, results)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ===================================================================
# Private sub-panel helpers
# ===================================================================


def _dcf_waterfall(
    ax: Axes,
    entity_id: int,
    results: AnalysisResults,
) -> None:
    """Waterfall chart of DCF valuation components on a per-share basis.

    Steps: Current FCF -> Growth Value -> Terminal Value -> Discount ->
    Safety Margin -> IV Per Share.

    Args:
        ax: Matplotlib axes to draw on.
        entity_id: Company identifier.
        results: Complete pipeline output.
    """
    ivs = _get_primary_ivs(results, entity_id)
    base_iv = ivs.get("base")

    if entity_id not in results.companies.index or base_iv is None:
        ax.set_title("DCF Waterfall — no data")
        ax.axis("off")
        return

    company = results.companies.loc[entity_id]
    shares = float(company["shares_diluted"])

    if shares <= 0:
        ax.set_title("DCF Waterfall — no data")
        ax.axis("off")
        return

    # Per-share values
    annual_cfs = base_iv.projected_annual_cash_flows
    current_fcf_ps = float(company["fcf"]) / shares
    total_projected_ps = sum(annual_cfs) / shares
    growth_value_ps = total_projected_ps - current_fcf_ps * len(annual_cfs)
    terminal_ps = base_iv.terminal_value / shares

    # Present value before margin of safety deduction
    pv_before_margin = base_iv.present_value / shares
    discount_ps = -(total_projected_ps + terminal_ps - pv_before_margin)
    safety_margin_ps = -(pv_before_margin - base_iv.iv_per_share)

    labels = [
        "Current FCF",
        "Growth Value",
        "Terminal Value",
        "Discount",
        "Safety Margin",
        "IV/Share",
    ]
    values = [
        current_fcf_ps * len(annual_cfs),
        growth_value_ps,
        terminal_ps,
        discount_ps,
        safety_margin_ps,
        base_iv.iv_per_share,
    ]

    # Waterfall: cumulative positioning
    cumulative = 0.0
    bottoms: list[float] = []
    for i, v in enumerate(values):
        if i == len(values) - 1:
            # Final bar starts from zero
            bottoms.append(0.0)
        else:
            if v >= 0:
                bottoms.append(cumulative)
            else:
                bottoms.append(cumulative + v)
            cumulative += v

    colours = []
    for i, v in enumerate(values):
        if i == len(values) - 1:
            colours.append(_C1)  # total bar
        elif v >= 0:
            colours.append(_C2)
        else:
            colours.append(_C0)

    bar_heights = [abs(v) for v in values]
    x = np.arange(len(labels))
    ax.bar(x, bar_heights, bottom=bottoms, color=colours, width=0.6)

    # Annotate values
    for i, (b, h, v) in enumerate(zip(bottoms, bar_heights, values, strict=False)):
        ax.text(
            i, b + h, f"${v:.2f}", ha="center", va="bottom", fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("$/Share")
    ax.set_title("DCF Waterfall")


def _sensitivity_analysis(
    ax: Axes,
    entity_id: int,
    results: AnalysisResults,
) -> None:
    """Heatmap of IV sensitivity to discount rate and growth adjustment.

    5x5 grid with discount rates 8%-12% and growth adjustments
    -2% to +2%. Approximation: base_iv * (base_dr / dr) * (1 + ga / base_gr).

    Args:
        ax: Matplotlib axes to draw on.
        entity_id: Company identifier.
        results: Complete pipeline output.
    """
    ivs = _get_primary_ivs(results, entity_id)
    base_iv = ivs.get("base")

    if base_iv is None:
        ax.set_title("Sensitivity — no data")
        ax.axis("off")
        return

    base_iv_ps = base_iv.iv_per_share
    base_dr = base_iv.discount_rate
    base_gr = base_iv.growth_rate

    discount_rates = np.linspace(0.08, 0.12, 5)
    growth_adjustments = np.linspace(-0.02, 0.02, 5)

    grid = np.zeros((5, 5))
    for i, dr in enumerate(discount_rates):
        for j, ga in enumerate(growth_adjustments):
            dr_factor = base_dr / dr if dr > 0 else 1.0
            if base_gr != 0:
                gr_factor = max(0.25, min(1 + ga / base_gr, 3.0))
            else:
                gr_factor = 1.0
            grid[i, j] = base_iv_ps * dr_factor * gr_factor

    im = ax.imshow(grid, cmap="viridis", aspect="auto")

    # Annotate cells
    for i in range(5):
        for j in range(5):
            ax.text(
                j, i, f"${grid[i, j]:.0f}",
                ha="center", va="center", fontsize=7,
                color="white" if grid[i, j] < np.median(grid) else "black",
            )

    ax.set_xticks(range(5))
    ax.set_xticklabels([f"{ga:+.0%}" for ga in growth_adjustments], fontsize=8)
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"{dr:.0%}" for dr in discount_rates], fontsize=8)
    ax.set_xlabel("Growth Adjustment")
    ax.set_ylabel("Discount Rate")
    ax.set_title("Sensitivity Analysis")

    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _growth_trajectories(
    ax: Axes,
    entity_id: int,
    results: AnalysisResults,
) -> None:
    """Line chart of pessimistic, base, and optimistic FCF trajectories.

    X-axis in years (quarterly values converted). Y-axis in $M.

    Args:
        ax: Matplotlib axes to draw on.
        entity_id: Company identifier.
        results: Complete pipeline output.
    """
    projs = _get_primary_projections(results, entity_id, "fcf")

    if not projs:
        ax.set_title("Growth Trajectories — no data")
        ax.axis("off")
        return

    scenario_colours = {
        "pessimistic": _C0,
        "base": _C1,
        "optimistic": _C2,
    }

    for scenario_name in ("pessimistic", "base", "optimistic"):
        proj = projs.get(scenario_name)
        if proj is None:
            continue

        quarters = np.arange(len(proj.quarterly_values))
        years = quarters / 4
        vals = np.array(proj.quarterly_values) / 1e6

        ax.plot(
            years, vals,
            color=scenario_colours[scenario_name],
            linewidth=2,
            label=f"{scenario_name.capitalize()} ({proj.annual_cagr * 100:.1f}%)",
        )

    ax.set_xlabel("Years")
    ax.set_ylabel("FCF ($M)")
    ax.set_title("FCF Trajectories")
    ax.legend(fontsize=7)


def _risk_return_profile(
    ax: Axes,
    entity_id: int,
    results: AnalysisResults,
) -> None:
    """Scatter of scenario points: risk vs return (IV upside %).

    Three points for pessimistic, base, and optimistic scenarios.
    Return is (iv_per_share / current_price - 1) * 100.
    Risk proxy is the scenario index scaled by (1 - composite_safety).

    Args:
        ax: Matplotlib axes to draw on.
        entity_id: Company identifier.
        results: Complete pipeline output.
    """
    ivs = _get_primary_ivs(results, entity_id)
    price = _current_price(results, entity_id)

    if not ivs or price <= 0:
        ax.set_title("Risk-Return — no data")
        ax.axis("off")
        return

    # Get composite safety as risk proxy (inverted: lower safety = higher risk)
    safety = 0.5
    if entity_id in results.combined_rankings.index:
        safety = float(results.combined_rankings.loc[entity_id, "composite_safety"])  # type: ignore[arg-type]

    risk_base = (1 - safety) * 100

    scenarios = ["pessimistic", "base", "optimistic"]
    risk_multipliers = [1.5, 1.0, 0.5]  # pessimistic = higher risk
    colours = [_C0, _C1, _C2]

    x_vals: list[float] = []
    y_vals: list[float] = []
    c_vals: list[tuple[float, ...]] = []
    labels: list[str] = []

    for scenario, risk_mult, colour in zip(scenarios, risk_multipliers, colours, strict=False):
        iv = ivs.get(scenario)
        if iv is None:
            continue

        upside = (iv.iv_per_share / price - 1) * 100
        risk = risk_base * risk_mult if risk_base > 0 else risk_mult * 10

        x_vals.append(risk)
        y_vals.append(upside)
        c_vals.append(colour)
        labels.append(scenario.capitalize())

    if not x_vals:
        ax.set_title("Risk-Return — no data")
        ax.axis("off")
        return

    ax.scatter(x_vals, y_vals, c=c_vals, s=100, zorder=3)

    for x, y, label in zip(x_vals, y_vals, labels, strict=False):
        ax.annotate(
            label, (x, y),
            textcoords="offset points", xytext=(8, 4),
            fontsize=8,
        )

    ax.axhline(y=0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Risk (Safety-Weighted)")
    ax.set_ylabel("Return (IV Upside %)")
    ax.set_title("Risk-Return Profile")
