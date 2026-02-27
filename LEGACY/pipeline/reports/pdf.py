"""PDF report generation using Jinja2 templates and WeasyPrint.

Generates a comprehensive PDF report from AnalysisResults by rendering
chart figures into a Jinja2 HTML template and converting to PDF via
WeasyPrint.
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jinja2
import matplotlib
import matplotlib.pyplot as plt
import weasyprint

from pipeline.charts import (
    acquirers_multiple_analysis,
    comprehensive_rankings_table,
    factor_contributions_bar,
    factor_heatmap,
    financial_health_dashboard,
    growth_comparison_historical,
    growth_comparison_projected,
    growth_fan,
    growth_projection,
    growth_value_scatter,
    historical_growth,
    projected_growth_stability,
    ranking_comparison_table,
    risk_adjusted_opportunity,
    scenario_comparison,
    valuation_matrix,
    valuation_upside_comparison,
    watchlist_summary,
)
from pipeline.charts.tables import filter_summary

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from pipeline.data.contracts import AnalysisResults

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


def generate_pdf(
    results: AnalysisResults,
    output_path: Path,
    detailed_symbols: list[str] | None = None,
) -> Path:
    """Generate a comprehensive PDF report from analysis results.

    Args:
        results: Complete pipeline output.
        output_path: Destination path for the PDF file.
        detailed_symbols: Symbols to include as detailed company pages.
            Defaults to the top N from the watchlist per
            config.detailed_report_count.

    Returns:
        Path to the generated PDF file.
    """
    # Use non-interactive backend for rendering
    matplotlib.use("Agg")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = results.config

    if detailed_symbols is None:
        detailed_symbols = results.watchlist[:config.detailed_report_count]

    # Resolve entity_ids for detailed companies
    symbol_to_entity: dict[str, int] = {}
    for eid in results.companies.index:
        sym = str(results.companies.loc[eid, "symbol"])
        symbol_to_entity[sym] = eid

    # Build template context
    context = _build_context(results, detailed_symbols, symbol_to_entity)

    # Render HTML
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=True,
    )
    template = env.get_template("report.html")
    html_content = template.render(**context)

    # Generate PDF
    pdf_doc = weasyprint.HTML(string=html_content).write_pdf()
    output_path.write_bytes(pdf_doc)

    logger.info("PDF report generated: %s", output_path)
    return output_path


def _build_context(
    results: AnalysisResults,
    detailed_symbols: list[str],
    symbol_to_entity: dict[str, int],
) -> dict[str, Any]:
    """Build the Jinja2 template context with all chart images."""
    config = results.config

    # Count companies at various stages
    post_filter_count = len(results.companies)
    priced_count = len(results.combined_rankings)
    pre_filter_count = post_filter_count
    for removed_list in results.filter_log.removed.values():
        pre_filter_count += len(removed_list)

    # Market overview charts
    market_charts = _render_market_charts(results)

    # Comparative charts (for watchlist)
    watchlist_symbols = results.watchlist[:20]
    comparative_charts = _render_comparative_charts(results, watchlist_symbols)

    # Detailed company pages
    detailed_companies = _render_company_pages(
        results, detailed_symbols, symbol_to_entity,
    )

    # Summary tables
    watchlist_chart = _fig_to_base64(watchlist_summary(results))
    filter_chart = _fig_to_base64(filter_summary(results))

    # Full rankings
    all_symbols = [
        str(results.combined_rankings.loc[eid, "symbol"])
        for eid in results.combined_rankings.index
    ]
    rankings_chart = _fig_to_base64(
        comprehensive_rankings_table(results, all_symbols[:25]),
    )

    return {
        "title": "Intrinsic Value Analysis Report",
        "market": config.market,
        "analysis_date": datetime.now().strftime("%d %B %Y"),
        "primary_period": config.primary_period,
        "discount_rate": f"{config.discount_rate:.1%}",
        "margin_of_safety": f"{config.margin_of_safety:.0%}",
        "terminal_growth_rate": f"{config.terminal_growth_rate:.1%}",
        "company_count": pre_filter_count,
        "filtered_count": post_filter_count,
        "priced_count": priced_count,
        "watchlist_size": len(results.watchlist),
        "detailed_count": len(detailed_symbols),
        "filter_chart": filter_chart,
        "market_charts": market_charts,
        "comparative_charts": comparative_charts,
        "detailed_companies": detailed_companies,
        "watchlist_chart": watchlist_chart,
        "rankings_chart": rankings_chart,
    }


def _render_market_charts(results: AnalysisResults) -> list[str]:
    """Render all market overview charts to base64 strings."""
    charts: list[str] = []
    chart_fns = [
        growth_value_scatter,
        growth_comparison_historical,
        growth_comparison_projected,
        risk_adjusted_opportunity,
        factor_contributions_bar,
        factor_heatmap,
    ]
    for fn in chart_fns:
        try:
            fig = fn(results)
            charts.append(_fig_to_base64(fig))
        except Exception:
            logger.exception("Failed to render market chart %s", fn.__name__)
    return charts


def _render_comparative_charts(
    results: AnalysisResults,
    symbols: list[str],
) -> list[str]:
    """Render all comparative charts to base64 strings."""
    charts: list[str] = []
    chart_fns = [
        projected_growth_stability,
        acquirers_multiple_analysis,
        valuation_upside_comparison,
        ranking_comparison_table,
    ]
    for fn in chart_fns:
        try:
            fig = fn(results, symbols)
            charts.append(_fig_to_base64(fig))
        except Exception:
            logger.exception("Failed to render comparative chart %s", fn.__name__)
    return charts


def _render_company_pages(
    results: AnalysisResults,
    symbols: list[str],
    symbol_to_entity: dict[str, int],
) -> list[dict[str, Any]]:
    """Render per-company detail charts."""
    pages: list[dict[str, Any]] = []

    for symbol in symbols:
        entity_id = symbol_to_entity.get(symbol)
        if entity_id is None:
            logger.warning("Symbol %s not found in company data, skipping", symbol)
            continue

        company_name = str(results.companies.loc[entity_id, "company_name"])
        company_charts: list[str] = []

        detail_fns = [
            historical_growth,
            growth_projection,
            valuation_matrix,
            growth_fan,
            financial_health_dashboard,
            scenario_comparison,
        ]
        for fn in detail_fns:
            try:
                fig = fn(results, entity_id)
                company_charts.append(_fig_to_base64(fig))
            except Exception:
                logger.exception(
                    "Failed to render %s for %s", fn.__name__, symbol,
                )

        pages.append({
            "symbol": symbol,
            "name": company_name,
            "charts": company_charts,
        })

    return pages


def _fig_to_base64(fig: Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    result = base64.b64encode(buf.read()).decode("ascii")
    buf.close()
    return result
