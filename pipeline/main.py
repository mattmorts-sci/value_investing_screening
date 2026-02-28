"""CLI entry point for the value investing screening pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pipeline.config import (
    DCFConfig,
    HeatmapConfig,
    PipelineConfig,
    SimulationConfig,
    SortOption,
    TrendsConfig,
)
from pipeline.data import load_company, load_universe, lookup_entity_ids
from pipeline.data.models import CompanyData
from pipeline.metrics.quality import compute_quality
from pipeline.metrics.safety import compute_safety
from pipeline.metrics.sentiment import compute_sentiment
from pipeline.metrics.trends import TrendMetrics, compute_trends
from pipeline.metrics.valuation import compute_valuation
from pipeline.output.csv_export import export_csv
from pipeline.output.detail_report import generate_detail_html
from pipeline.screening import CompanyAnalysis, ScreeningResult, compute_mos, screen_companies
from pipeline.simulation import SimulationInput, SimulationOutput, run_simulation

logger = logging.getLogger(__name__)


def _build_simulation_input(
    trends: TrendMetrics,
    company: CompanyData,
) -> SimulationInput:
    """Build SimulationInput from trends output and company data.

    Args:
        trends: Computed trend metrics.
        company: Company data with starting revenue and shares.

    Returns:
        SimulationInput ready for run_simulation.
    """
    # Starting revenue: latest quarterly revenue
    latest_revenue = float(company.financials["revenue"].iloc[-1])

    return SimulationInput(
        revenue_qoq_growth_mean=trends.revenue_qoq_growth_mean,
        revenue_qoq_growth_var=trends.revenue_qoq_growth_var,
        margin_intercept=trends.margin_intercept,
        margin_slope=trends.margin_slope,
        conversion_intercept=trends.conversion_intercept,
        conversion_slope=trends.conversion_slope,
        conversion_median=trends.conversion_median,
        conversion_is_fallback=trends.conversion_is_fallback,
        starting_revenue=latest_revenue,
        shares_outstanding=company.shares_outstanding,
        num_historical_quarters=len(company.financials),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Value investing screening pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # screen command
    screen_parser = subparsers.add_parser(
        "screen", help="Screen companies and export CSV"
    )
    screen_parser.add_argument(
        "--exchanges",
        nargs="+",
        required=True,
        help="Stock exchanges to screen (e.g. ASX NYSE)",
    )
    screen_parser.add_argument(
        "--sort",
        choices=["composite", "fcf_ev", "ebit_ev"],
        default="composite",
        help="Sort metric (default: composite)",
    )
    screen_parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/screening.csv"),
        help="Output CSV path (default: output/screening.csv)",
    )
    screen_parser.add_argument(
        "--min-market-cap",
        type=float,
        default=None,
        help="Minimum market cap filter (default: disabled)",
    )
    screen_parser.add_argument(
        "--min-interest-coverage",
        type=float,
        default=None,
        help="Minimum interest coverage filter (default: disabled)",
    )
    screen_parser.add_argument(
        "--min-ocf-to-debt",
        type=float,
        default=None,
        help="Minimum OCF/debt filter (default: disabled)",
    )
    screen_parser.add_argument(
        "--min-fcf-conversion-r2",
        type=float,
        default=None,
        help="Minimum FCF conversion R-squared filter (default: disabled)",
    )
    screen_parser.add_argument(
        "--include-negative-fcf",
        action="store_true",
        help="Include companies with negative TTM FCF (default: exclude)",
    )
    screen_parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="FMP database path (default: from config)",
    )
    screen_parser.add_argument(
        "--min-quarters",
        type=int,
        default=None,
        help="Minimum quarters of data required (default: 12)",
    )
    screen_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    # detail command
    detail_parser = subparsers.add_parser(
        "detail", help="Generate HTML detail reports for specific tickers"
    )
    detail_parser.add_argument(
        "tickers",
        nargs="+",
        help="Ticker symbols to generate reports for",
    )
    detail_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/detail"),
        help="Output directory for HTML reports (default: output/detail/)",
    )
    detail_parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="FMP database path (default: from config)",
    )
    detail_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args(argv)


SORT_MAP = {
    "composite": SortOption.COMPOSITE,
    "fcf_ev": SortOption.FCF_EV,
    "ebit_ev": SortOption.EBIT_EV,
}


def _process_company(
    company: CompanyData,
    config: PipelineConfig,
    trends_config: TrendsConfig,
    sim_config: SimulationConfig,
    dcf_config: DCFConfig,
) -> CompanyAnalysis:
    """Compute all metrics for a single company.

    Valuation, safety, quality, and sentiment are always computed.
    Trends and simulation may be None if computation fails.

    Args:
        company: Loaded company data.
        config: Pipeline configuration.
        trends_config: Trends configuration.
        sim_config: Simulation configuration.
        dcf_config: DCF configuration.

    Returns:
        CompanyAnalysis with all available metrics.
    """
    valuation = compute_valuation(company, config)
    safety = compute_safety(company)
    quality = compute_quality(company)
    sentiment = compute_sentiment(company)

    # Trends may return None (insufficient data for regressions)
    trends = compute_trends(company, trends_config)

    # Simulation requires successful trends, positive starting revenue,
    # and positive shares outstanding
    simulation: SimulationOutput | None = None
    if trends is not None:
        sim_input = _build_simulation_input(trends, company)
        if sim_input.starting_revenue <= 0:
            logger.debug(
                "%s: non-positive starting revenue, skipping simulation",
                company.symbol,
            )
        elif sim_input.shares_outstanding <= 0:
            logger.warning(
                "%s: non-positive shares outstanding, skipping simulation",
                company.symbol,
            )
        else:
            simulation = run_simulation(
                sim_input,
                company.latest_price,
                sim_config=sim_config,
                dcf_config=dcf_config,
            )

    return CompanyAnalysis(
        company=company,
        valuation=valuation,
        trends=trends,
        safety=safety,
        quality=quality,
        sentiment=sentiment,
        simulation=simulation,
    )


def run_screen(args: argparse.Namespace) -> None:
    """Execute the screen command.

    Args:
        args: Parsed CLI arguments.
    """
    # Build config
    config = PipelineConfig(
        exchanges=args.exchanges,
        sort_by=SORT_MAP[args.sort],
        filter_min_market_cap=args.min_market_cap,
        filter_min_interest_coverage=args.min_interest_coverage,
        filter_min_ocf_to_debt=args.min_ocf_to_debt,
        filter_min_fcf_conversion_r2=args.min_fcf_conversion_r2,
        filter_exclude_negative_ttm_fcf=not args.include_negative_fcf,
    )
    if args.db_path is not None:
        config.db_path = args.db_path
    if args.min_quarters is not None:
        config.min_quarters = args.min_quarters

    trends_config = TrendsConfig()
    sim_config = SimulationConfig()
    dcf_config = DCFConfig()

    # Load universe
    universe = load_universe(config)
    logger.info("Universe: %d symbols from %s", len(universe), config.exchanges)

    # Process each company
    analyses: list[CompanyAnalysis] = []
    skipped = 0

    for entity_id, symbol in universe:
        company = load_company(entity_id, config)
        if company is None:
            logger.warning("%s: failed to load, skipping", symbol)
            skipped += 1
            continue

        analyses.append(
            _process_company(company, config, trends_config, sim_config, dcf_config)
        )

    logger.info(
        "Processed %d companies (%d failed to load)", len(analyses), skipped
    )

    # Screen and export
    results = screen_companies(analyses, config)
    export_csv(results, args.output)

    # Summary
    n_filtered = sum(1 for r in results if r.filtered)
    n_passed = len(results) - n_filtered
    logger.info(
        "Results: %d passed filters, %d filtered, written to %s",
        n_passed,
        n_filtered,
        args.output,
    )


def run_detail(args: argparse.Namespace) -> None:
    """Execute the detail command.

    Looks up entity IDs for the given tickers, loads each company,
    computes all metrics, and generates an HTML detail report per ticker.

    Args:
        args: Parsed CLI arguments (tickers, output_dir, db_path).
    """
    config = PipelineConfig()
    if args.db_path is not None:
        config.db_path = args.db_path

    trends_config = TrendsConfig()
    sim_config = SimulationConfig()
    dcf_config = DCFConfig()
    heatmap_config = HeatmapConfig()

    # Look up entity IDs
    entity_map = lookup_entity_ids(args.tickers, config)
    not_found = [t for t in args.tickers if t not in entity_map]
    if not_found:
        logger.warning("Tickers not found: %s", ", ".join(not_found))

    # Create output directory
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for ticker in args.tickers:
        if ticker not in entity_map:
            continue

        entity_id = entity_map[ticker]
        company = load_company(entity_id, config)
        if company is None:
            logger.warning("%s: failed to load, skipping", ticker)
            continue

        analysis = _process_company(
            company, config, trends_config, sim_config, dcf_config,
        )

        # Build sim_input for heatmap (if trends available)
        sim_input: SimulationInput | None = None
        if analysis.trends is not None:
            candidate = _build_simulation_input(analysis.trends, company)
            if candidate.starting_revenue > 0:
                sim_input = candidate

        # Compute MoS from simulation output
        price = company.latest_price
        sim = analysis.simulation
        if sim is not None:
            mos_p25 = compute_mos(sim.iv_p25, price)
            mos_p50 = compute_mos(sim.iv_p50, price)
            mos_p75 = compute_mos(sim.iv_p75, price)
        else:
            mos_p25 = None
            mos_p50 = None
            mos_p75 = None

        result = ScreeningResult(
            analysis=analysis,
            mos_p25=mos_p25,
            mos_p50=mos_p50,
            mos_p75=mos_p75,
            filtered=False,
            filter_reasons="",
        )

        # Generate HTML
        html_content = generate_detail_html(
            result=result,
            sim_input=sim_input,
            heatmap_config=heatmap_config,
            sim_config=sim_config,
            dcf_config=dcf_config,
        )

        output_path = output_dir / f"{ticker}.html"
        output_path.write_text(html_content)
        logger.info("%s: report written to %s", ticker, output_path)
        generated += 1

    logger.info("Generated %d detail reports in %s", generated, output_dir)


def main(argv: list[str] | None = None) -> None:
    """Main entry point.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).
    """
    args = _parse_args(argv)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "screen":
        run_screen(args)
    elif args.command == "detail":
        run_detail(args)
    else:
        logger.error("Unknown command: %s", args.command)
        sys.exit(1)


if __name__ == "__main__":
    main()
