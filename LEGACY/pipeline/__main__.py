"""CLI entry point for the value investing screening pipeline.

Usage:
    python -m pipeline --market AU --mode shortlist
    python -m pipeline --market US --mode owned --owned AAPL MSFT GOOG
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pipeline.config.settings import MARKET_EXCHANGES, AnalysisConfig
from pipeline.runner import run_analysis

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Value investing screening pipeline.",
    )
    parser.add_argument(
        "--market",
        choices=sorted(MARKET_EXCHANGES.keys()),
        default="AU",
        help="Market to analyse (default: AU).",
    )
    parser.add_argument(
        "--mode",
        choices=("shortlist", "owned"),
        default="shortlist",
        help="Analysis mode (default: shortlist).",
    )
    parser.add_argument(
        "--owned",
        nargs="+",
        default=[],
        metavar="SYMBOL",
        help="Owned company symbols (required when mode=owned).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: output).",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the pipeline, and return exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    kwargs: dict[str, object] = {
        "market": args.market,
        "mode": args.mode,
    }
    if args.owned:
        kwargs["owned_companies"] = args.owned
    if args.output_dir is not None:
        kwargs["output_directory"] = Path(args.output_dir)

    try:
        config = AnalysisConfig(**kwargs)  # type: ignore[arg-type]
    except ValueError as exc:
        logger.error("Invalid configuration: %s", exc)
        return 1

    try:
        results = run_analysis(config)
    except Exception:
        logger.exception("Pipeline failed")
        return 1

    logger.info(
        "Done: %d companies ranked, %d on watchlist",
        len(results.combined_rankings),
        len(results.watchlist),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
