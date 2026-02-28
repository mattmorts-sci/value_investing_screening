"""Screening: data sufficiency, filters, MoS computation, and ranking."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pipeline.config import PipelineConfig, SortOption
from pipeline.data.models import CompanyData
from pipeline.metrics.quality import QualityMetrics
from pipeline.metrics.safety import SafetyMetrics
from pipeline.metrics.sentiment import SentimentMetrics
from pipeline.metrics.trends import TrendMetrics
from pipeline.metrics.valuation import ValuationMetrics
from pipeline.simulation import SimulationOutput

logger = logging.getLogger(__name__)


@dataclass
class CompanyAnalysis:
    """All computed analysis for a single company.

    Attributes:
        company: Core company data.
        valuation: Valuation metrics (EV, yields).
        trends: Trend and regression metrics. None if computation failed.
        safety: Debt coverage metrics.
        quality: F-Score and quality metrics.
        sentiment: Price return metrics.
        simulation: Monte Carlo IV estimates. None if trends unavailable.
    """

    company: CompanyData
    valuation: ValuationMetrics
    trends: TrendMetrics | None
    safety: SafetyMetrics
    quality: QualityMetrics
    sentiment: SentimentMetrics
    simulation: SimulationOutput | None


@dataclass
class ScreeningResult:
    """A company with its analysis, filter status, and margin of safety.

    Attributes:
        analysis: Full computed analysis.
        mos_p25: Margin of safety at P25 IV. None if IV <= 0 or no simulation.
        mos_p50: Margin of safety at P50 IV. None if IV <= 0 or no simulation.
        mos_p75: Margin of safety at P75 IV. None if IV <= 0 or no simulation.
        filtered: True if excluded by one or more filters.
        filter_reasons: Comma-separated filter reason codes.
    """

    analysis: CompanyAnalysis
    mos_p25: float | None
    mos_p50: float | None
    mos_p75: float | None
    filtered: bool
    filter_reasons: str


def compute_mos(iv: float, price: float) -> float | None:
    """Compute margin of safety.

    MoS = (IV - price) / IV. Positive when price is below IV.

    Args:
        iv: Intrinsic value per share.
        price: Current stock price.

    Returns:
        Margin of safety as a float, or None if IV <= 0.
    """
    if iv <= 0:
        return None
    return (iv - price) / iv


def apply_filters(
    analysis: CompanyAnalysis, config: PipelineConfig
) -> tuple[bool, str]:
    """Check data sufficiency and apply configurable filters.

    Stage 1 (data sufficiency): checks min_quarters and whether
    trends/simulation computed successfully. Stage 2 (configurable
    filters): each independently toggleable via config (None = disabled).

    Args:
        analysis: Full computed analysis for one company.
        config: Pipeline configuration with filter thresholds.

    Returns:
        (filtered, filter_reasons) where filtered is True if any
        check triggered, and filter_reasons is a comma-separated
        string of reason codes (e.g. "DATA", "MC,IC").
    """
    reasons: list[str] = []

    # Stage 1: Data sufficiency
    n_quarters = len(analysis.company.financials)
    if (
        n_quarters < config.min_quarters
        or analysis.trends is None
        or analysis.simulation is None
    ):
        reasons.append("DATA")
        # Skip Stage 2 filters — metrics may be missing
        return True, ",".join(reasons)

    # Stage 2: Configurable filters (only if data sufficient)

    # Market cap
    if (
        config.filter_min_market_cap is not None
        and analysis.company.market_cap < config.filter_min_market_cap
    ):
        reasons.append("MC")

    # Interest coverage
    if config.filter_min_interest_coverage is not None:
        ic = analysis.safety.interest_coverage
        if ic is None or ic < config.filter_min_interest_coverage:
            reasons.append("IC")

    # OCF to debt
    if config.filter_min_ocf_to_debt is not None:
        od = analysis.safety.ocf_to_debt
        if od is None or od < config.filter_min_ocf_to_debt:
            reasons.append("OD")

    # FCF conversion R-squared
    if config.filter_min_fcf_conversion_r2 is not None:
        cr = analysis.trends.conversion_r_squared
        if cr is None or cr < config.filter_min_fcf_conversion_r2:
            reasons.append("CR")

    # Negative TTM FCF
    if config.filter_exclude_negative_ttm_fcf and analysis.valuation.ttm_fcf <= 0:
        reasons.append("FCF")

    filtered = len(reasons) > 0
    filter_reasons = ",".join(reasons)

    return filtered, filter_reasons


def screen_companies(
    analyses: list[CompanyAnalysis], config: PipelineConfig
) -> list[ScreeningResult]:
    """Screen, filter, and rank companies.

    Checks data sufficiency, applies configurable filters, computes
    margin of safety for each IV scenario, and sorts by the chosen
    valuation metric.

    Args:
        analyses: List of company analyses (some may lack trends/simulation).
        config: Pipeline configuration.

    Returns:
        Sorted list of ScreeningResult, unfiltered companies first,
        then filtered companies. Within each group, sorted descending
        by the chosen yield metric.
    """
    results: list[ScreeningResult] = []

    for analysis in analyses:
        filtered, filter_reasons = apply_filters(analysis, config)

        price = analysis.company.latest_price
        sim = analysis.simulation
        if sim is not None:
            mos_p25 = compute_mos(sim.iv_p25, price)
            mos_p50 = compute_mos(sim.iv_p50, price)
            mos_p75 = compute_mos(sim.iv_p75, price)
        else:
            mos_p25 = None
            mos_p50 = None
            mos_p75 = None

        results.append(
            ScreeningResult(
                analysis=analysis,
                mos_p25=mos_p25,
                mos_p50=mos_p50,
                mos_p75=mos_p75,
                filtered=filtered,
                filter_reasons=filter_reasons,
            )
        )

    # Determine sort metric
    def get_sort_value(r: ScreeningResult) -> float:
        val = r.analysis.valuation
        if config.sort_by == SortOption.FCF_EV:
            v = val.fcf_ev
        elif config.sort_by == SortOption.EBIT_EV:
            v = val.ebit_ev
        else:
            v = val.composite_yield

        # None values get -inf so they sort last in descending order
        return v if v is not None else float("-inf")

    # Sort: unfiltered first (0), filtered second (1).
    # Within each group, descending by yield (negate for ascending sort).
    results.sort(key=lambda r: (int(r.filtered), -get_sort_value(r)))

    return results
