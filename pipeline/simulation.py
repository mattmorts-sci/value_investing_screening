"""Monte Carlo simulation for intrinsic value estimation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from pipeline.config import DCFConfig, HeatmapConfig, SimulationConfig

logger = logging.getLogger(__name__)


@dataclass
class SimulationInput:
    """Input parameters for Monte Carlo simulation.

    Constructed from TrendMetrics output and CompanyData fields.

    Attributes:
        revenue_qoq_growth_mean: Mean QoQ revenue growth rate.
        revenue_qoq_growth_var: Variance of QoQ revenue growth rates.
        margin_intercept: Operating margin regression intercept.
        margin_slope: Operating margin regression slope (per quarter).
        conversion_intercept: FCF conversion regression intercept
            (None if fallback).
        conversion_slope: FCF conversion regression slope
            (None if fallback).
        conversion_median: Median FCF conversion (set if fallback).
        conversion_is_fallback: True if using median instead of regression.
        starting_revenue: Latest quarterly revenue.
        shares_outstanding: Diluted shares outstanding.
        num_historical_quarters: Number of quarters in the historical
            financials DataFrame. Used to offset margin and conversion
            regression indices so projections continue from the current
            level rather than resetting to the intercept.
    """

    revenue_qoq_growth_mean: float
    revenue_qoq_growth_var: float
    margin_intercept: float
    margin_slope: float
    conversion_intercept: float | None
    conversion_slope: float | None
    conversion_median: float | None
    conversion_is_fallback: bool
    starting_revenue: float
    shares_outstanding: float
    num_historical_quarters: int


@dataclass
class PathData:
    """Single simulation path trajectory for charting.

    Attributes:
        quarterly_revenue: Revenue at each projected quarter.
        quarterly_fcf: Free cash flow at each projected quarter.
    """

    quarterly_revenue: np.ndarray
    quarterly_fcf: np.ndarray


@dataclass
class PercentileBands:
    """Percentile bands across all simulation paths per quarter.

    Each array has shape (num_quarters,) where num_quarters =
    projection_years × 4.

    Attributes:
        p10: 10th percentile at each quarter.
        p25: 25th percentile at each quarter.
        p50: 50th percentile (median) at each quarter.
        p75: 75th percentile at each quarter.
        p90: 90th percentile at each quarter.
    """

    p10: np.ndarray
    p25: np.ndarray
    p50: np.ndarray
    p75: np.ndarray
    p90: np.ndarray


@dataclass
class SimulationOutput:
    """Monte Carlo simulation results.

    Attributes:
        iv_p10: 10th percentile intrinsic value per share.
        iv_p25: 25th percentile intrinsic value per share.
        iv_p50: 50th percentile (median) intrinsic value per share.
        iv_p75: 75th percentile intrinsic value per share.
        iv_p90: 90th percentile intrinsic value per share.
        iv_spread: P75 - P25 spread.
        implied_cagr_p25: Implied CAGR from current price to P25 IV.
        implied_cagr_p50: Implied CAGR from current price to P50 IV.
        implied_cagr_p75: Implied CAGR from current price to P75 IV.
        sample_paths: Display paths for charting.
        revenue_bands: Percentile bands for revenue across all paths.
            None when num_display_paths is 0.
        fcf_bands: Percentile bands for FCF across all paths.
            None when num_display_paths is 0.
    """

    iv_p10: float
    iv_p25: float
    iv_p50: float
    iv_p75: float
    iv_p90: float
    iv_spread: float

    implied_cagr_p25: float
    implied_cagr_p50: float
    implied_cagr_p75: float

    sample_paths: list[PathData]
    revenue_bands: PercentileBands | None
    fcf_bands: PercentileBands | None


def _compute_lognormal_params(
    mean: float, variance: float, cv_cap: float
) -> tuple[float, float]:
    """Convert growth rate mean/variance to log-normal mu/sigma for (1+g).

    Models (1 + growth_rate) as log-normal so that growth rates are
    bounded below by -100% (revenue cannot go negative).

    If the coefficient of variation exceeds cv_cap, the variance is
    clamped to keep dispersion bounded.

    Args:
        mean: Mean of QoQ growth rates.
        variance: Variance of QoQ growth rates.
        cv_cap: Maximum coefficient of variation.

    Returns:
        (mu, sigma) parameters for the log-normal distribution.
    """
    m = 1.0 + mean  # E[1+g]

    if m <= 0:
        # Growth so negative that expected (1+g) is non-positive.
        # Fall back to zero-growth log-normal.
        logger.warning(
            "Expected (1+g) = %.4f is non-positive, falling back to zero growth",
            m,
        )
        return 0.0, 0.01  # Tiny sigma, centered on 1.0

    # Cap CV = sqrt(var) / |mean of (1+g)|
    std = math.sqrt(max(variance, 0.0))
    if m > 0 and std / m > cv_cap:
        std = cv_cap * m
        variance = std**2

    # Log-normal parameters: if X ~ LogNormal(mu, sigma),
    # E[X] = exp(mu + sigma^2/2), Var[X] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
    sigma_sq = math.log(1.0 + variance / (m**2))
    sigma = math.sqrt(sigma_sq)
    mu = math.log(m) - sigma_sq / 2.0

    return mu, sigma


def _apply_growth_constraints(
    growth: float,
    quarter_idx: int,
    cumulative_revenue: float,
    starting_revenue: float,
    historical_mean: float,
    sim_config: SimulationConfig,
) -> float:
    """Apply all 8 growth constraints to a sampled growth rate.

    Constraints applied in order:
    1. CV cap (applied at distribution level, not here)
    2. Per-quarter growth caps (early vs late, positive vs negative)
    3. Cumulative cap
    4. CAGR backstop
    5. Momentum exhaustion
    6. Time decay
    7. Size penalty

    Args:
        growth: Raw sampled growth rate.
        quarter_idx: Zero-based quarter index (0 to num_quarters-1).
        cumulative_revenue: Current projected revenue before this growth.
        starting_revenue: Initial quarterly revenue.
        historical_mean: Historical QoQ growth mean.
        sim_config: Simulation configuration.

    Returns:
        Constrained growth rate.
    """
    revenue_ratio = cumulative_revenue / starting_revenue if starting_revenue > 0 else 1.0

    # --- Constraint 2: Per-quarter growth caps (early vs late) ---
    if revenue_ratio <= sim_config.size_tier_threshold:
        # Early stage: larger allowed growth
        growth = max(sim_config.early_negative_cap, min(growth, sim_config.early_positive_cap))
    else:
        # Late stage: tighter caps
        growth = max(sim_config.late_negative_cap, min(growth, sim_config.late_positive_cap))

    # --- Constraint 3: Cumulative cap ---
    # Prevent revenue from exceeding cumulative_cap × starting revenue
    if starting_revenue > 0 and cumulative_revenue > 0:
        max_revenue = starting_revenue * sim_config.cumulative_cap
        if cumulative_revenue * (1.0 + growth) > max_revenue:
            growth = max_revenue / cumulative_revenue - 1.0

    # --- Constraint 4: CAGR backstop ---
    # Annualised growth from start cannot exceed cagr_backstop
    years_elapsed = (quarter_idx + 1) / 4.0
    if years_elapsed > 0 and starting_revenue > 0 and cumulative_revenue > 0:
        max_annualised = sim_config.cagr_backstop
        # Max allowed cumulative factor at this point
        max_factor = (1.0 + max_annualised) ** years_elapsed
        max_allowed_revenue = starting_revenue * max_factor
        if cumulative_revenue * (1.0 + growth) > max_allowed_revenue:
            growth = max_allowed_revenue / cumulative_revenue - 1.0

    # --- Constraint 5: Momentum exhaustion ---
    # Positive mean: cap growth at momentum_upper × historical mean.
    # Negative mean: floor growth at momentum_lower × historical mean
    # (constrains decline to be less severe than historical rate).
    # Skip for zero/near-zero mean to avoid killing all positive growth.
    if historical_mean > 1e-6:
        upper_bound = historical_mean * sim_config.momentum_upper
        growth = min(growth, upper_bound)
    elif historical_mean < -1e-6:
        lower_bound = historical_mean * sim_config.momentum_lower
        growth = max(growth, lower_bound)

    # --- Constraint 6: Time decay ---
    # High growth rates decay over time
    if growth > sim_config.time_decay_growth_threshold:
        decay = sim_config.time_decay_factor ** quarter_idx
        excess = growth - sim_config.time_decay_growth_threshold
        growth = sim_config.time_decay_growth_threshold + excess * decay

    # --- Constraint 7: Size penalty ---
    # Scale down positive growth as revenue grows
    if growth > 0 and starting_revenue > 0:
        # Linear interpolation between max and min penalty based on revenue ratio
        # At ratio=1 (same size): penalty = size_penalty_max (least penalty)
        # At ratio=cumulative_cap: penalty = size_penalty_min (most penalty)
        cap = sim_config.cumulative_cap
        t = min((revenue_ratio - 1.0) / (cap - 1.0), 1.0) if cap > 1.0 else 0.0
        t = max(t, 0.0)
        penalty = sim_config.size_penalty_max - t * (
            sim_config.size_penalty_max - sim_config.size_penalty_min
        )
        growth *= penalty

    return growth


def _simulate_single_path(
    sim_input: SimulationInput,
    rng: np.random.Generator,
    mu: float,
    sigma: float,
    num_quarters: int,
    sim_config: SimulationConfig,
    dcf_config: DCFConfig,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Simulate a single path and return total PV and trajectory.

    Args:
        sim_input: Simulation input parameters.
        rng: Random number generator.
        mu: Log-normal mu parameter.
        sigma: Log-normal sigma parameter.
        num_quarters: Number of quarters to project.
        sim_config: Simulation configuration.
        dcf_config: DCF configuration.

    Returns:
        (total_pv, quarterly_revenue, quarterly_fcf) tuple.
    """
    quarterly_discount = (1.0 + dcf_config.discount_rate) ** 0.25

    revenue = sim_input.starting_revenue
    total_pv = 0.0
    revenues = np.empty(num_quarters)
    fcfs = np.empty(num_quarters)

    for q in range(num_quarters):
        # Step 1: Sample growth from log-normal
        sample = rng.lognormal(mu, sigma)
        raw_growth = sample - 1.0

        # Step 2: Apply constraints
        growth = _apply_growth_constraints(
            raw_growth,
            q,
            revenue,
            sim_input.starting_revenue,
            sim_input.revenue_qoq_growth_mean,
            sim_config,
        )

        # Step 3: Project revenue
        revenue = revenue * (1.0 + growth)
        revenues[q] = revenue

        # Step 4: Compute margin
        # Continue regression from last historical quarter index.
        regression_idx = sim_input.num_historical_quarters + q
        margin = sim_input.margin_intercept + sim_input.margin_slope * regression_idx

        # Step 5: Operating income
        operating_income = revenue * margin

        # Step 6: FCF conversion
        if sim_input.conversion_is_fallback:
            conversion = sim_input.conversion_median if sim_input.conversion_median is not None else 1.0
        else:
            intercept = sim_input.conversion_intercept if sim_input.conversion_intercept is not None else 1.0
            slope = sim_input.conversion_slope if sim_input.conversion_slope is not None else 0.0
            conversion = intercept + slope * regression_idx

        # Step 7: FCF
        fcf = operating_income * conversion
        fcfs[q] = fcf

        # Step 8: Discount to present value
        discount_factor = quarterly_discount ** (q + 1)
        pv = fcf / discount_factor
        total_pv += pv

    # Terminal value: perpetuity growth model on last quarter's FCF.
    # Only applied when last FCF is positive — the Gordon Growth Model
    # assumes a going concern generating positive cash flow.
    if num_quarters > 0:
        last_fcf = fcfs[-1]
        if last_fcf > 0:
            annual_fcf = last_fcf * 4.0  # Annualise quarterly FCF
            terminal_rate = dcf_config.discount_rate - dcf_config.terminal_growth_rate
            if terminal_rate <= 0:
                logger.warning(
                    "Terminal rate is non-positive (discount=%.4f, terminal_growth=%.4f), "
                    "skipping terminal value",
                    dcf_config.discount_rate,
                    dcf_config.terminal_growth_rate,
                )
            if terminal_rate > 0:
                terminal_value = annual_fcf * (1.0 + dcf_config.terminal_growth_rate) / terminal_rate
                # Discount terminal value back to present
                terminal_discount = quarterly_discount ** num_quarters
                total_pv += terminal_value / terminal_discount

    return total_pv, revenues, fcfs


def _implied_cagr(
    current_price: float, iv: float, years: float
) -> float:
    """Back-calculate implied CAGR from current price to intrinsic value.

    Args:
        current_price: Current stock price.
        iv: Intrinsic value per share.
        years: Number of years for the CAGR calculation.

    Returns:
        Implied annualised return. Returns 0.0 if inputs are invalid.
    """
    if current_price <= 0 or iv <= 0 or years <= 0:
        return 0.0
    return (iv / current_price) ** (1.0 / years) - 1.0


def run_simulation(
    sim_input: SimulationInput,
    current_price: float,
    sim_config: SimulationConfig | None = None,
    dcf_config: DCFConfig | None = None,
    seed: int | None = None,
) -> SimulationOutput:
    """Run Monte Carlo simulation to estimate intrinsic value distribution.

    Args:
        sim_input: Simulation input parameters.
        current_price: Current stock price (for implied CAGR calculation).
        sim_config: Simulation configuration. Uses defaults if None.
        dcf_config: DCF configuration. Uses defaults if None.
        seed: Random seed for reproducibility. None for random.

    Returns:
        SimulationOutput with IV percentiles, implied CAGRs, and sample paths.
    """
    if sim_input.starting_revenue <= 0:
        raise ValueError(
            f"starting_revenue must be positive, got {sim_input.starting_revenue}"
        )
    if sim_input.shares_outstanding <= 0:
        raise ValueError(
            f"shares_outstanding must be positive, got {sim_input.shares_outstanding}"
        )

    if sim_config is None:
        sim_config = SimulationConfig()
    if dcf_config is None:
        dcf_config = DCFConfig()

    num_quarters = dcf_config.projection_years * 4
    rng = np.random.default_rng(seed)

    # Compute log-normal parameters
    mu, sigma = _compute_lognormal_params(
        sim_input.revenue_qoq_growth_mean,
        sim_input.revenue_qoq_growth_var,
        sim_config.cv_cap,
    )

    # Run all paths
    compute_bands = sim_config.num_display_paths > 0
    total_pvs = np.empty(sim_config.num_replicates)

    if compute_bands:
        all_revenues = np.empty((sim_config.num_replicates, num_quarters))
        all_fcfs = np.empty((sim_config.num_replicates, num_quarters))

    for i in range(sim_config.num_replicates):
        pv, revenues, fcfs = _simulate_single_path(
            sim_input, rng, mu, sigma, num_quarters, sim_config, dcf_config
        )
        total_pvs[i] = pv

        if compute_bands:
            all_revenues[i] = revenues
            all_fcfs[i] = fcfs

    # IV per share
    iv_per_share = total_pvs / sim_input.shares_outstanding

    # Percentiles
    p10, p25, p50, p75, p90 = np.percentile(iv_per_share, [10, 25, 50, 75, 90])

    # Implied CAGRs
    years = float(dcf_config.projection_years)

    # Build sample paths and percentile bands
    if compute_bands:
        sample_paths = [
            PathData(
                quarterly_revenue=all_revenues[i],
                quarterly_fcf=all_fcfs[i],
            )
            for i in range(min(sim_config.num_display_paths, sim_config.num_replicates))
        ]

        pct_levels = [10, 25, 50, 75, 90]
        rev_pcts = np.percentile(all_revenues, pct_levels, axis=0)
        fcf_pcts = np.percentile(all_fcfs, pct_levels, axis=0)

        revenue_bands = PercentileBands(
            p10=rev_pcts[0], p25=rev_pcts[1], p50=rev_pcts[2],
            p75=rev_pcts[3], p90=rev_pcts[4],
        )
        fcf_bands = PercentileBands(
            p10=fcf_pcts[0], p25=fcf_pcts[1], p50=fcf_pcts[2],
            p75=fcf_pcts[3], p90=fcf_pcts[4],
        )
    else:
        sample_paths = []
        revenue_bands = None
        fcf_bands = None

    return SimulationOutput(
        iv_p10=float(p10),
        iv_p25=float(p25),
        iv_p50=float(p50),
        iv_p75=float(p75),
        iv_p90=float(p90),
        iv_spread=float(p75 - p25),
        implied_cagr_p25=_implied_cagr(current_price, float(p25), years),
        implied_cagr_p50=_implied_cagr(current_price, float(p50), years),
        implied_cagr_p75=_implied_cagr(current_price, float(p75), years),
        sample_paths=sample_paths,
        revenue_bands=revenue_bands,
        fcf_bands=fcf_bands,
    )


def run_parameterised_dcf(
    sim_input: SimulationInput,
    current_price: float,
    discount_rate: float,
    growth_multiplier: float,
    sim_config: SimulationConfig | None = None,
    dcf_config: DCFConfig | None = None,
    heatmap_config: HeatmapConfig | None = None,
    seed: int | None = None,
) -> float:
    """Run a parameterised DCF for sensitivity analysis.

    For discount rate variation: re-discounts using a modified DCF config.
    For growth variation: scales revenue_qoq_growth_mean by the multiplier.

    Both variations are applied simultaneously to produce a single
    median IV per share for the given (discount_rate, growth_multiplier)
    combination.

    Args:
        sim_input: Base simulation input.
        current_price: Current stock price.
        discount_rate: Discount rate override.
        growth_multiplier: Multiplier applied to revenue_qoq_growth_mean.
        sim_config: Base simulation config. Uses defaults if None.
        dcf_config: Base DCF config. Uses defaults if None.
        heatmap_config: Heatmap config for replicate count. Uses defaults if None.
        seed: Random seed for reproducibility.

    Returns:
        Median IV per share for the given parameters.
    """
    if sim_config is None:
        sim_config = SimulationConfig()
    if dcf_config is None:
        dcf_config = DCFConfig()
    if heatmap_config is None:
        heatmap_config = HeatmapConfig()

    # Create modified inputs
    modified_input = SimulationInput(
        revenue_qoq_growth_mean=sim_input.revenue_qoq_growth_mean * growth_multiplier,
        revenue_qoq_growth_var=sim_input.revenue_qoq_growth_var,
        margin_intercept=sim_input.margin_intercept,
        margin_slope=sim_input.margin_slope,
        conversion_intercept=sim_input.conversion_intercept,
        conversion_slope=sim_input.conversion_slope,
        conversion_median=sim_input.conversion_median,
        conversion_is_fallback=sim_input.conversion_is_fallback,
        starting_revenue=sim_input.starting_revenue,
        shares_outstanding=sim_input.shares_outstanding,
        num_historical_quarters=sim_input.num_historical_quarters,
    )

    modified_dcf = DCFConfig(
        discount_rate=discount_rate,
        terminal_growth_rate=dcf_config.terminal_growth_rate,
        projection_years=dcf_config.projection_years,
    )

    modified_sim = SimulationConfig(
        num_replicates=heatmap_config.heatmap_replicates,
        num_display_paths=0,  # No sample paths needed for heatmap
        cv_cap=sim_config.cv_cap,
        early_positive_cap=sim_config.early_positive_cap,
        early_negative_cap=sim_config.early_negative_cap,
        late_positive_cap=sim_config.late_positive_cap,
        late_negative_cap=sim_config.late_negative_cap,
        size_tier_threshold=sim_config.size_tier_threshold,
        cumulative_cap=sim_config.cumulative_cap,
        cagr_backstop=sim_config.cagr_backstop,
        momentum_upper=sim_config.momentum_upper,
        momentum_lower=sim_config.momentum_lower,
        time_decay_growth_threshold=sim_config.time_decay_growth_threshold,
        time_decay_factor=sim_config.time_decay_factor,
        size_penalty_min=sim_config.size_penalty_min,
        size_penalty_max=sim_config.size_penalty_max,
    )

    result = run_simulation(
        modified_input,
        current_price,
        sim_config=modified_sim,
        dcf_config=modified_dcf,
        seed=seed,
    )

    return result.iv_p50
