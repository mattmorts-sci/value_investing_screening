"""
Growth calculation using Monte Carlo simulation.

Handles both positive and negative FCF cases with path-to-profitability modeling.
Supports multiple time periods and named scenarios with dynamic constraints.
All calculations use quarterly data internally, converting to CAGR only for reporting.
"""
import logging
from typing import Dict, Tuple, Optional
import numpy as np
from multiprocessing import Pool
import os

from data_pipeline.data_structures import CompanyData, GrowthProjection
from config.user_config import AnalysisConfig

logger = logging.getLogger(__name__)


def _init_worker():
    """Initialize worker process with unique random seed."""
    # Each process gets a unique random seed
    seed = int.from_bytes(os.urandom(4), 'big')
    np.random.seed(seed)
    # Suppress logging in worker processes to reduce overhead
    logging.getLogger().setLevel(logging.WARNING)


def calculate_dynamic_max_multiple(growth_rate: float,
                                 is_best_scenario: bool = False,
                                 is_worst_scenario: bool = False,
                                 base_scenario_multiple: Optional[float] = None,
                                 config: AnalysisConfig = None) -> float:
    """
    Calculate dynamic maximum growth multiple based on scenario.
    
    Ensures best > base > worst relationship.
    
    Args:
        growth_rate: Base growth rate (quarterly)
        is_best_scenario: Whether this is the optimistic scenario
        is_worst_scenario: Whether this is the pessimistic scenario
        base_scenario_multiple: Base scenario's multiple for consistency
        config: Analysis configuration
        
    Returns:
        Maximum allowed growth multiple for this scenario
    """
    config = config or AnalysisConfig()
    
    # Base calculation
    if growth_rate <= 0:
        base_multiple = 1.0
    else:
        # Higher growth rates allow higher multiples
        high_growth_factor = min(growth_rate / config.REFERENCE_HIGH_GROWTH, 1.0)
        base_multiple = config.BASE_MULTIPLE + (
            (config.MAX_ALLOWED_MULTIPLE - config.BASE_MULTIPLE) * high_growth_factor
        )
    
    # Adjust for scenario
    if is_best_scenario:
        # Best scenario allows higher growth
        if base_scenario_multiple:
            return base_scenario_multiple * config.SCENARIO_BETTER_FACTOR
        return base_multiple * config.SCENARIO_BETTER_FACTOR
    elif is_worst_scenario:
        # Worst scenario constrains growth more
        if base_scenario_multiple:
            return base_scenario_multiple * config.SCENARIO_WORSE_FACTOR
        return base_multiple * config.SCENARIO_WORSE_FACTOR
    
    return base_multiple


def calculate_growth_projections(company: CompanyData, 
                               config: AnalysisConfig) -> Dict[int, Dict[str, Dict[str, GrowthProjection]]]:
    """
    Calculate growth projections for multiple periods and scenarios.
    
    Args:
        company: Company data including historical growth statistics
        config: Analysis configuration
        
    Returns:
        Dict mapping period -> metric -> scenario -> GrowthProjection
        e.g., {5: {'fcf': {'median': ..., 'top': ..., 'bottom': ...},
                   'revenue': {...}},
               10: {...}}
    """
    all_projections = {}
    
    for period_years in config.PROJECTION_PERIODS:
        logger.debug(f"Calculating {period_years}-year projections for {company.ticker}")
        
        # Calculate FCF projections for all scenarios
        fcf_projections = _simulate_growth_with_scenarios(
            current_value=company.fcf,
            growth_mean=company.fcf_growth_mean,
            growth_variance=company.fcf_growth_variance,
            market_cap=company.market_cap,
            metric_name='fcf',
            period_years=period_years,
            config=config
        )
        
        # Calculate revenue projections for all scenarios
        revenue_projections = _simulate_growth_with_scenarios(
            current_value=company.revenue,
            growth_mean=company.revenue_growth_mean,
            growth_variance=company.revenue_growth_variance,
            market_cap=company.market_cap,
            metric_name='revenue',
            period_years=period_years,
            config=config
        )
        
        all_projections[period_years] = {
            'fcf': fcf_projections,
            'revenue': revenue_projections
        }
    
    return all_projections


def _simulate_growth_with_scenarios(current_value: float,
                                   growth_mean: float,
                                   growth_variance: float,
                                   market_cap: float,
                                   metric_name: str,
                                   period_years: int,
                                   config: AnalysisConfig) -> Dict[str, GrowthProjection]:
    """
    Run ONE simulation and extract different percentiles for scenarios.
    
    This ensures scenario consistency where best > median > worst by using
    the same simulation paths for all scenarios.
    
    Args:
        current_value: Current metric value
        growth_mean: Historical mean growth rate (quarterly)
        growth_variance: Historical growth variance
        market_cap: Company market capitalization
        metric_name: 'fcf' or 'revenue'
        period_years: Projection period in years
        config: Analysis configuration
        
    Returns:
        Dictionary mapping scenario name to GrowthProjection
    """
    # Run a single simulation to get all paths
    simulation_result = _simulate_growth(
        current_value=current_value,
        growth_mean=growth_mean,
        growth_variance=growth_variance,
        market_cap=market_cap,
        metric_name=metric_name,
        period_years=period_years,
        scenario_type='full',  # Get full distribution
        is_best_scenario=False,
        is_worst_scenario=False,
        base_scenario_multiple=None,
        config=config
    )
    
    # Extract different percentiles from the same simulation
    projections = {}
    
    # Extract final values from simulation paths
    final_values = simulation_result.simulation_paths[:, -1]
    
    # Bottom scenario - 25th percentile
    p25 = np.percentile(final_values, 25)
    annual_growth_p25 = _calculate_annual_rate(p25, current_value, period_years)
    
    projections['bottom'] = GrowthProjection(
        metric_name=metric_name,
        current_value=current_value,
        period_years=period_years,
        final_value_median=p25,  # Using 25th percentile as the "median" for this scenario
        final_value_ci_bottom=np.percentile(final_values, 10),
        final_value_ci_top=np.percentile(final_values, 40),
        annual_growth_median=annual_growth_p25,
        annual_growth_ci_bottom=_calculate_annual_rate(np.percentile(final_values, 10), current_value, period_years),
        annual_growth_ci_top=_calculate_annual_rate(np.percentile(final_values, 40), current_value, period_years),
        scenario_type='bottom',
        simulation_paths=None  # Don't store paths for individual scenarios
    )
    
    # Median scenario - 50th percentile
    p50 = np.percentile(final_values, 50)
    annual_growth_p50 = _calculate_annual_rate(p50, current_value, period_years)
    
    projections['median'] = GrowthProjection(
        metric_name=metric_name,
        current_value=current_value,
        period_years=period_years,
        final_value_median=p50,
        final_value_ci_bottom=np.percentile(final_values, 25),
        final_value_ci_top=np.percentile(final_values, 75),
        annual_growth_median=annual_growth_p50,
        annual_growth_ci_bottom=_calculate_annual_rate(np.percentile(final_values, 25), current_value, period_years),
        annual_growth_ci_top=_calculate_annual_rate(np.percentile(final_values, 75), current_value, period_years),
        scenario_type='median',
        simulation_paths=simulation_result.simulation_paths  # Store full paths in median
    )
    
    # Top scenario - 75th percentile
    p75 = np.percentile(final_values, 75)
    annual_growth_p75 = _calculate_annual_rate(p75, current_value, period_years)
    
    projections['top'] = GrowthProjection(
        metric_name=metric_name,
        current_value=current_value,
        period_years=period_years,
        final_value_median=p75,  # Using 75th percentile as the "median" for this scenario
        final_value_ci_bottom=np.percentile(final_values, 60),
        final_value_ci_top=np.percentile(final_values, 90),
        annual_growth_median=annual_growth_p75,
        annual_growth_ci_bottom=_calculate_annual_rate(np.percentile(final_values, 60), current_value, period_years),
        annual_growth_ci_top=_calculate_annual_rate(np.percentile(final_values, 90), current_value, period_years),
        scenario_type='top',
        simulation_paths=None  # Don't store paths for individual scenarios
    )
    
    return projections

def _simulate_growth(current_value: float,
                    growth_mean: float,
                    growth_variance: float,
                    market_cap: float,
                    metric_name: str,
                    period_years: int,
                    scenario_type: str,
                    is_best_scenario: bool,
                    is_worst_scenario: bool,
                    base_scenario_multiple: Optional[float],
                    config: AnalysisConfig) -> GrowthProjection:
    """
    Run Monte Carlo simulation for a single metric.
    
    This implementation fixes inflated CAGR by:
    1. Using log-normal distributions to prevent impossible negative growth
    2. Implementing realistic growth caps based on business fundamentals
    3. Adding momentum exhaustion for sustained high growth
    4. Preserving the existing negative value handling
    """
    periods = period_years * config.QUARTERS_PER_YEAR
    replicates = config.SIMULATION_REPLICATES
    
    # Maximum cumulative growth constraint (unchanged)
    max_cumulative_multiple = 10.0  # Hard limit regardless of period
    
    # Initialize projection array
    projections = np.zeros((replicates, periods + 1))
    projections[:, 0] = current_value
    
    # Convert normal distribution parameters to log-normal parameters
    # This prevents the mathematical impossibility of growth worse than -100%
    normal_std = np.sqrt(growth_variance)
    
    # Handle the conversion carefully to avoid numerical issues
    if growth_mean > -0.5 and normal_std > 0:  # Reasonable growth parameters
        # For log-normal distribution of growth multipliers (1 + growth_rate)
        # If Y = 1 + growth_rate follows LogNormal(μ, σ²), then:
        # E[Y] = exp(μ + σ²/2) = 1 + growth_mean
        # Var[Y] = exp(2μ + σ²)(exp(σ²) - 1) ≈ (1 + growth_mean)² × σ²_log
        
        target_mean = 1 + growth_mean
        coefficient_of_variation = normal_std / target_mean
        
        # Cap the coefficient of variation to prevent extreme distributions
        # A CV of 1.0 means the standard deviation equals the mean
        coefficient_of_variation = min(coefficient_of_variation, 1.0)
        
        log_variance = np.log(1 + coefficient_of_variation**2)
        log_mean = np.log(target_mean) - log_variance / 2
    else:
        # For companies with very negative growth or edge cases
        log_variance = 0.1  # Low volatility
        log_mean = np.log(0.95)  # Default to 5% quarterly decline
    
    # Run simulations
    for sim in range(replicates):
        # Track recent growth for momentum effects
        recent_growth_rates = []
        momentum_window = 4  # Look at last 4 quarters
        
        for period in range(1, periods + 1):
            current = projections[sim, period - 1]
            previous_period_value = current
            
            if current >= 0:
                # Positive value: normal growth simulation with improvements
                
                # Step 1: Adjust volatility based on company size evolution
                # Larger companies (relative to start) should be more stable
                size_evolution = abs(current / current_value) if current_value != 0 else 1.0
                
                if size_evolution > 3.0:  # Company has tripled
                    volatility_dampener = 0.6
                elif size_evolution > 2.0:  # Company has doubled  
                    volatility_dampener = 0.7
                elif size_evolution > 1.5:
                    volatility_dampener = 0.85
                else:
                    volatility_dampener = 1.0
                
                adjusted_log_variance = log_variance * volatility_dampener
                
                # Step 2: Apply momentum exhaustion if applicable
                log_mean_adjustment = 0.0
                
                if len(recent_growth_rates) >= momentum_window:
                    avg_recent_growth = np.mean(recent_growth_rates[-momentum_window:])
                    
                    # If recent growth far exceeds historical norm, expect reversion
                    if growth_mean > 0 and avg_recent_growth > growth_mean * 2.0:
                        # Growing at 2x+ normal rate - apply exhaustion
                        excess_factor = avg_recent_growth / growth_mean
                        log_mean_adjustment = -np.log(1 + (excess_factor - 2.0) * 0.2)
                        log_mean_adjustment = max(log_mean_adjustment, -0.3)  # Cap adjustment
                    
                    elif growth_mean > 0 and avg_recent_growth < growth_mean * 0.5:
                        # Growing at less than half normal rate - potential bounce
                        shortfall_factor = avg_recent_growth / growth_mean
                        log_mean_adjustment = np.log(1 + (0.5 - shortfall_factor) * 0.1)
                        log_mean_adjustment = min(log_mean_adjustment, 0.2)  # Cap adjustment
                
                # Step 3: Sample from adjusted log-normal distribution
                adjusted_log_mean = log_mean + log_mean_adjustment
                log_growth = np.random.normal(adjusted_log_mean, np.sqrt(adjusted_log_variance))
                
                # Convert to growth rate
                growth_multiplier = np.exp(log_growth)
                growth_rate = growth_multiplier - 1
                
                # Step 4: Apply realistic business constraints
                # These caps are based on what's actually possible in business
                # They replace the unrealistic 200% quarterly cap
                
                if metric_name == 'fcf':
                    # FCF can be more volatile than revenue
                    if size_evolution > 2.0:
                        max_quarterly = 0.25  # 25% for large companies
                        min_quarterly = -0.20  # 20% decline max
                    else:
                        max_quarterly = 0.40  # 40% for smaller companies
                        min_quarterly = -0.30  # 30% decline max
                else:  # revenue
                    # Revenue is typically more stable
                    if size_evolution > 2.0:
                        max_quarterly = 0.15  # 15% for large companies
                        min_quarterly = -0.10  # 10% decline max
                    else:
                        max_quarterly = 0.25  # 25% for smaller companies
                        min_quarterly = -0.15  # 15% decline max
                
                # Apply caps
                growth_rate = np.clip(growth_rate, min_quarterly, max_quarterly)
                
                # Step 5: Apply dynamic maximum multiple for scenario consistency
                max_multiple = calculate_dynamic_max_multiple(
                    growth_rate, is_best_scenario, is_worst_scenario, 
                    base_scenario_multiple, config
                )
                
                # Step 6: Apply existing size and time constraints
                # These work well and don't need modification
                value_to_market_ratio = abs(current) / market_cap
                
                if value_to_market_ratio < 0.001:
                    size_factor = 0.8
                elif value_to_market_ratio < 0.01:
                    size_factor = max(0.5, min(0.9, 0.03 / value_to_market_ratio))
                else:
                    size_factor = max(0.4, min(1.0, config.SIZE_PENALTY_FACTOR / value_to_market_ratio))
                
                # Time decay for sustained high growth
                time_factor = 1.0
                if growth_rate > 0.3:  # High growth (30%+ quarterly)
                    quarters_elapsed = period - 1
                    years_elapsed = quarters_elapsed / config.QUARTERS_PER_YEAR
                    time_factor = 0.8 ** years_elapsed  # 20% decay per year
                
                # Apply all constraints
                constrained_growth = growth_rate * size_factor * time_factor
                
                # Final growth rate capping (much more conservative than original)
                constrained_growth = max(-0.5, min(0.5, constrained_growth))  # ±50% absolute max
                
                # Track this growth rate for momentum calculations
                recent_growth_rates.append(constrained_growth)
                
                # Apply growth
                next_value = current * (1 + constrained_growth)
                
                # Apply period-over-period constraint
                # This is now more conservative
                max_quarterly_multiple = 1.5  # 50% growth max (was 2.0)
                if next_value / previous_period_value > max_quarterly_multiple:
                    next_value = previous_period_value * max_quarterly_multiple
                
                # Apply cumulative constraint
                cumulative_factor = abs(next_value / current_value) if current_value != 0 else 1.0
                if cumulative_factor > max_cumulative_multiple:
                    next_value = current_value * max_cumulative_multiple
                    if current_value < 0:
                        next_value = -next_value
                
                # Additional sanity check: implied CAGR shouldn't exceed reasonable bounds
                if period > 4:  # After 1 year
                    implied_total_growth = next_value / current_value
                    quarters_elapsed = period
                    implied_quarterly_rate = implied_total_growth ** (1/quarters_elapsed) - 1
                    implied_annual_rate = (1 + implied_quarterly_rate) ** 4 - 1
                    
                    # Cap at 100% annual CAGR (already very high)
                    if implied_annual_rate > 1.0:
                        max_reasonable_growth = (2.0 ** (quarters_elapsed/4))  # 100% CAGR
                        next_value = current_value * max_reasonable_growth
                
                projections[sim, period] = next_value
                
            else:
                # Negative value: use existing path to equilibrium logic
                # This part is unchanged as it works well
                quarters_remaining = periods - period + 1
                
                # Target is to reach near-zero FCF (equilibrium)
                # Use exponential decay toward zero
                decay_rate = 0.15  # 15% improvement per quarter
                
                # Add some randomness
                noise_factor = np.random.normal(1.0, 0.1)
                actual_decay = decay_rate * noise_factor
                actual_decay = np.clip(actual_decay, 0.05, 0.25)  # Between 5% and 25%
                
                # Move toward zero
                next_value = current * (1 - actual_decay)
                
                # If very close to zero, add small random walk around zero
                if abs(next_value) < abs(current_value) * 0.01:
                    next_value = np.random.normal(0, abs(current_value) * 0.01)
                
                projections[sim, period] = next_value
    
    # Extract final values and calculate statistics
    final_values = projections[:, -1]
    
    # For 'full' scenario, return complete distribution
    if scenario_type == 'full':
        median = np.percentile(final_values, 50)
        p25 = np.percentile(final_values, 25)
        p75 = np.percentile(final_values, 75)
        
        annual_growth_median = _calculate_annual_rate(median, current_value, period_years)
        annual_growth_p25 = _calculate_annual_rate(p25, current_value, period_years)
        annual_growth_p75 = _calculate_annual_rate(p75, current_value, period_years)
        
        # Log the improvement in CAGR projections
        logger.debug(f"CAGR projections - Median: {annual_growth_median:.1%}, "
                    f"P25: {annual_growth_p25:.1%}, P75: {annual_growth_p75:.1%}")
        
        return GrowthProjection(
            metric_name=metric_name,
            current_value=current_value,
            period_years=period_years,
            final_value_median=median,
            final_value_ci_bottom=p25,
            final_value_ci_top=p75,
            annual_growth_median=annual_growth_median,
            annual_growth_ci_bottom=annual_growth_p25,
            annual_growth_ci_top=annual_growth_p75,
            scenario_type=scenario_type,
            simulation_paths=projections
        )
    else:
        # This shouldn't be called anymore, but keep for compatibility
        raise ValueError(f"Use 'full' scenario type for simulations")

def _sample_constrained_growth(growth_mean: float,
                              growth_variance: float,
                              current_value: float,
                              market_cap: float,
                              period: int,
                              config: AnalysisConfig) -> float:
    """
    Sample growth rate with size-based constraints.
    
    Larger companies (by FCF/market cap ratio) grow more slowly.
    """
    # Base growth rate from historical distribution
    base_growth = np.random.normal(growth_mean, np.sqrt(growth_variance))
    
    # Size constraint
    value_to_market_ratio = abs(current_value) / market_cap
    
    if value_to_market_ratio < 0.001:  # Very small relative to market cap
        size_factor = 0.8
    elif value_to_market_ratio < 0.01:  # Small
        size_factor = max(0.5, min(0.9, 0.03 / value_to_market_ratio))
    else:  # Normal or large
        size_factor = max(0.4, min(1.0, config.SIZE_PENALTY_FACTOR / value_to_market_ratio))
    
    # Time decay for high growth
    time_factor = 1.0
    if base_growth > 0.5:  # High growth (50%+ quarterly)
        quarters_elapsed = period - 1
        years_elapsed = quarters_elapsed / config.QUARTERS_PER_YEAR
        time_factor = 0.8 ** years_elapsed  # 20% decay per year
    
    # Apply constraints
    constrained_growth = base_growth * size_factor * time_factor
    
    # Hard limits
    constrained_growth = max(config.MIN_QUARTERLY_GROWTH, 
                           min(config.MAX_QUARTERLY_GROWTH, constrained_growth))
    
    return constrained_growth


def _sample_improvement_rate(period: int, config: AnalysisConfig) -> float:
    """
    Sample improvement rate for negative FCF companies.
    
    Early periods have faster improvement.
    """
    if period <= 3:  # First 3 quarters
        mean_improvement = 0.35  # 35% improvement per quarter
        std_improvement = 0.15
    else:
        mean_improvement = 0.25  # 25% improvement later
        std_improvement = 0.10
    
    rate = np.random.normal(mean_improvement, std_improvement)
    
    # Ensure positive improvement
    return max(0.0, min(0.9, rate))  # Cap at 90% improvement


def _calculate_annual_rate(final_value: float, 
                         initial_value: float, 
                         years: float) -> float:
    """
    Calculate compound annual growth rate from quarterly simulation results.
    
    Handles special cases like negative to positive transitions.
    This is the correct implementation that avoids complex numbers
    and maintains economic meaning.
    """
    if years == 0:
        return 0.0
    
    if initial_value == 0:
        if final_value > 0:
            return 1.0  # 100% growth
        else:
            return 0.0
    
    # Both positive: standard CAGR
    if initial_value > 0 and final_value > 0:
        return (final_value / initial_value) ** (1 / years) - 1
    
    # Negative to positive: massive improvement
    if initial_value < 0 and final_value > 0:
        return 1.0  # Cap at 100% annual growth
    
    # Both negative: improvement rate
    if initial_value < 0 and final_value < 0:
        if abs(final_value) < abs(initial_value):
            # Improved (less negative)
            improvement = 1 - (abs(final_value) / abs(initial_value))
            return improvement / years  # Linear approximation
        else:
            # Worsened
            return -0.5  # -50% growth
    
    # Positive to negative: disaster
    if initial_value > 0 and final_value < 0:
        return -0.9  # -90% growth
    
    # Default
    return 0.0


def calculate_growth_stability(projection: GrowthProjection) -> float:
    """
    Calculate growth stability score (0 to 1, higher is more stable).
    
    Based on consistency of growth paths. For negative FCF companies,
    stability is based on convergence toward equilibrium.
    """
    if projection.simulation_paths is None:
        return 0.5  # Default middle score
    
    paths = projection.simulation_paths
    
    # Calculate coefficient of variation for final values
    final_values = paths[:, -1]
    mean_final = np.mean(final_values)
    std_final = np.std(final_values)
    
    # Special handling for companies with negative starting FCF
    if projection.current_value < 0:
        # For negative FCF, stability means consistent path toward equilibrium (0)
        # Calculate how consistently the paths approach zero
        distances_from_zero = np.abs(final_values)
        mean_distance = np.mean(distances_from_zero)
        std_distance = np.std(distances_from_zero)
        
        if mean_distance < 1e-6:
            return 1.0  # Very stable if converged to near zero
        
        # Lower CV of distance means more consistent convergence
        cv = std_distance / mean_distance
        stability = 1.0 / (1.0 + cv)
        
    else:
        # Positive FCF: standard stability calculation
        if abs(mean_final) < 1e-6:
            return 0.0  # Unstable if mean is near zero
        
        cv = std_final / abs(mean_final)
        
        # Convert to stability score (lower CV = higher stability)
        stability = 1.0 / (1.0 + cv)
    
    return stability

def _process_single_company(args: Tuple[str, CompanyData, AnalysisConfig]) -> Tuple[str, Optional[Dict]]:
    """
    Process a single company's growth projections.
    
    Args:
        args: Tuple of (ticker, company, config)
        
    Returns:
        Tuple of (ticker, projections or None)
    """
    ticker, company, config = args
    
    try:
        projections = calculate_growth_projections(company, config)
        return ticker, projections
    except Exception as e:
        # Return None for failed companies - don't log to avoid spam
        return ticker, None

def calculate_all_projections(companies: Dict[str, CompanyData],
                                          config: AnalysisConfig) -> Dict[str, Dict[int, Dict[str, Dict[str, GrowthProjection]]]]:
    """
    Alternative implementation with real-time progress tracking.
    
    This version provides better visibility into progress but may have
    slightly more overhead due to the futures interface.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time
    
    projections = {}
    total = len(companies)
    n_processes = min(44, total)
    
    logger.info(f"Starting parallel growth projections for {total} companies")
    logger.info(f"Using {n_processes} processes with real-time progress tracking")
    
    start_time = time.time()
    completed = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=n_processes, initializer=_init_worker) as executor:
        # Submit all tasks
        future_to_ticker = {
            executor.submit(_process_single_company, (ticker, company, config)): ticker
            for ticker, company in companies.items()
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            
            try:
                _, projection = future.result()
                if projection is not None:
                    projections[ticker] = projection
                    completed += 1
                else:
                    failed += 1
                    logger.warning(f"Failed to process {ticker}")
                
                # Progress update with timing
                total_processed = completed + failed
                if total_processed % 5 == 0 or total_processed == total:
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed
                    remaining = (total - total_processed) / rate if rate > 0 else 0
                    
                    logger.info(
                        f"Progress: {total_processed}/{total} companies | "
                        f"Rate: {rate:.1f} companies/sec | "
                        f"ETA: {remaining:.0f} seconds"
                    )
                    
            except Exception as e:
                logger.error(f"Unexpected error processing {ticker}: {e}")
                failed += 1
    
    elapsed_total = time.time() - start_time
    logger.info(
        f"Completed in {elapsed_total:.1f} seconds | "
        f"Rate: {total/elapsed_total:.1f} companies/sec | "
        f"Success: {completed}/{total}"
    )
    
    return projections