"""
Intrinsic value calculation using Discounted Cash Flow (DCF) model.

IMPORTANT: DCF calculations are performed ONLY on Free Cash Flow (FCF) as this represents
actual cash available to investors. Revenue is tracked separately for growth analysis
but is NOT used for valuation as it doesn't represent cash flows.

Calculates valuations for multiple scenarios and time periods using FCF projections only.
All calculations use quarterly data internally, converting to annual rates only for reporting.
"""
import logging
from typing import Dict, List, Tuple
import numpy as np

from data_pipeline.data_structures import CompanyData, GrowthProjection, IntrinsicValue
from config.user_config import AnalysisConfig

logger = logging.getLogger(__name__)


def calculate_intrinsic_values(company: CompanyData,
                             growth_projections: Dict[int, Dict[str, Dict[str, GrowthProjection]]],
                             config: AnalysisConfig) -> Dict[int, Dict[Tuple[str, str], IntrinsicValue]]:
    """
    Calculate intrinsic values using FCF-based DCF only.
    
    Revenue growth is tracked in projections but NOT used for DCF valuation
    as revenue doesn't represent actual cash flows. This provides a more
    theoretically sound valuation approach.
    
    Calculates 3 FCF-based values per period:
    - FCF median scenario
    - FCF bottom scenario (conservative)
    - FCF top scenario (optimistic)
    
    Args:
        company: Company data
        growth_projections: Growth projections by period/metric/scenario
        config: Analysis configuration
        
    Returns:
        Dictionary mapping period -> (metric, scenario) -> IntrinsicValue
        Note: metric will always be 'fcf' as we only calculate FCF-based valuations
    """
    all_intrinsic_values = {}
    
    for period_years in config.PROJECTION_PERIODS:
        period_projections = growth_projections[period_years]
        intrinsic_values = {}
        
        # FCF-based valuations ONLY - this is the theoretically correct approach
        # We calculate three scenarios to capture uncertainty
        for scenario in ['median', 'bottom', 'top']:
            fcf_proj = period_projections['fcf'][scenario]
            iv = _calculate_dcf(
                company=company,
                projection=fcf_proj,
                scenario=scenario,
                metric_name='fcf',
                period_years=period_years,
                config=config
            )
            intrinsic_values[('fcf', scenario)] = iv
            
            logger.debug(f"{company.ticker} - {period_years}yr FCF {scenario}: "
                        f"IV=${iv.intrinsic_value_per_share:.2f}")
        
        # Note: We do NOT calculate revenue-based or average intrinsic values
        # Revenue growth is tracked separately for business quality analysis
        # but should never be used in DCF calculations
        
        all_intrinsic_values[period_years] = intrinsic_values
    
    return all_intrinsic_values


def _calculate_dcf(company: CompanyData,
                  projection: GrowthProjection,
                  scenario: str,
                  metric_name: str,
                  period_years: int,
                  config: AnalysisConfig) -> IntrinsicValue:
    """Calculate DCF for a specific FCF projection and scenario."""
    growth_rate = projection.get_scenario_growth(scenario)
    base_value = projection.current_value
    
    return _calculate_dcf_with_growth_rate(
        company=company,
        base_value=base_value,
        growth_rate=growth_rate,
        scenario=scenario,
        metric_name=metric_name,
        period_years=period_years,
        config=config
    )


def _calculate_dcf_with_growth_rate(company: CompanyData,
                                   base_value: float,
                                   growth_rate: float,
                                   scenario: str,
                                   metric_name: str,
                                   period_years: int,
                                   config: AnalysisConfig) -> IntrinsicValue:
    """
    Calculate DCF valuation with given growth rate.
    
    Uses quarterly projections and quarterly discounting throughout.
    Only converts to annual for reporting purposes.
    This function should only be called with FCF data.
    """
    # Convert annual growth rate to quarterly
    # Growth rate from projection is already annual, so convert to quarterly
    quarterly_growth_rate = (1 + growth_rate) ** (1/config.QUARTERS_PER_YEAR) - 1
    
    # Convert annual discount and terminal rates to quarterly
    quarterly_discount_rate = (1 + config.DISCOUNT_RATE) ** (1/config.QUARTERS_PER_YEAR) - 1
    quarterly_terminal_rate = (1 + config.TERMINAL_GROWTH_RATE) ** (1/config.QUARTERS_PER_YEAR) - 1
    
    # Total number of quarters
    total_quarters = period_years * config.QUARTERS_PER_YEAR
    
    # Project quarterly cash flows
    quarterly_cash_flows = []
    current = base_value
    
    for quarter in range(1, total_quarters + 1):
        if current >= 0:
            # Normal growth
            current = current * (1 + quarterly_growth_rate)
        else:
            # Negative FCF: apply improvement logic
            # This models path to profitability for companies with negative FCF
            if growth_rate > 0:
                # Improving (moving toward positive)
                improvement_rate = min(0.25, abs(growth_rate) / 4)  # Quarterly improvement
                current = current * (1 - improvement_rate)
            else:
                # Getting worse
                current = current * (1 + quarterly_growth_rate)
        
        quarterly_cash_flows.append(current)
    
    # Calculate terminal value based on final quarterly cash flow
    final_quarterly_cf = quarterly_cash_flows[-1]
    
    # CRITICAL FIX: Convert quarterly cash flow to annual before terminal value calculation
    # Terminal value represents the present value of all ANNUAL cash flows beyond the projection period
    final_annual_cf = final_quarterly_cf * config.QUARTERS_PER_YEAR  # Multiply by 4 to annualize
    
    # Now calculate terminal value using annual values and annual rates
    terminal_cf = final_annual_cf * (1 + config.TERMINAL_GROWTH_RATE)
    terminal_value = terminal_cf / (config.DISCOUNT_RATE - config.TERMINAL_GROWTH_RATE)
    
    # Discount all quarterly cash flows to present value
    present_value = 0
    for quarter, cf in enumerate(quarterly_cash_flows, 1):
        discount_factor = (1 + quarterly_discount_rate) ** quarter
        present_value += cf / discount_factor
    
    # Add discounted terminal value
    # Note: We need to discount the terminal value from the END of the projection period
    # Since terminal value is in annual terms, we use the annual discount rate raised to the number of years
    terminal_discount = (1 + config.DISCOUNT_RATE) ** period_years
    present_value += terminal_value / terminal_discount
    
    # Apply margin of safety for conservative valuation
    intrinsic_value = present_value * (1 - config.MARGIN_OF_SAFETY)
    
    # Per share value
    intrinsic_value_per_share = intrinsic_value / company.shares_diluted
    
    # Convert quarterly cash flows to annual for reporting only
    # Sum the 4 quarters for each year to get annual cash flows
    annual_cash_flows = []
    for year in range(period_years):
        start_quarter = year * config.QUARTERS_PER_YEAR
        end_quarter = start_quarter + config.QUARTERS_PER_YEAR
        # Sum quarters for this year
        annual_cf = sum(quarterly_cash_flows[start_quarter:end_quarter])
        annual_cash_flows.append(annual_cf)
    
    return IntrinsicValue(
        metric_name=metric_name,
        scenario=scenario,
        period_years=period_years,
        projected_cash_flows=annual_cash_flows,  # Annual values for reporting
        terminal_value=terminal_value,
        present_value=present_value,
        intrinsic_value_per_share=intrinsic_value_per_share,
        growth_rate=growth_rate,  # This is already annual
        discount_rate=config.DISCOUNT_RATE,
        terminal_growth_rate=config.TERMINAL_GROWTH_RATE,
        margin_of_safety=config.MARGIN_OF_SAFETY
    )


def calculate_all_intrinsic_values(companies: Dict[str, CompanyData],
                                 projections: Dict[str, Dict[int, Dict[str, Dict[str, GrowthProjection]]]],
                                 config: AnalysisConfig) -> Dict[str, Dict[int, Dict[Tuple[str, str], IntrinsicValue]]]:
    """
    Calculate FCF-based intrinsic values for all companies.
    
    Note: This function now only calculates FCF-based valuations.
    Revenue projections are available in the projections dict but are
    not used for valuation purposes.
    
    Args:
        companies: Dictionary of CompanyData objects
        projections: Dictionary of growth projections (includes both FCF and revenue)
        config: Analysis configuration
        
    Returns:
        Dictionary mapping ticker to intrinsic values by period
        Each company will have FCF-based valuations only
    """
    all_values = {}
    
    total = len(companies)
    for i, ticker in enumerate(companies.keys()):
        if i % 10 == 0:
            logger.info(f"Calculating intrinsic values: {i}/{total} companies")
        
        company = companies[ticker]
        company_projections = projections[ticker]
        
        intrinsic_values = calculate_intrinsic_values(
            company=company,
            growth_projections=company_projections,
            config=config
        )
        
        all_values[ticker] = intrinsic_values
    
    logger.info(f"Completed FCF-based intrinsic value calculations for {total} companies")
    
    return all_values