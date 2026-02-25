"""
Company filtering based on financial health criteria.

Supports both shortlist and owned company analysis modes.
"""
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd

from data_pipeline.data_structures import CompanyData
from config.user_config import AnalysisConfig

logger = logging.getLogger(__name__)


class FilterResults:
    """Track which companies are removed by each filter."""
    
    def __init__(self):
        self.removed_by_filter: Dict[str, List[str]] = {}
        self.removal_reasons: Dict[str, str] = {}
        self.owned_companies_bypassed: List[str] = []
        self.owned_companies_tracking: Dict[str, Dict] = {}
    
    def add_removal(self, filter_name: str, ticker: str, reason: str):
        """Record a company removal."""
        if filter_name not in self.removed_by_filter:
            self.removed_by_filter[filter_name] = []
        
        self.removed_by_filter[filter_name].append(ticker)
        self.removal_reasons[ticker] = reason
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary of removals by filter."""
        return {
            filter_name: len(companies)
            for filter_name, companies in self.removed_by_filter.items()
        }


def apply_financial_filters(companies: Dict[str, CompanyData], 
                          config: AnalysisConfig) -> Tuple[Dict[str, CompanyData], FilterResults]:
    """
    Apply financial health filters to companies.
    
    In 'owned' mode, filters are applied but owned companies are re-added.
    
    Filters applied:
    1. Data consistency (operating income < revenue)
    2. Minimum market cap
    3. Maximum debt-to-cash ratio
    
    Note: Operating income filter has been removed per requirements.
    Enterprise value consistency filter has been removed per requirements.
    
    Args:
        companies: Dictionary of CompanyData objects
        config: Analysis configuration
        
    Returns:
        Tuple of (filtered companies dict, filter results)
    """
    results = FilterResults()
    remaining = companies.copy()
    
    initial_count = len(companies)
    logger.info(f"Applying filters to {initial_count} companies")
    logger.info(f"Analysis mode: {config.ANALYSIS_MODE}")
    
    # Track owned companies if in owned mode
    if config.ANALYSIS_MODE == 'owned' and config.OWNED_COMPANIES:
        logger.info(f"Tracking {len(config.OWNED_COMPANIES)} owned companies")
        for ticker in config.OWNED_COMPANIES:
            if ticker in companies:
                results.owned_companies_tracking[ticker] = {
                    'in_initial_data': True,
                    'filters_passed': [],
                    'filters_failed': []
                }
            else:
                logger.warning(f"Owned company {ticker} not found in initial data")
                results.owned_companies_tracking[ticker] = {
                    'in_initial_data': False,
                    'filters_passed': [],
                    'filters_failed': []
                }
    
    # FILTER: Negative FCF
    # Companies burning cash are not suitable for value investing
    to_remove = []
    for ticker, company in remaining.items():
        if company.fcf <= 0:
            to_remove.append(ticker)
            reason = f"Negative FCF (${company.fcf:,.0f}) - cash burning company"
            results.add_removal('negative_fcf', ticker, reason)
            
            # Track for owned companies
            if ticker in results.owned_companies_tracking:
                results.owned_companies_tracking[ticker]['filters_failed'].append({
                    'filter': 'negative_fcf',
                    'reason': reason
                })

    # Remove companies with negative FCF
    for ticker in to_remove:
        del remaining[ticker]

    logger.info(f"Negative FCF filter: removed {len(to_remove)} companies")

    # FILTER 1: Financial Data Consistency
    # Check that operating income doesn't exceed revenue (indicates data quality issues)
    to_remove = []
    for ticker, company in remaining.items():
        # All CompanyData objects have revenue field, so just check the value
        if company.revenue > 0 and company.operating_income > company.revenue:
            to_remove.append(ticker)
            reason = f"Operating income (${company.operating_income:,.0f}) exceeds revenue (${company.revenue:,.0f}) - data anomaly"
            results.add_removal('financial_consistency', ticker, reason)
            
            # Track for owned companies
            if ticker in results.owned_companies_tracking:
                results.owned_companies_tracking[ticker]['filters_failed'].append({
                    'filter': 'financial_consistency',
                    'reason': reason
                })
    
    # Remove companies with data anomalies
    for ticker in to_remove:
        del remaining[ticker]
    
    logger.info(f"Financial consistency filter: removed {len(to_remove)} companies with data anomalies")
    
    # Apply remaining financial filters
    # Note: Operating income filter has been removed from this list
    filters_to_apply = [
        ('market_cap', config.MIN_MARKET_CAP, 'ge'),            # greater equal
        ('debt_cash_ratio', config.MAX_DEBT_TO_CASH_RATIO, 'le')  # less equal
    ]
    
    for filter_name, threshold, comparison in filters_to_apply:
        to_remove = []
        
        for ticker, company in remaining.items():
            # Get the value to check
            if filter_name == 'debt_cash_ratio':
                if company.cash_and_equiv > 0:
                    value = company.lt_debt / company.cash_and_equiv
                else:
                    value = float('inf')  # No cash = infinite ratio
            else:
                value = getattr(company, filter_name)
            
            # Apply comparison
            passed = False
            reason = None
            
            if comparison == 'ge' and value < threshold:
                passed = False
                if filter_name == 'market_cap':
                    reason = f"{filter_name} ${value:,.0f} < ${threshold:,.0f}"
                else:
                    reason = f"{filter_name} {value:.2f} < {threshold}"
            elif comparison == 'le' and value > threshold:
                passed = False
                reason = f"{filter_name} {value:.2f} > {threshold}"
            else:
                passed = True
            
            # Track for owned companies
            if ticker in results.owned_companies_tracking:
                if passed:
                    results.owned_companies_tracking[ticker]['filters_passed'].append(filter_name)
                else:
                    results.owned_companies_tracking[ticker]['filters_failed'].append({
                        'filter': filter_name,
                        'reason': reason
                    })
            
            # Remove if failed (unless owned mode will restore it)
            if not passed:
                to_remove.append(ticker)
                results.add_removal(filter_name, ticker, reason)
        
        # Remove companies that failed
        for ticker in to_remove:
            del remaining[ticker]
        
        logger.info(f"{filter_name} filter: removed {len(to_remove)} companies")
    
    # In owned mode, re-add filtered out owned companies
    if config.ANALYSIS_MODE == 'owned' and config.OWNED_COMPANIES:
        filtered_out_owned = []
        for ticker in config.OWNED_COMPANIES:
            if ticker in companies and ticker not in remaining:
                filtered_out_owned.append(ticker)
                remaining[ticker] = companies[ticker]
                results.owned_companies_bypassed.append(ticker)
        
        if filtered_out_owned:
            logger.info(f"Re-added {len(filtered_out_owned)} owned companies that were filtered out:")
            for ticker in filtered_out_owned:
                failed_filters = results.owned_companies_tracking[ticker]['filters_failed']
                logger.info(f"  {ticker}: bypassed {len(failed_filters)} filters")
    
    # Summary
    final_count = len(remaining)
    logger.info(f"Filtering complete: {initial_count} → {final_count} companies ({final_count/initial_count:.1%})")
    
    return remaining, results


def apply_growth_filters(companies: Dict[str, CompanyData],
                        config: AnalysisConfig,
                        results: FilterResults) -> Dict[str, CompanyData]:
    """
    Apply growth-based filters.
    
    Note: Growth volatility filter has been removed per requirements.
    This function now serves as a pass-through but is kept for pipeline consistency.
    
    Args:
        companies: Dictionary of CompanyData objects
        config: Analysis configuration
        results: FilterResults object to update
        
    Returns:
        Filtered companies dictionary (currently unchanged)
    """
    # Growth volatility filter has been removed
    # This function is kept as a placeholder for potential future growth filters
    logger.info("Growth filter stage: No filters applied (volatility filter removed)")
    
    return companies


def create_filtered_dataframe(companies: Dict[str, CompanyData]) -> pd.DataFrame:
    """
    Create DataFrame from filtered companies for further analysis.
    
    Args:
        companies: Dictionary of CompanyData objects
        
    Returns:
        DataFrame with all company data
    """
    rows = []
    
    for ticker, company in companies.items():
        row = {
            'ticker': ticker,
            'country': company.country,
            'fcf': company.fcf,
            'revenue': company.revenue,
            'operating_income': company.operating_income,
            'market_cap': company.market_cap,
            'lt_debt': company.lt_debt,
            'cash_and_equiv': company.cash_and_equiv,
            'shares_diluted': company.shares_diluted,
            'enterprise_value': company.enterprise_value,
            'fcf_growth_mean': company.fcf_growth_mean,
            'fcf_growth_variance': company.fcf_growth_variance,
            'revenue_growth_mean': company.revenue_growth_mean,
            'revenue_growth_variance': company.revenue_growth_variance,
            'average_growth_mean': company.average_growth_mean,
            'average_growth_variance': company.average_growth_variance,
            'debt_cash_ratio': company.lt_debt / company.cash_and_equiv if company.cash_and_equiv > 0 else float('inf'),
            'fcf_per_share': company.fcf / company.shares_diluted,
            'acquirers_multiple': company.enterprise_value / company.operating_income if company.operating_income != 0 else float('inf'),
            'fcf_to_market_cap': company.fcf / company.market_cap
        }
        rows.append(row)
    
    df = pd.DataFrame(rows).set_index('ticker')
    
    # Sort by market cap descending
    df = df.sort_values('market_cap', ascending=False)
    
    return df