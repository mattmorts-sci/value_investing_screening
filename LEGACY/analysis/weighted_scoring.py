"""
Weighted scoring system for company ranking.

Implements complex ranking using debt-cash, market cap, and growth penalties.

The core philosophy here is that we want to identify companies that have:
1. Reasonable debt levels (not overleveraged)
2. Sufficient size for stability but not so large that growth is difficult
3. Strong and consistent growth patterns

We use a penalty-based system where lower penalties indicate better companies.
"""
import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd

from data_pipeline.data_structures import CompanyData, CompanyAnalysis
from config.user_config import AnalysisConfig

logger = logging.getLogger(__name__)


def calculate_weighted_scores(analyses: Dict[str, CompanyAnalysis], 
                            config: AnalysisConfig) -> pd.DataFrame:
    """
    Calculate weighted scores using DC, MC, and Growth penalties.
    
    This is the heart of our ranking system. We calculate three types of penalties:
    
    1. Debt-Cash (DC) Penalty: Companies with high debt relative to cash are riskier
    2. Market Cap (MC) Penalty: Very large companies have limited growth potential
    3. Growth Penalty: Companies with low or inconsistent growth are less attractive
    
    Lower total penalty is better - think of it as golf scoring.
    
    Args:
        analyses: Dictionary of company analyses
        config: Analysis configuration
        
    Returns:
        DataFrame with penalty scores for each company
    """
    rows = []
    companies_with_issues = []
    
    for ticker, analysis in analyses.items():
        company = analysis.company_data
        
        logger.debug(f"Calculating weighted scores for {ticker}")
        
        # Calculate debt-cash penalty
        # The idea here is that companies with more debt than cash are riskier
        # We square the ratio to heavily penalize high debt levels
        if company.cash_and_equiv > 0:
            dc_ratio = company.lt_debt / company.cash_and_equiv
        else:
            dc_ratio = float('inf')  # Infinite penalty for no cash
            companies_with_issues.append((ticker, "No cash on hand"))
        
        dc_penalty = (abs(dc_ratio) ** 2) * config.DC_WEIGHT
        
        # Calculate market cap penalty
        # Larger companies grow more slowly, so we penalize size logarithmically
        # This reflects the reality that doubling from $1B to $2B is easier than
        # doubling from $100B to $200B
        if company.market_cap > config.MIN_MARKET_CAP:
            mc_penalty = (np.log10(company.market_cap / config.MIN_MARKET_CAP) * config.MC_WEIGHT)
        else:
            mc_penalty = 0.0
            companies_with_issues.append((ticker, f"Market cap below minimum: ${company.market_cap:,.0f}"))
        
        # Calculate growth penalty
        # This is more complex as it considers multiple aspects of growth
        growth_penalty = _calculate_growth_penalty(analysis, config)
        
        # Total penalty (lower is better)
        total_penalty = dc_penalty + mc_penalty + growth_penalty
        
        # Store in analysis object for later use
        analysis.dc_penalty = dc_penalty
        analysis.mc_penalty = mc_penalty
        analysis.growth_penalty = growth_penalty
        analysis.total_penalty = total_penalty
        
        rows.append({
            'ticker': ticker,
            'dc_penalty': dc_penalty,
            'mc_penalty': mc_penalty,
            'growth_penalty': growth_penalty,
            'total_penalty': total_penalty,
            'debt_cash_ratio': dc_ratio,
            'market_cap': company.market_cap,
            'fcf': company.fcf,
            'revenue': company.revenue
        })
    
    if companies_with_issues:
        logger.warning(f"Found {len(companies_with_issues)} companies with scoring issues")
        for ticker, issue in companies_with_issues[:5]:  # Log first 5
            logger.warning(f"  {ticker}: {issue}")
    
    df = pd.DataFrame(rows).set_index('ticker')
    
    # Add rank columns
    df['weighted_rank'] = df['total_penalty'].rank()  # Lower penalty = better rank
    
    logger.info(f"Calculated weighted scores for {len(df)} companies")
    logger.info(f"Penalty ranges - DC: [{df['dc_penalty'].min():.2f}, {df['dc_penalty'].max():.2f}], "
                f"MC: [{df['mc_penalty'].min():.2f}, {df['mc_penalty'].max():.2f}], "
                f"Growth: [{df['growth_penalty'].min():.2f}, {df['growth_penalty'].max():.2f}]")
    
    return df.sort_values('weighted_rank')


def _calculate_growth_penalty(analysis: CompanyAnalysis, config: AnalysisConfig) -> float:
    """
    Calculate growth-based penalty component using PROJECTED growth rates.
    
    This penalty is used in the risk calculation but is NOT the primary
    ranking mechanism. In the risk-adjusted approach, this helps determine
    the growth component of risk.
    
    Args:
        analysis: Company analysis object
        config: Analysis configuration
        
    Returns:
        Growth penalty score (used as part of risk calculation)
    """
    company = analysis.company_data
    period = config.PRIMARY_PERIOD
    
    logger.debug(f"Calculating growth penalty for {company.ticker}")
    
    # Try to get projected growth rates
    try:
        fcf_proj = analysis.get_projection(period, 'fcf', 'median')
        rev_proj = analysis.get_projection(period, 'revenue', 'median')
        
        fcf_growth = fcf_proj.annual_growth_median
        rev_growth = rev_proj.annual_growth_median
        
    except (KeyError, AttributeError):
        # Projections not yet available (shouldn't happen in normal flow)
        logger.warning(f"Projections not available for {company.ticker}, using historical")
        fcf_growth = company.historical_fcf_cagr
        rev_growth = company.historical_revenue_cagr
    
    # Rest of the calculation remains the same
    growth_rates = [fcf_growth, rev_growth]
    avg_growth = np.mean(growth_rates)
    
    if avg_growth < config.MIN_ACCEPTABLE_GROWTH:
        rate_penalty = (config.MIN_ACCEPTABLE_GROWTH - avg_growth) * config.GROWTH_WEIGHT * 0.5
        logger.debug(f"  Growth rate penalty: {rate_penalty:.3f} (avg growth {avg_growth:.1%} < {config.MIN_ACCEPTABLE_GROWTH:.1%})")
    else:
        rate_penalty = 0.0
    
    # Stability penalty based on volatility
    growth_volatility = np.std(growth_rates)
    stability_penalty = growth_volatility * config.GROWTH_WEIGHT * 0.3
    
    # Divergence penalty
    divergence = abs(fcf_growth - rev_growth)
    divergence_penalty = divergence * config.GROWTH_WEIGHT * 0.2
    
    total_growth_penalty = rate_penalty + stability_penalty + divergence_penalty
    
    return total_growth_penalty


def create_quadrant_analysis(analyses: Dict[str, CompanyAnalysis],
                           scored_df: pd.DataFrame,
                           config: AnalysisConfig) -> pd.DataFrame:
    """
    Create quadrant analysis based on growth vs value.
    
    This analysis helps visualize companies in a 2x2 matrix:
    
    Quadrant 1 (Best): High Growth + High Value (undervalued growth stocks)
    Quadrant 2: High Growth + Low Value (expensive growth stocks)
    Quadrant 3: Low Growth + High Value (value traps or turnarounds)
    Quadrant 4 (Worst): Low Growth + Low Value (avoid)
    
    The sweet spot is Quadrant 1 - companies growing rapidly that the market
    hasn't fully recognized yet.
    
    Args:
        analyses: Dictionary of company analyses
        scored_df: DataFrame with weighted scores
        config: Analysis configuration
        
    Returns:
        DataFrame with quadrant assignments
    """
    rows = []
    quadrant_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    
    for ticker in scored_df.index:
        if ticker not in analyses:
            logger.warning(f"Company {ticker} in scores but not in analyses")
            continue
            
        analysis = analyses[ticker]
        
        # Get growth metrics - no error hiding
        period = config.PRIMARY_PERIOD
        
        fcf_proj = analysis.get_projection(period, 'fcf', 'median')
        rev_proj = analysis.get_projection(period, 'revenue', 'median')
        
        fcf_growth = fcf_proj.annual_growth_median
        rev_growth = rev_proj.annual_growth_median
        
        # Calculate weighted average growth
        avg_growth = (fcf_growth * config.FCF_GROWTH_WEIGHT + 
                     rev_growth * config.REVENUE_GROWTH_WEIGHT)
        
        # Get value metric (IV/Price ratio)
        iv_ratio = analysis.best_iv_to_price_ratio or 0.0
        
        # Determine thresholds
        growth_threshold = config.MIN_ACCEPTABLE_GROWTH
        value_threshold = config.MIN_IV_TO_PRICE_RATIO
        
        # Assign quadrant based on thresholds
        if avg_growth >= growth_threshold and iv_ratio >= value_threshold:
            quadrant = 1  # High Growth, High Value - BEST
        elif avg_growth >= growth_threshold and iv_ratio < value_threshold:
            quadrant = 2  # High Growth, Low Value
        elif avg_growth < growth_threshold and iv_ratio >= value_threshold:
            quadrant = 3  # Low Growth, High Value
        else:
            quadrant = 4  # Low Growth, Low Value - WORST
        
        quadrant_counts[quadrant] += 1
        
        rows.append({
            'ticker': ticker,
            'avg_growth': avg_growth,
            'iv_ratio': iv_ratio,
            'quadrant': quadrant,
            'quadrant_label': _get_quadrant_label(quadrant)
        })
    
    # Log quadrant distribution
    logger.info("Quadrant distribution:")
    for quad, count in quadrant_counts.items():
        logger.info(f"  Quadrant {quad} ({_get_quadrant_label(quad)}): {count} companies")
    
    return pd.DataFrame(rows).set_index('ticker')


def _get_quadrant_label(quadrant: int) -> str:
    """
    Get descriptive label for quadrant.
    
    These labels help interpret what each quadrant means for investment decisions.
    """
    labels = {
        1: "High Growth, High Value",  # Sweet spot - undervalued growth
        2: "High Growth, Low Value",   # Expensive growth stocks
        3: "Low Growth, High Value",   # Deep value or potential turnarounds
        4: "Low Growth, Low Value"     # Generally avoid
    }
    return labels.get(quadrant, "Unknown")
