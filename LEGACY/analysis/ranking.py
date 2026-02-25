"""
Company ranking using risk-adjusted scoring methodology.

Implements a Sharpe-ratio-inspired approach where companies are ranked by
Expected Return / Risk. This provides a coherent framework that balances
growth potential, valuation opportunity, and various risk factors.

Key principles:
- All rankings use PROJECTED growth rates from Monte Carlo simulations
- Historical CAGR is available for context but NOT used in rankings
- Risk combines financial risk (debt/leverage) and volatility risk
- Expected returns combine growth and valuation upside
"""
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from data_pipeline.data_structures import CompanyAnalysis, GrowthProjection
from analysis.growth_calculation import calculate_growth_stability
from analysis.weighted_scoring import calculate_weighted_scores, create_quadrant_analysis
from analysis.factor_analysis import calculate_factor_contributions
from config.user_config import AnalysisConfig

logger = logging.getLogger(__name__)


def rank_companies(analyses: Dict[str, CompanyAnalysis], 
                  config: AnalysisConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Rank companies using a risk-adjusted approach while maintaining full compatibility.
    
    This function implements a Sharpe-ratio-like ranking system where:
    - Expected Return = Projected Growth + Annualized Valuation Upside  
    - Risk = 70% Volatility Risk + 30% Financial Risk
    - Risk-Adjusted Score = Expected Return / Risk
    
    All original DataFrame structures and columns are maintained for compatibility.
    The key change is that rankings now use projected growth instead of historical.
    
    Args:
        analyses: Dictionary of company analyses
        config: Analysis configuration
        
    Returns:
        Tuple of (growth_rankings, value_rankings, weighted_rankings, combined_rankings)
    """
    logger.info(f"Ranking {len(analyses)} companies using risk-adjusted approach")
    
    # First calculate weighted scores (includes growth penalty based on projections)
    weighted_scores = calculate_weighted_scores(analyses, config)
    
    # Build ranking data
    ranking_data = []
    companies_without_prices = []
    
    for ticker, analysis in analyses.items():
        if analysis.current_price is None:
            companies_without_prices.append(ticker)
            continue
        
        logger.debug(f"Processing rankings for {ticker}")
        
        # Get projections for primary period
        period = config.PRIMARY_PERIOD
        
        # Get projections - let errors propagate
        fcf_proj = analysis.get_projection(period, 'fcf', 'median')
        rev_proj = analysis.get_projection(period, 'revenue', 'median')
        
        # Growth metrics using PROJECTED values
        fcf_growth = fcf_proj.annual_growth_median
        rev_growth = rev_proj.annual_growth_median
        combined_growth = (fcf_growth * config.FCF_GROWTH_WEIGHT + 
                          rev_growth * config.REVENUE_GROWTH_WEIGHT)
        
        # Calculate best IV/price ratio across all periods
        all_ratios = {}
        for p in config.PROJECTION_PERIODS:
            period_ratios = analysis.calculate_iv_to_price_ratios(p)
            all_ratios.update(period_ratios)
        
        if not all_ratios:
            logger.error(f"No IV/price ratios calculated for {ticker}")
            raise ValueError(f"No IV/price ratios for {ticker}")
        
        best_ratio = max(all_ratios.values())
        best_ratio_key = max(all_ratios, key=all_ratios.get)
        analysis.best_iv_to_price_ratio = best_ratio
        
        # Calculate expected return
        valuation_return = best_ratio - 1.0
        annualized_valuation_return = (1 + valuation_return) ** (1/period) - 1
        total_expected_return = combined_growth + annualized_valuation_return
        
        # Calculate risk components
        # 1. Volatility risk from growth variance
        fcf_variance = analysis.company_data.fcf_growth_variance
        rev_variance = analysis.company_data.revenue_growth_variance
        
        # Convert quarterly variance to annual standard deviation
        fcf_std = np.sqrt(fcf_variance) * 2  # Quarterly to annual
        rev_std = np.sqrt(rev_variance) * 2
        
        # Weighted average volatility
        avg_volatility_decimal = (fcf_std * config.FCF_GROWTH_WEIGHT + 
                                  rev_std * config.REVENUE_GROWTH_WEIGHT)
        avg_volatility_pct = avg_volatility_decimal * 100  # As percentage for display
        
        # 2. Financial risk from debt/cash
        if analysis.company_data.cash_and_equiv > 0:
            debt_cash_ratio = analysis.company_data.lt_debt / analysis.company_data.cash_and_equiv
        else:
            debt_cash_ratio = float('inf')
        
        # Normalize to 0-1 scale
        financial_risk = min(debt_cash_ratio / config.MAX_DEBT_TO_CASH_RATIO, 1.0)
        
        # 3. Combined risk (ensure minimum to avoid division by zero)
        combined_risk = max((avg_volatility_decimal * 0.7 + financial_risk * 0.3), 0.1)
        
        # 4. Risk-adjusted score
        risk_adjusted_score = total_expected_return / combined_risk
        
        # Calculate stability scores
        fcf_stability = calculate_growth_stability(fcf_proj)
        rev_stability = calculate_growth_stability(rev_proj)
        growth_stability = (fcf_stability + rev_stability) / 2
        analysis.growth_stability_score = growth_stability
        
        # Calculate divergence
        divergence = abs(fcf_growth - rev_growth)
        analysis.growth_divergence = divergence
        
        # Build complete row with ALL required columns
        row = {
            'ticker': ticker,
            'current_price': analysis.current_price,
            # Growth metrics
            'fcf_growth_annual': fcf_growth,
            'revenue_growth_annual': rev_growth, 
            'combined_growth': combined_growth,
            # Value metrics
            'best_iv_ratio': best_ratio,
            'best_iv_scenario': best_ratio_key,
            # Risk components
            'volatility_risk': avg_volatility_decimal,
            'financial_risk': financial_risk,
            'combined_risk': combined_risk,
            'avg_volatility': avg_volatility_pct,  # REQUIRED for multiple visualizations
            # Return metrics
            'total_expected_return': total_expected_return,
            'risk_adjusted_score': risk_adjusted_score,
            # Stability and divergence
            'growth_stability': growth_stability,
            'growth_divergence': divergence,
            'divergence_flag': divergence > config.GROWTH_DIVERGENCE_THRESHOLD,
            # Company fundamentals
            'fcf': analysis.company_data.fcf,
            'market_cap': analysis.company_data.market_cap,
            # From weighted scores (for compatibility)
            'total_penalty': weighted_scores.loc[ticker, 'total_penalty'] if ticker in weighted_scores.index else np.inf,
            # Additional for compatibility
            'volatility_penalty': avg_volatility_pct  # Used by risk chart
        }
        
        ranking_data.append(row)
    
    if companies_without_prices:
        logger.warning(f"{len(companies_without_prices)} companies skipped due to missing price data")
    
    # Create DataFrame
    df = pd.DataFrame(ranking_data)
    
    if df.empty:
        logger.error("No companies with valid ranking data")
        raise ValueError("No valid companies for ranking")
    
    logger.info(f"Created ranking data for {len(df)} companies")
    
    # Create individual ranking DataFrames
    
    # 1. Growth rankings (using PROJECTED growth)
    growth_rankings = df.copy()
    growth_rankings['fcf_growth_rank'] = growth_rankings['fcf_growth_annual'].rank(ascending=False)
    growth_rankings['revenue_growth_rank'] = growth_rankings['revenue_growth_annual'].rank(ascending=False)
    growth_rankings['combined_growth_rank'] = growth_rankings['combined_growth'].rank(ascending=False)
    growth_rankings['stability_rank'] = growth_rankings['growth_stability'].rank(ascending=False)
    
    # 2. Value rankings
    value_rankings = df.copy()
    value_rankings['value_rank'] = value_rankings['best_iv_ratio'].rank(ascending=False)
    
    # 3. Weighted rankings (includes penalty columns for factor analysis)
    weighted_rankings = df.copy()
    weighted_rankings['weighted_rank'] = weighted_rankings['total_penalty'].rank()
    # Add penalty columns from weighted_scores for factor analysis
    for col in ['dc_penalty', 'mc_penalty', 'growth_penalty']:
        weighted_rankings[col] = weighted_rankings['ticker'].map(weighted_scores[col])
    
    # 4. Combined rankings (risk-adjusted approach)
    combined_rankings = df.copy()
    
    # Normalize ranks to 0-100 scale for scoring
    n_companies = len(df)
    combined_rankings['growth_score'] = (growth_rankings['combined_growth_rank'] / n_companies) * 100
    combined_rankings['value_score'] = (value_rankings['value_rank'] / n_companies) * 100
    combined_rankings['weighted_score'] = (weighted_rankings['weighted_rank'] / n_companies) * 100
    combined_rankings['stability_score'] = (growth_rankings['stability_rank'] / n_companies) * 100
    combined_rankings['divergence_penalty'] = combined_rankings['divergence_flag'].astype(int) * 10
    
    # Risk-adjusted ranking
    combined_rankings['risk_adjusted_rank'] = combined_rankings['risk_adjusted_score'].rank(ascending=False)
    
    # For compatibility: opportunity_rank = risk_adjusted_rank
    combined_rankings['opportunity_rank'] = combined_rankings['risk_adjusted_rank']
    combined_rankings['opportunity_score'] = 100 - (combined_rankings['risk_adjusted_rank'] / n_companies * 100)
    
    # Sort DataFrames
    growth_rankings = growth_rankings.sort_values('combined_growth_rank')
    value_rankings = value_rankings.sort_values('value_rank')
    weighted_rankings = weighted_rankings.sort_values('weighted_rank')
    combined_rankings = combined_rankings.sort_values('risk_adjusted_rank')
    
    # Log summary
    logger.info("Ranking summary:")
    logger.info(f"  Projected growth rates: {df['combined_growth'].min():.1%} to {df['combined_growth'].max():.1%}")
    logger.info(f"  FCF IV/Price ratios: {df['best_iv_ratio'].min():.2f} to {df['best_iv_ratio'].max():.2f}")
    logger.info(f"  Risk-adjusted scores: {df['risk_adjusted_score'].min():.2f} to {df['risk_adjusted_score'].max():.2f}")
    
    return growth_rankings, value_rankings, weighted_rankings, combined_rankings


def select_target_companies(combined_rankings: pd.DataFrame,
                          analyses: Dict[str, CompanyAnalysis],
                          config: AnalysisConfig,
                          top_iv_companies: Optional[List[str]] = None) -> List[str]:
    """
    Select target companies for watchlist.
    
    In shortlist mode: Select from top IV ratio companies, then rank by opportunity
    In owned mode: Include all owned companies plus top ranked others from IV list
    
    The watchlist is a subset of the top IV ratio companies, sorted by opportunity rank.
    
    Args:
        combined_rankings: Combined ranking DataFrame
        analyses: Dictionary of company analyses
        config: Analysis configuration
        top_iv_companies: Optional list of pre-filtered companies by IV ratio
        
    Returns:
        List of selected tickers
    """
    # If top_iv_companies provided, filter combined_rankings to only those companies
    if top_iv_companies:
        filtered_rankings = combined_rankings[combined_rankings['ticker'].isin(top_iv_companies)]
        logger.info(f"Selecting from {len(top_iv_companies)} companies pre-filtered by IV ratio")
    else:
        filtered_rankings = combined_rankings
        logger.warning("No IV ratio pre-filtering applied - using all companies")
    
    # Start with owned companies if in owned mode
    if config.ANALYSIS_MODE == 'owned':
        selected_tickers = set(config.OWNED_COMPANIES)
        logger.info(f"Starting with {len(selected_tickers)} owned companies")
    else:
        selected_tickers = set()
    
    # Get top ranked companies by opportunity rank from the filtered set
    top_companies = filtered_rankings.nsmallest(config.TARGET_WATCHLIST_SIZE, 'opportunity_rank')
    selected_tickers.update(top_companies['ticker'].tolist())
    
    # Log some statistics about the selected companies (informational only)
    high_divergence = top_companies[top_companies['growth_divergence'] > config.GROWTH_DIVERGENCE_THRESHOLD]
    if len(high_divergence) > 0:
        logger.info(f"{len(high_divergence)} of selected companies have FCF/revenue divergence > {config.GROWTH_DIVERGENCE_THRESHOLD:.0%}")
    
    negative_growth = top_companies[
        (top_companies['fcf_growth_annual'] < 0) & (top_companies['revenue_growth_annual'] < 0)
    ]
    if len(negative_growth) > 0:
        logger.info(f"{len(negative_growth)} of selected companies have negative growth in both FCF and revenue")
    
    # Convert to list and sort by opportunity rank
    all_selected = list(selected_tickers)
    
    # Sort by opportunity rank
    ticker_ranks = {}
    for ticker in all_selected:
        if ticker in filtered_rankings['ticker'].values:
            rank = filtered_rankings[filtered_rankings['ticker'] == ticker]['opportunity_rank'].values[0]
            ticker_ranks[ticker] = rank
        else:
            ticker_ranks[ticker] = np.inf
    
    all_selected.sort(key=lambda x: ticker_ranks[x])
    
    logger.info(f"Selected {len(all_selected)} target companies sorted by opportunity rank")
    
    return all_selected


def create_analysis_summary(analyses: Dict[str, CompanyAnalysis],
                           rankings: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
                           target_companies: List[str],
                           config: AnalysisConfig) -> pd.DataFrame:
    """
    Create comprehensive summary table for analysis results.
    
    This summary brings together all the key metrics and rankings for each
    company in the watchlist, making it easy to compare opportunities.
    
    Note: Now includes only FCF-based intrinsic values and growth divergence.
    Growth rates are reported as CAGR.
    
    Args:
        analyses: Dictionary of company analyses
        rankings: Tuple of ranking DataFrames
        target_companies: List of selected tickers
        config: Analysis configuration
        
    Returns:
        Summary DataFrame
    """
    growth_rankings, value_rankings, weighted_rankings, combined_rankings = rankings
    
    # Build summary
    rows = []
    
    for ticker in target_companies:
        if ticker not in analyses:
            logger.warning(f"Target company {ticker} not found in analyses")
            continue
            
        analysis = analyses[ticker]
        
        # Get rankings
        growth_rank = growth_rankings[growth_rankings['ticker'] == ticker]['combined_growth_rank'].values[0]
        value_rank = value_rankings[value_rankings['ticker'] == ticker]['value_rank'].values[0]
        weighted_rank = weighted_rankings[weighted_rankings['ticker'] == ticker]['weighted_rank'].values[0]
        opportunity_rank = combined_rankings[combined_rankings['ticker'] == ticker]['opportunity_rank'].values[0]
        
        # Get intrinsic values for primary period (FCF only)
        period = config.PRIMARY_PERIOD
        fcf_iv = analysis.get_intrinsic_value(period, 'fcf', 'median')
        
        # Get growth projections - these already have CAGR
        fcf_proj = analysis.get_projection(period, 'fcf', 'median')
        rev_proj = analysis.get_projection(period, 'revenue', 'median')
        
        # Get divergence
        divergence = analysis.growth_divergence or 0.0
        
        # Get factor details
        debt_cash_ratio = analysis.company_data.lt_debt / analysis.company_data.cash_and_equiv if analysis.company_data.cash_and_equiv > 0 else float('inf')
        
        row = {
            'ticker': ticker,
            'current_price': analysis.current_price,
            'market_cap': analysis.company_data.market_cap,
            'fcf': analysis.company_data.fcf,
            'fcf_growth_annual': f"{fcf_proj.annual_growth_median:.1%}",  # CAGR
            'revenue_growth_annual': f"{rev_proj.annual_growth_median:.1%}",  # CAGR
            'growth_divergence': f"{divergence:.1%}",
            'divergence_flag': '⚠️' if divergence > config.GROWTH_DIVERGENCE_THRESHOLD else '',
            f'{period}yr_fcf_iv': fcf_iv,
            'best_iv_ratio': analysis.best_iv_to_price_ratio,
            'growth_stability': analysis.growth_stability_score,
            'growth_rank': int(growth_rank),
            'value_rank': int(value_rank),
            'weighted_rank': int(weighted_rank),
            'opportunity_rank': int(opportunity_rank),
            # Factor details
            'debt_cash_ratio': f"{debt_cash_ratio:.2f}",
            'dc_penalty': f"{analysis.dc_penalty:.2f}",
            'mc_penalty': f"{analysis.mc_penalty:.2f}",
            'growth_penalty': f"{analysis.growth_penalty:.2f}",
            'total_penalty': f"{analysis.total_penalty:.2f}"
        }
        
        # Add currency anomalies if present
        if analysis.currency_anomalies:
            row['anomalies'] = ', '.join(analysis.currency_anomalies.keys())
        
        rows.append(row)
    
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.set_index('ticker')
    
    logger.info(f"Created summary for {len(summary)} companies")
    
    return summary