"""
Main analysis pipeline.

Orchestrates the complete intrinsic value analysis from data loading to final results.
Supports multiple analysis modes and time periods.
"""
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd

from config.user_config import AnalysisConfig
from data_pipeline.data_loading import load_metrics_data, load_growth_data, merge_data, create_company_objects
from data_pipeline.filtering import apply_financial_filters, apply_growth_filters, create_filtered_dataframe
from analysis.growth_calculation import calculate_all_projections
from analysis.intrinsic_value import calculate_all_intrinsic_values
from data_sources.yahoo_finance import fetch_current_prices, calculate_current_metrics
from analysis.ranking import rank_companies, select_target_companies, create_analysis_summary
from analysis.weighted_scoring import create_quadrant_analysis
from analysis.factor_analysis import calculate_factor_contributions, analyze_factor_dominance
from data_pipeline.data_structures import CompanyAnalysis, AnalysisResults

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_analysis(config: AnalysisConfig) -> AnalysisResults:
    """
    Run complete intrinsic value analysis pipeline.
    
    Args:
        config: Analysis configuration
        
    Returns:
        Complete analysis results
    """
    logger.info("="*60)
    logger.info("Starting intrinsic value analysis")
    logger.info(f"Data source: {config.COLLECTION_DATE} {config.COUNTRY}")
    logger.info(f"Analysis mode: {config.ANALYSIS_MODE}")
    logger.info(f"Projection periods: {config.PROJECTION_PERIODS} years")
    logger.info("="*60)
    
    # Validate configuration
    config.validate()
    
    # Step 1: Load data
    logger.info("\nStep 1: Loading data")
    metrics_df = load_metrics_data(config)
    growth_df = load_growth_data(config)
    merged_df = merge_data(metrics_df, growth_df)
    
    initial_companies = create_company_objects(merged_df, config)
    logger.info(f"Loaded data for {len(initial_companies)} companies")
    
    # Step 2: Apply filters
    logger.info("\nStep 2: Applying financial filters")
    filtered_companies, filter_results = apply_financial_filters(initial_companies, config)
    filtered_companies = apply_growth_filters(filtered_companies, config, filter_results)
    
    logger.info(f"After filtering: {len(filtered_companies)} companies remain")
    
    # Validate remaining companies
    for ticker, company in filtered_companies.items():
        company.validate() 

    if len(filtered_companies) == 0:
        raise ValueError("No companies passed the filters")
    
    # Report on owned companies if in owned mode
    if config.ANALYSIS_MODE == 'owned' and filter_results.owned_companies_bypassed:
        logger.info(f"\nOwned companies that bypassed filters:")
        for ticker in filter_results.owned_companies_bypassed:
            tracking = filter_results.owned_companies_tracking.get(ticker, {})
            failed = tracking.get('filters_failed', [])
            if failed:
                logger.info(f"  {ticker}: Failed {len(failed)} filters")
                for failure in failed[:3]:  # Show first 3 failures
                    logger.info(f"    - {failure['reason']}")
       
    # No longer limiting to top N companies - analyze all that pass filters
    logger.info(f"Proceeding with all {len(filtered_companies)} companies that passed filters")

    # Step 3: Calculate growth projections
    logger.info("\nStep 3: Calculating growth projections")
    logger.info(f"Running {config.SIMULATION_REPLICATES} Monte Carlo simulations per company")
    projections = calculate_all_projections(filtered_companies, config)
    
    # Step 4: Calculate intrinsic values
    logger.info("\nStep 4: Calculating intrinsic values")
    intrinsic_values = calculate_all_intrinsic_values(filtered_companies, projections, config)
    
    # Step 5: Fetch current prices
    logger.info("\nStep 5: Fetching current market prices")
    tickers = list(filtered_companies.keys())
    yahoo_data = fetch_current_prices(tickers, config)
    
    # Step 6: Build complete analyses
    logger.info("\nStep 6: Building company analyses")
    analyses = {}
    companies_with_prices = 0
    companies_with_anomalies = 0
    
    for ticker in filtered_companies:
        company = filtered_companies[ticker]
        company_projections = projections[ticker]
        company_ivs = intrinsic_values[ticker]
        
        analysis = CompanyAnalysis(
            company_data=company,
            growth_projections=company_projections,
            intrinsic_values=company_ivs
        )
        
        # Add current price if available
        if ticker in yahoo_data and yahoo_data[ticker].api_success:
            ydata = yahoo_data[ticker]
            analysis.current_price = ydata.current_price
            analysis.current_market_cap = ydata.current_market_cap
            analysis.current_fcf = ydata.current_fcf
            
            # Store anomalies
            if ydata.anomalies:
                analysis.currency_anomalies = ydata.anomalies
                companies_with_anomalies += 1
            
            # Calculate additional metrics
            current_metrics = calculate_current_metrics(ydata)
            
            # DEBUG: Log what metrics we got
            logger.debug(f"{ticker}: Current metrics = {list(current_metrics.keys())}")
            
            if 'current_enterprise_value' in current_metrics:
                analysis.current_enterprise_value = current_metrics['current_enterprise_value']
            if 'current_acquirers_multiple' in current_metrics:
                analysis.current_acquirers_multiple = current_metrics['current_acquirers_multiple']
                logger.info(f"{ticker}: Set current AM = {analysis.current_acquirers_multiple:.2f}")
            else:
                logger.debug(f"{ticker}: No current AM calculated")
            
            companies_with_prices += 1
        
        analyses[ticker] = analysis
    
    logger.info(f"Got current prices for {companies_with_prices}/{len(analyses)} companies")
    if companies_with_anomalies:
        logger.warning(f"{companies_with_anomalies} companies have data anomalies")
    
    # Step 7: Rank companies
    logger.info("\nStep 7: Ranking companies")
    growth_rankings, value_rankings, weighted_rankings, combined_rankings = rank_companies(analyses, config)
    
    # Step 7a: Get top companies by IV ratio for pre-filtering
    # This is the key change - we identify the top companies by IV ratio first
    logger.info("\nStep 7a: Identifying top companies by IV ratio")
    companies_with_iv = [(ticker, analysis.best_iv_to_price_ratio) 
                         for ticker, analysis in analyses.items() 
                         if analysis.best_iv_to_price_ratio is not None]
    companies_with_iv.sort(key=lambda x: x[1], reverse=True)
    
    # For analysis, we'll use a larger set (e.g., top 100 by IV ratio)
    # This ensures all subsequent analysis focuses on high IV ratio companies
    analysis_company_count = min(100, len(companies_with_iv))  # Analyze top 100 or all if less
    top_iv_companies = [ticker for ticker, _ in companies_with_iv[:analysis_company_count]]
    logger.info(f"Selected top {len(top_iv_companies)} companies by IV ratio for further analysis")
    
    # Step 8: Additional analysis
    logger.info("\nStep 8: Performing additional analysis")
    
    # Calculate factor contributions
    weighted_scores_df = weighted_rankings[['ticker', 'dc_penalty', 'mc_penalty', 'growth_penalty', 'total_penalty']].set_index('ticker')

    factor_contributions = calculate_factor_contributions(weighted_scores_df)
    
    # Analyze factor dominance
    factor_dominance = analyze_factor_dominance(factor_contributions)
    logger.info("\nFactor dominance summary:")
    for _, row in factor_dominance.iterrows():
        logger.info(f"  {row['primary_factor']}: {row['company_count']} companies ({row['pct_of_total']:.1f}%)")
    
    # Create quadrant analysis
    quadrant_analysis = create_quadrant_analysis(analyses, weighted_scores_df, config)
    quadrant_counts = quadrant_analysis['quadrant'].value_counts().sort_index()
    logger.info("\nQuadrant distribution:")
    for quad, count in quadrant_counts.items():
        logger.info(f"  Quadrant {quad}: {count} companies")
    
    # Step 9: Select target companies - now passing top_iv_companies
    logger.info("\nStep 9: Selecting target companies")
    target_companies = select_target_companies(combined_rankings, analyses, config, top_iv_companies)
    
    # Build results
    results = AnalysisResults(
        companies=analyses,
        initial_company_count=len(initial_companies),
        filtered_company_count=len(filtered_companies),
        final_company_count=len(analyses),
        removed_by_filter=filter_results.removed_by_filter,
        removal_reasons=filter_results.removal_reasons, 
        owned_companies_bypassed=filter_results.owned_companies_bypassed,
        growth_rankings=growth_rankings,
        value_rankings=value_rankings,
        weighted_rankings=weighted_rankings,
        combined_rankings=combined_rankings,
        factor_contributions=factor_contributions,
        watchlist=target_companies,
        analysis_mode=config.ANALYSIS_MODE,
        primary_period=config.PRIMARY_PERIOD,
        config=config
    )
    
    logger.info(f"\nAnalysis complete: {len(target_companies)} companies selected for watchlist")
    
    return results


def save_results(results: AnalysisResults, config: AnalysisConfig):
    """
    Save analysis results to files.
    
    Args:
        results: Analysis results
        config: Analysis configuration
    """
    output_dir = config.output_directory
    logger.info(f"\nSaving results to: {output_dir}")
    
    # Save watchlist summary
    if results.watchlist:
        summary = create_analysis_summary(
            results.companies,
            (results.growth_rankings, results.value_rankings, 
             results.weighted_rankings, results.combined_rankings),
            results.watchlist,
            config
        )
        
        summary_file = output_dir / "watchlist_summary.csv"
        summary.to_csv(summary_file)
        logger.info(f"Saved watchlist summary to {summary_file}")
    
    # Save full rankings
    rankings_file = output_dir / "all_rankings.csv"
    results.combined_rankings.to_csv(rankings_file, index=False)
    logger.info(f"Saved full rankings to {rankings_file}")
    
    # Save weighted rankings with factor details
    weighted_file = output_dir / "weighted_rankings.csv"
    results.weighted_rankings.to_csv(weighted_file, index=False)
    logger.info(f"Saved weighted rankings to {weighted_file}")
    
    # Save factor contributions
    if results.factor_contributions is not None:
        factor_file = output_dir / "factor_contributions.csv"
        results.factor_contributions.to_csv(factor_file)
        logger.info(f"Saved factor contributions to {factor_file}")
    
    # Save removal reasons
    if results.removed_by_filter:
        removals = []
        for filter_name, companies in results.removed_by_filter.items():
            for company in companies:
                removal_entry = {
                    'company': company,
                    'filter': filter_name,
                    'reason': results.removal_reasons.get(company, 'No detailed reason available')
                }
                removals.append(removal_entry)
        
        if removals:
            removals_df = pd.DataFrame(removals)
            removals_file = output_dir / "removed_companies.csv"
            removals_df.to_csv(removals_file, index=False)
            logger.info(f"Saved removal reasons to {removals_file}")
    
    # Save owned companies analysis if in owned mode
    if config.ANALYSIS_MODE == 'owned' and results.owned_companies_bypassed:
        owned_analysis = []
        for ticker in config.OWNED_COMPANIES:
            if ticker in results.companies:
                analysis = results.companies[ticker]
                owned_analysis.append({
                    'ticker': ticker,
                    'in_watchlist': ticker in results.watchlist,
                    'bypassed_filters': ticker in results.owned_companies_bypassed,
                    'current_price': analysis.current_price,
                    'best_iv_ratio': analysis.best_iv_to_price_ratio,
                    'growth_stability': analysis.growth_stability_score
                })
        
        if owned_analysis:
            owned_df = pd.DataFrame(owned_analysis)
            owned_file = output_dir / "owned_companies_analysis.csv"
            owned_df.to_csv(owned_file, index=False)
            logger.info(f"Saved owned companies analysis to {owned_file}")
    
    # Save full analysis data
    full_data = results.export_summary()
    full_file = output_dir / "full_analysis.csv"
    full_data.to_csv(full_file)
    logger.info(f"Saved full analysis to {full_file}")


def main():
    """Main entry point."""
    # Create configuration
    config = AnalysisConfig()
    
    # Run analysis
    results = run_analysis(config)
    
    # Save results
    save_results(results, config)
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Initial companies: {results.initial_company_count}")
    print(f"After filtering: {results.filtered_company_count}")
    print(f"With price data: {results.final_company_count}")
    print(f"Watchlist size: {len(results.watchlist)}")
    
    if config.ANALYSIS_MODE == 'owned' and results.owned_companies_bypassed:
        print(f"\nOwned companies that bypassed filters: {len(results.owned_companies_bypassed)}")
        for ticker in results.owned_companies_bypassed:
            print(f"  - {ticker}")
    
    print(f"\nTop 10 companies by opportunity:")
    
    top10 = results.combined_rankings.head(10)
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        ticker = row['ticker']
        analysis = results.get_company(ticker)
        
        print(f"\n{i}. {ticker}:")
        print(f"  Current Price: ${analysis.current_price:.2f}")
        print(f"  FCF Growth: {analysis.get_projection(config.PRIMARY_PERIOD, 'fcf', 'CI_median').annual_growth_median:.1%}")
        print(f"  Revenue Growth: {analysis.get_projection(config.PRIMARY_PERIOD, 'revenue', 'CI_median').annual_growth_median:.1%}")
        print(f"  Best IV/Price: {analysis.best_iv_to_price_ratio:.2f}x")
        print(f"  Opportunity Rank: {int(row['opportunity_rank'])}")
        
        if analysis.currency_anomalies:
            print(f"  ⚠️  Anomalies: {', '.join(analysis.currency_anomalies.keys())}")


if __name__ == "__main__":
    main()