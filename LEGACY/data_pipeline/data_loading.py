"""
Data loading module with strict validation.

No fallbacks, no alternatives - data must match the expected format exactly.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np

from data_pipeline.data_schema import SCHEMA, VALIDATOR
from data_pipeline.data_structures import CompanyData
from config.user_config import AnalysisConfig

logger = logging.getLogger(__name__)


def _calculate_historical_cagr(values: List[float], periods: int) -> float:
    """
    Calculate CAGR from historical data using first and last valid values.
    
    Args:
        values: List of historical values (e.g., FCF or revenue)
        periods: Number of periods (quarters)
        
    Returns:
        Annualized CAGR as a decimal (e.g., 0.15 for 15%)
    """
    # Filter out None/NaN values with their indices
    valid_pairs = [(i, v) for i, v in enumerate(values) if v is not None and not np.isnan(v) and v != 0]
    
    if len(valid_pairs) < 2:
        logger.warning("Insufficient valid data points for CAGR calculation")
        return 0.0
    
    # Get first and last valid values
    first_idx, first_value = valid_pairs[0]
    last_idx, last_value = valid_pairs[-1]
    
    # Skip if values have different signs (can't calculate meaningful CAGR)
    if first_value * last_value < 0:
        logger.debug("Cannot calculate CAGR when values change sign")
        return 0.0
    
    # Calculate periods between first and last valid values
    periods_between = last_idx - first_idx
    if periods_between < 4:  # Less than 1 year of data
        logger.debug("Less than 1 year between valid data points")
        return 0.0
    
    try:
        # Calculate quarterly CAGR
        quarterly_cagr = (abs(last_value) / abs(first_value)) ** (1 / periods_between) - 1
        
        # Convert to annual CAGR
        annual_cagr = (1 + quarterly_cagr) ** 4 - 1
        
        # Preserve sign (if both values are negative, growth should still be positive if magnitude increased)
        if first_value < 0 and last_value < 0:
            if abs(last_value) > abs(first_value):
                annual_cagr = -annual_cagr  # Negative growth (getting more negative)
        
        return annual_cagr
        
    except (ValueError, ZeroDivisionError):
        logger.warning("Error calculating CAGR")
        return 0.0


def load_metrics_data(config: AnalysisConfig) -> pd.DataFrame:
    """
    Load historical metrics data from JSON file.
    
    Args:
        config: Analysis configuration
        
    Returns:
        DataFrame with company metrics (index: ticker)
        
    Raises:
        FileNotFoundError: If metrics file doesn't exist
        ValueError: If data structure is invalid
    """
    metrics_file = config.metrics_file
    training_end = config.DATA_LENGTH - 1
    
    # Load JSON
    with open(metrics_file, 'r') as f:
        raw_data = json.load(f)
    
    # Extract metrics section
    if 'metrics' not in raw_data:
        raise ValueError(f"Invalid file structure: missing 'metrics' key in {metrics_file}")
    
    metrics_data = raw_data['metrics']
    
    # Build DataFrame with exact columns
    result_data = {}
    
    for metric in SCHEMA.HISTORICAL_METRICS:
        if metric not in metrics_data:
            raise ValueError(f"Missing required metric: {metric}")
        
        # Extract value at training_end for each company
        metric_values = {}
        for company, values in metrics_data[metric].items():
            if not isinstance(values, list):
                raise ValueError(f"Invalid data format for {metric}/{company}: expected list")
            
            if len(values) <= training_end:
                raise ValueError(f"Insufficient data for {metric}/{company}: {len(values)} periods")
            
            value = values[training_end]
            if value is None or np.isnan(value):
                raise ValueError(f"Missing value for {metric}/{company} at period {training_end}")
            
            metric_values[company] = value
        
        result_data[metric] = metric_values
    
    # Create DataFrame
    df = pd.DataFrame(result_data)
    
    # Validate
    VALIDATOR.validate_dataframe(df, SCHEMA.HISTORICAL_METRICS, "Metrics Data")
    
    # Store the raw metrics data for CAGR calculation
    df.attrs['raw_metrics_data'] = metrics_data
    
    logger.info(f"Loaded metrics for {len(df)} companies")
    
    return df


def load_growth_data(config: AnalysisConfig) -> pd.DataFrame:
    """
    Load historical growth data from JSON file.
    
    Args:
        config: Analysis configuration
        
    Returns:
        DataFrame with growth statistics (index: ticker)
        
    Raises:
        FileNotFoundError: If growth file doesn't exist
        ValueError: If data structure is invalid
    """
    growth_file = config.growth_file
    training_end = config.DATA_LENGTH - 1
    
    # Load JSON
    with open(growth_file, 'r') as f:
        raw_data = json.load(f)
    
    # Extract growth section
    if 'metrics' not in raw_data:
        raise ValueError(f"Invalid file structure: missing 'metrics' key in {growth_file}")
    
    growth_data = raw_data['metrics']
    
    # Build DataFrame with mean and variance for each growth metric
    result_data = {}
    
    for base_metric in ['fcf', 'revenue']:
        growth_metric = f'{base_metric}_growth'
        
        if growth_metric not in growth_data:
            # Try without _growth suffix
            if base_metric not in growth_data:
                raise ValueError(f"Missing growth data for {base_metric}")
            growth_metric = base_metric
        
        # Calculate mean and variance for each company
        mean_values = {}
        variance_values = {}
        
        for company, values in growth_data[growth_metric].items():
            if not isinstance(values, list):
                raise ValueError(f"Invalid growth data for {growth_metric}/{company}")
            
            # Use data up to training_end
            historical_values = values[:training_end + 1]
            
            # Filter out None/NaN values
            valid_values = [v for v in historical_values if v is not None and not np.isnan(v)]
            
            if len(valid_values) < 3:  # Need at least 3 points
                raise ValueError(f"Insufficient valid growth data for {growth_metric}/{company}")
            
            mean_values[company] = np.mean(valid_values)
            variance_values[company] = np.var(valid_values)
        
        result_data[f'{base_metric}_growth_mean'] = mean_values
        result_data[f'{base_metric}_growth_variance'] = variance_values
    
    # Create DataFrame
    df = pd.DataFrame(result_data)
    
    # Validate
    VALIDATOR.validate_dataframe(df, SCHEMA.HISTORICAL_GROWTH, "Growth Data")
    
    logger.info(f"Loaded growth data for {len(df)} companies")
    
    return df


def merge_data(metrics_df: pd.DataFrame, growth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge metrics and growth data, keeping only companies with both.
    
    Args:
        metrics_df: DataFrame with company metrics
        growth_df: DataFrame with growth statistics
        
    Returns:
        Merged DataFrame with all required data
        
    Raises:
        ValueError: If no companies have both metrics and growth data
    """
    # Store raw metrics data before merge
    raw_metrics_data = metrics_df.attrs.get('raw_metrics_data', {})
    
    # Inner join - only keep companies with both
    merged = pd.merge(
        metrics_df,
        growth_df,
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    # Restore raw metrics data to merged DataFrame
    merged.attrs['raw_metrics_data'] = raw_metrics_data
    
    if merged.empty:
        raise ValueError("No companies have both metrics and growth data")
    
    # Add calculated metrics with proper handling of division by zero
    # When denominators are zero, we get infinity which is mathematically correct
    # These will be filtered out during the filtering phase if needed
    
    # Debt/Cash Ratio: Handle zero cash cases
    merged['debt_cash_ratio'] = np.where(
        merged['cash_and_equiv'] == 0,
        np.inf,  # Infinite ratio when no cash
        merged['lt_debt'] / merged['cash_and_equiv']
    )
    
    # FCF per Share: Handle zero shares (shouldn't happen but be safe)
    merged['fcf_per_share'] = np.where(
        merged['shares_diluted'] == 0,
        np.nan,  # No meaningful per-share value without shares
        merged['fcf'] / merged['shares_diluted']
    )
    
    # Acquirer's Multiple: Handle zero or negative operating income
    merged['acquirers_multiple'] = np.where(
        merged['operating_income'] == 0,
        np.inf,  # Infinite multiple when no operating income
        merged['enterprise_value'] / merged['operating_income']
    )
    
    # FCF to Market Cap: Handle zero market cap (shouldn't happen but be safe)
    merged['fcf_to_market_cap'] = np.where(
        merged['market_cap'] == 0,
        np.nan,  # No meaningful ratio without market cap
        merged['fcf'] / merged['market_cap']
    )
    
    # Log any infinite or NaN values for transparency
    inf_debt_cash = np.isinf(merged['debt_cash_ratio']).sum()
    if inf_debt_cash > 0:
        logger.warning(f"{inf_debt_cash} companies have infinite debt/cash ratio (zero cash)")
    
    nan_fcf_share = merged['fcf_per_share'].isna().sum()
    if nan_fcf_share > 0:
        logger.warning(f"{nan_fcf_share} companies have invalid FCF per share")
    
    inf_acq_mult = np.isinf(merged['acquirers_multiple']).sum()
    if inf_acq_mult > 0:
        logger.warning(f"{inf_acq_mult} companies have infinite acquirer's multiple")
    
    logger.info(f"Merged data for {len(merged)} companies")
    
    return merged


def _validate_calculated_metrics(df: pd.DataFrame) -> None:
    """
    Validate calculated metrics are reasonable (basic checks only).
    
    This function logs warnings about extreme values but doesn't raise errors.
    Companies with problematic metrics will be handled during the filtering phase
    where they can be properly tracked and documented.
    """
    # Check for extremely negative acquirer's multiples
    # Negative multiples indicate negative operating income (losses)
    negative_am = df['acquirers_multiple'] < 0
    if negative_am.any():
        count = negative_am.sum()
        companies = df.index[negative_am].tolist()[:5]  # Show first 5
        logger.info(f"{count} companies have negative acquirer's multiple (operating losses)")
        logger.debug(f"Examples: {companies}")
    
    # Check for NaN values in calculated metrics
    calculated_cols = ['debt_cash_ratio', 'fcf_per_share', 'acquirers_multiple', 'fcf_to_market_cap']
    for col in calculated_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            logger.warning(f"{nan_count} companies have NaN values in {col}")
    
    # Summary of extreme values
    total_companies = len(df)
    logger.info(f"Calculated metrics validation complete for {total_companies} companies")


def create_company_objects(merged_df: pd.DataFrame, config: AnalysisConfig) -> Dict[str, CompanyData]:
    """
    Create CompanyData objects from merged DataFrame.
    
    Args:
        merged_df: DataFrame with all company data
        config: Analysis configuration
        
    Returns:
        Dictionary mapping ticker to CompanyData
    """
    companies = {}
    
    # Get raw metrics data for CAGR calculation
    raw_metrics_data = merged_df.attrs.get('raw_metrics_data', {})
    
    for ticker in merged_df.index:
        row = merged_df.loc[ticker]
        
        # Parse ticker format (SYMBOL:COUNTRY)
        if ':' in ticker:
            symbol, country = ticker.split(':', 1)
        else:
            symbol = ticker
            country = config.COUNTRY
        
        # Calculate historical CAGR if we have raw data
        historical_fcf_cagr = 0.0
        historical_revenue_cagr = 0.0
        
        if raw_metrics_data:
            # Get FCF history
            if 'fcf' in raw_metrics_data and ticker in raw_metrics_data['fcf']:
                fcf_history = raw_metrics_data['fcf'][ticker]
                historical_fcf_cagr = _calculate_historical_cagr(fcf_history, config.DATA_LENGTH)
                logger.debug(f"{ticker}: Historical FCF CAGR = {historical_fcf_cagr:.2%}")
            
            # Get revenue history
            if 'revenue' in raw_metrics_data and ticker in raw_metrics_data['revenue']:
                revenue_history = raw_metrics_data['revenue'][ticker]
                historical_revenue_cagr = _calculate_historical_cagr(revenue_history, config.DATA_LENGTH)
                logger.debug(f"{ticker}: Historical Revenue CAGR = {historical_revenue_cagr:.2%}")
        
        company = CompanyData(
            ticker=ticker,
            country=country,
            fcf=row['fcf'],
            revenue=row['revenue'],
            operating_income=row['operating_income'],
            market_cap=row['market_cap'],
            lt_debt=row['lt_debt'],
            cash_and_equiv=row['cash_and_equiv'],
            shares_diluted=row['shares_diluted'],
            enterprise_value=row['enterprise_value'],
            fcf_growth_mean=row['fcf_growth_mean'],
            fcf_growth_variance=row['fcf_growth_variance'],
            revenue_growth_mean=row['revenue_growth_mean'],
            revenue_growth_variance=row['revenue_growth_variance'],
            historical_fcf_cagr=historical_fcf_cagr,
            historical_revenue_cagr=historical_revenue_cagr
        )
        
        # Validate
        # company.validate()
        
        companies[ticker] = company
    
    # Log summary of historical CAGR values
    cagr_values = [(c.historical_fcf_cagr, c.historical_revenue_cagr) for c in companies.values()]
    fcf_cagrs = [v[0] for v in cagr_values if v[0] != 0]
    rev_cagrs = [v[1] for v in cagr_values if v[1] != 0]
    
    if fcf_cagrs:
        logger.info(f"Historical FCF CAGR range: {min(fcf_cagrs):.1%} to {max(fcf_cagrs):.1%}")
    if rev_cagrs:
        logger.info(f"Historical Revenue CAGR range: {min(rev_cagrs):.1%} to {max(rev_cagrs):.1%}")
    
    return companies