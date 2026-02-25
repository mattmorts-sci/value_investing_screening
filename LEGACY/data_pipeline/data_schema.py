"""
Data schema definitions for the intrinsic value analysis tool.

This module defines the exact data structure expected at each stage of the pipeline.
No fallbacks, no alternatives - if data doesn't match this schema, the pipeline fails.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import pandas as pd


@dataclass(frozen=True)
class DataColumns:
    """Defines exact column names for each data stage."""
    
    # Stage 1: Raw historical data from JSON files
    HISTORICAL_METRICS: List[str] = field(default_factory=lambda: [
        'fcf',
        'revenue', 
        'operating_income',
        'market_cap',
        'lt_debt',
        'cash_and_equiv',
        'shares_diluted',
        'enterprise_value',
        'enterprise_value_to_book'
    ])
    
    HISTORICAL_GROWTH: List[str] = field(default_factory=lambda: [
        'fcf_growth_mean',
        'fcf_growth_variance',
        'revenue_growth_mean',
        'revenue_growth_variance'
    ])
    
    # Stage 2: Calculated metrics
    CALCULATED_METRICS: Tuple[str, ...] = (
        'debt_cash_ratio',      # lt_debt / cash_and_equiv
        'fcf_per_share',        # fcf / shares_diluted
        'acquirers_multiple',   # enterprise_value / operating_income
        'fcf_to_market_cap'     # fcf / market_cap
    )
    
    # Stage 3: Growth projections (5 year)
    GROWTH_PROJECTIONS: Tuple[str, ...] = (
        'fcf_growth_annual_median',
        'fcf_growth_annual_p25',     # CI bottom
        'fcf_growth_annual_p75',     # CI top
        'fcf_final_value_median',
        'fcf_final_value_p25',
        'fcf_final_value_p75',
        'revenue_growth_annual_median',
        'revenue_growth_annual_p25',
        'revenue_growth_annual_p75',
        'revenue_final_value_median',
        'revenue_final_value_p25',
        'revenue_final_value_p75'
    )
    
    # Stage 4: Intrinsic values
    INTRINSIC_VALUES: Tuple[str, ...] = (
        'fcf_iv_median',
        'fcf_iv_p25',
        'fcf_iv_p75',
        'revenue_iv_median', 
        'revenue_iv_p25',
        'revenue_iv_p75',
        'avg_iv_median',
        'avg_iv_p25',
        'avg_iv_p75'
    )
    
    # Stage 5: Real-time data
    REALTIME_DATA: Tuple[str, ...] = (
        'current_price',
        'current_market_cap',
        'current_fcf',
        'current_shares',
        'api_success',
        'api_currency'
    )
    
    # Stage 6: Final analysis
    ANALYSIS_METRICS: Tuple[str, ...] = (
        'iv_to_price_ratio',      # Best IV / current price
        'growth_rank',            # Rank by growth rate
        'value_rank',             # Rank by IV/price ratio
        'growth_stability_score', # Consistency of growth
        'opportunity_score'       # Combined ranking
    )


@dataclass(frozen=True)
class DataValidation:
    """Validation rules for data at each stage."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str], stage_name: str) -> None:
        """Validate that DataFrame has all required columns with no missing data."""
        # Check columns exist
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"{stage_name}: Missing required columns: {missing_columns}")
        
        # Check for null values in required columns
        null_counts = df[required_columns].isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]
        
        if not columns_with_nulls.empty:
            raise ValueError(f"{stage_name}: Null values found in columns: {columns_with_nulls.to_dict()}")
    
    @staticmethod
    def validate_numeric_ranges(df: pd.DataFrame, column_ranges: Dict[str, tuple], stage_name: str) -> None:
        """Validate that numeric columns are within expected ranges."""
        for column, (min_val, max_val) in column_ranges.items():
            if column not in df.columns:
                continue
                
            values = df[column]
            out_of_range = (values < min_val) | (values > max_val)
            
            if out_of_range.any():
                bad_companies = df.index[out_of_range].tolist()
                raise ValueError(
                    f"{stage_name}: Column '{column}' has values outside range [{min_val}, {max_val}] "
                    f"for companies: {bad_companies[:5]}{'...' if len(bad_companies) > 5 else ''}"
                )


# Export schema instance
SCHEMA = DataColumns()
VALIDATOR = DataValidation()