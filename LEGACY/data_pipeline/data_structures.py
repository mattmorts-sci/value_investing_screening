"""
Core data structures for the analysis pipeline.

These classes encapsulate data at each stage of processing,
ensuring type safety and preventing data corruption.

Note: Intrinsic values are calculated using FCF-based DCF only.
Revenue growth is tracked separately for business quality analysis.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import pandas as pd
import numpy as np

# Import AnalysisConfig for type annotation only
if TYPE_CHECKING:
    from config.user_config import AnalysisConfig


@dataclass
class CompanyData:
    """Historical data for a single company."""
    ticker: str
    country: str
    
    # Financial metrics
    fcf: float
    revenue: float
    operating_income: float
    market_cap: float
    lt_debt: float
    cash_and_equiv: float
    shares_diluted: float
    enterprise_value: float
    
    # Growth rates (quarterly)
    fcf_growth_mean: float
    fcf_growth_variance: float
    revenue_growth_mean: float
    revenue_growth_variance: float
    
    # Historical CAGR (point-to-point calculation)
    historical_fcf_cagr: float = 0.0
    historical_revenue_cagr: float = 0.0
    
    @property
    def average_growth_mean(self) -> float:
        """Calculate weighted average of FCF and revenue growth means.
        
        Note: This is used for growth analysis only, NOT for valuation.
        """
        # FCF weighted 70%, Revenue 30%
        return (self.fcf_growth_mean * 0.7 + self.revenue_growth_mean * 0.3)
    
    @property
    def average_growth_variance(self) -> float:
        """Calculate weighted average of FCF and revenue growth variances.
        
        Note: This is used for growth analysis only, NOT for valuation.
        """
        return (self.fcf_growth_variance * 0.7 + self.revenue_growth_variance * 0.3)
    
    def validate(self) -> None:
        """Ensure all data is valid."""
        if self.market_cap <= 0:
            raise ValueError(f"{self.ticker}: Market cap must be positive")
        
        if self.shares_diluted <= 0:
            raise ValueError(f"{self.ticker}: Shares must be positive")
        
        if self.cash_and_equiv < 0:
            raise ValueError(f"{self.ticker}: Cash cannot be negative")
        
        if self.lt_debt < 0:
            raise ValueError(f"{self.ticker}: Debt cannot be negative")


@dataclass
class GrowthProjection:
    """Results from Monte Carlo growth simulation for one metric."""
    metric_name: str  # 'fcf' or 'revenue'
    current_value: float
    period_years: int  # 5 or 10
    
    # Projected final values
    final_value_median: float
    final_value_ci_bottom: float  # CI_bottom scenario
    final_value_ci_top: float     # CI_top scenario
    
    # Annual growth rates  
    annual_growth_median: float
    annual_growth_ci_bottom: float
    annual_growth_ci_top: float
    
    # Scenario type for dynamic constraints
    scenario_type: str = 'base'  # 'base', 'best', or 'worst'
    
    # Path statistics
    simulation_paths: Optional[np.ndarray] = None
    
    def get_scenario_growth(self, scenario: str) -> float:
        """Get growth rate for a specific scenario."""
        if scenario in ['median', 'CI_median']:
            return self.annual_growth_median
        elif scenario in ['bottom', 'CI_bottom']:
            return self.annual_growth_ci_bottom
        elif scenario in ['top', 'CI_top']:
            return self.annual_growth_ci_top
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
    
    def get_scenario_value(self, scenario: str) -> float:
        """Get final value for a specific scenario."""
        if scenario in ['median', 'CI_median']:
            return self.final_value_median
        elif scenario in ['bottom', 'CI_bottom']:
            return self.final_value_ci_bottom
        elif scenario in ['top', 'CI_top']:
            return self.final_value_ci_top
        else:
            raise ValueError(f"Unknown scenario: {scenario}")


@dataclass
class IntrinsicValue:
    """Intrinsic value calculation results.
    
    Note: Only FCF-based DCF calculations are performed.
    Revenue is tracked separately but not used for valuation.
    """
    metric_name: str  # Always 'fcf' for valuation
    scenario: str     # 'median', 'bottom', or 'top'
    period_years: int # 5 or 10
    
    # DCF components
    projected_cash_flows: List[float]
    terminal_value: float
    present_value: float
    
    # Per share values
    intrinsic_value_per_share: float
    
    # Assumptions used
    growth_rate: float
    discount_rate: float
    terminal_growth_rate: float
    margin_of_safety: float


@dataclass
class CompanyAnalysis:
    """Complete analysis results for a company."""
    company_data: CompanyData
    
    # Growth projections by period and metric
    # Structure: {period: {metric: {scenario: GrowthProjection}}}
    # Note: Both FCF and revenue projections are stored but only FCF is used for valuation
    growth_projections: Dict[int, Dict[str, Dict[str, GrowthProjection]]]
    
    # Intrinsic values by period and metric
    # Structure: {period: {(metric, scenario): IntrinsicValue}}
    # Note: Only FCF-based intrinsic values are stored
    intrinsic_values: Dict[int, Dict[Tuple[str, str], IntrinsicValue]]
    
    # Real-time data
    current_price: Optional[float] = None
    current_market_cap: Optional[float] = None
    current_fcf: Optional[float] = None
    current_enterprise_value: Optional[float] = None
    current_acquirers_multiple: Optional[float] = None
    currency_anomalies: Optional[Dict[str, str]] = None
    
    # Analysis metrics
    best_iv_to_price_ratio: Optional[float] = None
    growth_stability_score: Optional[float] = None
    growth_divergence: Optional[float] = None  # FCF vs revenue growth divergence
    
    # Weighted scoring components
    dc_penalty: Optional[float] = None
    mc_penalty: Optional[float] = None
    growth_penalty: Optional[float] = None
    total_penalty: Optional[float] = None
    
    # Scenario name mapping
    SCENARIO_MAP = {
        'bottom': 'bottom',
        'median': 'median', 
        'top': 'top',
        # Legacy support
        'CI_bottom': 'bottom',
        'CI_median': 'median',
        'CI_top': 'top'
    }
    
    def get_projection(self, period: int, metric: str, scenario: str) -> GrowthProjection:
        """Get growth projection for specific period, metric, and scenario.
        
        Note: Both FCF and revenue projections are available.
        """
        # Map scenario name if needed
        mapped_scenario = self.SCENARIO_MAP.get(scenario, scenario)
        
        if period not in self.growth_projections:
            raise KeyError(f"No projections for period {period}")
        if metric not in self.growth_projections[period]:
            raise KeyError(f"No {metric} projection for period {period}")
        if mapped_scenario not in self.growth_projections[period][metric]:
            raise KeyError(f"No {mapped_scenario} scenario for {metric} in period {period}")
        
        return self.growth_projections[period][metric][mapped_scenario]
    
    def get_intrinsic_value(self, period: int, metric: str, scenario: str) -> float:
        """Get intrinsic value for specific period, metric, and scenario.
        
        Note: Only FCF-based intrinsic values are available.
        Requesting revenue or average will raise KeyError.
        """
        # Map scenario name if needed
        mapped_scenario = self.SCENARIO_MAP.get(scenario, scenario)
        
        # Only FCF intrinsic values exist
        if metric not in ['fcf']:
            raise KeyError(f"Only FCF-based intrinsic values are calculated. Requested: {metric}")
        
        key = (metric, mapped_scenario)
        if period not in self.intrinsic_values:
            raise KeyError(f"No intrinsic values for period {period}")
        if key not in self.intrinsic_values[period]:
            raise KeyError(f"No intrinsic value for {metric}/{mapped_scenario} in period {period}")
        
        return self.intrinsic_values[period][key].intrinsic_value_per_share
    
    def calculate_iv_to_price_ratios(self, period: int = None) -> Dict[str, float]:
        """Calculate IV/price ratios for all scenarios.
        
        Note: Only FCF-based ratios are calculated.
        """
        if self.current_price is None or self.current_price <= 0:
            return {}
        
        # Use specified period or first available
        if period is None:
            period = min(self.intrinsic_values.keys())
        
        ratios = {}
        for (metric, scenario), iv in self.intrinsic_values[period].items():
            # Only FCF-based intrinsic values exist
            ratio_key = f"{period}yr_{metric}_{scenario}_ratio"
            ratios[ratio_key] = iv.intrinsic_value_per_share / self.current_price
        
        return ratios
    
    def calculate_growth_divergence(self, period: int) -> float:
        """Calculate divergence between FCF and revenue growth for a given period.
        
        Returns the absolute difference between FCF and revenue growth rates.
        This helps identify companies where cash generation is diverging from revenue.
        """
        fcf_proj = self.get_projection(period, 'fcf', 'median')
        rev_proj = self.get_projection(period, 'revenue', 'median')
        
        divergence = abs(fcf_proj.annual_growth_median - rev_proj.annual_growth_median)
        self.growth_divergence = divergence
        
        return divergence


@dataclass
class AnalysisResults:
    """Final results of the complete analysis pipeline."""
    # All analyzed companies
    companies: Dict[str, CompanyAnalysis]
    
    # Filtering statistics
    initial_company_count: int
    filtered_company_count: int
    final_company_count: int
    
    # Companies removed at each stage
    removed_by_filter: Dict[str, List[str]]
    removal_reasons: Dict[str, str]  
    owned_companies_bypassed: List[str]  # Owned companies that bypassed filters
    
    # Rankings
    growth_rankings: pd.DataFrame
    value_rankings: pd.DataFrame
    weighted_rankings: pd.DataFrame  # Using DC/MC/Growth weights
    combined_rankings: pd.DataFrame
    
    # Target watchlist
    watchlist: List[str]
    
    # Configuration used
    analysis_mode: str
    primary_period: int
    config: 'AnalysisConfig'  # Added config field
    
    # Factor analysis (optional field moved to end)
    factor_contributions: Optional[pd.DataFrame] = None
    
    def get_company(self, ticker: str) -> CompanyAnalysis:
        """Get analysis for specific company."""
        if ticker not in self.companies:
            raise KeyError(f"Company not found: {ticker}")
        return self.companies[ticker]
    
    def export_summary(self) -> pd.DataFrame:
        """Export summary DataFrame for all companies."""
        rows = []
        
        for ticker, analysis in self.companies.items():
            # Get primary period data from stored config
            period = self.primary_period
            
            row = {
                'ticker': ticker,
                'current_price': analysis.current_price,
                'current_market_cap': analysis.current_market_cap,
                'historical_fcf': analysis.company_data.fcf,
                'historical_revenue': analysis.company_data.revenue,
                'historical_market_cap': analysis.company_data.market_cap,
                f'{period}yr_fcf_growth_annual': analysis.get_projection(period, 'fcf', 'median').annual_growth_median,
                f'{period}yr_revenue_growth_annual': analysis.get_projection(period, 'revenue', 'median').annual_growth_median,
                f'{period}yr_fcf_iv_median': analysis.get_intrinsic_value(period, 'fcf', 'median'),
                # Note: No revenue or average intrinsic values
                'best_iv_to_price': analysis.best_iv_to_price_ratio,
                'growth_stability': analysis.growth_stability_score,
                'growth_divergence': analysis.growth_divergence,
                'current_acquirers_multiple': analysis.current_acquirers_multiple,
                'debt_cash_ratio': analysis.company_data.lt_debt / analysis.company_data.cash_and_equiv,
                'dc_penalty': analysis.dc_penalty,
                'mc_penalty': analysis.mc_penalty,
                'growth_penalty': analysis.growth_penalty,
                'total_penalty': analysis.total_penalty
            }
            
            # Add currency anomalies if present
            if analysis.currency_anomalies:
                row['currency_anomalies'] = '; '.join([
                    f"{k}: {v}" for k, v in analysis.currency_anomalies.items()
                ])
            
            rows.append(row)
        
        return pd.DataFrame(rows).set_index('ticker')