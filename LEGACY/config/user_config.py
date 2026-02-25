"""
User configuration for intrinsic value analysis.

All user-definable parameters are centralized here.
Supports multiple analysis modes and time horizons.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class AnalysisConfig:
    """All user-configurable parameters for the analysis."""
    
    # === DATA LOCATION ===
    DATA_ROOT: Path = Path('/home/mattm/projects/Pers/data')
    COLLECTION_DATE: str = '20250602'
    COUNTRY: str = 'AU'  # 'AU' or 'US'
    PERIOD: str = 'FQ'   # Quarterly data
    DATA_LENGTH: int = 20  # Number of historical periods
    
    # === MEAN REVERSION ===
    REALISTIC_MAX_QUARTERLY_GROWTH: float = 0.5      # 50% max growth per quarter
    REALISTIC_MIN_QUARTERLY_GROWTH: float = -0.3     # 30% max decline per quarter
    MOMENTUM_EXHAUSTION_THRESHOLD: float = 2.0       # Recent growth > 2x mean triggers exhaustion
    MOMENTUM_BOUNCE_THRESHOLD: float = 0.5           # Recent growth < 0.5x mean allows bounce

    # === ANALYSIS MODE ===
    ANALYSIS_MODE: str = 'shortlist'  # 'shortlist' or 'owned'
    OWNED_COMPANIES: List[str] = None  # List of owned tickers for 'owned' mode
    
    # === PROJECTION PERIODS ===
    PROJECTION_PERIODS: List[int] = field(default_factory=lambda: [5, 10])  # Calculate for both 5 and 10 years
    PRIMARY_PERIOD: int = 5  # Primary period for ranking/selection
    
    # === FILTERING THRESHOLDS ===
    MIN_MARKET_CAP: float = 20_000_000.0           # $20M minimum
    MAX_DEBT_TO_CASH_RATIO: float = 2.5            # Max 2.5x debt vs cash
    
    # === RANKING WEIGHTS (must sum to 2.1) ===
    DC_WEIGHT: float = 0.7  # Debt-cash ratio weight
    MC_WEIGHT: float = 0.4  # Market cap weight  
    GROWTH_WEIGHT: float = 1.0  # Growth metrics weight
    
    # === GROWTH ANALYSIS ===
    SIMULATION_REPLICATES: int = 10000              # Number of simulations
    QUARTERS_PER_YEAR: int = 4                      # Don't change - quarterly data
    
    # Growth constraints
    MAX_QUARTERLY_GROWTH: float = 0.5               # 50% max growth per quarter
    MIN_QUARTERLY_GROWTH: float = -0.5              # -50% max decline per quarter
    SIZE_PENALTY_FACTOR: float = 0.1                # Growth penalty for large companies
    
    # Dynamic growth constraint factors
    BASE_MULTIPLE: float = 3.0                      # Base growth multiple
    MAX_ALLOWED_MULTIPLE: float = 15.0              # Maximum growth multiple
    REFERENCE_HIGH_GROWTH: float = 0.5              # Reference high growth rate
    SCENARIO_BETTER_FACTOR: float = 1.2             # Multiplier for best scenario
    SCENARIO_WORSE_FACTOR: float = 0.8              # Multiplier for worst scenario
    
    # === DIVERGENCE ANALYSIS ===
    GROWTH_DIVERGENCE_THRESHOLD: float = 0.10       # Flag if FCF and revenue growth differ by more than 10%
    
    # === VALUATION PARAMETERS ===
    DISCOUNT_RATE: float = 0.10                     # 10% required return
    TERMINAL_GROWTH_RATE: float = 0.01              # 1% perpetual growth
    MARGIN_OF_SAFETY: float = 0.50                  # 50% safety margin
    
    # === SELECTION CRITERIA ===
    MIN_ACCEPTABLE_GROWTH: float = 0.10             # 10% minimum annual growth
    MIN_IV_TO_PRICE_RATIO: float = 1.0              # Must be undervalued
    
    # Growth weighting for average calculations
    FCF_GROWTH_WEIGHT: float = 0.7                  # FCF weight in average
    REVENUE_GROWTH_WEIGHT: float = 0.3              # Revenue weight in average
    
    # === OUTPUT SETTINGS ===
    TARGET_WATCHLIST_SIZE: int = 40                 # Final watchlist size
    DETAILED_REPORT_COUNT: int = 5                  # Number of companies for detailed reports
    
    # === YAHOO FINANCE ===
    CACHE_DIRECTORY: Path = Path("cache")
    CACHE_HOURS: int = 12                           # Cache API results for 12 hours
    API_RETRY_ATTEMPTS: int = 3
    API_RETRY_DELAY: float = 1.0                    # Seconds between retries
    ENABLE_TICKER_MAPPING: bool = True              # Enable ticker change handling
    
    # === PATHS (computed) ===
    @property
    def data_directory(self) -> Path:
        """Get the directory containing data files."""
        training_end = self.DATA_LENGTH - 1
        return self.DATA_ROOT / f'training/{self.COLLECTION_DATE}_{self.COUNTRY}_{self.PERIOD}_{training_end}'
    
    @property
    def metrics_file(self) -> Path:
        """Get the path to metrics data file."""
        return self.data_directory / f'acquirers_value_data_{self.DATA_LENGTH}{self.PERIOD}_{self.COLLECTION_DATE}.json'
    
    @property
    def growth_file(self) -> Path:
        """Get the path to growth data file."""
        return self.data_directory / f'growth_data_{self.DATA_LENGTH}{self.PERIOD}_{self.COLLECTION_DATE}.json'
    
    @property
    def output_directory(self) -> Path:
        """Get the output directory for results."""
        output_dir = Path("output") / f"{self.COLLECTION_DATE}_{self.COUNTRY}_{self.ANALYSIS_MODE}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def __post_init__(self):
        """Initialize and validate configuration."""
        # Initialize owned companies list if needed
        if self.OWNED_COMPANIES is None:
            self.OWNED_COMPANIES = []
        
        # Validate ranking weights
        weight_sum = self.DC_WEIGHT + self.MC_WEIGHT + self.GROWTH_WEIGHT
        if abs(weight_sum - 2.1) > 0.01:
            raise ValueError(f"Ranking weights must sum to 2.1, got {weight_sum}")
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # Check files exist
        if not self.metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {self.metrics_file}")
        
        if not self.growth_file.exists():
            raise FileNotFoundError(f"Growth file not found: {self.growth_file}")
        
        # Validate numeric ranges
        if self.DISCOUNT_RATE <= self.TERMINAL_GROWTH_RATE:
            raise ValueError("Discount rate must be greater than terminal growth rate")
        
        if self.MARGIN_OF_SAFETY < 0 or self.MARGIN_OF_SAFETY >= 1:
            raise ValueError("Margin of safety must be between 0 and 1")
        
        if self.MIN_QUARTERLY_GROWTH >= self.MAX_QUARTERLY_GROWTH:
            raise ValueError("Min growth must be less than max growth")
        
        # Validate weights sum
        if abs(self.FCF_GROWTH_WEIGHT + self.REVENUE_GROWTH_WEIGHT - 1.0) > 0.01:
            raise ValueError("FCF and revenue weights must sum to 1.0")
        
        # Validate mode
        if self.ANALYSIS_MODE not in ['shortlist', 'owned']:
            raise ValueError("Analysis mode must be 'shortlist' or 'owned'")
        
        # Validate owned companies in owned mode
        if self.ANALYSIS_MODE == 'owned' and not self.OWNED_COMPANIES:
            raise ValueError("Owned mode requires OWNED_COMPANIES list")