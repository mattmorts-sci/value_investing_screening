"""Pipeline configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Prices below this floor are treated as corrupted or meaningless.
# Prevents sub-penny prices from producing extreme returns.
MIN_PRICE_FLOOR: float = 0.01


class SortOption(Enum):
    """Screening sort options."""

    COMPOSITE = "composite"
    FCF_EV = "fcf_ev"
    EBIT_EV = "ebit_ev"


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""

    # Database
    db_path: Path = Path("/home/mattm/projects/Pers/financial_db/data/fmp.db")
    exchanges: list[str] = field(default_factory=list)

    # Data sufficiency
    min_quarters: int = 12

    # Filters (None = disabled)
    filter_min_market_cap: float | None = None
    filter_min_interest_coverage: float | None = None
    filter_min_ocf_to_debt: float | None = None
    filter_min_fcf_conversion_r2: float | None = None
    filter_exclude_negative_ttm_fcf: bool = True

    # Valuation ranking
    fcf_ev_weight: float = 0.6
    ebit_ev_weight: float = 0.4
    sort_by: SortOption = SortOption.COMPOSITE


@dataclass
class SimulationConfig:
    """Monte Carlo simulation parameters."""

    num_replicates: int = 10_000
    num_display_paths: int = 25
    cv_cap: float = 1.0

    # Per-quarter growth caps
    early_positive_cap: float = 0.20
    early_negative_cap: float = -0.15
    late_positive_cap: float = 0.12
    late_negative_cap: float = -0.08
    size_tier_threshold: float = 2.0

    # Cumulative constraints
    cumulative_cap: float = 5.0
    cagr_backstop: float = 0.50

    # Momentum exhaustion
    momentum_upper: float = 2.0
    momentum_lower: float = 0.5

    # Time decay
    time_decay_growth_threshold: float = 0.20
    time_decay_factor: float = 0.8

    # Size penalty
    size_penalty_min: float = 0.4
    size_penalty_max: float = 1.0


@dataclass
class DCFConfig:
    """DCF valuation parameters."""

    discount_rate: float = 0.10
    terminal_growth_rate: float = 0.025
    projection_years: int = 10


@dataclass
class TrendsConfig:
    """Trend analysis parameters."""

    min_quarters_fcf_conversion: int = 8


@dataclass
class HeatmapConfig:
    """Sensitivity heatmap parameters."""

    discount_rate_min: float = 0.07
    discount_rate_max: float = 0.13
    discount_rate_step: float = 0.01
    growth_multiplier_min: float = 0.5
    growth_multiplier_max: float = 1.5
    growth_multiplier_step: float = 0.25
    heatmap_replicates: int = 1_000


@dataclass
class DisplayConfig:
    """Display and visual encoding thresholds."""

    fscore_strong_min: int = 7
    fscore_moderate_min: int = 4
    safety_amber_threshold: float = 1.5
