"""Pipeline configuration.

All user-configurable parameters in a single AnalysisConfig dataclass.
Column mapping specs (TableSpec, ColumnSpec) for config-driven data loading.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Column / table specs (frozen — structural, not user-configurable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnSpec:
    """Mapping from a database column to an internal pipeline column name.

    Attributes:
        db_column: Column name in the FMP database.
        internal_name: Column name used throughout the pipeline.
        required: If True, NaN in this column triggers company drop.
            If False, NaN passes through to downstream modules.
    """

    db_column: str
    internal_name: str
    required: bool = True


@dataclass(frozen=True)
class TableSpec:
    """Specification for loading one FMP database table.

    Attributes:
        table_name: FMP table name (e.g. "incomeStatement").
        columns: Column mappings to load from this table.
        join_type: How to join this table to the base ("inner" or "left").
            The first TableSpec in a sequence is the base; its join_type
            is ignored.
    """

    table_name: str
    columns: tuple[ColumnSpec, ...]
    join_type: str = "inner"

    def __post_init__(self) -> None:
        if self.join_type not in ("inner", "left"):
            raise ValueError(
                f"Invalid join_type {self.join_type!r} for table "
                f"{self.table_name!r}. Must be 'inner' or 'left'."
            )


# ---------------------------------------------------------------------------
# Market mappings
# ---------------------------------------------------------------------------

MARKET_EXCHANGES: dict[str, tuple[str, ...]] = {
    "US": ("NASDAQ", "NYSE", "AMEX"),
    "AU": ("ASX",),
    "UK": ("LSE",),
    "CA": ("TSX", "CNQ", "NEO", "TSXV"),
    "NZ": ("NZE",),
    "SG": ("SES",),
    "HK": ("HKSE",),
}

MARKET_CURRENCIES: dict[str, tuple[str, ...]] = {
    "US": ("USD",),
    "AU": ("AUD",),
    "UK": ("GBP",),
    "CA": ("CAD",),
    "NZ": ("NZD",),
    "SG": ("SGD",),
    "HK": ("HKD",),
}


# ---------------------------------------------------------------------------
# Default table specs
# ---------------------------------------------------------------------------

DEFAULT_ENTITY_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec(db_column="entityId", internal_name="entity_id"),
    ColumnSpec(db_column="currentSymbol", internal_name="symbol"),
    ColumnSpec(db_column="companyName", internal_name="company_name"),
    ColumnSpec(db_column="exchange", internal_name="exchange"),
    ColumnSpec(db_column="country", internal_name="country"),
)

DEFAULT_PRICE_COLUMN = ColumnSpec(
    db_column="adjClose", internal_name="adj_close", required=True,
)

DEFAULT_TABLE_SPECS: tuple[TableSpec, ...] = (
    TableSpec(
        table_name="incomeStatement",
        columns=(
            ColumnSpec(db_column="revenue", internal_name="revenue"),
            ColumnSpec(db_column="operatingIncome", internal_name="operating_income"),
            ColumnSpec(
                db_column="weightedAverageShsOutDil",
                internal_name="shares_diluted",
            ),
        ),
        join_type="inner",  # base table — join_type ignored
    ),
    TableSpec(
        table_name="balanceSheet",
        columns=(
            ColumnSpec(db_column="longTermDebt", internal_name="lt_debt"),
            ColumnSpec(
                db_column="cashAndCashEquivalents", internal_name="cash",
            ),
        ),
        join_type="inner",
    ),
    TableSpec(
        table_name="cashFlow",
        columns=(
            ColumnSpec(db_column="freeCashFlow", internal_name="fcf"),
        ),
        join_type="inner",
    ),
    TableSpec(
        table_name="cashFlowGrowth",
        columns=(
            ColumnSpec(
                db_column="growthFreeCashFlow",
                internal_name="fcf_growth",
                required=False,
            ),
        ),
        join_type="left",
    ),
    TableSpec(
        table_name="incomeStatementGrowth",
        columns=(
            ColumnSpec(
                db_column="growthRevenue",
                internal_name="revenue_growth",
                required=False,
            ),
        ),
        join_type="left",
    ),
)


# ---------------------------------------------------------------------------
# Main configuration
# ---------------------------------------------------------------------------


@dataclass
class AnalysisConfig:
    """All user-configurable parameters for the analysis pipeline."""

    # -- Market --
    market: str = "AU"
    mode: str = "shortlist"
    owned_companies: list[str] = field(default_factory=list)

    # -- Data loading --
    db_path: Path = Path("/home/mattm/projects/Pers/financial_db/data/fmp.db")
    period_type: str = "FQ"
    history_years: int = 5
    price_alignment_days: int = 7
    common_range_lower_pct: float = 0.10
    common_range_upper_pct: float = 0.90

    # -- Growth projection (negative FCF: fade-to-equilibrium) --
    equilibrium_growth_rate: float = 0.03
    base_fade_half_life_years: float = 2.5
    scenario_band_width: float = 1.0
    negative_fcf_improvement_cap: float = 0.15
    projection_periods: tuple[int, ...] = (5, 10)
    primary_period: int = 5

    # -- Growth projection (positive FCF: Monte Carlo) --
    simulation_replicates: int = 10_000
    cv_cap: float = 1.0
    cumulative_growth_cap: float = 10.0
    cumulative_decline_floor: float = 0.1
    annual_cagr_backstop: float = 1.0
    momentum_exhaustion_threshold: float = 2.0
    time_decay_base: float = 0.8
    high_growth_threshold: float = 0.3
    size_penalty_factor: float = 0.1

    # Per-quarter growth caps (asymmetric by metric and size evolution).
    # "small" = cumulative growth <= 2.0x, "large" = > 2.0x.
    fcf_small_pos_cap: float = 0.40
    fcf_small_neg_cap: float = -0.30
    fcf_large_pos_cap: float = 0.25
    fcf_large_neg_cap: float = -0.20
    revenue_small_pos_cap: float = 0.25
    revenue_small_neg_cap: float = -0.15
    revenue_large_pos_cap: float = 0.15
    revenue_large_neg_cap: float = -0.10

    # -- DCF --
    discount_rate: float = 0.10
    terminal_growth_rate: float = 0.01
    margin_of_safety: float = 0.50
    quarters_per_year: int = 4

    # -- Filtering --
    enable_negative_fcf_filter: bool = False
    enable_data_consistency_filter: bool = True
    enable_market_cap_filter: bool = True
    enable_debt_cash_filter: bool = True
    min_market_cap: float = 20_000_000
    max_debt_to_cash_ratio: float = 2.5

    # -- Ranking (risk-adjusted) --
    fcf_growth_weight: float = 0.7
    revenue_growth_weight: float = 0.3
    downside_exposure_weight: float = 0.35
    scenario_spread_weight: float = 0.25
    terminal_dependency_weight: float = 0.20
    fcf_reliability_weight: float = 0.20
    growth_divergence_threshold: float = 0.10
    min_acceptable_growth: float = 0.10

    # -- Weighted scoring (penalty system) --
    dc_weight: float = 0.7
    mc_weight: float = 0.4
    growth_weight: float = 1.0
    growth_rate_subweight: float = 0.5
    growth_stability_subweight: float = 0.3
    growth_divergence_subweight: float = 0.2

    # -- Watchlist selection (two-step) --
    iv_prefilter_count: int = 100
    target_watchlist_size: int = 40
    min_iv_to_price_ratio: float = 1.0

    # -- Output --
    output_directory: Path = Path("output")
    detailed_report_count: int = 5

    # -- Column specs (structural, rarely changed) --
    table_specs: tuple[TableSpec, ...] = DEFAULT_TABLE_SPECS
    entity_columns: tuple[ColumnSpec, ...] = DEFAULT_ENTITY_COLUMNS
    price_column: ColumnSpec = DEFAULT_PRICE_COLUMN

    # -- Derived properties --

    @property
    def exchanges(self) -> tuple[str, ...]:
        """Exchange codes for the configured market."""
        return MARKET_EXCHANGES[self.market]

    @property
    def currencies(self) -> tuple[str, ...]:
        """Reporting currencies for the configured market."""
        return MARKET_CURRENCIES[self.market]

    @property
    def min_fiscal_year(self) -> int:
        """Earliest fiscal year to load, based on history_years."""
        return datetime.now(UTC).year - self.history_years

    @property
    def max_fiscal_year(self) -> int:
        """Latest fiscal year to load (current year)."""
        return datetime.now(UTC).year

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.market not in MARKET_EXCHANGES:
            raise ValueError(
                f"Unknown market {self.market!r}. "
                f"Must be one of {sorted(MARKET_EXCHANGES.keys())}."
            )

        if self.mode not in ("shortlist", "owned"):
            raise ValueError(
                f"Invalid mode {self.mode!r}. Must be 'shortlist' or 'owned'."
            )

        if self.mode == "owned" and not self.owned_companies:
            raise ValueError("'owned' mode requires a non-empty owned_companies list.")

        if self.period_type not in ("FY", "FQ"):
            raise ValueError(
                f"Invalid period_type {self.period_type!r}. Must be 'FY' or 'FQ'."
            )

        if self.discount_rate <= self.terminal_growth_rate:
            raise ValueError(
                f"discount_rate ({self.discount_rate}) must exceed "
                f"terminal_growth_rate ({self.terminal_growth_rate})."
            )

        if not 0 < self.margin_of_safety < 1:
            raise ValueError(
                f"margin_of_safety must be between 0 and 1 exclusive, "
                f"got {self.margin_of_safety}."
            )

        if self.base_fade_half_life_years <= 0:
            raise ValueError(
                f"base_fade_half_life_years must be positive, "
                f"got {self.base_fade_half_life_years}."
            )

        growth_sub_sum = (
            self.growth_rate_subweight
            + self.growth_stability_subweight
            + self.growth_divergence_subweight
        )
        if abs(growth_sub_sum - 1.0) > 0.01:
            raise ValueError(
                f"Growth sub-weights must sum to 1.0, got {growth_sub_sum:.3f}."
            )

        if abs(self.fcf_growth_weight + self.revenue_growth_weight - 1.0) > 0.01:
            raise ValueError(
                f"fcf_growth_weight + revenue_growth_weight must sum to 1.0, "
                f"got {self.fcf_growth_weight + self.revenue_growth_weight:.3f}."
            )

        risk_weight_sum = (
            self.downside_exposure_weight
            + self.scenario_spread_weight
            + self.terminal_dependency_weight
            + self.fcf_reliability_weight
        )
        if abs(risk_weight_sum - 1.0) > 0.01:
            raise ValueError(
                f"Risk factor weights must sum to 1.0, got {risk_weight_sum:.3f}."
            )
