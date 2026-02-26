# Design Brief: Value Investing Screening Pipeline

Status: **Ready for implementation**

---

## Implementation Workflow

Six phases, one per session. Each session follows the same pattern:

1. `/coord` referencing DESIGN.md Phase N — selects mode, delegates work
2. Implement the modules in that phase
3. `/verify` checks output against DESIGN.md contracts
4. Update TECHNICAL_REFERENCE.md and USER_GUIDE.md
5. `/commit` runs pre-commit checks (ruff, mypy, pytest) and commits

| Session | Phase | Modules | Depends on |
|---------|-------|---------|------------|
| 1 | Foundation | config/settings.py, data/contracts.py, data/loader.py | — (install dev tools first: ruff, mypy, pytest) |
| 2 | Enrichment | derived_metrics.py, growth_stats.py, filtering.py | Phase 1 |
| 3 | Valuation | growth_projection.py, dcf.py, live_prices.py | Phase 2 |
| 4 | Selection | weighted_scoring.py, ranking.py, factor_analysis.py, watchlist.py | Phase 3 |
| 5 | Orchestration | pipeline/runner.py, __main__.py | Phases 1-4 |
| 6 | Visualisation | charts/, notebooks/analysis.ipynb, reports/pdf.py | Phase 5 |

Reference documents: REQUIREMENTS.md (what), DESIGN.md (how),
TECHNICAL_REFERENCE.md (as-built).

---

## 1. Legacy Summary

17 Python modules across 6 directories. Single-script orchestration
(`main_pipeline.py`) calling modules in sequence. Data flows as Python
dicts of dataclass instances (`CompanyData`, `GrowthProjection`,
`IntrinsicValue`, `CompanyAnalysis`) keyed by ticker strings.

Major subsystems:

1. **Data loading** — JSON files -> `CompanyData` dict (CAGR, growth
   stats, derived metrics all computed at load)
2. **Filtering** — 4 financial health filters -> reduced dict
   (non-toggleable)
3. **Growth projection** — Monte Carlo, 10K sims, 44 processes, 20+
   hardcoded constraints -> `GrowthProjection` per
   company/metric/scenario
4. **DCF valuation** — FCF-only, quarterly internal, margin of safety
   -> `IntrinsicValue` per company/period/scenario
5. **Live data** — Yahoo Finance -> live prices + fundamentals, caching,
   anomaly detection
6. **Ranking** — Sharpe-like risk-adjusted score as primary rank, plus
   penalty-based weighted scoring as secondary overlay. Produces 4
   DataFrames.
7. **Reporting** — matplotlib PdfPages + standalone PNGs + CSVs
   (separate entry point)

---

## 2. Target Design

### Package Structure

```
value_investing_screening/
+-- __init__.py
+-- __main__.py                  # CLI entry point
+-- config/
|   +-- __init__.py
|   +-- settings.py              # AnalysisConfig (flat dataclass)
+-- data/
|   +-- __init__.py
|   +-- loader.py                # FMP SQLite data loading
|   +-- live_prices.py           # PriceProvider protocol + implementations
|   +-- contracts.py             # Data contracts
+-- analysis/
|   +-- __init__.py
|   +-- derived_metrics.py       # market_cap, EV, ratios from raw data
|   +-- growth_stats.py          # mean, variance, CAGR from growth series
|   +-- filtering.py             # Toggleable financial health filters
|   +-- growth_projection.py     # Fade-to-equilibrium model
|   +-- dcf.py                   # DCF intrinsic value
|   +-- weighted_scoring.py      # Penalty-based scoring (DC, MC, growth)
|   +-- ranking.py               # Sharpe-like + weighted ranking, 4 DataFrames
|   +-- factor_analysis.py       # Factor contribution analysis
|   +-- watchlist.py             # Two-step watchlist selection
+-- pipeline/
|   +-- __init__.py
|   +-- runner.py                # Pipeline orchestrator
+-- charts/
|   +-- __init__.py
|   +-- market_overview.py       # Market-wide charts (6)
|   +-- comparative.py           # Multi-company comparison charts (9)
|   +-- company_detail.py        # Per-company detail charts (9)
|   +-- tables.py                # Summary table rendering
+-- reports/
|   +-- __init__.py
|   +-- templates/               # Jinja2 HTML templates
|   +-- pdf.py                   # WeasyPrint PDF generation
+-- input/                       # Static reference CSV data
+-- output/                      # Generated outputs (gitignored)
+-- notebooks/
    +-- analysis.ipynb           # Development notebook
```

### Data Flow

```
AnalysisConfig
  |
  v
loader.py -----------> RawFinancialData
  |                     (multi-period DataFrame: one row per entity x quarter)
  |
  +-> growth_stats.py -> GrowthStats (one row per entity: mean, var, std, CAGR)
  |
  +-> derived_metrics.py -> CompanyMetrics (one row per entity: latest-period ratios)
  |
  +-> (kept as time_series for charts)
  |
  v
merge stats + metrics -> per-company DataFrame
  |
  v
filtering.py ----------> filtered per-company DataFrame + FilterLog
  |
  v
growth_projection.py --> projections dict (per entity: period x metric x scenario)
  |
  v
dcf.py ----------------> IVs dict (per entity: period x scenario)
  |
  v
live_prices.py --------> {symbol: current_price}
  |
  v
(compute live metrics) -> updated per-company DataFrame
  |
  v
weighted_scoring.py --> penalty scores per company (DC, MC, growth penalties)
  |
  v
ranking.py ------------> 4 ranking DataFrames (growth, value, weighted, combined)
  |
  v
factor_analysis.py ----> factor contributions + quadrant analysis
  |
  v
watchlist.py ----------> selected symbols
  |
  v
AnalysisResults -------> charts/ (notebook or PDF) + CSV exports
```

---

### Module Interfaces

#### config/settings.py

Flat dataclass (legacy approach). All configurable parameters in a
single `AnalysisConfig`. Phantom parameters removed. Validation cleaned
up.

```python
@dataclass
class AnalysisConfig:
    # -- Market --
    market: str = "AU"              # "US", "AU", "UK", "CA", "NZ", "SG", "HK"
    mode: str = "shortlist"         # "shortlist" or "owned"
    owned_companies: list[str] = field(default_factory=list)

    # -- Data loading --
    db_path: Path = Path("/home/mattm/projects/Pers/financial_db/data/fmp.db")
    period_type: str = "FQ"         # Quarterly
    history_years: int = 5          # -> 20 quarters
    # exchanges, currencies derived from market via property/factory

    # -- Growth projection (fade-to-equilibrium) --
    equilibrium_growth_rate: float = 0.03
    base_fade_half_life_years: float = 2.5
    scenario_band_width: float = 1.0
    negative_fcf_improvement_cap: float = 0.15
    projection_periods: tuple[int, ...] = (5, 10)
    primary_period: int = 5

    # -- DCF --
    discount_rate: float = 0.10
    terminal_growth_rate: float = 0.01
    margin_of_safety: float = 0.50
    quarters_per_year: int = 4

    # -- Filtering --
    enable_negative_fcf_filter: bool = True
    enable_data_consistency_filter: bool = True
    enable_market_cap_filter: bool = True
    enable_debt_cash_filter: bool = True
    min_market_cap: float = 20_000_000
    max_debt_to_cash_ratio: float = 2.5

    # -- Ranking (Sharpe-like risk-adjusted) --
    fcf_growth_weight: float = 0.7
    revenue_growth_weight: float = 0.3
    volatility_risk_weight: float = 0.7
    financial_risk_weight: float = 0.3
    growth_divergence_threshold: float = 0.10
    min_acceptable_growth: float = 0.10

    # -- Weighted scoring (penalty system) --
    dc_weight: float = 0.7              # Debt-cash penalty weight
    mc_weight: float = 0.4              # Market cap penalty weight
    growth_weight: float = 1.0          # Growth penalty weight
    # Growth sub-weights (must sum to 1.0)
    growth_rate_subweight: float = 0.5
    growth_stability_subweight: float = 0.3
    growth_divergence_subweight: float = 0.2

    # -- Watchlist selection (two-step) --
    iv_prefilter_count: int = 100       # Step 1: top N by IV/price ratio
    target_watchlist_size: int = 40     # Step 2: top M by opportunity rank
    min_iv_to_price_ratio: float = 1.0  # For quadrant analysis threshold

    # -- Output --
    output_directory: Path = Path("output")
    detailed_report_count: int = 5

    # -- Derived properties --
    @property
    def exchanges(self) -> tuple[str, ...]: ...
    @property
    def currencies(self) -> tuple[str, ...]: ...
    @property
    def min_fiscal_year(self) -> int: ...
    @property
    def max_fiscal_year(self) -> int: ...
```

Market-specific exchange/currency mappings:

| Market | Exchanges | Currencies |
|--------|-----------|------------|
| US | NASDAQ, NYSE, AMEX | USD |
| AU | ASX | AUD |
| UK | LSE | GBP |
| CA | TSX, CNQ, NEO, TSXV | CAD |
| NZ | NZE | NZD |
| SG | SES | SGD |
| HK | HKSE | HKD |

#### data/loader.py

Adapted from `/home/mattm/projects/Pers/algorithmic-investing/data/loader.py`.

Same pattern: config-driven `TableSpec`/`ColumnSpec`, read-only SQLite
connection, SQL identifier validation, entity filtering, temporal
alignment (common range, gap detection), price alignment (+/-7 day
window), currency filtering, data quality filtering, dropped-companies
CSV logging. Returns `RawFinancialData`.

Table specs for this pipeline:

| TableSpec | Table | Columns | Join |
|-----------|-------|---------|------|
| financials_income | incomeStatement | revenue, operatingIncome, weightedAverageShsOutDil | base |
| financials_balance | balanceSheet | longTermDebt, cashAndCashEquivalents | inner |
| financials_cashflow | cashFlow | freeCashFlow | inner |
| growth_cashflow | cashFlowGrowth | growthFreeCashFlow | left |
| growth_income | incomeStatementGrowth | growthRevenue | left |

Growth tables use left join (missing growth rates = NaN, handled
downstream). Financial tables use inner join (missing fundamentals =
drop company).

Price column: `eodPrice.adjClose`, aligned to period end dates.

Entity columns: `entityId`, `currentSymbol` (mapped to `symbol`),
`companyName`, `exchange`, `country`.

#### data/live_prices.py

Copied from `/home/mattm/projects/Pers/algorithmic-investing/prediction/live_data.py`.

- `PriceProvider` protocol: `get_prices(symbols) -> dict[str, float]`
- `FMPPriceProvider`: batch-quote endpoint, retry + exponential backoff
- `YFinancePriceProvider`: fallback, lazy yfinance import
- `auto_select_provider()`: FMP if `FMP_API_KEY` set, else yfinance

#### data/contracts.py

```python
@dataclass
class RawFinancialData:
    """Output of the data loader."""
    data: pd.DataFrame              # Multi-period: one row per entity x quarter
    query_metadata: dict[str, Any]
    row_count: int
    company_count: int
    period_range: tuple[int, int]   # (min_fiscal_year, max_fiscal_year)
    dropped_companies_path: Path

@dataclass
class FilterLog:
    """Track which companies are removed by each filter."""
    removed: dict[str, list[str]]   # filter_name -> [symbols]
    reasons: dict[str, str]         # symbol -> reason string
    owned_bypassed: list[str]       # Owned companies that bypassed filters
    owned_tracking: dict[str, dict] # Per-owned-company filter pass/fail

@dataclass
class Projection:
    """Growth projection for one company, one metric, one scenario."""
    entity_id: int
    metric: str                     # "fcf" or "revenue"
    period_years: int
    scenario: str                   # "base", "pessimistic", "optimistic"
    quarterly_growth_rates: list[float]
    quarterly_values: list[float]
    annual_cagr: float              # Implied CAGR over horizon
    current_value: float            # Starting value

@dataclass
class IntrinsicValue:
    """DCF valuation result for one scenario."""
    scenario: str
    period_years: int
    projected_annual_cash_flows: list[float]
    terminal_value: float
    present_value: float
    iv_per_share: float
    growth_rate: float
    discount_rate: float
    terminal_growth_rate: float
    margin_of_safety: float

@dataclass
class AnalysisResults:
    """Complete pipeline output."""
    time_series: pd.DataFrame           # Multi-period (for charts)
    companies: pd.DataFrame             # Per-company summary (all metrics)
    projections: dict[int, dict]        # entity_id -> {period: {metric: {scenario: Projection}}}
    intrinsic_values: dict[int, dict]   # entity_id -> {period: {scenario: IntrinsicValue}}
    # 4 ranking DataFrames (legacy structure)
    growth_rankings: pd.DataFrame
    value_rankings: pd.DataFrame
    weighted_rankings: pd.DataFrame     # Penalty-based
    combined_rankings: pd.DataFrame     # Risk-adjusted (primary)
    factor_contributions: pd.DataFrame  # Factor contribution percentages
    factor_dominance: pd.DataFrame      # Factor dominance summary
    quadrant_analysis: pd.DataFrame     # Growth-value quadrant assignments
    watchlist: list[str]                # Selected symbols
    filter_log: FilterLog
    live_prices: dict[str, float]
    config: AnalysisConfig
```

#### analysis/derived_metrics.py

```python
def compute_derived_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Extract latest-period snapshot per company. Compute derived ratios.

    Input: multi-period DataFrame from loader.
    Output: one-row-per-company DataFrame with:
        - Latest-period financials (fcf, revenue, operating_income, etc.)
        - market_cap = price x shares_diluted
        - enterprise_value = market_cap + lt_debt - cash
        - debt_cash_ratio = lt_debt / cash (inf if cash = 0)
        - fcf_per_share = fcf / shares_diluted
        - acquirers_multiple = enterprise_value / operating_income (inf if OI = 0)
        - fcf_to_market_cap = fcf / market_cap
    """
```

#### analysis/growth_stats.py

```python
def compute_growth_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Compute per-company growth statistics from time series.

    Input: multi-period DataFrame with fcf_growth and revenue_growth columns.
    Output: one-row-per-company DataFrame with:
        - fcf_growth_mean, fcf_growth_var, fcf_growth_std
        - revenue_growth_mean, revenue_growth_var, revenue_growth_std
        - fcf_cagr, revenue_cagr (annualised, legacy algorithm)
        - combined_growth_mean = fcf_mean * fcf_weight + rev_mean * rev_weight
        - growth_stability = 1 / (1 + avg_std)  -- bounded [0, 1]
          where avg_std = mean(fcf_growth_std, revenue_growth_std).
          Replaces legacy CV-of-simulation-paths formula.

    CAGR algorithm (from legacy):
        - Filter out None/NaN/zero values
        - Require min 4 quarters between valid points
        - Return 0.0 on sign change
        - Quarterly CAGR then annualise by compounding 4 quarters
        - Special sign preservation for both-negative cases

    Growth stats use population variance (ddof=0) with minimum 3
    data points required per metric.
    """
```

#### analysis/filtering.py

```python
def apply_filters(
    companies: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, FilterLog]:
    """Apply individually toggleable financial health filters.

    Filters (each controlled by config.enable_*):
    1. Negative/zero FCF (fcf <= 0)
    2. Data consistency (operating_income > revenue, only when revenue > 0)
    3. Minimum market cap (< config.min_market_cap)
    4. Maximum debt-to-cash ratio (> config.max_debt_to_cash_ratio)

    In owned mode, owned companies bypass active filters with failures tracked.
    Post-filter, sorted by market cap descending.
    """
```

#### analysis/growth_projection.py

```python
def project_growth(
    entity_id: int,
    fcf_stats: dict,        # mean, std, latest_value, latest_revenue_growth
    revenue_stats: dict,     # mean, std, latest_value
    market_cap: float,
    config: AnalysisConfig,
) -> dict[int, dict[str, dict[str, Projection]]]:
    """Fade-to-equilibrium growth projection.

    Returns {period: {metric: {scenario: Projection}}}

    Model: g(t) = g_eq + (g_0 - g_eq) * exp(-lambda * t)
    - g_0 = mean historical quarterly growth rate
    - g_eq = config.equilibrium_growth_rate (annual, converted to quarterly
      internally: g_eq_q = (1 + g_eq)^(1/4) - 1)
    - lambda = ln(2) / (half_life_years * 4), adjusted for company size
      (half_life converted from years to quarters)

    Size adjustment: lambda = lambda_base * (1 + size_adj)
    where size_adj is log-scaled from market cap.

    Three scenarios per metric per period:
    - base: fade from g_0
    - optimistic: fade from g_0 + k * sigma
    - pessimistic: fade from g_0 - k * sigma

    Negative FCF handling:
    - Growth rates meaningless on negative values
    - Linear improvement toward zero at rate derived from revenue growth
    - Rate = min(cap, revenue_growth * scaling_factor) per quarter
    - If revenue also declining, use slower default improvement rate
    - Once FCF crosses zero, switch to standard fade with conservative g_0

    Value(t) = Value(0) * prod(1 + g(q)) for each quarter q = 1..t
    """

def project_all(
    companies: pd.DataFrame,
    growth_stats: pd.DataFrame,
    config: AnalysisConfig,
) -> dict[int, dict]:
    """Project growth for all companies."""
```

#### analysis/dcf.py

Carries forward legacy DCF formulae. Interface changed: receives a
`Projection` object with quarterly growth rates from the fade model,
rather than a single annual growth rate from Monte Carlo.

```python
def calculate_dcf(
    base_fcf: float,
    shares: float,
    projection: Projection,
    config: AnalysisConfig,
) -> IntrinsicValue:
    """FCF-only DCF with quarterly internal discounting.

    Quarterly rates:
        quarterly_growth = (1 + annual_growth) ^ (1/4) - 1
        quarterly_discount = (1 + discount_rate) ^ (1/4) - 1

    Cash flow projection:
        For each quarter t = 1..N:
            if current >= 0: current *= (1 + quarterly_growth)
            else (negative FCF, positive growth):
                improvement = min(0.25, abs(growth_rate) / 4)
                current *= (1 - improvement)

    Terminal value:
        final_annual_cf = final_quarterly_cf * 4
        terminal = final_annual_cf * (1 + terminal_growth) / (discount - terminal_growth)

    Present value:
        PV = sum(cf_q / (1 + quarterly_discount)^q) + terminal / (1 + discount)^years

    IV per share:
        PV * (1 - margin_of_safety) / shares
    """

def calculate_all_dcf(
    companies: pd.DataFrame,
    projections: dict[int, dict],
    config: AnalysisConfig,
) -> dict[int, dict]:
    """Calculate DCF for all companies, all periods, all scenarios."""
```

#### analysis/weighted_scoring.py

Penalty-based scoring (legacy approach). Fixes: all weights
configurable via config. Penalty weights (dc_weight, mc_weight,
growth_weight) are unconstrained — no normalisation applied (legacy's
arbitrary sum=2.1 constraint removed). Growth sub-weights configurable
(were hardcoded 0.5/0.3/0.2; should sum to 1.0).

```python
def calculate_weighted_scores(
    companies: pd.DataFrame,
    growth_stats: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Calculate penalty-based weighted scores.

    Three penalty types (lower = better):

    1. DC penalty = (abs(debt_cash_ratio) ** 2) * dc_weight
       Squared ratio to heavily penalise high debt.
       abs() retained from legacy for defensive robustness.
       (inf penalty if cash = 0)

    2. MC penalty = log10(market_cap / min_market_cap) * mc_weight
       Logarithmic size penalty — reflects diminishing growth
       potential at scale. Zero if below min_market_cap.

    3. Growth penalty (3 configurable sub-components):
       a. Rate penalty: if avg_growth < min_acceptable_growth,
          penalty = (threshold - avg_growth) * growth_weight * rate_subweight
       b. Stability penalty: std(fcf_growth, rev_growth) * growth_weight * stability_subweight
       c. Divergence penalty: |fcf_growth - rev_growth| * growth_weight * divergence_subweight

    Total penalty = dc_penalty + mc_penalty + growth_penalty
    Weighted rank = rank by total_penalty ascending (lower = better)

    Output: DataFrame indexed by ticker with dc_penalty, mc_penalty,
    growth_penalty, total_penalty, weighted_rank columns.
    """
```

#### analysis/factor_analysis.py

Factor contribution analysis and quadrant analysis (legacy approach).

```python
def calculate_factor_contributions(
    weighted_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate percentage contribution of each penalty factor.

    For each company: factor_pct = factor_penalty / total_penalty * 100
    Output: DataFrame with dc_pct, mc_pct, growth_pct per company.
    (Column names shortened from legacy dc_penalty_pct, mc_penalty_pct,
    growth_penalty_pct.)
    """

def analyze_factor_dominance(
    contributions: pd.DataFrame,
    weighted_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Identify which factor dominates each company's penalty.

    Output: Summary showing primary_factor, company_count, pct_of_total,
    avg_contribution, avg_total_penalty.
    """

def create_quadrant_analysis(
    companies: pd.DataFrame,
    intrinsic_values: dict[int, dict],
    live_prices: dict[str, float],
    growth_stats: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Assign companies to growth-value quadrants.

    Quadrant 1 (best): high growth + high value (IV/price >= threshold)
    Quadrant 2: high growth + low value
    Quadrant 3: low growth + high value
    Quadrant 4 (worst): low growth + low value

    Thresholds: min_acceptable_growth, min_iv_to_price_ratio from config.
    """
```

#### analysis/ranking.py

Combines Sharpe-like risk-adjusted scoring with penalty-based weighted
scoring. Produces 4 ranking DataFrames (carried forward from legacy
with modifications).

```python
def rank_companies(
    companies: pd.DataFrame,
    growth_stats: pd.DataFrame,
    projections: dict[int, dict],
    intrinsic_values: dict[int, dict],
    weighted_scores: pd.DataFrame,
    live_prices: dict[str, float],
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rank companies using both risk-adjusted and weighted approaches.

    Returns (growth_rankings, value_rankings, weighted_rankings, combined_rankings).

    For each company with a live price:

    --- Risk-adjusted scoring (primary, for combined_rankings) ---

    1. Combined growth = fcf_growth * fcf_weight + rev_growth * rev_weight
       (projected annual growth rates from fade model, base scenario)

    2. Best IV/price ratio across all periods and scenarios

    3. Expected return = combined_growth + annualised_valuation_return
       where valuation_return = (1 + (iv_ratio - 1)) ^ (1/period) - 1

    4. Volatility risk = weighted std dev of growth rates (annualised)
       fcf_std * fcf_weight + rev_std * rev_weight
       (quarterly std * 2 for annualisation)

    5. Financial risk = min(debt_cash_ratio / max_debt_to_cash_ratio, 1.0)

    6. Combined risk = max(volatility * vol_weight + financial * fin_weight, 0.1)
       (weights configurable, default 0.7/0.3; 0.1 floor prevents div/zero)

    7. Risk-adjusted score = expected_return / combined_risk

    8. Growth divergence = |fcf_growth - rev_growth|
       (flagged when > threshold)

    9. Growth stability = 1 / (1 + avg_std)
       where avg_std = mean(fcf_growth_std, revenue_growth_std).
       Uses historical std (replaces legacy CV-of-simulation-paths).

    --- Penalty scoring (for weighted_rankings, via weighted_scoring.py) ---

    10. DC, MC, growth penalties -> total_penalty -> weighted_rank

    --- Four output DataFrames ---

    growth_rankings: sorted by combined_growth desc
        + fcf_growth_rank, revenue_growth_rank, combined_growth_rank, stability_rank

    value_rankings: sorted by best_iv_ratio desc
        + value_rank

    weighted_rankings: sorted by total_penalty asc
        + weighted_rank, dc_penalty, mc_penalty, growth_penalty

    combined_rankings: sorted by risk_adjusted_rank asc
        + risk_adjusted_rank, opportunity_rank (= risk_adjusted_rank),
          opportunity_score (= 100 - rank/n * 100)
        + growth_score, value_score, weighted_score, stability_score
          (normalised rank-based scores, 0-100)
        + divergence_penalty (binary: 0 or 10, from divergence_flag)

    All four DataFrames share the same base columns:
        ticker, symbol (new), entity_id (new), current_price,
        fcf_growth_annual, revenue_growth_annual, combined_growth,
        best_iv_ratio, best_iv_scenario,
        volatility_risk, financial_risk, combined_risk,
        total_expected_return, risk_adjusted_score,
        growth_stability, growth_divergence, divergence_flag,
        fcf, market_cap, debt_cash_ratio (new), total_penalty, avg_volatility
    """
```

#### analysis/watchlist.py

Two-step selection (legacy approach).

```python
def select_watchlist(
    rankings: pd.DataFrame,
    config: AnalysisConfig,
) -> list[str]:
    """Select watchlist using two-step approach.

    Step 1: Take top iv_prefilter_count companies by best_iv_ratio.
    Step 2: From that set, take top target_watchlist_size by opportunity_rank.

    In owned mode: owned companies are always included regardless of
    rank, plus fill remaining slots from the ranked list.

    Returns symbols sorted by opportunity_rank.
    """
```

#### pipeline/runner.py

```python
def run_analysis(config: AnalysisConfig) -> AnalysisResults:
    """Execute the 12-step analysis pipeline.

    1. Validate config (discount > terminal, growth sub-weights sum to 1.0, etc.)
    2. Load data (FMP SQLite via loader)
    3. Compute derived metrics (latest period: market_cap, EV, ratios)
    4. Compute growth statistics (per-company: mean, var, std, CAGR)
    5. Filter (4 toggleable filters + owned bypass)
    6. Growth projection (fade-to-equilibrium, 3 scenarios x N periods)
    7. DCF intrinsic value (FCF-only, quarterly, 3 scenarios x N periods)
    8. Fetch live prices (FMP API primary, yfinance fallback)
    9. Compute live metrics (market_cap, EV, acquirers_multiple from
       live price + DB fundamentals)
    10. Rank (Sharpe-like risk-adjusted + penalty-based weighted scoring
        -> 4 ranking DataFrames)
    11. Additional analysis (factor contributions, quadrant analysis)
    12. Select watchlist (two-step: IV pre-filter then opportunity rank)

    Returns AnalysisResults with all data for charts and reports.
    """
```

Entry points:
- CLI: `python -m value_investing_screening --market US --mode shortlist`
- Notebook: `config = AnalysisConfig(market="US", mode="shortlist"); results = run_analysis(config)`
- PDF report: separate entry point consuming `AnalysisResults`

---

## 3. Keep / Rewrite / Delete

| Legacy Module | Action | Target | Notes |
|---|---|---|---|
| config/user_config.py | Rewrite | config/settings.py | Flat dataclass. Remove 4 phantom params (MOMENTUM_EXHAUSTION/BOUNCE_THRESHOLD, REALISTIC_MAX/MIN_QUARTERLY_GROWTH). Remove 2 Monte Carlo-specific params (MAX/MIN_QUARTERLY_GROWTH) made obsolete by fade model. Remove weight-sum=2.1 validation. Add market-derived properties. |
| data_pipeline/data_loading.py | Replace | data/loader.py | New data source (FMP SQLite). Adapt reference loader pattern. |
| data_pipeline/data_schema.py | Delete | Absorbed into contracts.py | Stale entries, static validation unnecessary. |
| data_pipeline/data_structures.py | Rewrite | data/contracts.py + analysis dataclasses | CompanyData becomes a DataFrame row. Projections/IVs are new dataclasses. |
| data_pipeline/filtering.py | Rework | analysis/filtering.py | Add per-filter toggles. Remove no-op `apply_growth_filters` (called but performs no filtering) and unused `create_filtered_dataframe`. |
| data_pipeline/ticker_mapping.py | Delete | N/A | Only served Yahoo Finance. Irrelevant with FMP. |
| data_sources/yahoo_finance.py | Replace | data/live_prices.py | Copy PriceProvider from reference project. FMP primary, yfinance fallback. |
| analysis/growth_calculation.py | Rewrite | analysis/growth_projection.py | Monte Carlo -> fade-to-equilibrium. ~716 -> ~200 lines. |
| analysis/intrinsic_value.py | Keep + cleanup | analysis/dcf.py | Same DCF logic. New dataclasses as input. Remove verbose comments. |
| analysis/ranking.py | Rework | analysis/ranking.py | Keep Sharpe-like + penalty-based. 4 DataFrames. Make all weights configurable. |
| analysis/weighted_scoring.py | Rework | analysis/weighted_scoring.py | Keep penalty system (DC squared, MC log10, growth sub-penalties). Fix: all weights configurable, remove arbitrary sum=2.1 constraint, configurable growth sub-weights. `create_quadrant_analysis` moves to factor_analysis.py. |
| analysis/factor_analysis.py | Rework | analysis/factor_analysis.py | Keep factor contributions. Absorb `create_quadrant_analysis` from weighted_scoring.py. Updated to use configurable weights. |
| reports/report_generator.py | Rework | reports/pdf.py | matplotlib PdfPages -> Jinja2+WeasyPrint. |
| reports/report_visualizations.py | Rework | charts/ (3 modules) | 2,027-line monolith -> market_overview.py, comparative.py, company_detail.py. |
| reports/visualization.py | Delete | Absorbed into charts/ | Thin wrapper, 69 lines, indentation bug. |
| reports/methodology_generator.py | Rework | Jinja2 template | matplotlib text rendering -> HTML template. |
| scripts/main_pipeline.py | Rewrite | pipeline/runner.py | Clean orchestration. Logging not print. No hardcoded pre-filter. |

---

## 4. Migration Strategy (Build Order)

Six phases. Each independently testable. Dependencies flow downward.

### Phase 1: Foundation

No internal dependencies. Everything else depends on these.

1. **config/settings.py** — `AnalysisConfig` dataclass. Market-derived
   properties (exchanges, currencies, fiscal year range). Column mapping
   specs (TableSpec/ColumnSpec). Validation.
2. **data/contracts.py** — `RawFinancialData`, `FilterLog`,
   `Projection`, `IntrinsicValue`, `AnalysisResults`.
3. **data/loader.py** — Adapted from reference. Connect to FMP SQLite,
   load 5 table specs (income, balance, cashflow, cashflow growth,
   income growth), join, filter, align prices, return
   `RawFinancialData`.

**Test:** Load data for AU market. Verify row counts, column names,
no NaN in required columns. Spot-check a known company's values
against manual DB query.

### Phase 2: Enrichment

Depends on Phase 1.

4. **analysis/derived_metrics.py** — Latest-period extraction. Computed
   ratios (market_cap, EV, debt_cash_ratio, etc.).
5. **analysis/growth_stats.py** — Per-company mean, variance, std,
   CAGR. Growth stability score.
6. **analysis/filtering.py** — 4 toggleable filters + owned-company
   bypass.

**Test:** Verify derived metrics against manual calculation for 3
companies. Verify CAGR matches legacy algorithm on test cases (sign
change, < 4 quarters, both-negative). Verify each filter toggle works
independently. Verify owned-company bypass.

### Phase 3: Valuation

Depends on Phase 2.

7. **analysis/growth_projection.py** — Fade-to-equilibrium model.
8. **analysis/dcf.py** — DCF intrinsic value.
9. **data/live_prices.py** — Copied from reference project.

**Test:** Verify fade converges to g_eq as t -> infinity. Verify
scenarios bracket base case (pessimistic < base < optimistic). Verify
negative FCF transitions to positive. Verify DCF output matches legacy
for identical inputs (same growth rate, same base FCF, same config).
Verify PriceProvider returns valid prices.

### Phase 4: Selection

Depends on Phase 3.

10. **analysis/weighted_scoring.py** — Penalty-based scoring (DC, MC,
    growth). Configurable weights.
11. **analysis/ranking.py** — Sharpe-like risk-adjusted scoring +
    weighted scoring integration. Produces 4 DataFrames.
12. **analysis/factor_analysis.py** — Factor contributions, factor
    dominance, quadrant analysis.
13. **analysis/watchlist.py** — Two-step selection.

**Test:** Verify penalty scores change with weight config. Verify
ranking is monotone in risk_adjusted_score. Verify opportunity_rank =
risk_adjusted_rank. Verify 4 DataFrames have correct sort orders.
Verify factor contributions sum to ~100%. Verify quadrant assignments
match thresholds. Verify two-step selection (IV pre-filter then
opportunity rank). Verify owned-company inclusion. Verify watchlist
size <= target.

### Phase 5: Orchestration

Depends on Phases 1-4.

14. **pipeline/runner.py** — 12-step pipeline.
15. **__main__.py** — CLI entry point.

**Test:** Full pipeline for US and AU markets. Verify `AnalysisResults`
has all expected fields populated. Verify CSV exports. Verify logging
output (no print statements).

### Phase 6: Visualisation

Depends on Phase 5.

16. **charts/** — All chart functions. Shared between notebook and PDF.
17. **notebooks/analysis.ipynb** — Thin harness (config cell, pipeline
    call, one cell per chart).
18. **reports/pdf.py** — Jinja2+WeasyPrint PDF generation.
19. **Methodology template** — Jinja2 HTML.

**Test:** Notebook runs end-to-end. All chart functions produce valid
matplotlib figures. PDF generates without errors.

---

## 5. Edge Cases and Error Handling

| Situation | Handling |
|---|---|
| Company has < 4 quarters of growth data | CAGR returns 0.0 (legacy). Growth stats computed from available data. |
| Company has negative FCF | Passes through loader. Filtered by default (`enable_negative_fcf_filter=True`). If filter disabled: fade model uses revenue-derived improvement rate toward zero. |
| Company has zero cash | debt_cash_ratio = inf. Filtered by debt/cash filter if enabled. |
| Growth rate series all NaN | Company dropped by loader NaN-in-required-columns filter (growth tables use left join, so NaN growth = loader passes through, but derived_metrics requires FCF/revenue). |
| Live price unavailable for a company | Excluded from ranking (no IV/price ratio). Logged as warning. |
| Selected price provider fails for some symbols | live_prices dict is empty for affected symbols. Pipeline continues with available prices. Warning logged. (`auto_select_provider` picks one provider at startup, not both.) |
| Division by zero in metrics | operating_income = 0 -> acquirers_multiple = inf. shares = 0 -> caught by loader validation (required column). cash = 0 -> debt_cash_ratio = inf. |
| Growth rates span sign change | CAGR returns 0.0 (legacy). Fade model starts from g_0 (mean growth rate, which can be near zero). |
| equilibrium_growth_rate > g_0 | Fade projects growth increasing toward equilibrium. Valid -- below-equilibrium companies mean-revert upward. |
| Market with few companies passing filters | Pipeline proceeds with whatever passes. Watchlist may be smaller than target. Warning logged. |
| Config validation failures | Hard failure with descriptive ValueError. discount_rate <= terminal_growth_rate, weight sums != 1.0, unsupported market, etc. |
| Large market (US: ~3,500 entities) | Sequential processing. Loader handles large queries efficiently (parameterised SQL, no ORM overhead). Rate-limited live price fetching. |

---

## 6. Verification

### Unit Tests (per module)

- **growth_stats.py**: Known input series -> expected mean, variance,
  CAGR. Edge cases: all zeros, sign change, < 4 points, both-negative.
- **derived_metrics.py**: Known financials -> expected ratios. Division
  by zero cases.
- **filtering.py**: Each filter toggle on/off. Owned-company bypass.
  Correct removal reasons in FilterLog.
- **growth_projection.py**: Fade converges to g_eq. Scenarios bracket
  base. Negative FCF transitions. Size adjustment increases lambda.
- **dcf.py**: Known projection -> expected PV, terminal value, IV/share.
  Compare against hand calculation. Compare against legacy output for
  same inputs.
- **weighted_scoring.py**: Penalty scores change with weight config.
  DC penalty squared, MC penalty logarithmic. Growth sub-weights sum
  to 1.0. Companies with zero cash get inf DC penalty.
- **ranking.py**: Monotone in risk_adjusted_score. 4 DataFrames have
  correct sort orders. Weights configurable. Companies without live
  price excluded.
- **factor_analysis.py**: Factor contributions sum to ~100%.
  Quadrant assignments match thresholds.
- **watchlist.py**: Two-step selection correct. Owned-company inclusion.
  Watchlist size.

### Integration Tests

- Full pipeline for a small test set (~10 companies).
- Verify `AnalysisResults` has all expected fields populated.
- Verify CSV export matches `AnalysisResults` content.
- Round-trip: config -> pipeline -> results -> charts (no errors).

### Regression Comparison

- Run new pipeline on FMP data.
- Spot-check: for companies present in both old and new runs, verify
  that DCF values are within expected tolerance (not exact match due to
  different growth model).
- Verify top-20 overlap is reasonable (many of the same companies
  should appear, though ranks will differ).

---

## 7. Decisions Log

All decisions resolved. No remaining decisions require human input.

| # | Decision | Resolution | Source |
|---|---|---|---|
| 1 | Ranking system | Keep both: Sharpe-like risk-adjusted (primary, for combined_rankings and watchlist) + penalty-based weighted scoring (secondary, for weighted_rankings). 4 DataFrames output. Keep factor analysis and quadrant analysis. Fix: all weights configurable, remove sum=2.1 constraint, configurable growth sub-weights. | Legacy ranking.py, weighted_scoring.py, factor_analysis.py |
| 2 | Config structure | Flat dataclass (legacy approach). Remove phantom params. Add market-derived properties. | Legacy user_config.py |
| 3 | Watchlist selection | Two-step: top N by IV/price ratio, then top M by opportunity rank. Both N and M configurable (default 100/40). | Legacy main_pipeline.py:167-168 |
| 4 | FMP column mapping | Finalised. longTermDebt, computed market_cap/EV, pre-computed growth tables. | REQUIREMENTS.md |
| 5 | Growth projection | Deterministic fade-to-equilibrium. 5 parameters. | REQUIREMENTS.md |
| 6 | Parallelisation | None. Sequential pipeline. | REQUIREMENTS.md |
| 7 | PDF engine | Jinja2 + WeasyPrint. Matplotlib for charts only. | REQUIREMENTS.md |
| 8 | Notebook workflow | Thin harness. Config -> pipeline call -> chart calls. | REQUIREMENTS.md |

---

## 8. Chart Migration Notes

All legacy chart types are kept. Data sources updated to use new
pipeline output.

| Legacy Chart | Status | Data Source |
|---|---|---|
| Factor contributions stacked bar | Keep | factor_contributions DataFrame (dc_pct, mc_pct, growth_pct — shortened from legacy dc_penalty_pct etc.) |
| Factor heatmap | Keep (use matplotlib, not seaborn) | factor_contributions DataFrame |
| Weighted rankings table | Keep | weighted_rankings DataFrame |
| Quadrant analysis (growth vs value) | Keep | quadrant analysis from factor_analysis.py |
| All other charts | Keep | AnalysisResults fields |

Chart functions receive `AnalysisResults` and extract what they need.
Same functions used by notebook cells and PDF template engine.

---

## 9. Reference Files

These files from the algorithmic-investing project are the basis for
adaptation/copying:

| File | Purpose | Adaptation |
|---|---|---|
| `/home/mattm/projects/Pers/algorithmic-investing/data/loader.py` (1,145 lines) | FMP data loading | Adapt column specs, table specs, entity filters for value investing pipeline |
| `/home/mattm/projects/Pers/algorithmic-investing/prediction/live_data.py` (235 lines) | Live price fetching | Copy as-is (PriceProvider protocol + FMP/yfinance implementations) |

---

## 10. Pipeline Issue Fixes

Fixes for issues found in pipeline output review. Reference: `ISSUES.md`
for full issue descriptions, evidence, and decisions.

Four phases, dependency-ordered. Each phase is independently committable
and can be picked up in a separate session.

| Phase | Scope | Files | Depends on |
|-------|-------|-------|------------|
| 7A | TTM growth rates | growth_stats.py | — |
| 7B | Ranking overhaul | ranking.py, settings.py, contracts.py | 7A |
| 7C | Chart fixes | market_overview.py, comparative.py, company_detail.py, analysis.ipynb | 7B |
| 7D | Documentation | TECHNICAL_REFERENCE.md, USER_GUIDE.md | 7C |

---

### Phase 7A: TTM Growth Rates (Issue 2)

**File:** `pipeline/analysis/growth_stats.py`

Root cause fix. Raw quarterly FCF/revenue growth rates have extreme
volatility (e.g. MAT.AX std ~257), producing astronomical starting
rates in the fade model's `mean + k * sigma` formula.

**Change:** Replace raw quarterly growth with TTM (trailing twelve
month) growth:

1. At each quarter, compute TTM values: `TTM_FCF(q) = sum(FCF[q-3..q])`,
   same for revenue.
2. Compute growth of the TTM series:
   `ttm_growth(q) = (TTM(q) - TTM(q-1)) / abs(TTM(q-1))`
3. Compute mean/var/std from the TTM growth series.

The first 4 quarters are consumed forming the initial TTM window.
With 20 quarters of history this yields ~16 TTM growth observations —
enough for meaningful statistics while eliminating quarterly lumpiness.

**Unblocks:** Issues 3 (pessimistic > base IV), 6 (projected growth
"no data"), 9 (growth vs stability all outliers).

---

### Phase 7B: Ranking Overhaul (Issues 1, 7)

**Files:** `pipeline/analysis/ranking.py`, `pipeline/config/settings.py`,
`pipeline/data/contracts.py` (if AnalysisResults needs new fields)

Two changes to the ranking algorithm:

#### Issue 1 — Composite IV ratio

Replace `_best_iv_ratio()` (cherry-picks highest IV/price across all
scenarios and periods) with weighted composite at primary period:

```
composite_iv_ratio = 0.25 * pessimistic + 0.50 * base + 0.25 * optimistic
```

Safety gate: filter companies where `pessimistic_iv < current_price`
from ranking.

Output must expose: composite, base, pessimistic, and optimistic IV
ratios alongside each other.

#### Issue 7 — Risk framework

Replace `volatility_risk` (fcf_std * 2 * 0.7 + rev_std * 2 * 0.3) and
`combined_risk` (volatility * 0.7 + financial * 0.3) with four
value-investing risk factors.

**Raw risk factors:**

| Factor | Computation | Interpretation |
|--------|-------------|----------------|
| Scenario spread | `(optimistic_iv - pessimistic_iv) / base_iv` | Projection confidence. Wide = unreliable base case. |
| Downside exposure | `pessimistic_iv / current_price` | Capital protection. Pessimistic IV already includes 50% margin of safety from DCF. |
| Terminal value dependency | `terminal_pv / (terminal_pv + projected_cf_pv)` from base-case DCF | Valuation speculativeness. High = rests on far-future assumptions. |
| FCF reliability | Quarters with positive FCF / total quarters in time series | Business cash generation consistency. Range [0, 1]. |

**Normalisation (all to [0, 1] where 1 = safest):**

| Factor | Raw: higher = | Normalised score | Direction |
|--------|---------------|------------------|-----------|
| Scenario spread | worse | `1 / (1 + spread)` | Inverted |
| Downside exposure | better | `min(value, 2.0) / 2.0` | Kept, capped |
| Terminal value dependency | worse | `1 - terminal_pct` | Inverted |
| FCF reliability | better | As-is | Kept |

**Composite risk score (for ranking):**

Weighted linear combination of normalised scores:

| Factor | Weight |
|--------|--------|
| Downside exposure | 35% |
| Scenario spread | 25% |
| Terminal value dependency | 20% |
| FCF reliability | 20% |

Higher composite = safer. Replaces `combined_risk` in:
`risk_adjusted_score = expected_return / combined_risk`

Note: since higher composite now = safer (lower risk), the division
needs adjustment — either invert the composite or restructure the
formula so that safer companies score higher.

Config: add risk factor weights as configurable parameters.

---

### Phase 7C: Chart Fixes (Issues 4, 5, 7-vis, 11)

**Files:** `pipeline/charts/market_overview.py`,
`pipeline/charts/comparative.py`,
`pipeline/charts/company_detail.py`,
`notebooks/analysis.ipynb`

Four independent changes (can be implemented in parallel):

#### Issue 4 — Growth vs Value scatter (market_overview.py)

`growth_value_scatter()`: plot all companies as unlabelled dots for
context. Only annotate labels on watchlist companies.

#### Issue 5 — Acquirer's multiple chart (comparative.py)

`acquirers_multiple_analysis()`: floor y-axis at -10. Companies with
actual AM below -10 get their true value annotated as text next to the
capped position. Violin/distribution still shown for the visible range.

#### Issue 7 — Risk visualisation

**Market level** — `risk_adjusted_opportunity()` in market_overview.py:
Replace average volatility axis with composite risk score.

**Company detail** — new spider chart in company_detail.py:
Four axes, one per normalised safety score (higher = safer):

| Axis | Score |
|------|-------|
| Downside exposure | `min(pessimistic_iv / price, 2.0) / 2.0` |
| Scenario spread | `1 / (1 + normalised_spread)` |
| Terminal value dependency | `1 - terminal_pct` |
| FCF reliability | positive_quarters / total_quarters |

Risk factors only — no mixing of opportunity metrics.

#### Issue 11 — Duplicate charts (analysis.ipynb)

Add `plt.close(fig)` after `display(fig)` in notebook loop cells, or
suppress return value with semicolons.

---

### Phase 7D: Documentation

**Files:** `TECHNICAL_REFERENCE.md`, `USER_GUIDE.md`

Update both documents with:

- Exact risk factor definitions and formulas
- Composite risk score weights and rationale
- Normalisation formulas (raw → safety score)
- Distinction between margin of safety (50% DCF discount, uniform) and
  downside exposure (pessimistic scenario position after all discounts)
- Composite IV ratio formula (25/50/25) and pessimistic safety gate
- TTM growth computation method
