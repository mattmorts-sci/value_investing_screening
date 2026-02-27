# Technical Reference

## Package Structure

```
pipeline/
├── __init__.py
├── __main__.py                  # CLI entry point
├── runner.py                    # 12-step pipeline orchestrator
├── config/
│   ├── __init__.py
│   └── settings.py              # AnalysisConfig, ColumnSpec, TableSpec
├── data/
│   ├── __init__.py
│   ├── contracts.py             # Data contracts (RawFinancialData, FilterLog, etc.)
│   ├── loader.py                # FMP database loader
│   └── live_prices.py           # PriceProvider protocol + FMP/yfinance
├── analysis/
│   ├── __init__.py
│   ├── derived_metrics.py       # Latest-period snapshot + derived ratios
│   ├── growth_stats.py          # Per-company growth statistics + CAGR
│   ├── filtering.py             # Toggleable financial health filters
│   ├── growth_projection.py     # Fade-to-equilibrium growth model
│   ├── dcf.py                   # DCF intrinsic value calculation
│   ├── weighted_scoring.py      # Penalty-based scoring (DC, MC, growth)
│   ├── ranking.py               # Sharpe-like + weighted ranking, 4 DataFrames
│   ├── factor_analysis.py       # Factor contributions + quadrant analysis
│   └── watchlist.py             # Two-step watchlist selection
├── charts/
│   ├── __init__.py              # Public API re-exports
│   ├── market_overview.py       # Market-wide charts (6)
│   ├── comparative.py           # Multi-company comparison charts (5)
│   ├── company_detail.py        # Per-company detail charts (7)
│   └── tables.py                # Summary table rendering (2)
└── reports/
    ├── __init__.py
    ├── pdf.py                   # Jinja2+WeasyPrint PDF generation
    └── templates/
        └── report.html          # HTML template for PDF report
notebooks/
└── analysis.ipynb               # Development notebook
tests/
├── __init__.py
├── test_settings.py
├── test_contracts.py
├── test_loader.py
├── test_derived_metrics.py
├── test_growth_stats.py
├── test_filtering.py
├── test_growth_projection.py
├── test_dcf.py
├── test_live_prices.py
├── test_weighted_scoring.py
├── test_ranking.py
├── test_factor_analysis.py
├── test_watchlist.py
├── test_runner.py
├── test_charts.py
└── test_pdf.py
```

## Modules

### pipeline.config.settings

Central configuration. All configurable values live here.

**Classes:**

`ColumnSpec` (frozen dataclass) — maps a database column to a pipeline
column name.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| db_column | str | — | Column name in FMP database |
| internal_name | str | — | Column name in pipeline DataFrames |
| required | bool | True | If True, NaN triggers company drop |

`TableSpec` (frozen dataclass) — specification for loading one FMP
table.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| table_name | str | — | FMP table name |
| columns | tuple[ColumnSpec, ...] | — | Column mappings |
| join_type | str | "inner" | "inner" or "left" |

`AnalysisConfig` (mutable dataclass) — all pipeline parameters.

| Field | Type | Default |
|-------|------|---------|
| market | str | "AU" |
| mode | str | "shortlist" |
| owned_companies | list[str] | [] |
| db_path | Path | /home/mattm/projects/Pers/financial_db/data/fmp.db |
| period_type | str | "FQ" |
| history_years | int | 5 |
| price_alignment_days | int | 7 |
| common_range_lower_pct | float | 0.10 |
| common_range_upper_pct | float | 0.90 |
| equilibrium_growth_rate | float | 0.03 |
| base_fade_half_life_years | float | 2.5 |
| scenario_band_width | float | 1.0 |
| negative_fcf_improvement_cap | float | 0.15 |
| projection_periods | tuple[int, ...] | (5, 10) |
| primary_period | int | 5 |
| discount_rate | float | 0.10 |
| terminal_growth_rate | float | 0.01 |
| margin_of_safety | float | 0.50 |
| quarters_per_year | int | 4 |
| enable_negative_fcf_filter | bool | True |
| enable_data_consistency_filter | bool | True |
| enable_market_cap_filter | bool | True |
| enable_debt_cash_filter | bool | True |
| min_market_cap | float | 20,000,000 |
| max_debt_to_cash_ratio | float | 2.5 |
| fcf_growth_weight | float | 0.7 |
| revenue_growth_weight | float | 0.3 |
| downside_exposure_weight | float | 0.35 |
| scenario_spread_weight | float | 0.25 |
| terminal_dependency_weight | float | 0.20 |
| fcf_reliability_weight | float | 0.20 |
| growth_divergence_threshold | float | 0.10 |
| min_acceptable_growth | float | 0.10 |
| dc_weight | float | 0.7 |
| mc_weight | float | 0.4 |
| growth_weight | float | 1.0 |
| growth_rate_subweight | float | 0.5 |
| growth_stability_subweight | float | 0.3 |
| growth_divergence_subweight | float | 0.2 |
| iv_prefilter_count | int | 100 |
| target_watchlist_size | int | 40 |
| min_iv_to_price_ratio | float | 1.0 |
| output_directory | Path | output |
| detailed_report_count | int | 5 |
| table_specs | tuple[TableSpec, ...] | DEFAULT_TABLE_SPECS |
| entity_columns | tuple[ColumnSpec, ...] | DEFAULT_ENTITY_COLUMNS |
| price_column | ColumnSpec | DEFAULT_PRICE_COLUMN |

Derived properties (computed, not stored):

| Property | Type | Logic |
|----------|------|-------|
| exchanges | tuple[str, ...] | Looked up from MARKET_EXCHANGES[market] |
| currencies | tuple[str, ...] | Looked up from MARKET_CURRENCIES[market] |
| min_fiscal_year | int | current year - history_years |
| max_fiscal_year | int | current year |

Validation (`__post_init__`):
- market must be in MARKET_EXCHANGES
- mode must be "shortlist" or "owned"
- "owned" mode requires non-empty owned_companies
- period_type must be "FY" or "FQ"
- discount_rate > terminal_growth_rate
- 0 < margin_of_safety < 1
- growth sub-weights sum to 1.0 (±0.01)
- fcf_growth_weight + revenue_growth_weight sum to 1.0 (±0.01)
- risk factor weights (downside_exposure + scenario_spread + terminal_dependency + fcf_reliability) sum to 1.0 (±0.01)

**Market mappings:**

| Market | Exchanges | Currencies |
|--------|-----------|------------|
| US | NASDAQ, NYSE, AMEX | USD |
| AU | ASX | AUD |
| UK | LSE | GBP |
| CA | TSX, CNQ, NEO, TSXV | CAD |
| NZ | NZE | NZD |
| SG | SES | SGD |
| HK | HKSE | HKD |

**Default table specs:**

| Table | DB Columns → Internal | Join | Required |
|-------|-----------------------|------|----------|
| incomeStatement | revenue, operatingIncome → operating_income, weightedAverageShsOutDil → shares_diluted | inner (base) | yes |
| balanceSheet | longTermDebt → lt_debt, cashAndCashEquivalents → cash | inner | yes |
| cashFlow | freeCashFlow → fcf | inner | yes |
| cashFlowGrowth | growthFreeCashFlow → fcf_growth | left | no |
| incomeStatementGrowth | growthRevenue → revenue_growth | left | no |

Entity columns: entityId → entity_id, currentSymbol → symbol,
companyName → company_name, exchange, country.

Price column: eodPrice.adjClose → adj_close.

---

### pipeline.data.contracts

Data contracts passed between pipeline stages.

**RawFinancialData** — output of the loader.

| Field | Type |
|-------|------|
| data | pd.DataFrame |
| query_metadata | dict[str, Any] |
| row_count | int |
| company_count | int |
| period_range | tuple[int, int] |
| dropped_companies_path | Path |

**FilterLog** — tracks companies removed by filters.

| Field | Type | Default |
|-------|------|---------|
| removed | dict[str, list[str]] | {} |
| reasons | dict[str, str] | {} |
| owned_bypassed | list[str] | [] |
| owned_tracking | dict[str, dict[str, Any]] | {} |

**Projection** — growth projection for one company/metric/scenario.

| Field | Type |
|-------|------|
| entity_id | int |
| metric | str |
| period_years | int |
| scenario | str |
| quarterly_growth_rates | list[float] |
| quarterly_values | list[float] |
| annual_cagr | float |
| current_value | float |

**IntrinsicValue** — DCF result for one scenario.

| Field | Type |
|-------|------|
| scenario | str |
| period_years | int |
| projected_annual_cash_flows | list[float] |
| terminal_value | float |
| present_value | float |
| iv_per_share | float |
| growth_rate | float |
| discount_rate | float |
| terminal_growth_rate | float |
| margin_of_safety | float |

**AnalysisResults** — complete pipeline output.

| Field | Type |
|-------|------|
| time_series | pd.DataFrame |
| companies | pd.DataFrame |
| projections | dict[int, dict[str, Any]] |
| intrinsic_values | dict[int, dict[str, Any]] |
| growth_rankings | pd.DataFrame |
| value_rankings | pd.DataFrame |
| weighted_rankings | pd.DataFrame |
| combined_rankings | pd.DataFrame |
| factor_contributions | pd.DataFrame |
| factor_dominance | pd.DataFrame |
| quadrant_analysis | pd.DataFrame |
| watchlist | list[str] |
| filter_log | FilterLog |
| live_prices | dict[str, float] |
| config | AnalysisConfig |

---

### pipeline.data.loader

Loads raw financial data from the FMP SQLite database.

**Public function:**

```python
def load_raw_data(
    config: AnalysisConfig,
    output_dir: Path | None = None,
) -> RawFinancialData
```

**Data flow:**

1. Validate all SQL identifiers from config
2. Open read-only SQLite connection
3. Load eligible entities (not ETF/ADR/fund, actively trading, exchange
   in config)
4. Load 5 financial tables per config.table_specs
5. Join tables: first table is base, subsequent joined per join_type.
   Inner joins drop entities missing from the joining table (logged).
   Left joins preserve base rows with NaN.
6. Check for duplicate (entity_id, fiscal_year, period) rows
7. Filter by reporting currency (incomeStatement.reportedCurrency must
   be in config.currencies). Cross-validates against
   companyProfile.currency.
8. Merge entity metadata (symbol, company_name, exchange, country)
9. Align prices: for each (entity_id, date), find nearest eodPrice
   within ±config.price_alignment_days days. Uses ROW_NUMBER() window
   function. NaN if no price found.
10. Determine common fiscal year range: config.common_range_lower_pct
    percentile of per-company min years, config.common_range_upper_pct
    percentile of per-company max years. Clamped to
    [min_fiscal_year, max_fiscal_year].
11. Filter companies: drop those ending before common_max, with temporal
    gaps, or with NaN in required columns.
12. Preprocessing: inf → NaN for all numeric columns. Raises
    RuntimeError if any NaN remains in required columns after filtering.
    NaN in optional columns preserved.
13. Assign period_idx: FQ = (fiscal_year - common_min) × 4 +
    quarter_offset. FY = fiscal_year - common_min.
14. Write dropped companies CSV log.
15. Return RawFinancialData.

**Output DataFrame columns:**

Entity metadata: entity_id, symbol, company_name, exchange, country.

Temporal: fiscal_year, period, date, period_idx.

Financial (required): revenue, operating_income, shares_diluted,
lt_debt, cash, fcf, adj_close.

Growth (optional, may contain NaN): fcf_growth, revenue_growth.

**Errors raised:**

| Error | Condition |
|-------|-----------|
| FileNotFoundError | Database file does not exist |
| ValueError | Unsafe SQL identifier in config |
| RuntimeError | Required columns missing from DB table |
| RuntimeError | Duplicate (entity_id, fiscal_year, period) rows |
| RuntimeError | No eligible entities for configured exchanges |
| RuntimeError | No companies retained after filtering |
| RuntimeError | NaN in required columns after filtering |
| ValueError | Unmapped quarter value in FQ mode |

**Dropped companies CSV:** Written to output_dir with timestamp. Columns:
symbol, company_name, reason, periods_present, periods_required,
missing_fields. Drop reasons: missing_from:{table}, ends_before_common_max,
temporal_gap, partial_nan:{columns}, non_allowed_currency:{currency}.

---

### pipeline.analysis.derived_metrics

Extracts the latest-period snapshot per company and computes derived
ratios.

**Public function:**

```python
def compute_derived_metrics(data: pd.DataFrame) -> pd.DataFrame
```

Input: multi-period DataFrame from loader (one row per entity per
quarter).

Output: one-row-per-company DataFrame indexed by entity_id.

**Carried forward from latest period:** symbol, company_name, exchange,
country, fcf, revenue, operating_income, shares_diluted, lt_debt, cash,
adj_close, period_idx.

**Computed columns:**

| Column | Formula | Zero-denominator |
|--------|---------|------------------|
| market_cap | adj_close × shares_diluted | — |
| enterprise_value | market_cap + lt_debt - cash | — |
| debt_cash_ratio | lt_debt / cash | inf if cash = 0 |
| fcf_per_share | fcf / shares_diluted | — (loader validates shares > 0) |
| acquirers_multiple | enterprise_value / operating_income | inf if OI = 0 |
| fcf_to_market_cap | fcf / market_cap | inf if market_cap = 0 |

---

### pipeline.analysis.growth_stats

Per-company growth statistics from time series data. Growth rates are
derived from TTM (trailing twelve month) sums of the raw quarterly FCF
and revenue series, smoothing quarterly lumpiness (capex timing, working
capital swings).

**Public function:**

```python
def compute_growth_statistics(
    data: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame
```

Input: multi-period DataFrame from loader. Uses columns: entity_id,
period_idx, fcf, revenue.

Output: one-row-per-company DataFrame indexed by entity_id.

**TTM smoothing:** For each metric (FCF, revenue), quarterly absolute
values are summed in a rolling 4-quarter window to produce a TTM series.
Quarter-over-quarter growth of the TTM series is then computed:
`growth = (TTM_t - TTM_{t-1}) / |TTM_{t-1}|`. With 20 quarters of
input this yields ~16 TTM growth observations. The resulting growth
rates feed the mean/var/std calculations below.

**Output columns:**

| Column | Source | Method |
|--------|--------|--------|
| fcf_growth_mean | TTM FCF growth series | mean, min 3 data points |
| fcf_growth_var | TTM FCF growth series | population variance (ddof=0) |
| fcf_growth_std | TTM FCF growth series | population std (ddof=0) |
| revenue_growth_mean | TTM revenue growth series | mean, min 3 data points |
| revenue_growth_var | TTM revenue growth series | population variance (ddof=0) |
| revenue_growth_std | TTM revenue growth series | population std (ddof=0) |
| fcf_cagr | fcf absolute values | CAGR algorithm (see below) |
| revenue_cagr | revenue absolute values | CAGR algorithm (see below) |
| combined_growth_mean | fcf/rev means + config weights | fcf_mean × fcf_weight + rev_mean × rev_weight |
| growth_stability | fcf/rev std | 1 / (1 + avg_std), where avg_std = mean(fcf_std, rev_std) |
| fcf_reliability | fcf absolute values | positive_quarters / total_non_nan_quarters |

Companies with fewer than 3 valid TTM growth data points for a metric
get 0.0 for that metric's mean, variance, and standard deviation.

**CAGR algorithm** (from legacy, applied to absolute FCF and revenue
values):

1. Filter out None, NaN, and zero values, keeping position indices.
2. Require at least 2 valid points with ≥ 4 quarters between them.
3. Return 0.0 on sign change between first and last valid values.
4. Quarterly CAGR = (|last| / |first|)^(1/periods) - 1.
5. Annualise: (1 + quarterly)^4 - 1.
6. Both-negative sign correction: if both values negative and magnitude
   increased (more negative), negate the result.

---

### pipeline.analysis.filtering

Individually toggleable financial health filters.

**Public function:**

```python
def apply_filters(
    companies: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, FilterLog]
```

Input: per-company DataFrame indexed by entity_id (output of
derived_metrics, optionally merged with growth_stats). Required
columns: symbol, fcf, operating_income, revenue, market_cap,
debt_cash_ratio.

Output: filtered DataFrame (sorted by market_cap descending) and a
FilterLog recording all removals.

**Filters (applied in order):**

| # | Name | Condition | Config toggle |
|---|------|-----------|---------------|
| 1 | negative_fcf | fcf ≤ 0 | enable_negative_fcf_filter |
| 2 | data_consistency | operating_income > revenue (when revenue > 0) | enable_data_consistency_filter |
| 3 | market_cap | market_cap < min_market_cap | enable_market_cap_filter |
| 4 | debt_cash | debt_cash_ratio > max_debt_to_cash_ratio | enable_debt_cash_filter |

**Owned-company bypass:** In owned mode, companies in
config.owned_companies bypass all active filters. Their pass/fail per
filter is tracked in FilterLog.owned_tracking. Companies that failed at
least one filter are listed in FilterLog.owned_bypassed.

**FilterLog.reasons:** Records the *first* filter that caught each
symbol. FilterLog.removed records each symbol under every filter it
failed.

---

### pipeline.analysis.growth_projection

Fade-to-equilibrium growth projection model. Deterministic replacement
for the legacy Monte Carlo simulation.

**Public functions:**

```python
def project_growth(
    entity_id: int,
    fcf_stats: dict[str, float],
    revenue_stats: dict[str, float],
    market_cap: float,
    config: AnalysisConfig,
) -> dict[int, dict[str, dict[str, Projection]]]

def project_all(
    companies: pd.DataFrame,
    growth_stats: pd.DataFrame,
    config: AnalysisConfig,
) -> dict[int, dict]
```

`project_growth` returns `{period_years: {metric: {scenario: Projection}}}`.
`project_all` returns `{entity_id: {period_years: {metric: {scenario: Projection}}}}`.

**Fade model:** g(t) = g_eq + (g_0 - g_eq) × exp(-λ × t)

| Symbol | Meaning | Source |
|--------|---------|--------|
| g_0 | Starting quarterly growth rate | Mean historical growth from growth_stats |
| g_eq | Quarterly equilibrium rate | (1 + config.equilibrium_growth_rate)^0.25 - 1 |
| λ | Decay constant (per quarter) | ln(2) / (half_life_years × 4), adjusted for company size |

**Size adjustment:** λ = λ_base × (1 + size_adj), where
size_adj = log10(market_cap / min_market_cap) × 0.1 for companies above
min_market_cap. Larger companies fade faster.

**Three scenarios per metric per period:**

| Scenario | Starting growth rate |
|----------|---------------------|
| base | g_0 (mean historical) |
| optimistic | g_0 + k × σ |
| pessimistic | g_0 - k × σ |

k = config.scenario_band_width (default 1.0), σ = historical std.

**Negative FCF handling:** Growth rates are meaningless on negative
values. Instead, constant fractional improvement toward zero each
quarter:

- Improvement rate derived from revenue growth: min(cap, revenue_growth)
  where cap = config.negative_fcf_improvement_cap.
- If revenue declining: rate = cap × 0.3 (slower default).
- Floor at 1% per quarter.
- When |FCF| drops below 1% of original magnitude, snaps to a small
  positive value and switches to conservative fade (g_0 = g_eq × 0.5).
- Scenarios vary the revenue growth assumption (optimistic uses higher
  revenue growth → faster improvement).

**Config parameters used:** equilibrium_growth_rate,
base_fade_half_life_years, scenario_band_width,
negative_fcf_improvement_cap, projection_periods, quarters_per_year,
min_market_cap.

---

### pipeline.analysis.dcf

FCF-only discounted cash flow with quarterly internal discounting.

**Public functions:**

```python
def calculate_dcf(
    base_fcf: float,
    shares: float,
    projection: Projection,
    config: AnalysisConfig,
) -> IntrinsicValue

def calculate_all_dcf(
    companies: pd.DataFrame,
    projections: dict[int, dict],
    config: AnalysisConfig,
) -> dict[int, dict]
```

`calculate_dcf` values one scenario for one company.
`calculate_all_dcf` returns `{entity_id: {period: {scenario: IntrinsicValue}}}`.
Only FCF projections are valued; revenue projections are ignored.

**Algorithm:**

1. Quarterly discount rate: (1 + discount_rate)^0.25 - 1.
2. PV of projected cash flows: sum of each quarterly CF discounted by
   (1 + quarterly_discount)^q.
3. Terminal value (Gordon Growth Model): final_annual_cf × (1 +
   terminal_growth) / (discount - terminal_growth), where
   final_annual_cf = final_quarterly_cf × 4.
4. Terminal PV: terminal_value / (1 + discount_rate)^period_years.
5. Present value = PV of CFs + terminal PV.
6. IV per share = present_value × (1 - margin_of_safety) / shares.

Uses projected quarterly values from the Projection directly (the
growth model handles negative FCF transitions).

**Output detail:** projected_annual_cash_flows (sum of each 4-quarter
group), terminal_value, present_value, iv_per_share, growth_rate
(annual CAGR from projection), discount_rate, terminal_growth_rate,
margin_of_safety.

**Config parameters used:** discount_rate, terminal_growth_rate,
margin_of_safety, quarters_per_year, projection_periods.

**Edge cases:** Companies with zero shares_diluted are skipped
(warning logged). Companies missing from projections dict are skipped.
Zero FCF → zero IV. Negative terminal CF → negative terminal value
(and negative IV).

---

### pipeline.data.live_prices

Live price fetching with two provider implementations.

**Protocol:**

```python
class PriceProvider(Protocol):
    def get_prices(self, symbols: list[str]) -> dict[str, float]: ...
```

**Implementations:**

| Class | Source | Notes |
|-------|--------|-------|
| FMPPriceProvider | FMP /stable/batch-quote endpoint | Primary. Batches of 100 symbols. Retry with exponential backoff on 429/5xx. |
| YFinancePriceProvider | yfinance.download | Fallback. Lazy import. Single-ticker and multi-ticker handling. |

**Factory function:**

```python
def auto_select_provider() -> PriceProvider
```

Returns FMPPriceProvider if `FMP_API_KEY` environment variable is set,
otherwise YFinancePriceProvider.

**FMP parameters:** batch size 100, max 3 retries, 1.0s backoff factor,
30s request timeout, 0.3s sleep between batches.

Symbols with zero, negative, or missing prices are omitted from results.

---

### pipeline.analysis.weighted_scoring

Penalty-based weighted scoring. Three penalty types, all weights
configurable via AnalysisConfig.

**Public function:**

```python
def calculate_weighted_scores(
    companies: pd.DataFrame,
    growth_stats: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame
```

Input: per-company DataFrame indexed by entity_id (requires: symbol,
debt_cash_ratio, market_cap) and growth stats DataFrame (requires:
fcf_growth_mean, fcf_growth_std, revenue_growth_mean,
revenue_growth_std, combined_growth_mean).

Output: DataFrame indexed by entity_id with: symbol, dc_penalty,
mc_penalty, growth_penalty, total_penalty, weighted_rank.

**Penalty formulae:**

| Penalty | Formula | Weight |
|---------|---------|--------|
| DC | (abs(debt_cash_ratio))^2 × dc_weight | dc_weight (0.7) |
| MC | log10(market_cap / min_market_cap) × mc_weight | mc_weight (0.4) |
| Growth (rate) | (threshold - avg_growth) × growth_weight × rate_subweight | When avg_growth < min_acceptable_growth |
| Growth (stability) | avg_std × growth_weight × stability_subweight | avg_std = mean(fcf_std, rev_std) |
| Growth (divergence) | |fcf_mean - rev_mean| × growth_weight × divergence_subweight | Always |

Total penalty = DC + MC + growth. Weighted rank = rank by total_penalty
ascending (lower = better). Companies with inf debt_cash_ratio (zero
cash) get inf DC penalty. Setting any penalty weight to 0 disables
that penalty (returns 0.0 regardless of input).

MC penalty is zero for companies at or below min_market_cap.

**Config parameters used:** dc_weight, mc_weight, growth_weight,
growth_rate_subweight, growth_stability_subweight,
growth_divergence_subweight, min_acceptable_growth, min_market_cap.

---

### pipeline.analysis.ranking

Combines composite IV ratio with value-investing risk factors to produce
risk-adjusted scores. Produces 4 ranking DataFrames. Companies without
a live price are excluded. Companies where pessimistic IV < current
price are excluded (safety gate).

**Public function:**

```python
def rank_companies(
    companies: pd.DataFrame,
    growth_stats: pd.DataFrame,
    projections: dict[int, dict],
    intrinsic_values: dict[int, dict],
    weighted_scores: pd.DataFrame,
    live_prices: dict[str, float],
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
```

Returns (growth_rankings, value_rankings, weighted_rankings,
combined_rankings).

**Composite IV ratio:** Weighted average of IV/price ratios at the
primary period: `0.25 × pessimistic + 0.50 × base + 0.25 × optimistic`.
Each scenario ratio = iv_per_share / current_price. Requires base
scenario ratio > 0; otherwise composite is 0.0.

**Safety gate:** Companies where `pessimistic_iv_ratio < 1.0` (i.e.
pessimistic IV per share < current price) are excluded from all ranking
DataFrames.

**Risk-adjusted scoring algorithm:**

1. Combined growth = fcf_growth_annual × fcf_weight + revenue_growth_annual × rev_weight (from fade model base scenario, primary period).
2. Composite IV ratio at primary period (see above).
3. Expected return = combined_growth + annualised_valuation_return, where annualised_valuation_return = composite_iv_ratio^(1/period) - 1.
4. Four risk factors, each normalised to [0, 1] where 1 = safest:
   - **Downside exposure:** min(pessimistic_iv_ratio, 2.0) / 2.0 (weight: downside_exposure_weight, default 0.35)
   - **Scenario spread:** 1 / (1 + spread), where spread = (optimistic - pessimistic) / base IV ratio (weight: scenario_spread_weight, default 0.25)
   - **Terminal dependency:** 1 - (terminal_PV / present_value), from base-case DCF (weight: terminal_dependency_weight, default 0.20)
   - **FCF reliability:** proportion of quarters with positive FCF, from growth_stats (weight: fcf_reliability_weight, default 0.20)
5. Composite safety = weighted sum of the four normalised risk factor scores.
6. Risk-adjusted score = total_expected_return × composite_safety.

**Four output DataFrames (shared base columns + extras):**

| DataFrame | Sort order | Extra columns |
|-----------|-----------|---------------|
| growth_rankings | combined_growth desc | fcf_growth_rank, revenue_growth_rank, combined_growth_rank, stability_rank |
| value_rankings | composite_iv_ratio desc | value_rank |
| weighted_rankings | total_penalty asc | weighted_rank |
| combined_rankings | risk_adjusted_rank asc | risk_adjusted_rank, opportunity_rank, opportunity_score, growth_score, value_score, weighted_score, stability_score, divergence_penalty |

Base columns: symbol, current_price, fcf_growth_annual,
revenue_growth_annual, combined_growth, composite_iv_ratio,
pessimistic_iv_ratio, base_iv_ratio, optimistic_iv_ratio,
scenario_spread, downside_exposure, terminal_dependency,
fcf_reliability, downside_exposure_score, scenario_spread_score,
terminal_dependency_score, fcf_reliability_score, composite_safety,
total_expected_return, risk_adjusted_score, growth_stability,
growth_divergence, divergence_flag, fcf, market_cap, debt_cash_ratio,
dc_penalty, mc_penalty, growth_penalty, total_penalty.

Note: scenario_spread, downside_exposure, terminal_dependency, and
fcf_reliability store raw values. The corresponding *_score columns
store normalised [0, 1] scores (1 = safest). composite_safety is the
weighted sum of the four score columns.

opportunity_rank = risk_adjusted_rank. opportunity_score = 100 -
(rank / n × 100). Normalised scores (growth, value, weighted,
stability) are rank-based 0–100. divergence_penalty is binary: 0 or 10.

**Config parameters used:** fcf_growth_weight, revenue_growth_weight,
downside_exposure_weight, scenario_spread_weight,
terminal_dependency_weight, fcf_reliability_weight,
growth_divergence_threshold, primary_period.

---

### pipeline.analysis.factor_analysis

Factor contribution analysis and quadrant classification.

**Public functions:**

```python
def calculate_factor_contributions(
    weighted_scores: pd.DataFrame,
) -> pd.DataFrame

def analyze_factor_dominance(
    contributions: pd.DataFrame,
    weighted_scores: pd.DataFrame,
) -> pd.DataFrame

def create_quadrant_analysis(
    companies: pd.DataFrame,
    intrinsic_values: dict[int, dict],
    live_prices: dict[str, float],
    growth_stats: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame
```

**calculate_factor_contributions:** Computes percentage contribution of
each penalty factor: dc_pct, mc_pct, growth_pct. Formula: factor_penalty
/ total_penalty × 100. Zero total penalty → 0% for all factors.

**analyze_factor_dominance:** Identifies which factor dominates each
company's penalty (highest %). Takes the contributions DataFrame and
the weighted_scores DataFrame (for total_penalty access). Returns
summary: primary_factor, company_count, pct_of_total,
avg_contribution, avg_total_penalty.

**create_quadrant_analysis:** Assigns companies to growth-value
quadrants based on combined_growth_mean vs min_acceptable_growth and
composite IV/price ratio vs min_iv_to_price_ratio. The composite ratio
uses the same 25/50/25 weighting as ranking: `0.25 × pessimistic +
0.50 × base + 0.25 × optimistic`, computed at the primary period.

| Quadrant | Growth | Value |
|----------|--------|-------|
| 1 (best) | ≥ threshold | ≥ threshold |
| 2 | ≥ threshold | < threshold |
| 3 | < threshold | ≥ threshold |
| 4 (worst) | < threshold | < threshold |

Output: DataFrame indexed by entity_id with: symbol, combined_growth,
composite_iv_ratio, high_growth, high_value, quadrant.

**Config parameters used:** min_acceptable_growth,
min_iv_to_price_ratio, primary_period.

---

### pipeline.analysis.watchlist

Two-step watchlist selection.

**Public function:**

```python
def select_watchlist(
    rankings: pd.DataFrame,
    config: AnalysisConfig,
) -> list[str]
```

Input: combined_rankings DataFrame indexed by entity_id (requires:
symbol, composite_iv_ratio, opportunity_rank).

**Algorithm:**

1. Step 1: Take top `iv_prefilter_count` companies by composite_iv_ratio.
2. Step 2: From that set, take top `target_watchlist_size` by
   opportunity_rank (ascending = best).

In owned mode: owned companies in the rankings are always included
regardless of rank. Remaining slots filled from the ranked list.

Output: list of symbols sorted by opportunity_rank. May be fewer than
target_watchlist_size if insufficient companies pass the pre-filter.

**Config parameters used:** iv_prefilter_count, target_watchlist_size,
mode, owned_companies.

---

### pipeline.runner

Pipeline orchestrator. Executes the 12-step analysis pipeline.

**Public function:**

```python
def run_analysis(config: AnalysisConfig) -> AnalysisResults
```

**Pipeline steps:**

| Step | Module | Action |
|------|--------|--------|
| 1 | config | Validate primary_period in projection_periods |
| 2 | loader | Load raw financial data from FMP SQLite |
| 3 | derived_metrics | Compute latest-period snapshot + derived ratios |
| 4 | growth_stats | Compute per-company growth statistics |
| 5 | filtering | Merge stats into companies, apply toggleable filters |
| 6 | growth_projection | Fade-to-equilibrium, 3 scenarios per metric per period |
| 7 | dcf | DCF intrinsic value from projections |
| 8 | live_prices | Fetch current prices (FMP primary, yfinance fallback) |
| 9 | runner | Update market_cap, EV, acquirers_multiple from live prices |
| 10 | weighted_scoring + ranking | Penalty scores + risk-adjusted ranking, 4 DataFrames |
| 11 | factor_analysis | Factor contributions, factor dominance, quadrant analysis |
| 12 | watchlist | Two-step selection (IV pre-filter then opportunity rank) |

After step 12, exports CSV files to `{config.output_directory}/{market}/`:
growth_rankings.csv, value_rankings.csv, weighted_rankings.csv,
combined_rankings.csv, factor_contributions.csv, watchlist.csv.

**Internal functions:**

`_update_live_metrics(companies, live_prices)` — updates adj_close,
market_cap, enterprise_value, acquirers_multiple, fcf_to_market_cap
from live prices. Returns a copy; does not mutate input. Companies
without a live price retain DB-derived values.

`_export_csv(results, output_dir)` — writes ranking DataFrames and
watchlist to CSV files. Creates the output directory if needed.

**Errors raised:**

| Error | Condition |
|-------|-----------|
| ValueError | primary_period not in projection_periods |

All other errors propagate from the modules called in each step.

---

### pipeline.__main__

CLI entry point. Invoked via `python -m pipeline`.

**Public function:**

```python
def main(argv: list[str] | None = None) -> int
```

Returns 0 on success, 1 on failure (invalid config or pipeline error).

**CLI arguments:**

| Flag | Type | Default | Maps to |
|------|------|---------|---------|
| --market | choice | AU | AnalysisConfig.market |
| --mode | choice | shortlist | AnalysisConfig.mode |
| --owned | list[str] | [] | AnalysisConfig.owned_companies |
| --output-dir | path | output | AnalysisConfig.output_directory |
| --log-level | choice | INFO | logging level |

Configures logging with format: `%(asctime)s %(name)s %(levelname)s %(message)s`.

---

### pipeline.charts

Chart functions for visualising analysis results. All chart functions
take `AnalysisResults` as their first argument and return
`matplotlib.figure.Figure`. Organised into four modules.

#### pipeline.charts.market_overview

Six market-wide charts:

| Function | Description |
|----------|-------------|
| `growth_value_scatter` | Scatter: growth rank vs composite IV/price ratio. Bubble size = market cap, colour = opportunity rank. Quadrant labels. Watchlist companies annotated. |
| `growth_comparison_historical` | Grouped bar chart: historical FCF CAGR vs revenue CAGR for watchlist companies. |
| `growth_comparison_projected` | Grouped bar chart with error bars from pessimistic to optimistic scenario CAGR. |
| `risk_adjusted_opportunity` | Scatter: opportunity rank vs composite safety. Bubble size = market cap, colour = risk-adjusted rank. Quadrant labels. |
| `factor_contributions_bar` | Stacked horizontal bar: dc_pct, mc_pct, growth_pct per company. |
| `factor_heatmap` | Heatmap of factor contribution percentages (matplotlib imshow). |

#### pipeline.charts.comparative

Five multi-company comparison charts. All take `symbols: list[str]` in
addition to `AnalysisResults`.

| Function | Description |
|----------|-------------|
| `projected_growth_stability` | Scatter: projected FCF growth % vs growth stability. Quality rank colouring. |
| `acquirers_multiple_analysis` | Violin plots of historical acquirer's multiple with current AM overlay. Extreme AM values capped at -10.0 floor with true value annotated. |
| `valuation_upside_comparison` | Horizontal bars: upside/downside % for 3 scenarios per company. |
| `ranking_comparison_table` | Table: Symbol, Price, growth rates, IV/Price, ranks. |
| `comprehensive_rankings_table` | 13-column table sorted by risk-adjusted rank. Colour-coded: low safety (<0.40) yellow, high growth (>20 %) green. |

#### pipeline.charts.company_detail

Seven per-company charts. All take `entity_id: int` in addition to
`AnalysisResults`.

| Function | Description |
|----------|-------------|
| `historical_growth` | Bar chart of quarterly FCF and revenue growth rates. |
| `growth_projection` | Line chart of projected values with confidence bands. |
| `valuation_matrix` | Bar chart of IV per share across 3 scenarios with current price line. |
| `growth_fan` | Fan chart of FCF and revenue trajectories. |
| `financial_health_dashboard` | Text-based gauge display of 4 health metrics. |
| `risk_spider` | Radar chart of four normalised safety scores (downside_exposure_score, scenario_spread_score, terminal_dependency_score, fcf_reliability_score), all [0, 1] where 1 = safest. Composite safety displayed at centre. |
| `scenario_comparison` | 2x2 grid: DCF waterfall, sensitivity heatmap, growth trajectories, risk-return scatter. |

#### pipeline.charts.tables

Two summary table renderers:

| Function | Description |
|----------|-------------|
| `watchlist_summary` | Watchlist companies as a formatted table (symbol, price, growth, IV/price, rank, market cap). |
| `filter_summary` | Filter results: how many removed per filter, owned bypasses. |

---

### pipeline.reports.pdf

Jinja2 + WeasyPrint PDF report generation.

**Public function:**

```python
def generate_pdf(
    results: AnalysisResults,
    output_path: Path,
    detailed_symbols: list[str] | None = None,
) -> Path
```

Renders all chart functions into an HTML template
(`reports/templates/report.html`), embeds them as base64-encoded PNG
images, and converts to PDF via WeasyPrint.

**Report sections:**
1. Title page (market, date, key parameters)
2. Executive summary (company counts, filter summary)
3. Market overview (6 charts)
4. Comparative analysis (4 charts for watchlist)
5. Company detail pages (6 charts per company, for top N by
   config.detailed_report_count)
6. Watchlist summary table
7. Full rankings table (top 25)

`detailed_symbols` defaults to the first `config.detailed_report_count`
symbols from the watchlist.

**Dependencies:** matplotlib, jinja2, weasyprint.

---

### notebooks/analysis.ipynb

Thin Jupyter notebook harness. Cells:

1. Configuration (create `AnalysisConfig`)
2. Run pipeline (`run_analysis(config)`)
3. Market overview charts (one cell per chart)
4. Comparative charts (watchlist top 20)
5. Company detail charts (top N companies)
6. Watchlist summary table
7. PDF report generation

---

## Data Source

FMP SQLite database at config.db_path. Tables used: entity, eodPrice,
incomeStatement, balanceSheet, cashFlow, cashFlowGrowth,
incomeStatementGrowth, companyProfile (for currency cross-validation).

## Tests

301 tests across 16 files.

- test_settings.py (21): ColumnSpec frozen, TableSpec validation,
  AnalysisConfig defaults, all markets, validation error cases,
  half-life validation, risk factor weights validation.
- test_contracts.py (5): Construction of all 5 dataclasses.
- test_loader.py (6): Integration tests against real FMP database (AU
  market). Skipped if database not available. Tests: loads successfully,
  correct columns, no NaN in required columns, correct metadata,
  monotonic period_idx, no duplicates.
- test_derived_metrics.py (14): Latest-period selection, all 6 computed
  ratios, division-by-zero edge cases (cash=0, OI=0, market_cap=0),
  multiple companies, metadata preservation, input validation.
- test_growth_stats.py (31): CAGR algorithm (normal growth, sign change,
  both-negative, <4 quarters, all zeros, all NaN, NaN filtering),
  TTM growth (constant growth, insufficient periods, exactly five,
  zero denominator, negative values), mean/var/std with constant and
  variable growth, combined growth mean, growth stability, CAGR from
  absolute values, min data points, NaN absolute values, very short
  series, multiple companies, FCF reliability (all positive, mixed,
  all negative, NaN excluded), input validation.
- test_filtering.py (21): Each filter toggle on/off, owned-company
  bypass and tracking, shortlist mode, owned-not-in-data, sort order,
  all filters disabled, multi-filter reason (first recorded), empty
  DataFrame, disabled filter tracking for missing owned, input
  validation.
- test_growth_projection.py (41): Quarterly rate conversion (including
  extreme negative clamping), fade lambda computation (base case, size
  adjustment, below-min, zero market cap), fade growth rates
  (convergence, monotonicity, count), value projection, negative FCF
  (improvement toward zero, crossing, revenue-derived rate, declining
  revenue, improvement cap), annual CAGR (all sign combos),
  project_growth integration (periods, scenarios, bracket ordering,
  negative FCF, convergence), project_all (multiple companies, NaN
  growth stats, empty).
- test_dcf.py (20): Core DCF (returns IntrinsicValue, positive IV,
  margin of safety, share dilution, discount rate sensitivity, scenario/
  period preservation, config passthrough, annual CF count and sums),
  hand calculation verification, edge cases (negative terminal, zero FCF,
  growth rate passthrough, empty quarterly values), calculate_all_dcf
  (all companies, all scenarios, skip zero shares, skip missing
  projections, empty companies, FCF-only).
- test_live_prices.py (10): FMPPriceProvider (successful batch, zero/null
  price skip, empty list, retry on 5xx, max retries, batch splitting),
  YFinancePriceProvider (import error fallback), auto_select_provider
  (FMP key present, FMP key absent).
- test_weighted_scoring.py (22): DC penalty (squared, weight, zero, inf,
  abs), MC penalty (log scaling, weight, below/at min), growth penalty
  (above threshold, below threshold, stability, divergence),
  calculate_weighted_scores (all columns, rank ordering, weight config
  sensitivity, inf DC, total = sum of components, empty, missing stats
  skip, missing columns raises).
- test_ranking.py (31): _compute_iv_ratios (all scenario ratios,
  composite weighted, no IVs, zero price, negative IV),
  _compute_terminal_dependency (normal values, no IVs, no base scenario,
  negative present value), structure (returns 4 DataFrames, all
  companies included, company without price excluded, output has new
  columns, old columns removed), safety gate (excludes below pessimistic,
  no pessimistic IV excluded, exactly 1.0 passes), sort orders (growth
  desc, composite_iv_ratio desc, penalty asc, risk-adjusted rank asc),
  ranking properties (monotone, opportunity = risk-adjusted, divergence
  binary, composite safety bounded, score columns bounded [0,1],
  risk-adjusted score = return × safety),
  edge cases (empty, no prices, missing columns, missing fcf_reliability).
- test_factor_analysis.py (14): Factor contributions (sum to 100%,
  correct percentages, zero total, empty, missing columns), factor
  dominance (identifies factors, company counts, pct sums to 100%,
  empty), quadrant analysis (all 4 quadrants, no-price exclusion,
  empty, threshold boundary, composite IV ratio).
- test_watchlist.py (10): Two-step selection (basic, IV pre-filter
  limits, sort order, fewer than target), owned-company inclusion
  (always included, no duplication, shortlist mode, not in rankings),
  edge cases (empty, missing columns).
- test_runner.py (22): _update_live_metrics (price/market_cap update,
  unchanged without price, EV update, acquirers_multiple, no mutation,
  empty prices), _export_csv (creates files, watchlist content, creates
  directory), run_analysis (returns AnalysisResults, all steps called,
  fields populated, invalid primary_period, all filtered out, no live
  prices, partial live prices, empty watchlist), __main__ (parser
  defaults, all flags, success, pipeline failure, invalid config).
- test_charts.py (26): Market overview (growth_value_scatter,
  growth_comparison_historical, growth_comparison_projected,
  risk_adjusted_opportunity, factor_contributions_bar, factor_heatmap),
  comparative (projected_growth_stability, acquirers_multiple_analysis,
  valuation_upside_comparison, ranking_comparison_table,
  comprehensive_rankings_table), company detail (historical_growth,
  growth_projection, valuation_matrix, growth_fan,
  financial_health_dashboard, risk_spider, scenario_comparison), tables
  (watchlist_summary, filter_summary), edge cases (empty watchlist,
  empty rankings, nonexistent company, NaN market cap, NaN factor
  contributions).
- test_pdf.py (7): _fig_to_base64 (valid base64 PNG), _build_context
  (required keys, market charts list, detailed companies), generate_pdf
  (creates file, creates parent directory, defaults detailed symbols).
