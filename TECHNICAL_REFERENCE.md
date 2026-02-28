# Technical Reference

## Architecture Overview

```
CLI (pipeline/__main__.py, pipeline/main.py)
 │
 ├── Data Layer (pipeline/data/)
 │   ├── fmp.py          → FMP SQLite database (fundamentals, prices, profiles)
 │   ├── live_price.py   → FMP API / yfinance live price
 │   └── models.py       → CompanyData dataclass
 │
 ├── Metrics (pipeline/metrics/)
 │   ├── valuation.py    → EV, yield ratios
 │   ├── trends.py       → Growth stats, regressions, ROIC
 │   ├── safety.py       → Debt coverage ratios
 │   ├── quality.py      → Piotroski F-Score, profitability
 │   └── sentiment.py    → Price momentum
 │
 ├── Simulation (pipeline/simulation.py)
 │   └── Monte Carlo DCF → IV percentile distribution
 │
 ├── Screening (pipeline/screening.py)
 │   └── Filters, MoS, ranking
 │
 └── Output (pipeline/output/)
     ├── csv_export.py     → CSV export
     ├── detail_report.py  → HTML detail report generation
     └── charts.py         → Chart rendering (matplotlib → base64 PNG)
```

**Data flow:** FMP database + live price → `CompanyData` → five metric modules (independent, no cross-dependencies) → simulation (depends on trends) → screening (consumes all metrics + simulation) → CSV export or HTML detail report.

**Module file:** `pipeline/config.py` — all configuration dataclasses.

---

## Data Layer

### CompanyData

Defined in `pipeline/data/models.py`. Central data contract consumed by all metric modules.

| Field | Type | Description |
|---|---|---|
| `symbol` | `str` | Stock ticker symbol |
| `company_name` | `str` | Company name |
| `sector` | `str` | Business sector |
| `exchange` | `str` | Stock exchange |
| `financials` | `pd.DataFrame` | Quarterly financials, one row per quarter, sorted by date |
| `latest_price` | `float` | Most recent stock price |
| `market_cap` | `float` | Market capitalisation |
| `shares_outstanding` | `float` | Diluted shares outstanding (from latest quarter) |
| `price_history` | `pd.DataFrame` | Daily price history |

**`financials` DataFrame columns:**

`date`, `fiscal_year`, `period`, `revenue`, `gross_profit`, `operating_income`, `ebit`, `net_income`, `income_before_tax`, `income_tax_expense`, `interest_expense`, `weighted_average_shs_out_dil`, `total_assets`, `total_current_assets`, `total_current_liabilities`, `total_debt`, `long_term_debt`, `cash_and_cash_equivalents`, `operating_cash_flow`, `free_cash_flow`

**`price_history` DataFrame columns:** `date`, `close`

### Data Sources

**FMP database** (`pipeline/data/fmp.py`): SQLite, opened read-only. Tables queried:

- `entity` — symbol, name, sector, exchange, ETF/ADR/fund flags, trading status
- `incomeStatement`, `balanceSheet`, `cashFlow` — joined on `(entityId, date, period)`, filtered to quarterly periods (Q1–Q4)
- `eodPrice` — daily close prices
- `companyProfile` — latest price and market cap

Column names are mapped from FMP camelCase to pipeline snake_case at load time.

**Live price** (`pipeline/data/live_price.py`): FMP batch-quote API (requires `FMP_API_KEY` environment variable), with yfinance fallback. Returns a single current price per symbol.

### Loading Sequence

`pipeline/data/__init__.py` orchestrates loading:

1. `load_universe(config)` — query `entity` table filtered by configured exchanges, excluding ETFs, ADRs, funds, and inactive symbols. Returns `list[tuple[int, str]]` of `(entity_id, symbol)` sorted by symbol.
2. `lookup_entity_ids(symbols, config)` — resolve a list of ticker symbols to entity IDs. Queries `entity` table by `currentSymbol`, preferring actively-trading entities (then highest entity ID). Returns `dict[str, int]` mapping found symbols to entity IDs. Symbols not in the database are omitted. Used by the `detail` command.
3. `load_company(entity_id, config)` — for each entity:
   1. Load entity metadata, quarterly financials (income + balance + cash flow joined), price history, and company profile from the FMP database.
   2. Fetch current price via `fetch_current_price` (FMP API, then yfinance fallback).
   3. If a live price is available, overwrite `latest_price` and recalculate `market_cap` as `live_price * shares_outstanding`.
   4. Returns `CompanyData`, or `None` if critical data is missing (no entity row, no financials, no company profile).

---

## Metrics Modules

All five modules are independent — they consume `CompanyData` and produce their own output dataclass. No metric module depends on the output of another.

### Valuation

**Module:** `pipeline/metrics/valuation.py`
**Function:** `compute_valuation(company, config) -> ValuationMetrics`
**Inputs:** `CompanyData`, `PipelineConfig`

#### ValuationMetrics

| Field | Type | Description |
|---|---|---|
| `enterprise_value` | `float` | Market cap + total debt - cash |
| `ttm_fcf` | `float` | Sum of last 4 quarters' free cash flow |
| `ttm_ebit` | `float` | Sum of last 4 quarters' EBIT |
| `fcf_ev` | `float \| None` | TTM FCF / EV. `None` if EV <= 0 |
| `ebit_ev` | `float \| None` | TTM EBIT / EV. `None` if EV <= 0 |
| `composite_yield` | `float \| None` | `fcf_ev * fcf_ev_weight + ebit_ev * ebit_ev_weight`. `None` if EV <= 0 |

**Formulas:**

- `EV = market_cap + total_debt - cash_and_cash_equivalents` (from latest quarter)
- TTM values use the last 4 rows (or fewer if unavailable), NaN treated as zero
- If EV <= 0, all yield ratios are `None`

### Trends

**Module:** `pipeline/metrics/trends.py`
**Function:** `compute_trends(company, config) -> TrendMetrics | None`
**Inputs:** `CompanyData`, `TrendsConfig`

Returns `None` if any sub-computation lacks sufficient data.

#### TrendMetrics

| Field | Type | Description |
|---|---|---|
| `revenue_cagr` | `float` | Annualised compound growth (TTM start vs TTM end) |
| `revenue_yoy_growth_std` | `float` | Std dev of YoY quarterly revenue growth rates |
| `revenue_qoq_growth_mean` | `float` | Mean QoQ revenue growth (simulation input) |
| `revenue_qoq_growth_var` | `float` | Variance of QoQ revenue growth (simulation input) |
| `margin_intercept` | `float` | Operating margin regression intercept |
| `margin_slope` | `float` | Operating margin regression slope (per quarter, decimal) |
| `margin_r_squared` | `float` | Operating margin regression R-squared |
| `conversion_intercept` | `float \| None` | FCF conversion regression intercept (`None` if fallback) |
| `conversion_slope` | `float \| None` | FCF conversion regression slope (per quarter; `None` if fallback) |
| `conversion_r_squared` | `float \| None` | FCF conversion regression R-squared (`None` if fallback) |
| `conversion_median` | `float \| None` | Median FCF conversion (set if fallback) |
| `conversion_is_fallback` | `bool` | `True` if fewer than `min_quarters_fcf_conversion` valid quarters |
| `roic_latest` | `float` | Most recent quarter's ROIC |
| `roic_slope` | `float` | ROIC regression slope (per quarter) |
| `roic_detrended_std` | `float` | Std dev of ROIC regression residuals |
| `roic_minimum` | `float` | Minimum ROIC across valid quarters |

**Sub-computations:**

- **Revenue CAGR:** `(end_ttm / start_ttm)^(1/years) - 1`. Uses TTM sums (first 4 and last 4 quarters) where >= 4 quarters exist, otherwise single-quarter start/end. Years derived from calendar date span.
- **Revenue YoY growth std:** Each quarter compared to 4 quarters prior. Requires >= 2 valid rates. Sample std (ddof=1).
- **Revenue QoQ stats:** Each quarter compared to the prior quarter. Requires >= 2 valid rates. Sample variance (ddof=1).
- **Operating margin regression:** `margin = operating_income / revenue` per quarter, regressed against quarter index via `scipy.stats.linregress`. Quarters with zero revenue excluded. Requires >= 2 data points.
- **FCF conversion regression:** `conversion = free_cash_flow / operating_income` per quarter (excluding `operating_income <= 0`). Full regression if >= `min_quarters_fcf_conversion` valid quarters; otherwise falls back to median conversion.
- **ROIC:** `NOPAT / invested_capital` where `NOPAT = EBIT * (1 - effective_tax_rate)`, `effective_tax_rate = income_tax_expense / income_before_tax`, `invested_capital = total_assets - total_current_liabilities`. Excludes quarters with `income_before_tax <= 0` or `invested_capital <= 0`. Slope and detrended std from `linregress`. Requires >= 2 valid quarters.

### Safety

**Module:** `pipeline/metrics/safety.py`
**Function:** `compute_safety(company) -> SafetyMetrics`
**Inputs:** `CompanyData`

#### SafetyMetrics

| Field | Type | Description |
|---|---|---|
| `interest_coverage` | `float \| None` | TTM EBIT / TTM interest expense. `None` if interest expense is zero or missing |
| `ocf_to_debt` | `float \| None` | TTM OCF / latest total debt. `None` if total debt is zero or missing |

TTM values sum the last 4 quarters (or fewer if unavailable). Balance sheet values from the latest quarter.

### Quality

**Module:** `pipeline/metrics/quality.py`
**Function:** `compute_quality(company) -> QualityMetrics`
**Inputs:** `CompanyData`

#### QualityMetrics

| Field | Type | Description |
|---|---|---|
| `f_roa_positive` | `bool` | TTM net income / total assets > 0 |
| `f_ocf_positive` | `bool` | TTM OCF > 0 |
| `f_roa_improving` | `bool` | Current TTM ROA > prior year TTM ROA |
| `f_accruals_negative` | `bool` | TTM OCF > TTM net income |
| `f_leverage_decreasing` | `bool` | Current LTD/TA < prior year LTD/TA |
| `f_current_ratio_improving` | `bool` | Current ratio improved YoY |
| `f_no_dilution` | `bool` | Diluted shares not increased YoY |
| `f_gross_margin_improving` | `bool` | TTM gross margin improved YoY |
| `f_asset_turnover_improving` | `bool` | TTM asset turnover improved YoY |
| `f_score` | `int` | Sum of all 9 boolean signals (0-9) |
| `gross_profitability` | `float \| None` | TTM gross profit / total assets (Novy-Marx). `None` if total assets is zero |
| `accruals_ratio` | `float \| None` | (TTM net income - TTM OCF) / total assets. `None` if total assets is zero |

**TTM periods:** Current year = last 4 quarters. Prior year = quarters 5-8 back. YoY comparisons require >= 8 quarters of data (`has_prior`); signals default to `False` when insufficient data.

### Sentiment

**Module:** `pipeline/metrics/sentiment.py`
**Function:** `compute_sentiment(company) -> SentimentMetrics`
**Inputs:** `CompanyData`

#### SentimentMetrics

| Field | Type | Description |
|---|---|---|
| `return_6m` | `float \| None` | 6-month simple price return (182 calendar days) |
| `return_12m` | `float \| None` | 12-month simple price return (365 calendar days) |

Looks up the close price on the nearest available trading date to the target date. Returns `None` if price history does not span the lookback period or the reference close is zero.

---

## Simulation

**Module:** `pipeline/simulation.py`

### SimulationInput

Constructed in `pipeline/main.py._build_simulation_input` from `TrendMetrics` and `CompanyData`.

| Field | Type | Source |
|---|---|---|
| `revenue_qoq_growth_mean` | `float` | `TrendMetrics.revenue_qoq_growth_mean` |
| `revenue_qoq_growth_var` | `float` | `TrendMetrics.revenue_qoq_growth_var` |
| `margin_intercept` | `float` | `TrendMetrics.margin_intercept` |
| `margin_slope` | `float` | `TrendMetrics.margin_slope` |
| `conversion_intercept` | `float \| None` | `TrendMetrics.conversion_intercept` |
| `conversion_slope` | `float \| None` | `TrendMetrics.conversion_slope` |
| `conversion_median` | `float \| None` | `TrendMetrics.conversion_median` |
| `conversion_is_fallback` | `bool` | `TrendMetrics.conversion_is_fallback` |
| `starting_revenue` | `float` | Latest quarterly revenue from `CompanyData.financials` |
| `shares_outstanding` | `float` | `CompanyData.shares_outstanding` |

### PercentileBands

Percentile bands across all simulation paths per projected quarter. Each array has shape `(num_quarters,)` where `num_quarters = projection_years × 4`.

| Field | Type | Description |
|---|---|---|
| `p10` | `np.ndarray` | 10th percentile at each quarter |
| `p25` | `np.ndarray` | 25th percentile at each quarter |
| `p50` | `np.ndarray` | 50th percentile (median) at each quarter |
| `p75` | `np.ndarray` | 75th percentile at each quarter |
| `p90` | `np.ndarray` | 90th percentile at each quarter |

### SimulationOutput

| Field | Type | Description |
|---|---|---|
| `iv_p10` | `float` | 10th percentile IV per share |
| `iv_p25` | `float` | 25th percentile IV per share |
| `iv_p50` | `float` | Median IV per share |
| `iv_p75` | `float` | 75th percentile IV per share |
| `iv_p90` | `float` | 90th percentile IV per share |
| `iv_spread` | `float` | P75 - P25 spread |
| `implied_cagr_p25` | `float` | Implied CAGR from current price to P25 IV |
| `implied_cagr_p50` | `float` | Implied CAGR from current price to P50 IV |
| `implied_cagr_p75` | `float` | Implied CAGR from current price to P75 IV |
| `sample_paths` | `list[PathData]` | Display paths for charting (each has `quarterly_revenue` and `quarterly_fcf` arrays) |
| `revenue_bands` | `PercentileBands \| None` | Revenue percentile bands across all paths. `None` when `num_display_paths` is 0 |
| `fcf_bands` | `PercentileBands \| None` | FCF percentile bands across all paths. `None` when `num_display_paths` is 0 |

**Implied CAGR formula:** `(IV / current_price)^(1 / projection_years) - 1`. Returns `0.0` if current price, IV, or years <= 0.

### Per-Path Pipeline (7 Steps)

Each of the `num_replicates` paths runs `projection_years * 4` quarters:

1. **Sample growth** -- draw from log-normal distribution. Growth rates are modelled as `(1 + g) ~ LogNormal(mu, sigma)` so revenue cannot go negative. If CV of the growth distribution exceeds `cv_cap`, variance is clamped.
2. **Apply constraints** -- see constraint list below.
3. **Project revenue** -- `revenue = revenue * (1 + constrained_growth)`.
4. **Compute margin** -- `margin = margin_intercept + margin_slope * quarter_offset`.
5. **Operating income** -- `revenue * margin`.
6. **FCF conversion** -- if fallback: use `conversion_median`. Otherwise: `conversion_intercept + conversion_slope * quarter_offset`.
7. **FCF and discount** -- `FCF = operating_income * conversion`. Discount each quarter's FCF to present value using `quarterly_discount = (1 + discount_rate)^0.25`.

**Terminal value:** After the last projected quarter, a perpetuity growth model is applied: `TV = annual_FCF * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)`, discounted back to present. `annual_FCF = last_quarterly_FCF * 4`. Skipped if `discount_rate <= terminal_growth_rate`.

**IV per share:** `total_PV / shares_outstanding`. Percentiles computed across all replicates.

### Growth Constraints (7)

Applied in order to each sampled growth rate:

| # | Constraint | Description |
|---|---|---|
| 1 | **CV cap** | Applied at the distribution level. If `std / E[1+g] > cv_cap`, variance is clamped. Default: `1.0`. |
| 2 | **Per-quarter growth caps** | Two tiers based on `revenue_ratio = current_revenue / starting_revenue`. **Early** (ratio <= `size_tier_threshold`): clamped to `[early_negative_cap, early_positive_cap]`. **Late** (ratio > threshold): `[late_negative_cap, late_positive_cap]`. |
| 3 | **Cumulative cap** | Revenue cannot exceed `cumulative_cap * starting_revenue`. Growth reduced to enforce. |
| 4 | **CAGR backstop** | Annualised growth from start cannot exceed `cagr_backstop`. Uses `(1 + cagr_backstop)^years_elapsed` as maximum factor. |
| 5 | **Momentum exhaustion** | Positive mean: growth capped at `momentum_upper * historical_mean`. Negative mean: growth floored at `momentum_upper * historical_mean`. Skipped for near-zero mean. |
| 6 | **Time decay** | Growth above `time_decay_growth_threshold` decays by `time_decay_factor^quarter_idx`. Only the excess above the threshold decays. |
| 7 | **Size penalty** | Positive growth scaled down as revenue grows. Linear interpolation from `size_penalty_max` (at ratio=1) to `size_penalty_min` (at ratio=`cumulative_cap`). |

### Parameterised DCF

`run_parameterised_dcf(sim_input, current_price, discount_rate, growth_multiplier, ...) -> float`

For sensitivity analysis (heatmaps). Scales `revenue_qoq_growth_mean` by `growth_multiplier` and overrides the discount rate. Runs `heatmap_replicates` paths (fewer than the main simulation) with no sample paths. Returns median IV per share.

---

## Screening

**Module:** `pipeline/screening.py`

### Two-Stage Filter Pipeline

**Stage 1 -- Data sufficiency:**

- Number of quarters in `financials` < `min_quarters`, OR
- `trends` is `None`, OR
- `simulation` is `None`

If any of these hold, the company is tagged with reason code `DATA` and Stage 2 is skipped.

**Stage 2 -- Configurable filters** (each independently toggleable, `None` = disabled):

| Filter | Config field | Reason code | Logic |
|---|---|---|---|
| Market cap | `filter_min_market_cap` | `MC` | `market_cap < threshold` |
| Interest coverage | `filter_min_interest_coverage` | `IC` | `interest_coverage` is `None` or `< threshold` |
| OCF to debt | `filter_min_ocf_to_debt` | `OD` | `ocf_to_debt` is `None` or `< threshold` |
| FCF conversion R-squared | `filter_min_fcf_conversion_r2` | `CR` | `conversion_r_squared` is `None` or `< threshold` |
| Negative TTM FCF | `filter_exclude_negative_ttm_fcf` | `FCF` | `ttm_fcf <= 0` |

### Margin of Safety

`MoS = (IV - price) / IV`

Computed at P25, P50, and P75 IV levels. `None` if IV <= 0 or no simulation.

### Sort and Ranking

Sort metric is selected by `sort_by`:

| `SortOption` | Metric |
|---|---|
| `COMPOSITE` | `composite_yield` |
| `FCF_EV` | `fcf_ev` |
| `EBIT_EV` | `ebit_ev` |

**Sort order:** Unfiltered companies first, filtered companies second. Within each group, descending by chosen yield metric. `None` yield values sort last.

### Output Dataclasses

**`CompanyAnalysis`:** Bundles `CompanyData` + all five metric outputs + `SimulationOutput | None`.

**`ScreeningResult`:** Wraps `CompanyAnalysis` with `mos_p25`, `mos_p50`, `mos_p75`, `filtered` (bool), `filter_reasons` (comma-separated reason codes).

---

## CSV Export

**Module:** `pipeline/output/csv_export.py`
**Function:** `export_csv(results, output_path)`

Creates parent directories if they do not exist.

### Column List (in order)

**Identity:** `symbol`, `company_name`, `sector`, `price`, `market_cap`

**Valuation:** `fcf_ev`, `ebit_ev`, `composite_yield`

**IV estimates:** `iv_p25`, `iv_p50`, `iv_p75`, `iv_spread`

**Margin of safety:** `mos_p25`, `mos_p50`, `mos_p75`

**Implied CAGR:** `implied_cagr_p25`, `implied_cagr_p50`, `implied_cagr_p75`

**Trends:** `revenue_cagr`, `revenue_yoy_growth_std`, `margin_slope`, `margin_r_squared`, `conversion_slope`, `conversion_r_squared`

**Quality:** `f_score`, `f_roa_positive`, `f_ocf_positive`, `f_roa_improving`, `f_accruals_negative`, `f_leverage_decreasing`, `f_current_ratio_improving`, `f_no_dilution`, `f_gross_margin_improving`, `f_asset_turnover_improving`, `gross_profitability`, `accruals_ratio`

**ROIC:** `roic_latest`, `roic_slope`, `roic_detrended_std`, `roic_minimum`

**Safety:** `interest_coverage`, `ocf_to_debt`

**Sentiment:** `return_6m`, `return_12m`

**Filter status:** `filtered`, `filter_reasons`

### Slope Annualisation

`margin_slope`, `conversion_slope`, and `roic_slope` are stored per quarter in `TrendMetrics` (for simulation use) but annualised (×4) in the CSV output.

### IV Rounding Rules

- Price >= $10: IV rounded to nearest dollar
- Price < $10: IV rounded to nearest 10 cents

Applied to `iv_p25`, `iv_p50`, `iv_p75`, `iv_spread`.

### Format Conventions

- `None` → empty string
- Booleans → `TRUE` / `FALSE`
- NaN / Inf → empty string
- All other values → `str()` representation

---

## Detail Reports

### Charts

**Module:** `pipeline/output/charts.py`

All charts render to in-memory PNG via matplotlib (Agg backend), then base64-encode to data URIs for embedding in HTML. Each chart function returns a `matplotlib.figure.Figure`. The `figure_to_data_uri(fig)` utility renders and closes the figure.

| Function | Inputs | Description |
|---|---|---|
| `revenue_projection_chart` | `sample_paths`, `revenue_bands`, `historical_revenues` | Historical revenue line + projected sample paths (grey) + P10–P90 and P25–P75 shaded bands + median line |
| `fcf_projection_chart` | `sample_paths`, `fcf_bands`, `historical_fcfs` | Same structure as revenue projection but for FCF |
| `iv_scenario_chart` | `iv_p25`, `iv_p50`, `iv_p75`, `current_price` | Horizontal bars for P25/P50/P75 IV with vertical current price reference line |
| `sensitivity_heatmap_chart` | `sim_input`, `current_price`, configs, `seed` | Grid of (discount_rate × growth_multiplier) cells. Each cell runs `run_parameterised_dcf`. Coloured by margin of safety (RdYlGn), with break-even contour at IV = price |
| `margin_time_series_chart` | `revenues`, `operating_incomes`, `margin_intercept`, `margin_slope` | Scatter of actual operating margins + regression trend line |
| `revenue_growth_chart` | `revenues` | YoY quarterly revenue growth bars (coloured by sign) + mean and ±1σ reference lines |

Projection charts include zero on the y-axis when negative outcomes are possible.

### HTML Report

**Module:** `pipeline/output/detail_report.py`
**Function:** `generate_detail_html(result, sim_input, display_config, heatmap_config, sim_config, dcf_config, seed) -> str`

Produces a self-contained HTML document with embedded chart images (base64 data URIs). The report has five sections:

1. **Header** — symbol, name, price, market cap, FCF/EV, EBIT/EV, MoS (median), F-Score
2. **Charts** — all available charts (see above). Revenue growth and margin charts render when financials exist. Projection and IV charts render when simulation output exists. Heatmap renders when `sim_input` is provided.
3. **Quality** — F-Score components as checkmarks/crosses, gross profitability, accruals ratio, sentiment (6m/12m returns)
4. **Trends** — revenue CAGR, YoY growth std, margin slope/R², conversion slope/R², ROIC metrics. Slopes are annualised (×4) for display.
5. **Safety** — interest coverage, OCF/total debt

**Visual encoding:**

| Element | Rule |
|---|---|
| F-Score | Green (≥ `fscore_strong_min`, default 7), amber (≥ `fscore_moderate_min`, default 4), red (below) |
| Safety metrics | Amber when below `safety_amber_threshold` (default 1.5×) |
| F-Score components | Green checkmark (✓) for pass, red cross (✗) for fail |

---

## CLI

Entry point: `python -m pipeline`

### Commands

#### `screen`

Screen companies and export CSV.

```
python -m pipeline screen --exchanges ASX [NYSE ...] [options]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--exchanges` | Yes | -- | Stock exchanges to screen |
| `--sort` | No | `composite` | Sort metric: `composite`, `fcf_ev`, `ebit_ev` |
| `--output` | No | `output/screening.csv` | Output CSV path |
| `--min-market-cap` | No | disabled | Minimum market cap filter |
| `--min-interest-coverage` | No | disabled | Minimum interest coverage filter |
| `--min-ocf-to-debt` | No | disabled | Minimum OCF/debt filter |
| `--min-fcf-conversion-r2` | No | disabled | Minimum FCF conversion R-squared filter |
| `--include-negative-fcf` | No | `False` (excluded) | Include companies with negative TTM FCF |
| `--db-path` | No | from config | FMP database path |
| `--min-quarters` | No | `12` | Minimum quarters of data required |
| `-v`, `--verbose` | No | `False` | Enable DEBUG logging |

#### `detail`

Generate HTML detail reports for specific tickers.

```
python -m pipeline detail AAPL [MSFT ...] [options]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `tickers` | Yes | -- | Ticker symbols (positional, one or more) |
| `--output-dir` | No | `output/detail/` | Output directory for HTML reports |
| `--db-path` | No | from config | FMP database path |
| `-v`, `--verbose` | No | `False` | Enable DEBUG logging |

Resolves tickers to entity IDs via `lookup_entity_ids`, loads each company, computes all metrics and simulation, generates an HTML detail report per ticker. Unknown tickers are skipped with a warning. Each report is written as `{TICKER}.html` in the output directory.

**Logging:** `INFO` by default, `DEBUG` with `-v`. Format: `HH:MM:SS LEVEL    module: message`.

---

## Configuration

All configuration is Python dataclasses in `pipeline/config.py`.

### PipelineConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `db_path` | `Path` | `/home/mattm/projects/Pers/financial_db/data/fmp.db` | FMP database path |
| `exchanges` | `list[str]` | `[]` | Stock exchanges to screen |
| `min_quarters` | `int` | `12` | Minimum quarters for data sufficiency |
| `filter_min_market_cap` | `float \| None` | `None` | Min market cap filter (disabled) |
| `filter_min_interest_coverage` | `float \| None` | `None` | Min interest coverage filter (disabled) |
| `filter_min_ocf_to_debt` | `float \| None` | `None` | Min OCF/debt filter (disabled) |
| `filter_min_fcf_conversion_r2` | `float \| None` | `None` | Min FCF conversion R-squared filter (disabled) |
| `filter_exclude_negative_ttm_fcf` | `bool` | `True` | Exclude negative TTM FCF |
| `fcf_ev_weight` | `float` | `0.6` | FCF/EV weight in composite yield |
| `ebit_ev_weight` | `float` | `0.4` | EBIT/EV weight in composite yield |
| `sort_by` | `SortOption` | `COMPOSITE` | Ranking sort metric |

### SimulationConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `num_replicates` | `int` | `10,000` | Monte Carlo paths |
| `num_display_paths` | `int` | `25` | Sample paths saved for charting |
| `cv_cap` | `float` | `1.0` | Max coefficient of variation for growth distribution |
| `early_positive_cap` | `float` | `0.20` | Max positive QoQ growth (early stage) |
| `early_negative_cap` | `float` | `-0.15` | Max negative QoQ growth (early stage) |
| `late_positive_cap` | `float` | `0.12` | Max positive QoQ growth (late stage) |
| `late_negative_cap` | `float` | `-0.08` | Max negative QoQ growth (late stage) |
| `size_tier_threshold` | `float` | `2.0` | Revenue ratio threshold for early/late tier |
| `cumulative_cap` | `float` | `5.0` | Max revenue as multiple of starting revenue |
| `cagr_backstop` | `float` | `0.50` | Max annualised growth rate |
| `momentum_upper` | `float` | `2.0` | Upper momentum exhaustion multiplier |
| `momentum_lower` | `float` | `0.5` | Lower momentum exhaustion multiplier |
| `time_decay_growth_threshold` | `float` | `0.20` | Growth above this threshold decays |
| `time_decay_factor` | `float` | `0.8` | Decay factor per quarter |
| `size_penalty_min` | `float` | `0.4` | Size penalty at max revenue ratio |
| `size_penalty_max` | `float` | `1.0` | Size penalty at starting revenue |

### DCFConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `discount_rate` | `float` | `0.10` | Annual discount rate |
| `terminal_growth_rate` | `float` | `0.025` | Terminal perpetuity growth rate |
| `projection_years` | `int` | `10` | Years to project (quarters = years * 4) |

### TrendsConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `min_quarters_fcf_conversion` | `int` | `8` | Minimum valid quarters for FCF conversion regression (below this, falls back to median) |

### HeatmapConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `discount_rate_min` | `float` | `0.07` | Heatmap discount rate range min |
| `discount_rate_max` | `float` | `0.13` | Heatmap discount rate range max |
| `discount_rate_step` | `float` | `0.01` | Heatmap discount rate step |
| `growth_multiplier_min` | `float` | `0.5` | Heatmap growth multiplier range min |
| `growth_multiplier_max` | `float` | `1.5` | Heatmap growth multiplier range max |
| `growth_multiplier_step` | `float` | `0.25` | Heatmap growth multiplier step |
| `heatmap_replicates` | `int` | `1,000` | Monte Carlo paths per heatmap cell |

### DisplayConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `fscore_strong_min` | `int` | `7` | F-Score threshold for "strong" rating |
| `fscore_moderate_min` | `int` | `4` | F-Score threshold for "moderate" rating |
| `safety_amber_threshold` | `float` | `1.5` | Safety metric amber warning threshold |

### SortOption (Enum)

| Value | String | Description |
|---|---|---|
| `COMPOSITE` | `"composite"` | Weighted composite yield |
| `FCF_EV` | `"fcf_ev"` | FCF/EV yield |
| `EBIT_EV` | `"ebit_ev"` | EBIT/EV yield |
