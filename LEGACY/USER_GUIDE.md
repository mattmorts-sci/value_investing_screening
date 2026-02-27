# User Guide

## What This Pipeline Does

Screens publicly listed companies for value investing opportunities.
Loads financial data from an FMP database, computes derived metrics and
growth statistics, projects future growth, estimates intrinsic value via
DCF, ranks companies by risk-adjusted returns, and produces a watchlist.

The pipeline is being built in phases. This guide covers what is
currently implemented.

## Current Status: Phase 7 (Issue Fixes)

All phases are complete. The pipeline runs end-to-end from data loading
through to charts, notebooks, and PDF reports.

## Configuration

All parameters are in a single Python dataclass: `AnalysisConfig` in
`pipeline/config/settings.py`. Create an instance with defaults or
override individual fields:

```python
from pipeline.config.settings import AnalysisConfig

# Australian market, defaults
config = AnalysisConfig()

# US market
config = AnalysisConfig(market="US")

# Owned portfolio mode
config = AnalysisConfig(mode="owned", owned_companies=["BHP", "CBA", "WES"])
```

### Key Parameters

**Market selection:**

- `market` — which stock exchange to screen. One of: AU, US, UK, CA,
  NZ, SG, HK. Default: AU.
- `mode` — "shortlist" screens all companies on the exchange. "owned"
  analyses specific companies you already hold.

**Data loading:**

- `db_path` — path to the FMP SQLite database file.
- `period_type` — "FQ" for quarterly data (default), "FY" for annual.
- `history_years` — how many years of historical data to load.
  Default: 5.
- `price_alignment_days` — when matching share prices to financial
  statement dates, how many days either side to search. Default: 7.

**Filtering:**

Four filters can be toggled on or off independently:

- `enable_negative_fcf_filter` — exclude companies with negative free
  cash flow.
- `enable_data_consistency_filter` — exclude companies with
  inconsistent financial data.
- `enable_market_cap_filter` — exclude companies below
  `min_market_cap` (default $20M).
- `enable_debt_cash_filter` — exclude companies with debt-to-cash
  ratio above `max_debt_to_cash_ratio` (default 2.5).

**Valuation:**

- `discount_rate` — required rate of return for DCF. Default: 10%.
- `terminal_growth_rate` — perpetual growth rate after projection
  period. Default: 1%.
- `margin_of_safety` — discount applied to intrinsic value. Default:
  50%.

**Output:**

- `output_directory` — where results and logs are written. Default:
  `output/`.
- `target_watchlist_size` — number of companies in the final watchlist.
  Default: 40.

All parameters and their defaults are listed in TECHNICAL_REFERENCE.md.

## Running the Pipeline

### Command Line

```bash
# Australian market (default)
python -m pipeline

# US market
python -m pipeline --market US

# Owned portfolio mode
python -m pipeline --market AU --mode owned --owned BHP CBA WES

# Custom output directory and debug logging
python -m pipeline --market US --output-dir results/ --log-level DEBUG
```

CLI flags:

| Flag | Description | Default |
|------|-------------|---------|
| `--market` | Market to analyse (AU, US, UK, CA, NZ, SG, HK) | AU |
| `--mode` | "shortlist" or "owned" | shortlist |
| `--owned` | Ticker symbols for owned mode | — |
| `--output-dir` | Output directory | output |
| `--log-level` | DEBUG, INFO, WARNING, ERROR | INFO |

### From Python

```python
from pipeline.config.settings import AnalysisConfig
from pipeline.runner import run_analysis

config = AnalysisConfig(market="US", mode="shortlist")
results = run_analysis(config)

print(f"Ranked {len(results.combined_rankings)} companies")
print(f"Watchlist: {results.watchlist}")
```

### Pipeline Steps

The pipeline executes 12 steps in sequence:

1. **Validate config** — check parameter consistency.
2. **Load data** — read financial data from FMP SQLite database.
3. **Derived metrics** — compute market cap, EV, ratios from latest
   period.
4. **Growth statistics** — compute mean, variance, CAGR from time
   series.
5. **Filter** — apply financial health filters (4 toggleable).
6. **Growth projection** — fade-to-equilibrium model, 3 scenarios.
7. **DCF valuation** — intrinsic value per share, 3 scenarios.
8. **Live prices** — fetch current market prices (FMP or yfinance).
9. **Live metrics** — update market cap and EV with live prices.
10. **Ranking** — risk-adjusted + penalty-based scoring, 4 DataFrames.
11. **Factor analysis** — penalty contributions + quadrant analysis.
12. **Watchlist** — two-step selection (IV pre-filter then opportunity
    rank).

### Output Files

CSV files are exported to `{output_directory}/{market}/`:

| File | Content |
|------|---------|
| growth_rankings.csv | Companies ranked by growth rate |
| value_rankings.csv | Companies ranked by IV/price ratio |
| weighted_rankings.csv | Companies ranked by penalty score |
| combined_rankings.csv | Companies ranked by risk-adjusted score |
| factor_contributions.csv | Penalty factor percentages per company |
| watchlist.csv | Selected watchlist symbols |

## Loading Data

The loader reads financial data from the FMP SQLite database and returns
a validated, cleaned dataset.

```python
from pipeline.config.settings import AnalysisConfig
from pipeline.data.loader import load_raw_data

config = AnalysisConfig(market="AU")
result = load_raw_data(config)

print(f"Loaded {result.company_count} companies, {result.row_count} rows")
print(f"Period range: {result.period_range[0]}-{result.period_range[1]}")
print(f"Dropped companies log: {result.dropped_companies_path}")
```

### What the Loader Does

1. Connects to the FMP database (read-only).
2. Finds all actively traded companies on the configured exchange(s),
   excluding ETFs, ADRs, and funds.
3. Loads income statement, balance sheet, cash flow, and growth data.
4. Joins the tables together. Companies missing from any required table
   are dropped.
5. Filters by reporting currency (e.g. only AUD for the AU market).
6. Matches share prices to financial statement dates (nearest trading
   day within the configured window).
7. Determines a common fiscal year range where most companies have data.
8. Drops companies with gaps in their data or missing values in
   required financial columns.
9. Assigns a sequential period index for time-series analysis.
10. Writes a CSV log of all dropped companies with reasons.

### Output

`load_raw_data` returns a `RawFinancialData` object containing:

- `data` — a pandas DataFrame with one row per company per quarter (or
  year). Columns include financials (revenue, free cash flow, debt,
  etc.), share price, and company metadata.
- `company_count` — number of companies retained after all filtering.
- `row_count` — total rows in the DataFrame.
- `period_range` — (earliest year, latest year) of data retained.
- `dropped_companies_path` — path to the CSV log of companies that
  were dropped, with reasons.

### Dropped Companies Log

Written to `output/data_loading/dropped_companies_{timestamp}.csv`.
Each row records one dropped company:

- `symbol` — ticker symbol.
- `reason` — why it was dropped (e.g. "temporal_gap",
  "partial_nan:revenue", "non_allowed_currency:USD").
- `missing_fields` — which columns or years were problematic.

## Derived Metrics

After loading, the pipeline extracts the latest-period snapshot for each
company and computes derived ratios:

```python
from pipeline.analysis.derived_metrics import compute_derived_metrics

companies = compute_derived_metrics(result.data)

# One row per company, indexed by entity_id
print(companies[["symbol", "market_cap", "enterprise_value", "debt_cash_ratio"]])
```

Computed columns: market_cap, enterprise_value, debt_cash_ratio,
fcf_per_share, acquirers_multiple, fcf_to_market_cap.

## Growth Statistics

Computes per-company growth statistics from the time series data:

```python
from pipeline.analysis.growth_stats import compute_growth_statistics

growth = compute_growth_statistics(result.data, config)

# One row per company
print(growth[["fcf_growth_mean", "revenue_growth_mean", "fcf_cagr", "growth_stability"]])
```

Outputs include mean, variance, and standard deviation for FCF and
revenue growth rates, annualised CAGR for both metrics, a combined
growth mean (weighted by config), and a growth stability score (higher
is more stable, bounded 0–1).

CAGR is computed from absolute values (not growth rates) using a
point-to-point calculation from the first to last valid quarterly value.
Requires at least 4 quarters between valid points. Returns 0.0 when
data is insufficient or values change sign.

## Filtering

Four financial health filters can be toggled on or off independently:

```python
from pipeline.analysis.filtering import apply_filters

filtered, filter_log = apply_filters(companies, config)

print(f"Retained {len(filtered)} of {len(companies)} companies")
print(f"Removed by filter: {filter_log.removed}")
```

| Filter | What it removes | Config toggle |
|--------|----------------|---------------|
| Negative FCF | Companies with FCF ≤ 0 | enable_negative_fcf_filter |
| Data consistency | Operating income exceeds revenue | enable_data_consistency_filter |
| Market cap | Below min_market_cap ($20M default) | enable_market_cap_filter |
| Debt-to-cash | Above max_debt_to_cash_ratio (2.5 default) | enable_debt_cash_filter |

**Owned mode:** When `mode="owned"`, companies in `owned_companies`
bypass all filters. The filter log still tracks which filters they would
have failed, so you can see the health profile of your holdings.

## Growth Projection

Projects future growth for FCF and revenue using a fade-to-equilibrium
model. Each company gets three scenarios (base, optimistic, pessimistic)
for each configured projection period (default: 5 and 10 years).

```python
from pipeline.analysis.growth_projection import project_all

projections = project_all(companies, growth_stats, config)

# Access a specific projection
proj = projections[entity_id][5]["fcf"]["base"]
print(f"5-year FCF CAGR (base): {proj.annual_cagr:.1%}")
print(f"Projected quarterly values: {len(proj.quarterly_values)} quarters")
```

The model starts from a company's historical mean growth rate and
exponentially fades toward a long-run equilibrium rate (default 3%
annual). Larger companies fade faster (less room to grow). Scenarios
adjust the starting growth rate by +/- one standard deviation.

Companies with negative free cash flow use a separate model that
improves FCF toward zero at a rate derived from revenue growth, then
switches to conservative positive growth once near breakeven.

### Key Parameters

- `equilibrium_growth_rate` — long-run growth rate all companies
  converge toward. Default: 3% annual.
- `base_fade_half_life_years` — time for growth to move halfway to
  equilibrium. Default: 2.5 years.
- `scenario_band_width` — standard deviations for optimistic/pessimistic
  scenarios. Default: 1.0.
- `projection_periods` — years to project. Default: (5, 10).

## DCF Valuation

Estimates intrinsic value per share using a discounted cash flow model
with quarterly internal discounting. Uses the growth projection output.

```python
from pipeline.analysis.dcf import calculate_all_dcf

intrinsic_values = calculate_all_dcf(companies, projections, config)

# Access a specific valuation
iv = intrinsic_values[entity_id][5]["base"]
print(f"IV/share (base, 5yr): ${iv.iv_per_share:.2f}")
print(f"Present value: ${iv.present_value:,.0f}")
print(f"Terminal value: ${iv.terminal_value:,.0f}")
```

The DCF projects quarterly cash flows from the fade model, discounts
them to present value, adds a Gordon Growth Model terminal value, and
applies the margin of safety before converting to per-share value.

### Key Parameters

- `discount_rate` — required rate of return. Default: 10%.
- `terminal_growth_rate` — perpetual growth rate after projection.
  Default: 1%.
- `margin_of_safety` — discount applied to intrinsic value. Default:
  50% (a $10 calculated fair value reports as $5 IV per share).

## Live Prices

Fetches current market prices for ranking and IV/price ratio
calculations. Two providers available:

```python
from pipeline.data.live_prices import auto_select_provider

provider = auto_select_provider()
prices = provider.get_prices(["AAPL", "MSFT", "BHP.AX"])
```

- **FMP** (primary) — uses the FMP API batch-quote endpoint. Requires
  `FMP_API_KEY` environment variable.
- **yfinance** (fallback) — used when no FMP API key is available.
  Requires the `yfinance` package.

`auto_select_provider()` picks FMP if the API key is set, otherwise
falls back to yfinance. Symbols with no price available are omitted
from the result.

## Weighted Scoring

Scores companies using a penalty system. Three penalty types: debt-cash
(penalises high leverage), market-cap (penalises large size), and growth
(penalises low growth, instability, and FCF/revenue divergence).

```python
from pipeline.analysis.weighted_scoring import calculate_weighted_scores

weighted_scores = calculate_weighted_scores(companies, growth_stats, config)

# Lower total_penalty = better
print(weighted_scores[["symbol", "dc_penalty", "mc_penalty", "growth_penalty", "total_penalty", "weighted_rank"]])
```

All penalty weights are configurable: `dc_weight`, `mc_weight`,
`growth_weight`, and the three growth sub-weights (`growth_rate_subweight`,
`growth_stability_subweight`, `growth_divergence_subweight`).

## Ranking

Ranks companies using a risk-adjusted score (primary) and a
penalty-based weighted score (secondary). Produces four ranking
DataFrames. Only companies with a live price are included. Companies
where the pessimistic intrinsic value is below the current price are
excluded (safety gate).

```python
from pipeline.analysis.ranking import rank_companies

growth_rankings, value_rankings, weighted_rankings, combined_rankings = rank_companies(
    companies, growth_stats, projections, intrinsic_values,
    weighted_scores, live_prices, config,
)

# Combined rankings is the primary output
print(combined_rankings[["symbol", "risk_adjusted_score", "opportunity_rank", "opportunity_score"]])
```

| DataFrame | What it ranks by | Use case |
|-----------|-----------------|----------|
| growth_rankings | Combined growth rate (desc) | Find fastest growers |
| value_rankings | Composite IV/price ratio (desc) | Find most undervalued |
| weighted_rankings | Total penalty (asc) | Find lowest-risk companies |
| combined_rankings | Risk-adjusted score (desc) | Primary ranking for watchlist selection |

The composite IV/price ratio is a weighted average across scenarios at
the primary period: 25% pessimistic + 50% base + 25% optimistic.

The risk-adjusted score combines expected return (growth + valuation
upside) with a composite safety score. Four safety factors are
evaluated: downside exposure (how far pessimistic IV exceeds current
price), scenario spread (tightness of optimistic-pessimistic range),
terminal value dependency (how much of the DCF depends on the terminal
value), and FCF reliability (proportion of quarters with positive free
cash flow). Each factor is normalised to 0–1 (higher = safer) and
weighted to produce a composite safety score. The final risk-adjusted
score = expected return × composite safety.

## Factor Analysis

Analyses what drives each company's penalty score and classifies
companies into growth-value quadrants.

```python
from pipeline.analysis.factor_analysis import (
    calculate_factor_contributions,
    analyze_factor_dominance,
    create_quadrant_analysis,
)

contributions = calculate_factor_contributions(weighted_scores)
print(contributions[["dc_pct", "mc_pct", "growth_pct"]])  # Percentages

dominance = analyze_factor_dominance(contributions, weighted_scores)
print(dominance)  # Which factor dominates most companies

quadrants = create_quadrant_analysis(
    companies, intrinsic_values, live_prices, growth_stats, config,
)
print(quadrants[["symbol", "quadrant"]])  # 1=best, 4=worst
```

Quadrant 1 (high growth + high value) companies are the strongest
candidates. Quadrant 4 (low growth + low value) are the weakest.

## Watchlist Selection

Selects the final watchlist using a two-step process:

```python
from pipeline.analysis.watchlist import select_watchlist

watchlist = select_watchlist(combined_rankings, config)
print(f"Watchlist ({len(watchlist)} companies): {watchlist}")
```

1. **Pre-filter:** Take the top N companies by composite IV/price ratio
   (`iv_prefilter_count`, default 100).
2. **Select:** From that set, take the top M by opportunity rank
   (`target_watchlist_size`, default 40).

In owned mode, companies in your `owned_companies` list are always
included in the watchlist regardless of their rank.

## Supported Markets

| Code | Country | Exchanges |
|------|---------|-----------|
| AU | Australia | ASX |
| US | United States | NASDAQ, NYSE, AMEX |
| UK | United Kingdom | LSE |
| CA | Canada | TSX, CNQ, NEO, TSXV |
| NZ | New Zealand | NZE |
| SG | Singapore | SES |
| HK | Hong Kong | HKSE |

## Charts

All chart functions take `AnalysisResults` and return matplotlib
figures. Three chart modules cover different scopes:

### Market Overview Charts

```python
from pipeline.charts import (
    growth_value_scatter,
    growth_comparison_historical,
    growth_comparison_projected,
    risk_adjusted_opportunity,
    factor_contributions_bar,
    factor_heatmap,
)

fig = growth_value_scatter(results)
fig.savefig("growth_value.png", dpi=150, bbox_inches="tight")
```

| Chart | What it shows |
|-------|--------------|
| growth_value_scatter | Growth rank vs composite IV/price ratio, watchlist annotated |
| growth_comparison_historical | FCF vs revenue CAGR for watchlist companies |
| growth_comparison_projected | Projected CAGR with scenario error bars |
| risk_adjusted_opportunity | Opportunity rank vs composite safety score |
| factor_contributions_bar | Penalty factor breakdown per company |
| factor_heatmap | Factor contribution percentages as heatmap |

### Comparative Charts

Take a list of symbols to compare:

```python
from pipeline.charts import (
    projected_growth_stability,
    acquirers_multiple_analysis,
    valuation_upside_comparison,
    ranking_comparison_table,
    comprehensive_rankings_table,
)

symbols = results.watchlist[:20]
fig = projected_growth_stability(results, symbols)
```

### Company Detail Charts

Take an entity_id for a specific company:

```python
from pipeline.charts import (
    historical_growth,
    growth_projection,
    valuation_matrix,
    growth_fan,
    financial_health_dashboard,
    risk_spider,
    scenario_comparison,
)

# Look up entity_id from symbol
for eid in results.companies.index:
    if str(results.companies.loc[eid, "symbol"]) == "BHP":
        fig = scenario_comparison(results, eid)
        break
```

The risk_spider chart shows four normalised safety scores
(downside_exposure_score, scenario_spread_score,
terminal_dependency_score, fcf_reliability_score) as a radar plot,
all bounded [0, 1] where 1 = safest. Composite safety score displayed
at the centre.

The scenario_comparison chart is a 2x2 grid showing DCF waterfall,
sensitivity analysis, growth trajectories, and risk-return profile.

### Summary Tables

```python
from pipeline.charts import watchlist_summary, filter_summary

fig = watchlist_summary(results)  # Watchlist companies table
fig = filter_summary(results)     # How many companies each filter removed
```

## PDF Reports

Generate a comprehensive PDF report with all charts:

```python
from pathlib import Path
from pipeline.reports.pdf import generate_pdf

pdf_path = Path("output/AU/report.pdf")
generate_pdf(results, pdf_path)
```

The report includes:
- Title page with key parameters
- Executive summary with filter statistics
- Market overview charts (6 pages)
- Comparative analysis for watchlist companies (4 pages)
- Detailed company analysis for top companies (6 charts each)
- Watchlist summary table
- Full rankings table

To customise which companies get detailed pages:

```python
generate_pdf(results, pdf_path, detailed_symbols=["BHP", "CBA", "WES"])
```

By default, the top N companies from the watchlist are used, where N is
`config.detailed_report_count` (default: 5).

**Requirements:** WeasyPrint requires system-level dependencies for PDF
rendering (Cairo, Pango, GDK-PixBuf). On Ubuntu/Debian:
`apt install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0`.

## Analysis Notebook

A Jupyter notebook at `notebooks/analysis.ipynb` provides an
interactive environment for running the pipeline and exploring results.

```bash
cd notebooks
jupyter notebook analysis.ipynb
```

The notebook is a thin harness: configure, run the pipeline, then
display charts one cell at a time. It uses the same chart functions as
the PDF report.

## Running Tests

```bash
source .venv/value_investing/bin/activate
pytest -v tests/
```

The loader tests require the FMP database to be present at the
configured path. They are automatically skipped if the file is not
found.

## Pre-commit Checks

```bash
ruff check pipeline/ tests/    # Linting
mypy pipeline/ tests/          # Type checking
pytest -v tests/               # Tests
```
