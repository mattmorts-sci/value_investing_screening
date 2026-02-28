# User Guide

## Overview

The value investing screening pipeline screens companies from a financial database, computes valuation, quality, safety, and trend metrics, runs a Monte Carlo simulation to estimate intrinsic value, and produces ranked output. Two modes of operation:

- **Screen** — bulk-screen companies from one or more exchanges and export a ranked CSV.
- **Detail** — generate an HTML report for specific tickers with embedded charts, metric breakdowns, and sensitivity analysis.

The pipeline reads quarterly financial data from a local FMP (Financial Modeling Prep) database and computes all metrics locally.


## Prerequisites

- **Python 3.12** or later
- **FMP database** — a local SQLite database populated by the FMP data collector (default path: `~/projects/Pers/financial_db/data/fmp.db`)
- Price history in the database (used for sentiment metrics)


## Installation

From the project root:

    pip install -e .

This installs the `pipeline` package and its dependencies (pandas, numpy, scipy, matplotlib, yfinance, requests).


## Running the Pipeline

The pipeline is invoked as a Python module with a command (`screen` or `detail`).

### Screening

#### Basic usage

    python -m pipeline screen --exchanges ASX

#### Multiple exchanges

    python -m pipeline screen --exchanges NYSE NASDAQ

#### Custom sort and output path

    python -m pipeline screen --exchanges ASX --sort fcf_ev --output results.csv

#### With filters enabled

    python -m pipeline screen --exchanges ASX --min-market-cap 500000000 --min-interest-coverage 3.0

#### Verbose logging

    python -m pipeline screen --exchanges ASX -v

#### All screen arguments

| Argument | Description | Default |
|---|---|---|
| `--exchanges` | Stock exchanges to screen (required). Space-separated list. | — |
| `--sort` | Sort metric: `composite`, `fcf_ev`, or `ebit_ev`. | `composite` |
| `--output` | Output CSV file path. | `output/screening.csv` |
| `--min-market-cap` | Minimum market capitalisation filter. | Disabled |
| `--min-interest-coverage` | Minimum interest coverage ratio filter. | Disabled |
| `--min-ocf-to-debt` | Minimum OCF-to-debt ratio filter. | Disabled |
| `--min-fcf-conversion-r2` | Minimum FCF conversion R-squared filter. | Disabled |
| `--include-negative-fcf` | Include companies with negative trailing twelve-month free cash flow. | Excluded |
| `--db-path` | Path to the FMP SQLite database. | `~/projects/Pers/financial_db/data/fmp.db` |
| `--min-quarters` | Minimum quarters of financial data required. | 12 |
| `-v`, `--verbose` | Enable debug-level logging. | Info level |

### Detail Reports

Generate an HTML detail report for one or more tickers. Each report includes embedded charts, metric breakdowns, and sensitivity analysis.

#### Single ticker

    python -m pipeline detail CBA.AX

#### Multiple tickers

    python -m pipeline detail AAPL MSFT GOOG

#### Custom output directory

    python -m pipeline detail CBA.AX --output-dir reports/

#### All detail arguments

| Argument | Description | Default |
|---|---|---|
| `tickers` | Ticker symbols to generate reports for (positional, one or more). | — |
| `--output-dir` | Directory for output HTML files. Created if it does not exist. | `output/detail` |
| `--db-path` | Path to the FMP SQLite database. | `~/projects/Pers/financial_db/data/fmp.db` |
| `-v`, `--verbose` | Enable debug-level logging. | Info level |

Each report is written as `{TICKER}.html` in the output directory. Tickers not found in the database are skipped with a warning.

### Report contents

The HTML report is a self-contained file with embedded charts (no external dependencies). It contains up to five sections (charts section omitted when insufficient data):

**Header** — price, market cap, FCF/EV, EBIT/EV, margin of safety (median), and F-Score.

**Charts** — revenue growth and operating margin charts appear when sufficient financial data exists. Projection, IV, and sensitivity charts require successful simulation:

| Chart | Description |
|---|---|
| Revenue Growth | Year-over-year quarterly revenue growth bars with mean and ±1σ reference lines. |
| Operating Margin | Historical operating margins as a scatter with regression trend line. |
| Revenue Projection | Historical revenue + projected sample paths with P10–P90 and P25–P75 shaded bands. |
| FCF Projection | Same structure as revenue projection, for free cash flow. |
| IV Scenarios | Horizontal bars for P25/P50/P75 intrinsic value estimates with current price reference line. |
| Sensitivity Analysis | Heatmap of intrinsic value across discount rate and growth multiplier combinations. Coloured by margin of safety with a break-even contour. |

**Quality** — F-Score breakdown (9 components as checkmarks/crosses), gross profitability, accruals ratio, and sentiment (6- and 12-month returns).

**Trends** — revenue CAGR, growth volatility, margin and FCF conversion trends (slope and R²), ROIC metrics.

**Safety** — interest coverage and OCF-to-debt ratio. Values below 1.5× are highlighted in amber.

**Visual encoding:** F-Score is colour-coded — green for 7–9 (strong), amber for 4–6 (moderate), red for 0–3 (weak).


## Understanding the Output

The sections below describe the CSV output from the `screen` command.

The output CSV contains one row per company. Companies that pass all filters appear first, sorted by the chosen valuation yield (highest first). Filtered companies follow, also sorted by yield.

### Identity

| Column | Description |
|---|---|
| `symbol` | Stock ticker symbol. |
| `company_name` | Company name. |
| `sector` | Business sector classification. |
| `price` | Latest stock price. |
| `market_cap` | Market capitalisation. |

### Valuation Yields

All yields are expressed as decimals (e.g., 0.08 = 8%).

| Column | Description |
|---|---|
| `fcf_ev` | Trailing twelve-month free cash flow divided by enterprise value. Higher means cheaper on a cash flow basis. |
| `ebit_ev` | Trailing twelve-month EBIT divided by enterprise value. Higher means cheaper on an earnings basis. |
| `composite_yield` | Weighted blend of FCF/EV and EBIT/EV. Default weights: 60% FCF/EV, 40% EBIT/EV. |

Enterprise value = market cap + total debt - cash.

### Intrinsic Value Estimates

Intrinsic value (IV) per share is estimated by a Monte Carlo simulation that projects revenue, operating margins, and FCF conversion forward, then discounts the resulting cash flows back to the present. The simulation runs 10,000 paths and reports percentiles of the resulting distribution.

| Column | Description |
|---|---|
| `iv_p25` | 25th percentile IV per share. 75% of simulated outcomes produced a higher value. A conservative estimate. |
| `iv_p50` | 50th percentile (median) IV per share. The central estimate. |
| `iv_p75` | 75th percentile IV per share. Only 25% of simulated outcomes exceeded this. An optimistic estimate. |
| `iv_spread` | P75 minus P25. Measures how wide the range of outcomes is. A large spread means high uncertainty. |

IVs are rounded to the nearest dollar for stocks priced at $10 or above, and to the nearest 10 cents below $10.

### Margin of Safety

Margin of safety (MoS) measures how far the current price sits below the estimated intrinsic value. Expressed as a decimal.

    MoS = (IV - price) / IV

| Column | Description |
|---|---|
| `mos_p25` | Margin of safety using the P25 (conservative) IV estimate. |
| `mos_p50` | Margin of safety using the P50 (median) IV estimate. |
| `mos_p75` | Margin of safety using the P75 (optimistic) IV estimate. |

A positive MoS means the stock is priced below the IV estimate (potential undervaluation). A negative MoS means the stock is priced above the IV estimate. For example, MoS of 0.30 means the price is 30% below the estimated IV.

### Implied CAGR

The implied compound annual growth rate (CAGR) is the annualised return you would earn if you bought at the current price and the stock eventually reached the estimated IV over the projection period (default: 10 years).

| Column | Description |
|---|---|
| `implied_cagr_p25` | Implied annualised return to P25 IV. |
| `implied_cagr_p50` | Implied annualised return to P50 IV. |
| `implied_cagr_p75` | Implied annualised return to P75 IV. |

### Trends

| Column | Description |
|---|---|
| `revenue_cagr` | Annualised compound growth rate of revenue over the full data history. |
| `revenue_yoy_growth_std` | Standard deviation of year-over-year quarterly revenue growth rates. Measures growth volatility — lower is more consistent. |
| `margin_slope` | Annualised slope of the operating margin regression. Positive means margins are improving. Expressed as a decimal — a value of 0.02 means the margin ratio increases by 2 percentage points per year. |
| `margin_r_squared` | R-squared of the operating margin regression. Higher means the margin trend is more consistent. |
| `conversion_slope` | Annualised slope of the FCF conversion regression. FCF conversion is free cash flow divided by operating income. Positive means conversion is improving. |
| `conversion_r_squared` | R-squared of the FCF conversion regression. Higher means the conversion trend is more consistent. |

### Quality

#### Piotroski F-Score

The F-Score is a 0-to-9 composite of binary financial health signals. Each component is TRUE (1 point) or FALSE (0 points). Higher is stronger.

| Column | Description |
|---|---|
| `f_score` | Total F-Score (sum of the 9 components below). |
| `f_roa_positive` | TTM return on assets is positive. |
| `f_ocf_positive` | TTM operating cash flow is positive. |
| `f_roa_improving` | TTM ROA has improved year-on-year. |
| `f_accruals_negative` | TTM operating cash flow exceeds TTM net income (earnings quality). |
| `f_leverage_decreasing` | Long-term debt to total assets has decreased year-on-year. |
| `f_current_ratio_improving` | Current ratio has improved year-on-year. |
| `f_no_dilution` | Diluted shares outstanding have not increased year-on-year. |
| `f_gross_margin_improving` | TTM gross margin has improved year-on-year. |
| `f_asset_turnover_improving` | TTM asset turnover has improved year-on-year. |

Year-on-year comparisons use the most recent 4 quarters against the prior 4 quarters. Requires at least 8 quarters of data for year-on-year signals; those signals default to FALSE when data is insufficient.

#### Other Quality Metrics

| Column | Description |
|---|---|
| `gross_profitability` | TTM gross profit divided by latest total assets (Novy-Marx gross profitability). Higher indicates stronger profitability per unit of assets deployed. |
| `accruals_ratio` | (TTM net income - TTM operating cash flow) / latest total assets. Lower (more negative) is better — it means more of the earnings are backed by real cash flow. |

### ROIC

Return on invested capital, where NOPAT = EBIT x (1 - effective tax rate) and invested capital = total assets - total current liabilities.

| Column | Description |
|---|---|
| `roic_latest` | ROIC of the most recent valid quarter. |
| `roic_slope` | Annualised slope from a linear regression of ROIC over time. Positive means ROIC is trending upward. |
| `roic_detrended_std` | Standard deviation of the ROIC regression residuals. Measures ROIC stability — lower is more consistent. |
| `roic_minimum` | Minimum ROIC across all valid quarters. Shows worst-case capital efficiency. |

### Safety

| Column | Description |
|---|---|
| `interest_coverage` | TTM EBIT divided by TTM interest expense. Measures ability to service debt from operating earnings. Higher is safer. Empty if the company has no interest expense. |
| `ocf_to_debt` | TTM operating cash flow divided by latest total debt. Measures ability to retire debt from cash flow. Higher is safer. Empty if the company has no debt. |

### Sentiment

| Column | Description |
|---|---|
| `return_6m` | Simple price return over the past 6 months. |
| `return_12m` | Simple price return over the past 12 months. |

### Filter Status

| Column | Description |
|---|---|
| `filtered` | TRUE if the company was excluded by one or more filters. FALSE if it passed all filters. |
| `filter_reasons` | Comma-separated reason codes explaining why the company was filtered. Empty if not filtered. |


## Filters

Filters flag companies that fail data sufficiency or configurable threshold checks. Filtered companies remain in the CSV with `filtered = TRUE` and the relevant reason codes. They are sorted to the bottom of the output.

Filtering happens in two stages.

### Stage 1: Data Sufficiency

This filter cannot be disabled. It triggers when:

- The company has fewer quarters of data than `--min-quarters` (default: 12).
- Trend metrics could not be computed (insufficient data for regressions).
- The Monte Carlo simulation could not run (depends on successful trends and positive starting revenue).

**Reason code:** `DATA`

### Stage 2: Configurable Filters

These filters only apply to companies that pass Stage 1. Each is independently togglable.

| Filter | CLI argument | Reason code | Triggers when |
|---|---|---|---|
| Market cap | `--min-market-cap` | `MC` | Market cap is below the specified threshold. |
| Interest coverage | `--min-interest-coverage` | `IC` | Interest coverage is below the threshold, or is unavailable. |
| OCF to debt | `--min-ocf-to-debt` | `OD` | OCF-to-debt ratio is below the threshold, or is unavailable. |
| FCF conversion R² | `--min-fcf-conversion-r2` | `CR` | FCF conversion regression R-squared is below the threshold, or is unavailable. |
| Negative TTM FCF | `--include-negative-fcf` | `FCF` | Trailing twelve-month free cash flow is zero or negative. Enabled by default; pass `--include-negative-fcf` to disable. |

A company can trigger multiple filters. When it does, all applicable reason codes appear comma-separated (e.g., `MC,IC`).


## Sort Options

The `--sort` argument controls how companies are ranked within the passed and filtered groups.

| Option | Description |
|---|---|
| `composite` | Weighted blend of FCF/EV and EBIT/EV (default weights: 60/40). This is the default. |
| `fcf_ev` | Sort by FCF/EV yield only. |
| `ebit_ev` | Sort by EBIT/EV yield only. |

Within both the passed and filtered groups, companies are sorted from highest yield to lowest.


## Configuration

### Parameters you might adjust

| Parameter | CLI argument | Default | Description |
|---|---|---|---|
| Minimum quarters | `--min-quarters` | 12 | Minimum quarters of financial data required. Companies with fewer quarters are filtered with reason code `DATA`. |
| Market cap filter | `--min-market-cap` | Disabled | Set a minimum market cap to exclude micro-caps. |
| Interest coverage filter | `--min-interest-coverage` | Disabled | Set a minimum interest coverage ratio. |
| OCF/debt filter | `--min-ocf-to-debt` | Disabled | Set a minimum operating cash flow to debt ratio. |
| FCF conversion R² filter | `--min-fcf-conversion-r2` | Disabled | Set a minimum R-squared for the FCF conversion regression. Filters companies with unpredictable cash flow conversion. |
| Negative FCF | `--include-negative-fcf` | Excluded | Include companies that are currently burning cash (negative TTM FCF). |
| Sort metric | `--sort` | `composite` | Which yield metric to rank by. |
| Database path | `--db-path` | `~/projects/Pers/financial_db/data/fmp.db` | Path to the FMP SQLite database. |

### DCF parameters

These are set to reasonable defaults and are not exposed as CLI arguments. They can be changed by modifying `DCFConfig` in the source.

| Parameter | Default | Description |
|---|---|---|
| Discount rate | 10% | Rate used to discount projected cash flows to present value. |
| Terminal growth rate | 2.5% | Assumed perpetual growth rate for the terminal value calculation. |
| Projection years | 10 | Number of years of cash flows projected in the simulation. |

### Simulation parameters

The Monte Carlo simulation uses multiple constraints to keep projected growth paths realistic. These are set in `SimulationConfig` and rarely need changing. The simulation runs 10,000 paths by default.
