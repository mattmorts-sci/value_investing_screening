# Conceptual Plan

## Purpose

A screening tool that identifies companies likely to be undervalued
relative to their future cash generation. It produces a candidate list
for further human analysis, not a buy list.

## Screening Sequence

### Stage 1: Data Sufficiency

Exclude companies where required fields are missing or fewer than
12 quarters (3 years) of financial history are available. Threshold
configurable; 12 is the minimum for meaningful regression fits and
growth distribution parameterisation. Pure data quality gate.

### Stage 2: Configurable Filters

All filters toggleable and threshold-configurable. Nothing hardcoded
as mandatory. Filter candidates:

- Minimum market cap
- Minimum interest coverage
- Minimum OCF / Total Debt threshold
- Minimum FCF conversion R²
- Negative TTM FCF exclusion

A company excluded by a filter remains visible (greyed out or flagged)
so the user knows what was filtered and why.

### Stage 3: Valuation Ranking

Two complementary valuation approaches:

**Ratio-based (primary sort):**

- EBIT yield (EBIT / EV)
- FCF / EV
- Weighted composite of both (default: FCF/EV 0.6, EBIT/EV 0.4)

FCF/EV weighted higher because cash flow is harder to manipulate than
operating income. EBIT/EV retained because it smooths capex timing
noise. Weights configurable. Three sort options: weighted composite
(default), FCF/EV only, or EBIT/EV only.

**IV-based (forward-looking):**

Monte Carlo simulation projecting revenue growth, applying margin and
FCF conversion trajectories to derive projected FCF, discounted to
present value. Produces IV per share and margin of safety relative to
current price.

The ratios say "is it cheap now?" The IV estimate says "is it cheap
relative to what it could produce over the next N years?"

### Stage 4: Quality Context

All quality metrics displayed alongside the valuation ranking. No
filtering, no scoring. Informs the user's judgement about each
company.

## Trend and Model Fit Metrics

For each core time series, the pipeline computes a trend and a measure
of how well that trend describes the data. These are display-only
quality context — not used for filtering, scoring, or modifying the
Monte Carlo simulation.

The fit metrics serve a specific purpose: the Monte Carlo applies
operating margin and FCF conversion as deterministic trend lines. The
R² of each regression tells the user how reliable that assumption is
for a given company. A high R² means the trend captures most of the
variation. A low R² means the deterministic assumption is a poor fit
and the IV estimate rests on weak foundations.

### Revenue

- **Trend:** historical CAGR (display only — computed from first and
  last revenue values as a human-readable summary)
- **Dispersion:** standard deviation of quarterly year-over-year
  revenue growth rates (display only — YoY removes seasonality,
  making the number meaningful to a human). Reported directly
  (e.g., "std 3pp") — not divided by the mean. Avoids the failure
  mode where low-growth companies appear artificially unstable.

Note: the Monte Carlo uses a separate calculation — mean and variance
of sequential QoQ growth rates (see Growth Input below). CAGR and
YoY std are display metrics; QoQ mean/variance are simulation inputs.

### Operating Margin

- **Trend:** slope of linear regression over available quarters
  (direction and rate of change — improving, flat, or deteriorating)
- **Model fit:** R² of the regression. Directly indicates how well
  the deterministic trend line (used in the Monte Carlo) describes
  the historical data.

### FCF Conversion (FCF / Operating Income)

- **Trend:** slope of linear regression over available quarters
- **Model fit:** R² of the regression. Same purpose as operating
  margin R².

Quarters where operating income is ≤ 0 are excluded from the
regression (the ratio is meaningless when the denominator is zero or
negative). If fewer than 8 valid quarters remain, the trend and R²
are not computed — the Monte Carlo uses a flat conversion assumption
(median of valid quarters) instead of a trend line.

### ROIC

- **Trend:** slope of linear regression over available quarters,
  calculated from raw quarterly financials:
  - NOPAT = EBIT × (1 − effective tax rate), where effective tax
    rate = incomeTaxExpense / incomeBeforeTax
  - Invested Capital = totalAssets − totalCurrentLiabilities
  - Quarters where incomeBeforeTax ≤ 0 are excluded (effective tax
    rate is undefined)
- **Dispersion:** standard deviation of residuals from the fitted
  trend (detrended). Separates noise from trend — a company improving
  from 10% to 20% ROIC is not "unstable."
- **Floor:** minimum ROIC over the available period. A company that
  never dropped below 15% is a different proposition from one that
  hit 2%.

## Monte Carlo Simulation

### Growth Input

Revenue growth (not FCF growth). Revenue is more stable and harder to
fake. QoQ growth rates computed from quarterly revenue data. Mean and
variance parameterise the log-normal sampling distribution.

### Deriving FCF from Revenue

The simulation projects revenue, then applies:

1. **Operating margin trajectory** — trend-based, fitted from
   historical regression. If margins are trending down, the projection
   reflects that rather than assuming they stay flat.
2. **FCF conversion trajectory** — trend-based, fitted from historical
   FCF/operating-income ratio. Same principle: observed trend
   extrapolated, not averaged.

This produces projected FCF each period. Margin and conversion
trajectories are deterministic (trend-based); the stochastic element
is in revenue growth only. Primary uncertainty lives in the signal we
trust most.

### Constraint System

Carried over from existing design, recalibrated for revenue (less
volatile than FCF, so caps are tighter).

1. **Log-normal sampling with CV cap at 1.0.** The log-normal
   distribution itself prevents structurally impossible growth
   (< -100%). The CV cap tames extreme right-tail outcomes by
   limiting the dispersion parameter.

2. **Per-quarter revenue growth caps:**

   | Size evolution | Positive cap | Negative cap |
   |----------------|-------------|-------------|
   | ≤ 2.0x        | +20%        | -15%        |
   | > 2.0x        | +12%        | -8%         |

   Tighter than existing FCF caps (+40%/-30% and +25%/-20%) because
   revenue is inherently less volatile than FCF.

3. **Cumulative 5x cap.** Hard ceiling on total revenue growth over
   the projection period. Down from 10x for FCF.

4. **50% annual CAGR backstop.** At each quarter beyond Q4, if
   cumulative growth from the start implies > 50% annualised CAGR,
   clamp the projected value to start × 1.50^(years elapsed). Down
   from 100% for FCF.

5. **Percentile aggregation** of per-path present values
   (P10/P25/P50/P75/P90). P25/P50/P75 used for IV scenarios;
   P10/P90 used for chart bands only. Jensen's inequality naturally
   dampens the right tail.

6. **Momentum exhaustion.** Mean reversion within simulation paths
   when recent 4-quarter average growth exceeds 2x the historical
   mean. Upward bounce when below 0.5x mean. For companies with
   negative or near-zero historical mean growth, use absolute
   deviation bands instead of ratio-based thresholds.

7. **Time decay for high growth.** When quarterly growth exceeds 20%,
   multiply the full quarterly growth rate by 0.8^(years elapsed).
   Threshold lowered from 30% for revenue.

8. **Size-based growth penalty.** Dampens growth as projected revenue
   grows relative to starting revenue (0.4–1.0x multiplier).
   Sector-neutral — uses the same basis as the cumulative 5x cap.

### DCF Parameters

All configurable with stated defaults:

- **Discount rate:** 10% (rough equity cost of capital approximation;
  per-company WACC is false precision given Monte Carlo uncertainty)
- **Terminal growth rate:** 2.5% (long-term nominal GDP growth proxy)
- **Projection period:** 10 years (40 quarters)

### DCF and Scenario Extraction

The DCF is computed within the simulation: each Monte Carlo path
produces period-by-period FCF (revenue × margin trajectory ×
conversion trajectory at each step), discounted to present value.
This preserves the trending margin and conversion assumptions
rather than flattening them into a smooth growth rate.

Percentiles are taken from the distribution of per-path present
values:

- Pessimistic → 25th percentile
- Base → 50th percentile (median)
- Optimistic → 75th percentile

Annual CAGR is back-calculated from each percentile for display
only — so the user sees "this P50 implies X% annual growth."

## Balance Sheet Safety

Two metrics replacing the current debt-to-cash ratio:

1. **Interest coverage ratio** (EBIT / Interest Expense) — strongest
   empirical support for predicting financial distress (Federal Reserve
   research, 2019). Uses GAAP EBIT, avoids EBITDA manipulation.
2. **Operating cash flow / Total Debt** — measures actual cash
   available to service and repay debt.

## Piotroski F-Score: Display Only

All 9 binary signals displayed individually, plus the composite score
(0–9) colour-coded by strength. Not used as a filter, not used in
ranking or scoring. Informational only — the user interprets it.

The 9 signals:

1. Positive return on assets
2. Positive operating cash flow
3. ROA improved year-over-year
4. Cash flow exceeds net income (earnings quality)
5. Long-term debt ratio decreased
6. Current ratio improved
7. No new shares issued
8. Gross margin improved
9. Asset turnover improved

## Quality Metrics: Display Only

### Gross Profitability (Gross Profit / Total Assets)

Quality signal separating "cheap and earning well" from "cheap and
earning poorly." Novy-Marx (2013, *Journal of Financial Economics*)
showed this has the same return-predicting power as book-to-market.
Gross profit sits above the income statement items most subject to
accounting discretion. Data fields already required for F-Score.

### Accruals Ratio ((Net Income − Operating Cash Flow) / Total Assets)

Continuous measure of how much of reported earnings is backed by cash.
Complements F-Score signal #4 (which is binary — cash flow > net
income yes/no). Sloan (1996, *The Accounting Review*) showed high
accruals predict lower future returns. Data fields already required
for other items in this plan.

### ROIC (Return on Invested Capital)

Calculated from raw quarterly financials as defined in the Trend and
Model Fit section (NOPAT = EBIT × (1 − effective tax rate), Invested
Capital = totalAssets − totalCurrentLiabilities). Produces a full
quarterly time series for trend, detrended residual std, and minimum
analysis. The FMP pre-calculated `returnOnInvestedCapitalTTM` (single
snapshot) is not suitable — it lacks historical depth.

### Market Sentiment (6-Month and 12-Month Price Return)

Neutral presentation of how the market has priced the company over
recent periods. Not colour-coded, not framed as a warning or signal.
Prompts the question: "Why does the market disagree?" — in either
direction. A stock down 30% and a stock up 40% both warrant the
question. Derived from existing price data.

## Valuation Denominator

FCF/EV replaces FCF/market_cap. Enterprise value (market cap + total
debt − cash) prices the whole business, not just the equity slice. Two
companies with identical FCF but different debt levels should not
screen the same. Already computed for the Acquirer's Multiple.

## Data Presentation

### Two-Phase Output

**Phase 1: CSV export (primary output).** One row per company, all
computed metrics. The user imports this into Google Sheets for
screening: sorting, filtering, removing non-starters, highlighting,
annotating. The pipeline produces the data; the spreadsheet handles
presentation and interaction.

**Phase 2: Per-company detail reports (on demand).** The user passes
back a list of tickers. The pipeline generates detailed HTML reports
with charts for those specific companies.

### CSV Structure

Sorted by valuation composite (default). No rank column — the sort
position serves as implicit ranking. Raw metric values let the user
see when the top candidates are clustered (FCF/EV of 6.1% vs 6.3%)
vs clearly separated.

Filtered companies are included in the same CSV with a `filtered`
boolean column and a `filter_reasons` string column (e.g., "MC,IC"
for market cap and interest coverage). The user filters on these in
the spreadsheet.

All computed metrics as columns, including:

- Identity: symbol, company name, price, market cap, sector
- Valuation: FCF/EV, EBIT/EV, weighted composite
- IV estimates: pessimistic (P25), base (P50), optimistic (P75) IV
  per share, rounded to nearest dollar (nearest 10 cents for stocks
  with current price under $10). MoS for each scenario.
- Model fit: revenue growth std, operating margin R², FCF conversion
  R²
- Simulation uncertainty: P25-P75 IV spread (an output of the Monte
  Carlo, distinct from the regression-based fit metrics above)
- Quality: F-Score composite + 9 individual signal columns, gross
  profitability, accruals ratio, ROIC (latest), ROIC trend slope,
  ROIC detrended residual std, ROIC minimum
- Trends: revenue CAGR, operating margin slope, FCF conversion slope
- Safety: interest coverage, OCF/total debt
- Sentiment: 6-month price return, 12-month price return
- Filter status: filtered (boolean), filter_reasons (string)

### Per-Company Detail Reports

Generated on demand for a user-specified list of tickers. HTML with
structured data as native HTML tables (searchable, selectable) and
charts as embedded PNGs.

**Header:** Symbol, company name, price, market cap, FCF/EV, EBIT/EV,
MoS (base), F-Score composite.

**Quality block:** F-Score 9-signal checklist (text checkmarks, not
chart — binary signals do not benefit from visualisation). Quality
metrics table (gross profitability, accruals ratio, ROIC). Market
sentiment (plain numbers, no colour).

**Trend and fit block:** Revenue (CAGR, growth std), operating margin
(slope, R²), FCF conversion (slope, R²), ROIC (slope, detrended
residual std, minimum). Presented so the user sees e.g., "margin
improving at +0.8pp/yr (R² = 0.82)".

**Safety block:** Interest coverage, OCF/total debt.

**Charts:**

1. **Revenue projection (spaghetti plot).** 20-30 individual Monte
   Carlo paths as thin semi-transparent lines. P10-P90 outer band
   labelled "80% of simulated outcomes." P25-P75 inner band labelled
   "50% range." Historical quarterly revenue on the left.
2. **Derived FCF projection.** Same spaghetti structure, showing how
   revenue translates to cash through the margin/conversion pipeline.
   Variance in FCF paths derives entirely from revenue growth
   variation; margin and conversion follow their deterministic trends.
3. **IV scenario bars.** Horizontal bars for P25/P50/P75 IV per
   share, current price as a reference line. No "upside/downside"
   arrows — the user sees the gap directly.
4. **Sensitivity heatmap.** Discount rate (rows, 7%-13% in 1pp
   steps around the 10% default) vs growth adjustment (columns,
   0.5×-1.5× multiplier on base CAGR in 0.25 steps). Cell values
   show IV per share, recomputed by re-running the DCF with adjusted
   parameters. Current price as a contour line showing the break-even
   boundary.
5. **Operating margin time series.** Historical quarterly margins
   with the regression trend line overlaid. Lets the user judge
   whether the R² and slope are reasonable.
6. **Historical revenue growth bars.** Quarterly YoY growth rates
   with mean and ±1 std lines. Shows the raw data feeding the Monte
   Carlo.

### Visual Encoding

- **F-Score composite:** Colour-coded (7-9 strong, 4-6 moderate,
  0-3 weak). Individual signals as monochrome checkmarks.
- **Balance sheet safety:** Amber below distress thresholds (e.g.,
  interest coverage < 1.5x). No colour above — unmarked is neutral,
  not "good."
- **Everything else:** No colour. MoS is a model output, not a fact.
  Sentiment is neutral context. Stability/fit metrics are information
  for the user to interpret. Growth rates are not good or bad without
  context.
- **Neutral naming:** "Cheapest by FCF/EV" not "Best Value."
  No evaluative language in column headers or chart labels.
- **Chart axes:** Y-axis includes zero when negative outcomes are
  possible. No truncated axes that hide downside scenarios.

## Data Source

All required fields are already present in the shared FMP database
(`/home/mattm/projects/Pers/financial_db/data/fmp.db`). No changes
to fmp_download are needed. The algorithmic-investing pipeline is
isolated (explicit column selects, read-only mode, no dynamic schema
discovery) and unaffected by the screening pipeline's queries.

Key tables: `incomeStatement`, `balanceSheet`, `cashFlow`,
`companyProfile`, `eodPrice`, `entity`. ROIC is calculated from raw
quarterly financials (not from the single-snapshot `keyMetricsTtm`).
