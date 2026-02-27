# Value Investing Screening — Requirements & Refactoring Notes

Status: **Draft — iterative discussion in progress**

---

## 1. Goals

1. **Clean and tidy** the legacy code for maintainability and extensibility.
2. **Remove Jupyter notebook dependency** — the pipeline must run standalone (CLI or script).
3. **Switch data source** from pre-collected JSON files to the FMP SQLite database at `/home/mattm/projects/Pers/financial_db/data/fmp.db`.
4. **Reuse the data loader** from `/home/mattm/projects/Pers/algorithmic-investing/data/loader.py` (adapt, don't reinvent).
5. **Templated PDF reports** as the ultimate output format, but first get all charts working in a `.ipynb` development notebook.

---

## 2. Data Source

### Current (Legacy)

- Pre-collected JSON files at an external path (`~/projects/Pers/data/training/...`)
- Two files per run: `acquirers_value_data_*.json` (metrics) and `growth_data_*.json` (growth stats)
- Growth JSON contains raw per-quarter growth rate time series; mean and variance computed at load time (population variance via `np.var()` ddof=0, minimum 3 data points required per metric)
- Yahoo Finance API for live data at analysis time — fetches price, market cap, shares, FCF, operating income (from income statement), long-term debt, cash, and currency per company (not just prices)

### Target

- **FMP SQLite database** (~17 GB, 28 tables, ~90K entities, ~3.7M rows per financial statement table)
- Key tables: `incomeStatement`, `balanceSheet`, `cashFlow`, `*Growth` tables, `eodPrice` (93M rows), `entity`, `companyProfile`
- Reuse `loader.py` pattern: config-driven table/column specs, read-only connection, SQL identifier validation, entity filtering, temporal alignment, price alignment, data quality filtering, dropped-companies logging
- Loader returns a `RawFinancialData` contract: a single DataFrame with entity metadata, temporal columns, financials, growth rates, and period-end prices
- Growth statistics (mean, variance, CAGR) to be computed internally from the loaded time series — no longer pre-computed
- Legacy CAGR calculation: filters out None/NaN/zero values, requires minimum 4 quarters between valid points, returns 0.0 on sign change, computes quarterly CAGR then annualises by compounding 4 quarters, special sign preservation for both-negative cases
- Derived metrics computed at load time: `debt_cash_ratio`, `fcf_per_share`, `acquirers_multiple`, `fcf_to_market_cap`

### Open Questions — Data

- [x] ~~Which exchanges/markets to support?~~ → **Single market per run. Seven markets initially:**
  - US: NASDAQ, NYSE, AMEX (3,544 entities)
  - AU: ASX (1,655)
  - UK: LSE (2,204)
  - CA: TSX, CNQ, NEO, TSXV (1,750)
  - NZ: NZE (112)
  - SG: SES (269)
  - HK: HKSE (2,439)
  - Small/micro-cap included — broad coverage desired, financial filters handle quality.
- [x] ~~Which FMP columns map to the legacy pipeline's required fields?~~ → Resolved. See finalised mapping below.
- [x] ~~Do we still need Yahoo Finance for live prices, or can eodPrice serve that role?~~ → **Yes, live prices needed. FMP API primary, yfinance fallback.**
- [x] ~~Do we need live fundamentals (FCF, debt, cash) from Yahoo at runtime?~~ → **No. Fundamentals only change on quarterly/annual filings. Use latest period from FMP database. Only price is truly live. Market cap, EV, acquirer's multiple computed at runtime from live price + DB fundamentals.**
- [x] ~~What period type: FQ (quarterly) or FY (annual)?~~ → FQ (quarterly).
- [x] ~~Fiscal year range for historical data?~~ → Default 5 years (20 quarters), configurable.

### Column Mapping (FMP → Pipeline)

| Pipeline Field | FMP Table | FMP Column | Notes |
|---|---|---|---|
| `fcf` | cashFlow | freeCashFlow | Direct match |
| `revenue` | incomeStatement | revenue | Direct match |
| `operating_income` | incomeStatement | operatingIncome | Direct match |
| `market_cap` | Computed | `eodPrice.adjClose × incomeStatement.weightedAverageShsOutDil` | companyProfile/keyMetricsTtm are single snapshots — not per-period. Compute from components. |
| `lt_debt` | balanceSheet | longTermDebt | `totalDebt` also exists but includes short-term. Use long-term only (matches legacy). |
| `cash_and_equiv` | balanceSheet | cashAndCashEquivalents | Direct match |
| `shares_diluted` | incomeStatement | weightedAverageShsOutDil | Direct match |
| `enterprise_value` | Computed | `market_cap + longTermDebt - cashAndCashEquivalents` | keyMetricsTtm is a single snapshot. Compute from components for per-period values. |
| `fcf_growth` | cashFlowGrowth | growthFreeCashFlow | Pre-computed period-over-period rates. Stats (mean, variance, CAGR) computed by pipeline. |
| `revenue_growth` | incomeStatementGrowth | growthRevenue | Same as above. |
| `price` | eodPrice | adjClose | Aligned to period end dates |

FMP database schema notes:
- 28 tables, ~90K entities, ~3.7M rows per financial statement table, 93M eodPrice rows
- Period values: FY, Q1, Q2, Q3, Q4. Pipeline uses quarterly (Q1-Q4).
- `companyProfile` and `keyMetricsTtm` are single-snapshot tables (one row per entity, not historical) — not suitable for per-period values.
- Both `longTermDebt` and `totalDebt` columns exist in `balanceSheet`. Decision: use `longTermDebt` only.
- `*Growth` tables share the same `entityId`/`fiscalYear`/`period`/`date` keys as parent statement tables.

---

## 3. Pipeline Architecture

### Current (Legacy) — 9 Steps

1. Validate config (`config.validate()` — checks file existence, numeric ranges, weight sums, mode validity)
2. Load data (JSON) — extracts single point-in-time snapshot per metric at index `training_end`; computes CAGR from full time series; computes four derived metrics (debt_cash_ratio, fcf_per_share, acquirers_multiple, fcf_to_market_cap); computes weighted average growth (70% FCF + 30% revenue, configurable via `FCF_GROWTH_WEIGHT`/`REVENUE_GROWTH_WEIGHT`)
3. Filter (zero/negative FCF, data consistency, market cap, debt/cash)
4. Monte Carlo growth projection (10K sims, log-normal for positive values only, exponential decay for negative FCF, 20+ hardcoded constraints)
5. DCF intrinsic value (FCF-only, quarterly internal, margin of safety)
6. Fetch current data (Yahoo Finance, cached) — fetches price, market cap, shares, FCF, operating income, long-term debt, cash, currency; computes live EV, acquirer's multiple, debt-cash ratio, FCF yield
7. Build analyses (combine DB data with live data)
8. Rank (growth, value, weighted penalties, risk-adjusted) — also computes IV/price ratios for all periods/scenarios, growth stability score, growth divergence metric, penalty assignments
9. Additional analysis (factor contributions, quadrant analysis)
10. Select watchlist (top min(100, n) by IV ratio → top 40 by opportunity; 100 is hardcoded)

Note: PDF report generation is a **separate entry point** — not called from the analysis pipeline.

### Target Architecture

Sequential pipeline, no parallelisation:

1. Validate config
2. Load data (FMP SQLite — financials, growth rates, period-end prices)
3. Compute derived metrics (market cap from price×shares, EV, debt/cash ratio, FCF/share, acquirer's multiple, FCF/market cap)
4. Compute growth statistics (mean, variance, CAGR from loaded growth rate time series)
5. Filter (4 individually toggleable filters + owned-company bypass)
6. Growth projection (deterministic fade-to-equilibrium, 3 scenarios)
7. DCF intrinsic value (FCF-only, quarterly internal, 3 scenarios × N periods)
8. Fetch live prices (FMP API primary, yfinance fallback)
9. Compute live metrics (market cap, EV, acquirer's multiple from live price + DB fundamentals)
10. Rank (Sharpe-like risk-adjusted + penalty-based weighted scoring → 4 ranking DataFrames)
11. Additional analysis (factor contributions, quadrant analysis)
12. Select watchlist

Entry points:
- CLI: `python -m value_investing_screening --market US --mode shortlist`
- Notebook: `config = AnalysisConfig(market="US", mode="shortlist"); results = run_analysis(config)`
- PDF report: separate entry point consuming `AnalysisResults`

### Growth Projection — Redesign Required

The legacy Monte Carlo simulation will not be carried forward. The intent and problems are documented here for redesign.

**What it was trying to do:**
- Project future FCF and revenue values over 5 and 10 year horizons
- Capture uncertainty by producing a distribution of outcomes (bottom/median/top scenarios)
- Account for the fact that high growth rates are unsustainable (mean reversion)
- Account for the fact that larger companies grow more slowly
- Handle companies with negative FCF (model path to profitability)
- Feed projected growth rates into the DCF intrinsic value calculation

**Problems:**
- Weak base model (log-normal from historical quarterly mean/variance) papered over with 20+ hardcoded constraints, each added reactively — cumulative growth caps, volatility dampeners, momentum exhaustion/bounce adjustments, per-metric per-size quarterly caps, time decay, period-over-period multiples, implied CAGR caps, absolute floors/ceilings
- Log-normal distribution only applies to positive-value path; negative FCF uses a completely separate exponential decay model (decay rate 0.15/quarter) with no smooth transition between the two
- Several config parameters exist for these constraints (`MOMENTUM_EXHAUSTION_THRESHOLD`, `MOMENTUM_BOUNCE_THRESHOLD`, `REALISTIC_MAX/MIN_QUARTERLY_GROWTH`, `MAX/MIN_QUARTERLY_GROWTH`) but are **never actually referenced** — the code uses hardcoded values instead (phantom configurability)
- Misleading field names (scenario "median" fields hold percentile values from a different percentile)
- 10K simulations per company is slow; parallelisation (44 processes, hardcoded) compensates rather than fixes
- Two dead functions exist (`_sample_constrained_growth`, `_sample_improvement_rate`) with different thresholds from the active simulation — risk of accidental reimplementation
- No validation that the complexity produces better rankings than simpler projection methods

**Requirements for replacement:**
- Must produce projected FCF and revenue values for configurable time horizons
- Must produce scenario bands (pessimistic / base / optimistic)
- Must account for mean reversion and size-based growth constraints
- Must handle negative FCF companies
- Parameters must be derived from data or grounded in financial theory, not arbitrary
- Simpler is better if ranking quality is comparable

### Open Questions — Architecture

- [x] ~~Growth projection approach (to replace Monte Carlo)~~ → Deterministic fade-to-equilibrium with analytical scenario bands. See details below.
- [x] ~~DCF intrinsic value calculation~~ → Carry forward with cleanup. Standard textbook DCF (FCF-only, quarterly internal, configurable discount/terminal/margin-of-safety).
- [x] ~~Parallelisation strategy~~ → None. Sequential pipeline. The fade model eliminates the only CPU-bound bottleneck (Monte Carlo). SQLite reads are fast (single query). Live price fetching is rate-limited by the API, not CPU. Async/concurrent fetches can be added later if needed.

### Growth Projection — Fade-to-Equilibrium Model

Replaces the legacy Monte Carlo simulation. Deterministic, closed-form, 5 parameters.

**Base case projection:**

For each quarter `t` in the projection horizon:
```
g(t) = g_eq + (g_0 - g_eq) × exp(-λt)
```
- `g_0` = starting growth rate (mean of historical quarterly growth rates from FMP `*Growth` tables)
- `g_eq` = equilibrium growth rate (long-run nominal growth, configurable)
- `λ` = fade speed derived from `base_fade_half_life`: `λ = ln(2) / half_life`

At `t=0`, growth equals the company's historical rate. As `t→∞`, growth converges to equilibrium. Half-life of excess growth is configurable.

**Fade speed adjustment for company size:**
```
λ = λ_base × (1 + size_adjustment)
```
Where `size_adjustment` is derived from market cap relative to a reference threshold (log-scaled). Larger companies fade faster — supported by empirical research (Chan, Karceski & Lakonishok 2003; Fama & French).

**Scenario bands (analytical, not simulated):**
- **Base case:** Fade from `g_0`
- **Optimistic:** Fade from `g_0 + k × σ`
- **Pessimistic:** Fade from `g_0 - k × σ`

Where `σ` = standard deviation of historical quarterly growth rates, `k` = configurable band width (default 1.0 ≈ 68% interval). All three scenarios use the same fade structure — optimistic > base > pessimistic by construction.

**Negative FCF handling:**
- Growth rates are meaningless on negative values. Model as linear improvement toward zero at a rate derived from the company's revenue growth.
- Improvement rate = `min(cap, revenue_growth_rate × scaling_factor)` per quarter.
- If revenue is also declining, use a slower default improvement rate.
- Once FCF crosses zero, switch to the standard fade model with a conservative starting growth rate.

**Projected values:**
```
Value(t) = Value(0) × ∏(1 + g(q)) for each quarter q = 1..t
```

**Parameters:**

| Parameter | Default | Basis |
|---|---|---|
| `equilibrium_growth_rate` | 0.03 (3%) | Long-run nominal GDP growth |
| `base_fade_half_life_years` | 2.5 | Empirical growth persistence research |
| `size_fade_factor` | Derived from market cap | Larger companies fade faster |
| `scenario_band_width` | 1.0 (1 std dev) | Statistical confidence interval |
| `negative_fcf_improvement_cap` | 0.15/quarter | Matches legacy decay rate |

**Growth stability score:** Derived directly from historical growth rate variance (standard deviation of the quarterly growth series), rather than from simulated path variance. More honest — measures actual historical consistency.

### DCF — Carry Forward Details

Legacy DCF specifics to preserve or consciously change:
- **Negative FCF handling**: When current FCF < 0 and growth rate > 0, applies a capped quarterly improvement rate of `min(0.25, abs(growth_rate) / 4)`. This is separate from the Monte Carlo negative-FCF model.
- **Terminal value**: Converts final quarterly cash flow to annual (×4) before applying Gordon Growth Model. The rest of the DCF uses quarterly discounting — the switch to annual at the terminal step is a deliberate design choice.
- **Margin of safety**: Applied as a discount to the aggregate present value (`PV × (1 - margin_of_safety)`), before dividing by shares. Not a comparison threshold.
- **Configurable parameters**: `DISCOUNT_RATE` (default 0.10), `TERMINAL_GROWTH_RATE` (default 0.01), `MARGIN_OF_SAFETY` (default 0.50).

### Ranking & Scoring — Redesign Required

The legacy ranking system will not be carried forward as-is. The intent and problems are documented here for redesign.

**What it was trying to do:**
- Rank companies from multiple perspectives: growth, value, financial health, combined — produces 4 ranking DataFrames (growth, value, weighted, combined) but 5+ scoring perspectives
- Produce a single "opportunity rank" for watchlist selection — `opportunity_score = 100 - (risk_adjusted_rank / n × 100)`
- Penalise companies with high debt, excessive size, and weak growth
- Balance growth potential against risk (Sharpe-like ratio)

**Key metrics computed during ranking (needed by any replacement):**
- **Weighted average growth**: 70% FCF + 30% revenue growth (configurable via `FCF_GROWTH_WEIGHT`/`REVENUE_GROWTH_WEIGHT`)
- **Growth stability**: 0–1 score from coefficient of variation of Monte Carlo simulation paths (median scenario only); negative-FCF companies use distance-from-zero consistency instead
- **Growth divergence**: |FCF growth − revenue growth|; flagged when exceeds `GROWTH_DIVERGENCE_THRESHOLD` (default 0.10); used as a penalty and a watchlist flag
- **Combined risk**: `max(avg_volatility × 0.7 + financial_risk × 0.3, 0.1)` — financial risk normalised by dividing debt-cash ratio by `MAX_DEBT_TO_CASH_RATIO` (couples risk metric to filter threshold)
- **Expected return**: combined_growth + annualised valuation return, where valuation return = `(1 + (IV_ratio - 1)) ^ (1/period) - 1`
- **IV/price ratio**: Computed for all periods and scenarios; best ratio selected for ranking

**Problems:**
- Five+ scoring perspectives with unclear relationship to each other
- Penalty weights are arbitrary (must sum to 2.1 — no basis; enforced by `config.validate()`)
- DC penalty uses squared ratio: `(dc_ratio² × DC_WEIGHT)` — unjustified
- MC penalty uses log₁₀: `log₁₀(market_cap / MIN_MARKET_CAP) × MC_WEIGHT`
- Growth penalty has 3 hardcoded sub-weights (rate 50%, stability 30%, divergence 20%) — not configurable
- 70/30 volatility/financial risk split is arbitrary
- Stability penalty and divergence penalty are correlated by construction (`np.std([fcf_growth, rev_growth])` ≈ `|fcf - rev| / 2`)
- Factor analysis and quadrant analysis exist to explain the penalty system — complexity explaining complexity

**Requirements for replacement:**
- Must rank companies by overall investment attractiveness
- Must balance growth potential, valuation upside, and financial risk
- Must produce a single primary ranking for watchlist selection
- Supporting views (growth-only, value-only) are useful for the report but secondary
- Weights and parameters must be configurable and justifiable
- Simpler is better
- [x] ~~Keep the two analysis modes (shortlist vs owned)?~~ → Yes. Shortlist screens the market; owned analyses held stocks (bypass filters, track failures).
### Filtering

Carry forward with cleanup. Four filters:
1. Non-positive FCF (`fcf <= 0` — catches zero as well as negative)
2. Operating income > revenue (data anomaly) — only checked when `revenue > 0`
3. Minimum market cap (configurable threshold, default $20M)
4. Maximum debt-to-cash ratio (configurable threshold, default 2.5)

**Target requirement:** Each filter individually toggleable via config (on/off). The legacy code has no toggle mechanism — all four filters are always applied unconditionally.

Thresholds configurable where applicable. In owned mode, owned companies still bypass active filters with failures tracked.

Post-filter, companies are sorted by market cap descending.

- [x] ~~Parallelisation strategy?~~ → None needed. See Architecture section.

---

## 4. Output & Reporting

### Current (Legacy)

**CSV files (6, or 7 in owned mode):**
- `watchlist_summary.csv` — summary for watchlist companies
- `all_rankings.csv` — combined rankings
- `weighted_rankings.csv` — weighted rankings with factor details
- `factor_contributions.csv` — factor contribution percentages
- `removed_companies.csv` — companies removed by filters with reasons
- `full_analysis.csv` — full analysis export
- `owned_companies_analysis.csv` — owned mode only

Note: 4 ranking DataFrames exist in memory (growth, value, weighted, combined) but only 2 are written as separate CSVs.

**PDF report (~28 chart/table types across 2,000+ lines of matplotlib):**

Market overview (6 charts, 3 pages):
1. Size distribution (histogram)
2. Growth distribution (histogram)
3. Valuation distribution (histogram)
4. Filter impact (bar chart)
5. Projected growth vs stability (scatter)
6. Acquirer's multiple analysis (bar/scatter)

Comparative analysis (9 items, 9 pages):
1. Valuation upside comparison (bar)
2. Growth-value scatter (scatter with quadrants)
3. Risk-adjusted opportunity chart
4. Growth comparison — historical (grouped bar)
5. Growth comparison — projected (grouped bar)
6. Comprehensive rankings table
7. Growth projection chart for top 4 companies (2×2 grid)
8. Ranking comparison table
9. Acquirer's multiple analysis (duplicate of market overview chart 6)

Per-company detail (9 charts across 2 pages, only for top N companies via `DETAILED_REPORT_COUNT`, default 5):
- Page 1: metrics summary table, growth fan chart, valuation matrix, historical growth chart, financial health dashboard
- Page 2 (scenario comparison 2×2): DCF waterfall, sensitivity heatmap, growth trajectories, risk-return profile

Summary tables (4):
1. Watchlist summary (top 30)
2. Comprehensive rankings
3. Ranking comparison
4. Full rankings (top 40)

**Standalone PNG outputs** (separate from PDF, via `visualization.py` and `factor_analysis.py`):
- `growth_value_matrix.png`
- `growth_comparison_historical.png`
- `growth_comparison_projected.png`
- `valuation_upside_comparison.png`
- `acquirers_multiple_analysis.png`
- `factor_contributions.png` (stacked bar)
- Factor heatmap (seaborn)

**Methodology PDF** (7 pages): title, data sources, filtering methodology, growth calculation, DCF/IV, ranking methodology, example calculations.

Chart behaviours: outlier companies (>100% CAGR) excluded from chart scales and listed in annotation boxes; IV/price ratio capped at 5 for histograms; viridis colourmap throughout.

### Target

- **Phase 1**: Get all charts working in a `.ipynb` development notebook
- **Phase 2**: Templated PDF reports
- CSV outputs likely retained for data export

### Open Questions — Output

- [x] ~~Which charts/visualisations to keep, modify, or drop?~~ → Keep all. Market overview (6 charts), comparative analysis (9 charts/tables), per-company detail (9 charts across 2 pages), summary tables (4), standalone PNGs (7). See legacy output inventory above for full breakdown.
- [x] ~~Carry forward the methodology PDF?~~ → Yes, carry forward. Content will need to be updated to reflect the new pipeline.
- [x] ~~PDF template engine preference?~~ → Jinja2 + WeasyPrint. Jinja2 templates for HTML structure (text, tables, layout). WeasyPrint renders HTML/CSS to PDF. Matplotlib charts embedded as images (render to PNG/SVG in memory, insert into template). No LaTeX dependency.
- [x] ~~What should the notebook workflow look like?~~ → Thin harness. Cell 1: config. Cell 2: `results = run_analysis(config)`. Cell 3+: one cell per chart, each calling a chart function on the results object. No pipeline logic in the notebook. Same chart functions reused by PDF template engine in Phase 2.

---

## 5. Configuration

### Current (Legacy)

- Single `AnalysisConfig` dataclass with ~50 parameters
- Hardcoded defaults, no external config file
- Paths hardcoded to developer machine
- Several config parameters are phantom — declared but never referenced in code (`MOMENTUM_EXHAUSTION_THRESHOLD`, `MOMENTUM_BOUNCE_THRESHOLD`, `REALISTIC_MAX/MIN_QUARTERLY_GROWTH`, `MAX/MIN_QUARTERLY_GROWTH`)
- ~50+ hardcoded magic numbers in analysis files that should be configurable but aren't (see Growth Projection problems section)
- `__post_init__` validation enforces weight sum = 2.1

### Target

- Python dataclasses only (per CLAUDE.md — no YAML)
- Central config, no hardcoded values in source
- Separate concerns: data loading config vs analysis config vs output config

### Open Questions — Config

- [x] ~~How should the user specify which run to perform?~~ → Two entry points: CLI with arguments (`python -m value_investing_screening --market US --mode shortlist`) and notebook cell (`config = AnalysisConfig(market="US", mode="shortlist"); results = run_analysis(config)`). Same config object and pipeline underneath.

---

## 6. Code Quality Requirements

Per CLAUDE.md:
- Python 3.12.8
- `logging` module only (no `print()`)
- Type hints on all function signatures (mypy enforced)
- Docstrings on all public functions and classes
- Config values imported from central config — never hardcoded
- Hard failure on data integrity issues; graceful handling of expected operational conditions
- Australian English

---

## 7. Decisions Made

| # | Decision | Date | Context |
|---|---|---|---|
| 1 | Data source switches from JSON to FMP SQLite DB | 2025-02-25 | User requirement |
| 2 | Reuse loader pattern from algorithmic-investing | 2025-02-25 | Avoid reinventing |
| 3 | Pipeline must run without Jupyter notebook | 2025-02-25 | User requirement |
| 4 | Charts developed in .ipynb first, then templated PDF | 2025-02-25 | User requirement |
| 5 | Live prices: FMP API primary, yfinance fallback | 2025-02-25 | IV changes with price; need current data |
| 6 | Reuse `live_data.py` from algorithmic-investing (PriceProvider protocol) | 2025-02-25 | Already written, copy across |
| 7 | Fundamentals from FMP DB only; live API for price only | 2025-02-25 | Fundamentals only change on filings. Market cap/EV/AM computed at runtime from live price + DB fundamentals. |
| 8 | FMP column mapping finalised | 2026-02-25 | `longTermDebt` (not `totalDebt`); `market_cap` and `enterprise_value` computed from components; growth rates from pre-computed `*Growth` tables |
| 9 | Growth projection: deterministic fade-to-equilibrium | 2026-02-25 | Replaces Monte Carlo. 5 parameters, all data-driven or research-grounded. Closed-form, no simulation. |
| 10 | No parallelisation | 2026-02-25 | Fade model eliminates CPU bottleneck. Sequential pipeline. |
| 11 | PDF engine: Jinja2 + WeasyPrint | 2026-02-25 | Matplotlib for charts, Jinja2+WeasyPrint for document structure. No LaTeX dependency. |
| 12 | Notebook: thin harness over pipeline | 2026-02-25 | Config → pipeline call → chart calls. Same chart functions reused by PDF engine. |

---

## 8. Out of Scope (for now)

- The external data collection pipeline (FMP DB is assumed populated)
- Real-time trading or portfolio management
- Web UI

---

## 9. Legacy Code Inventory

### Modules (by directory)

| Module | Files | Purpose | Keep/Rework/Drop |
|---|---|---|---|
| `config/` | `user_config.py` | Single dataclass with ~50 params | Rework |
| `data_pipeline/` | `data_loading.py`, `data_schema.py`, `data_structures.py`, `filtering.py`, `ticker_mapping.py` | JSON loading, schema validation, filtering, ticker mapping | Replace (new data source) |
| `data_sources/` | `yahoo_finance.py` | Live data fetch (price + fundamentals), caching, anomaly detection — only supports AU and US ticker formats | Replace with `live_data.py` (FMP primary, yfinance fallback) |
| `analysis/` | `growth_calculation.py`, `intrinsic_value.py`, `ranking.py`, `weighted_scoring.py`, `factor_analysis.py` | Core analysis engine | Rework |
| `reports/` | `report_generator.py`, `report_visualizations.py` (2027 lines), `visualization.py`, `methodology_generator.py` | PDF generation, charts, standalone PNGs | Rework (phase 1: notebook, phase 2: templates) |
| `scripts/` | `main_pipeline.py` | Orchestrator | Rework |
| `input/` | CSV files (ASX listings, market caps) | Static reference data | TBD |
| `cache/` | `yahoo_cache_AU.json`, `yahoo_cache_US.json`, `ticker_mappings.json` | Yahoo Finance cache, ticker remapping | TBD |
| (root) | `intrinsic_value_analysis.ipynb` | Jupyter notebook entry point | Drop (Goal 2) |

### Structural Notes

- No `__init__.py` files anywhere — imports work via `sys.path` manipulation from the notebook
- `seaborn` is imported in `factor_analysis.py` but missing from `requirements.txt`
- `data_schema.py` lists `enterprise_value_to_book` and revenue-based IV columns in the schema, but neither is used in any analysis — stale schema
- Bug in `report_generator.py`: `create_growth_comparison_historical(results)` references bare `results` instead of `self.results` — would crash at runtime
