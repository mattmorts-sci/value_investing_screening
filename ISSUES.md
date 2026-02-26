# Known Issues

## Open

### 1. Inflated growth rates and intrinsic values

**Status:** Root cause identified, design decisions resolved
**Affects:** `pipeline/analysis/growth_stats.py`,
`pipeline/analysis/growth_projection.py`

**Problem:** The refactored pipeline produces massive inflated growth
rates and IVs. The legacy code did not have this problem. Root cause
analysis identified two compounding failures: the growth rate input
method and the projection model.

#### 1a. Root cause: TTM QoQ growth rates

The refactored pipeline computes TTM (trailing twelve month) QoQ
growth rates from rolling 4-quarter sums. Two problems:

1. **Small-denominator explosion.** When a TTM sum is near zero
   (company transitioning between negative and positive FCF, or
   quarterly FCF netting to near-zero), the denominator in the QoQ
   calculation produces extreme growth rates (+500%, +1000%).

2. **Autocorrelation from overlapping windows.** Each TTM observation
   shares 3 of 4 quarters with its neighbour. Lag-1 autocorrelation
   ~0.75, effective sample size drops from ~16 to ~2.3. All statistics
   (mean, median, std, MAD) computed on this series are unreliable.

The legacy code used raw QoQ growth rates from an external source —
up to 20 independent observations from 20 quarters of data. No
overlap, no autocorrelation, no small-denominator problem from
rolling sums.

#### 1b. Root cause: deterministic fade model vs Monte Carlo

The refactored pipeline replaced the legacy Monte Carlo simulation
(10,000 paths, percentile aggregation) with a deterministic
fade-to-equilibrium model using the arithmetic mean of TTM QoQ
historical growth rates as starting point.

This removed the constraint system and structural features that the
legacy pipeline used. The table below describes the current state
(before the fix):

| Guardrail | Legacy | Refactored |
|-----------|--------|------------|
| Aggregation | P25/P50/P75 of 10k paths | Deterministic mean ± k×σ |
| Per-quarter growth cap | +25-40% / -10-30% (metric/size) | None |
| Cumulative growth cap | 10x | None |
| Annual CAGR backstop | 100% | None |
| Period-over-period cap | 1.5x | None |
| Volatility dampening | 15-40% as company grows | None |
| Momentum exhaustion | Mean reversion at 2x | None |
| Time decay for high growth | 0.8^years | None |
| Size-based growth penalty | 0.4-1.0x multiplier | Weak (fade λ adj) |
| Log-normal with CV cap | CV ≤ 1.0 | N/A (deterministic) |
| Negative FCF filter | Excluded pre-analysis | Excluded (filter on by default) |
| Growth rate input | Raw QoQ | TTM QoQ |

**Jensen's inequality** is the primary mathematical explanation.
For any distribution with positive variance, the median of compound
growth paths is strictly below the path implied by the mean growth
rate. The more volatile the company, the larger the gap. The
deterministic model uses the arithmetic mean — systematically higher
than the Monte Carlo median. For a moderately volatile company
(quarterly std ~20%), the mean final value is ~43% higher than the
median. For high-variance companies, the gap can be 2x or more.

#### 1c. Rejected alternatives

Several alternative designs were explored and rejected through two
rounds of adversarial review. Documented here for reference.

**Revenue × Margin decomposition.** Decompose FCF = Revenue × FCF
Margin, project each separately. Rejected: the assumption that
FCF/Revenue is more stable than raw FCF fails for asset-heavy and
cyclical companies (capex timing dominates margin). Insufficient data
for margin trend regression (~3-4 effective observations).
Double-penalises cyclicals at trough.

**g_0 = slope / FCF_T conversion.** Fit OLS regression to absolute
FCF, convert slope to implied growth rate for the fade model.
Rejected: recreates the small-denominator problem near zero-crossings
(FCF_T approaches zero, g_0 explodes).

**Dual-path projection.** Use fade model for stable companies, OLS
regression for volatile companies. Rejected: classification boundary
creates discontinuous IV jumps; OLS on overlapping TTM data is
fundamentally compromised (autocorrelation); cross-path ranking is
incommensurable (different valuation methods on a single scale); no
value investing tradition supports this.

**Median + MAD on TTM QoQ rates.** Replace mean/std with robust
statistics. Rejected as standalone fix: does not address the
structural problem (when every growth rate is extreme because FCF is
small relative to its variation, the median is also extreme). Does
not address autocorrelation (n_eff ≈ 2.3 makes the median estimate
unreliable). Does not replace the missing Monte Carlo guardrails.

**R² as confidence metric.** Use regression R² to adjust margin of
safety. Rejected: penalises stable companies (no trend = low R²,
exactly backwards); inflated by autocorrelation in overlapping data;
measures in-sample fit, not out-of-sample predictive power.

#### 1d. Design: restore Monte Carlo with raw QoQ growth

**Status:** Design decisions resolved

Restore Monte Carlo simulation as the projection method for
positive-FCF companies, using raw QoQ growth rates as input. This is
the combination the legacy code used — it worked because:

- **Percentile aggregation** (P25/P50/P75) naturally dampens the
  right tail without arbitrary caps. Jensen's inequality does the
  work.
- **Per-path constraints** encode real economic limits on how fast a
  company can grow.
- **Raw QoQ growth rates** provide ~19 independent observations with
  no autocorrelation and no small-denominator problem from rolling
  sums.
- **Distribution-based output** gives percentiles for scenarios
  rather than mean ± k×σ.

##### D1. Constraints to keep (resolved)

Keep 8 of the legacy constraints. Drop 3 redundant ones.

**Keep:**
1. Log-normal sampling with CV cap at 1.0 — prevents structurally
   impossible growth (<-100%) and extreme distribution tails.
   Log-normal parameterisation: `target_mean = 1 + m`,
   `CV = min(s / target_mean, 1.0)`,
   `σ² = ln(1 + CV²)`,
   `μ = ln(target_mean) - σ²/2`.
   Edge case: if `m ≤ -0.5` or `s = 0`, default to low-volatility
   5% quarterly decline (`μ = ln(0.95)`, `σ² = 0.1`).
   See `LEGACY/analysis/growth_calculation.py:260-281`.
2. Per-quarter growth caps, asymmetric by metric and size evolution:

   | Metric | Size evolution | Positive cap | Negative cap |
   |--------|---------------|-------------|-------------|
   | FCF | ≤ 2.0x | +40% | -30% |
   | FCF | > 2.0x | +25% | -20% |
   | Revenue | ≤ 2.0x | +25% | -15% |
   | Revenue | > 2.0x | +15% | -10% |

   See `LEGACY/analysis/growth_calculation.py:342-360`.
3. Cumulative 10x cap — hard ceiling, prevents runaway compounding.
4. 100% annual CAGR backstop — safety net, catches anything the
   per-quarter caps miss. After year 1, if implied annual CAGR
   exceeds 100%, value is clamped to what 100% CAGR would produce.
5. Percentile aggregation of final values (P25/P50/P75) — Jensen's
   inequality dampens the right tail naturally.
6. Momentum exhaustion — mean reversion within simulation paths when
   recent 4-quarter average growth exceeds 2x the historical mean.
   Adjustment capped at -0.3 in log space. Conversely, if recent
   growth falls below 0.5x mean, a small upward bounce is applied
   (capped at +0.2 in log space).
7. Time decay for high growth — `0.8^years_elapsed` multiplier on
   quarterly growth exceeding 30%.
8. Size-based growth penalty (FCF/market-cap ratio) — dampens growth
   as FCF grows relative to market cap (0.4-1.0x multiplier).

**Drop:**
- Volatility dampening as company grows — redundant with cumulative
  cap and per-quarter caps.
- Period-over-period 1.5x cap — redundant with per-quarter growth
  caps (40% positive cap means max 1.4x per quarter).
- Dynamic max multiple system — vestigial in the legacy code
  (computed at `growth_calculation.py:363` but the `max_multiple`
  variable is never read afterwards).

##### D2. Growth rate input (resolved)

Compute raw QoQ growth rates in `growth_stats.py`, replacing
`_compute_ttm_growth()` with a new `_compute_raw_qoq_growth()`
function: `g = (Q_t - Q_{t-1}) / |Q_{t-1}|` (absolute-value
denominator handles sign transitions without producing NaN). With
~20 quarters (~5 years, using `history_years = 5`), this produces
~19 independent observations. Compute mean and variance from these
to parameterise the log-normal sampling distribution (see D1 item 1).

The Monte Carlo module receives mean and variance from
`growth_stats.py` — same interface as the current fade model
(`fcf_growth_mean`, `fcf_growth_std`, etc.).

Replaces the TTM QoQ approach. No overlapping windows, no
autocorrelation, no small-denominator problem from rolling sums.
Quarterly lumpiness (seasonal capex, working capital) is accepted —
the Monte Carlo's many paths and median aggregation handle noisy
inputs.

##### D3. Scenario extraction (resolved)

Extract percentiles of simulated final values. Map to existing
scenario structure:
- Pessimistic → 25th percentile
- Base → 50th percentile (median)
- Optimistic → 75th percentile

Only these three percentiles are extracted. The legacy code's
additional CI bands (10th/40th/60th/90th) are not carried forward —
they were used only for the legacy `GrowthProjection` dataclass.

Back-calculate annual CAGR from each percentile's final value
relative to the starting value. Use the CAGR to project a smooth
compound path for the DCF. (The legacy code extracted percentile
final values and back-calculated CAGR similarly, but did not project
smooth compound paths — this is a pragmatic simplification to fit
the refactored `Projection` contract.)

##### D4. Integration with DCF and ranking (resolved)

No `Projection` dataclass field changes needed. Monte Carlo output
maps to the existing fields:
- `quarterly_values`: smooth compound path from current value to
  percentile final value
- `quarterly_growth_rates`: constant quarterly rate implied by CAGR
  (all elements identical — the per-quarter variation from the fade
  model is lost, but no downstream consumer relies on per-quarter
  variation; DCF uses `quarterly_values`, ranking uses `annual_cagr`)
- `annual_cagr`: back-calculated from percentile final value
- `current_value`: unchanged

The DCF module reads `quarterly_values` (dcf.py:54) — works as-is.
The ranking module reads `annual_cagr` (ranking.py:227) — works
as-is.

**Chart visual change:** The growth projection chart will show smooth
compound-growth curves instead of the current fade-to-equilibrium
curves. The shaded band between pessimistic and optimistic represents
the IQR of simulated final values rather than mean ± k×σ. No chart
code changes needed, but the visual interpretation changes.

##### D5. Negative FCF handling (resolved)

Keep the existing negative-FCF improvement-toward-zero model
(`_project_negative_fcf` in `growth_projection.py`). This is a
separate code path that does not use percentage growth on a negative
base. In the restored Monte Carlo design, it runs independently of
the Monte Carlo simulation for positive-FCF companies.

Negative-FCF companies are included in the screening universe (not
excluded pre-analysis as in the legacy code). Set
`enable_negative_fcf_filter` to `False` (currently `True` in
`settings.py:189`). The Monte Carlo simulation runs only for
positive-FCF companies; negative-FCF companies use the existing
improvement model.

Routing condition: if the company's latest FCF (`current_value`) is
negative, use the improvement model; otherwise, use Monte Carlo.
Companies with positive latest FCF but historical sign transitions
will have some extreme QoQ rates — the Monte Carlo's constraints
(per-quarter caps, CV cap on log-normal) handle these.

##### D6. Module changes (resolved)

**`growth_stats.py`:** Replace `_compute_ttm_growth()` with
`_compute_raw_qoq_growth()`. All other functions (CAGR,
`combined_growth_mean`, `growth_stability`, `fcf_reliability`)
unchanged.

**`growth_projection.py`:** Delete fade-model functions
(`_fade_growth_rates`, `_compute_fade_lambda`, `_project_values`,
`_quarterly_rate`). Replace `project_growth` with Monte Carlo
simulation for positive-FCF companies. Keep `_project_negative_fcf`
for negative-FCF companies. Keep `_compute_annual_cagr`. Modify
`project_all` to route positive vs negative FCF.

**`dcf.py`:** No changes.

**`ranking.py`:** No changes.

**`contracts.py`:** No field changes to `Projection` or
`IntrinsicValue`.

**`settings.py`:** Change `enable_negative_fcf_filter` default to
`False`. Existing fade-model parameters (`equilibrium_growth_rate`,
`base_fade_half_life_years`) retained — still used by
`_project_negative_fcf`. `scenario_band_width` no longer used by
positive-FCF projection (scenarios come from percentiles). New
parameters needed: `simulation_replicates` (int, default 10000),
`cv_cap` (float, default 1.0), `cumulative_growth_cap` (float,
default 10.0), `annual_cagr_backstop` (float, default 1.0),
`momentum_exhaustion_threshold` (float, default 2.0),
`time_decay_base` (float, default 0.8), `high_growth_threshold`
(float, default 0.3), `size_penalty_factor` (float, default 0.1).
Per-quarter growth caps: add a structure or individual parameters
matching the legacy values in D1 item 2.

**`runner.py`:** Pipeline call sequence unchanged.

##### D7. Performance (resolved)

The legacy code parallelised Monte Carlo across companies using
`ProcessPoolExecutor` with up to 44 workers
(`LEGACY/analysis/growth_calculation.py:650-715`). For ~300-500
companies × 10,000 paths × 20-40 quarter steps, single-process
execution may be slow. Evaluate whether NumPy vectorisation of the
inner simulation loop (simulating all 10,000 paths for one company
in a single NumPy operation) eliminates the need for multiprocessing.
If not, add parallelisation.

---

### 2. AM chart percentage change always shows "+0%"

**Status:** Root cause identified, fix pending
**Affects:** `pipeline/charts/comparative.py`, `pipeline/runner.py`

`_update_live_metrics` (runner.py:35, overwrite at line 60) replaces
`adj_close` with the live price in the companies DataFrame before
results are assembled. The AM chart then reads `acquirers_multiple`
from this already-updated DataFrame AND computes current AM from the
same live prices. Identical inputs produce 0% change for every
company.

**Fix:** Preserve the filing-period AM (computed from the original
`adj_close`) before the live-price overwrite, so the chart can show
the actual change from last filing to current market price.
