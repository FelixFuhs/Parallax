# Goal Brief: Parallax AI Label Decomposition Redesign

Use this from the repo root with:

```text
/goal follow the objective in inputs/goal_prompts/2026-05-19_ai_label_decomposition.md
```

## Objective

Redesign Parallax from a "single AI DCF upside label distilled into five features" project into an evidence-producing AI label decomposition research workflow.

The core scientific question is not just whether today's AI DCF upside can be compressed into historical accounting features. The new core question is:

```text
Does any part of the AI-implied expected-return signal predict future returns after separating:
1. raw AI signal,
2. mechanical DCF signal,
3. factor-compressible AI signal,
4. non-factor AI residual signal?
```

Build the project so it can answer these three questions cleanly:

```text
1. Does the raw AI label predict future returns?
2. Does the factor-compressible part of the AI label predict returns?
3. Does the non-factor AI residual predict returns?
```

Do not complete this goal by making the existing pipeline marginally wider or by adding random features. The target state is an "AI label decomposition project" with implemented label panels, diagnostics, benchmarks, data-quality separation, holdings-level outputs, tests, and documentation.

## Current Context

Read the repo before editing. As of this goal brief, the important files are:

- `README.md`: frames Parallax as a negative finding for AI DCF label distillation.
- `openrouter.py`: generates AI DCF reports.
- `parser.py`: normalizes reports, applies defaults, and records quality flags.
- `dcf.py`: computes three-scenario FCFF DCF valuation artifacts.
- `edgar.py`: builds current EDGAR feature rows.
- `historical.py`: builds accepted-date historical feature matrices.
- `distill.py`: fits Elastic Net, XGBoost regressor, and related distillation artifacts.
- `stability.py`: still has ElasticNet-centered stability paths despite the freeze doc saying XGBoost is primary.
- `backtest.py`: scores historical features, forms monthly buckets, and compares simple baselines.
- `docs/model_freeze.md`: declares the frozen primary model as XGBoost regressor.
- `results/backtest_summary.json`: has prior backtest outputs and may contain stale absolute paths.
- `tests/`: has parser, EDGAR, historical, distill, stability, and OpenRouter tests.

The current research workflow mostly tests:

```text
2026 AI DCF base-case upside
    -> five-feature surrogate
    -> historical portability of that surrogate
```

The redesign must produce and use a label panel:

```text
raw_ai_upside
raw_ai_implied_irr
mechanical_dcf_implied_irr
ai_minus_mechanical_irr
factor_compressible_ai_score
ai_factor_residual
ai_label_uncertainty
quality_flags / failure reasons
```

## Primary Deliverables

Deliver working code and reproducible artifacts, not only a plan.

At minimum, the repo should gain a coherent v2 workflow that can run on the existing saved reports and public data without requiring new paid data or API calls. If optional external data, paid data, or API credentials are missing, implement the interfaces, tests, and honest blockers rather than fabricating data or silently weakening the claim.

Required deliverables:

- A label-panel builder that extracts raw AI upside and computes AI implied IRR where possible.
- A mechanical DCF baseline that produces `mechanical_dcf_implied_irr` from standardized assumptions.
- Residual labels: `ai_minus_mechanical_irr`, factor-compressible fitted value, and AI factor residual.
- Rank IC diagnostics before portfolio backtests.
- Global, within-sector, and across-sector decomposition for each signal.
- A serious pre-registered value-quality-momentum-investment-accruals benchmark.
- A corrected price model that separates raw close for valuation/market cap/EV from adjusted close for returns and momentum.
- Holdings-level backtest output, turnover, costs, and exposure/audit tables.
- Fixes for freeze-artifact inconsistency between XGBoost and Elastic Net.
- Tests that protect the new behavior and catch the known failure modes.
- Updated docs that state what is supported, what remains prototype-grade, and what conclusions are not justified.

## Must Do

### 1. Inspect and map the current system

- Read the implementation, docs, tests, model artifacts, and existing result artifacts before making edits.
- Identify where labels, DCF outputs, features, historical prices, sectors, backtest returns, and model metadata currently flow.
- Preserve user and prior-generated artifacts unless changing them is necessary for the new workflow.
- Note any dirty worktree changes and avoid reverting unrelated changes.

### 2. Redesign labels around implied IRR

Implement the new label panel as a first-class artifact. Prefer a new module if that fits the repo, for example `labels.py` or `label_panel.py`, but follow existing style.

The panel must include, with explicit missingness and quality flags:

```text
ticker
report_id / report_path
report_date
model_id / tier when available
sector when available
raw_ai_upside
raw_ai_implied_irr
mechanical_dcf_implied_irr
ai_minus_mechanical_irr
factor_compressible_ai_score
ai_factor_residual
ai_label_uncertainty
parse_failure_rate or quality/failure fields where available
source paths / artifact IDs
```

Use implied IRR as the main AI label when feasible. Raw DCF upside should remain a secondary label.

Do not use the mechanical DCF as "the AI label." It is a benchmark/control. The interesting AI object is:

```text
ai_minus_mechanical_irr = raw_ai_implied_irr - mechanical_dcf_implied_irr
```

If current saved reports do not contain enough cash-flow detail to compute a clean IRR for every name, do the best implementation possible and mark missing/low-quality rows explicitly. Do not backfill critical missing assumptions with quiet defaults and call the resulting labels clean.

### 3. Build strict label-quality and DCF sanity checks

Failed or default-heavy AI reports are data quality observations, not labels to silently rescue.

Implement or centralize flags for at least:

```text
missing_price
missing_wacc
default_terminal_growth
missing_scenario
scenario_order_fail
internal_inconsistency
suspiciously_round_forecast
missing_comps
terminal_value_dominates
wacc_terminal_spread_too_small
share_count_outlier
ev_negative
forecast_discontinuity
stale_price
```

Define which flags exclude a row, which downweight it, and which are warning-only. Report failure counts and failure rates by sector and, where possible, market-cap bucket or model tier.

Scenario sanity must check:

```text
bear_value <= base_value <= bull_value
bear_irr <= base_irr <= bull_irr
WACC > terminal growth
terminal value share is not absurd
current price date matches report date or is flagged
```

### 4. Add mechanical DCF implied IRR

Add a standardized mechanical DCF baseline. It should be transparent and reproducible, using public/current repo data where possible.

The mechanical DCF should:

- Use consistent assumptions rather than AI narrative assumptions.
- Produce an implied IRR-like expected-return label comparable to AI IRR.
- Be clearly documented as a benchmark/control, not as an AI result.
- Be tested on synthetic cases where the answer is obvious.
- Handle missing inputs honestly.

If only an approximate public-data baseline is feasible in this repo, implement that and document the limits.

### 5. Add factor residualization

For each usable cross-section, estimate:

```text
ai_label_i = alpha
           + beta_1 value_i
           + beta_2 quality_i
           + beta_3 momentum_i
           + beta_4 investment_i
           + beta_5 accruals_i
           + beta_6 size_i
           + sector fixed effects
           + epsilon_i
```

Then store:

```text
factor_compressible_ai_score = fitted value
ai_factor_residual = epsilon_i
```

Use sensible robust handling for missing features, winsorization, and sector fixed effects. Do not optimize residualization choices on future return performance.

### 6. Build rank IC diagnostics before portfolio backtests

Implement monthly rank IC diagnostics for every important signal:

```text
raw_ai_irr
mechanical_irr
ai_minus_mechanical_irr
factor_compressible_ai_score
ai_factor_residual
fcf_to_ev
composite_vqmia
value
quality
momentum
```

For each date and horizon:

```text
IC_t,h = SpearmanRankCorr(score_i,t, return_i,t_to_t+h)
```

Report:

```text
mean IC
median IC
IC standard deviation
Newey-West or defensible t-stat where feasible
positive IC hit rate
IC by year
IC by sector
IC decay at 1m, 3m, 6m, 12m
coverage per month
```

Do not let portfolio backtests be the first or only evidence of signal quality.

### 7. Add within-sector and across-sector decomposition

For every signal, compute and test:

```text
global_score_i
within_sector_score_i = score_i - sector_mean(score)
sector_score_s = sector_mean(score_i within sector s)
```

Produce diagnostics and backtests for:

```text
global IC
within-sector IC
across-sector IC
sector-neutral portfolio
unconstrained portfolio
```

The final reporting table should be able to compare at least:

```text
Signal                      Global   Sector-neutral   Across-sector
AI IRR
AI residual
Mechanical IRR
Composite VQMIA
FCF/EV
```

### 8. Build a serious composite benchmark

Replace weak benchmark framing with an equal-block-weight composite benchmark:

```text
Composite VQMIA =
    value block
  + quality block
  + momentum block
  + investment block
  + accruals / cash-conversion block
  + leverage / distress control
```

Use feature blocks rather than random feature hunting. Start with available fields and add a clear feature-engineering path. Examples:

```text
Value: fcf_to_ev, ebit_to_ev, ebitda_to_ev, book_to_market, earnings_yield
Quality: gross_profitability_assets, roic, roe, operating_margin, cash_conversion
Momentum: momentum_12_1, 6m momentum, 1m reversal
Investment: asset_growth_1y, capex_growth, working_capital_growth, issuance/buyback yield if available
Accruals: cash_earnings_gap, accruals, working capital accruals
Balance sheet: net_debt_to_ebitda, interest coverage, current ratio, leverage
```

Winsorize each feature cross-sectionally, z-score globally and within sector where appropriate, average within blocks, then average blocks with equal weights. Do not tune weights after seeing returns.

### 9. Fix adjusted-price versus raw-price usage

Audit `historical.py`, `edgar.py`, and `backtest.py` for price usage.

Store or compute two distinct price panels:

```text
raw_close_price       -> market cap, enterprise value, valuation features
adjusted_close_price  -> returns and momentum
```

Then enforce:

```text
market_cap = raw_close * shares_outstanding
returns = adjusted_close_exit / adjusted_close_entry - 1
momentum = adjusted-close based
```

Add tests that would fail if adjusted dividend-adjusted prices leak into market cap or EV construction.

### 10. Add holdings, turnover, costs, and audit outputs

Extend backtesting so serious portfolio results are auditable.

Write artifacts such as:

```text
results/holdings.parquet
results/monthly_returns.parquet
results/turnover.parquet
results/exposures.parquet
```

Each holding row should include:

```text
date
ticker
sector
market_cap
score
signal_name
bucket
weight
features
feature_null_count
label fields if available
entry_price
exit_price
raw_return
transaction_cost
net_return
```

Report:

```text
monthly turnover
average number of names
sector weights
market-cap exposure
beta exposure if feasible
gross and net returns
transaction-cost drag
capacity/liquidity caveats
```

At minimum test 0 bps, 10 bps, 25 bps, and 50 bps costs. State whether costs are one-way or round-trip.

### 11. Improve data model boundaries

Move toward a clear separation between:

```text
security_master
prices
fundamentals
labels
experiments
```

Do not overbuild a database if a file-based design is enough, but make the boundaries explicit in code and artifacts.

Where feasible, include:

```text
ticker history / CIK mapping caveats
sector history or current-sector caveat
index membership history or survivor-bias caveat
filing accession
accepted timestamp
period end
XBRL tags used
raw values
derived values
prompt/report IDs
model IDs
quality flags
```

### 12. Add quarterly fundamentals if feasible in scope

The current historical pipeline is annual-filing heavy. Add 10-Q support or a well-tested scaffold for it:

```text
quarterly revenue
quarterly gross profit
quarterly operating cash flow
quarterly capex
quarterly assets
quarterly debt
quarterly cash
quarterly shares
TTM rollups
QoQ changes
YoY quarterly changes
```

If full quarterly implementation would exceed the run, land the minimal abstraction and tests needed so it is a real next step, not a vague TODO.

### 13. Address survivor bias and point-in-time universe honestly

Do not claim institutional point-in-time validity unless the data supports it.

Implement the best public-data improvement feasible, or produce a concrete ingestion interface and blocker document. Minimum acceptable behavior:

- Current survivor-only universe is clearly flagged in outputs.
- Universe membership assumptions are recorded in experiment metadata.
- Backtest claims do not overstate alpha or historical validity.
- Any public S&P 500 changes table use is documented as approximate, not CRSP-quality.

### 14. Fix freeze and artifact inconsistency

The docs say XGBoost regressor is primary, while `stability.py` and artifacts remain ElasticNet-centered in places.

Split or clarify artifacts such as:

```text
docs/freeze_xgb_regressor.md
docs/freeze_elasticnet_baseline.md
models/frozen_xgb_regressor.json
models/frozen_xgb_regressor_metadata.json
models/frozen_elasticnet_baseline.pkl
models/frozen_elasticnet_metadata.json
```

Add a test equivalent to:

```text
test_backtest_model_matches_declared_freeze_doc
```

The test should fail if a script silently writes a frozen artifact for the wrong model family.

### 15. Add experiment registry and reproducibility metadata

Every major run should write a metadata artifact with at least:

```json
{
  "experiment_id": "...",
  "git_commit": "...",
  "data_snapshot_hash": "...",
  "label_snapshot_hash": "...",
  "feature_config": "...",
  "model_config": "...",
  "universe_config": "...",
  "backtest_config": "...",
  "generated_at": "..."
}
```

Remove absolute local paths from generated JSON summaries. Use repo-relative paths.

Add or improve:

```text
pyproject.toml or equivalent project metadata
locked or pinned dependencies where practical
ruff or explicit lint decision
pytest / pytest-cov configuration
CI workflow if feasible
```

Do not spend most of the run on packaging polish if the core research redesign remains unimplemented.

### 16. Add repeated-label and uncertainty support

Do not launch expensive bulk label regeneration unless credentials, budget, and explicit permission are available.

But implement the data model and aggregator for:

```text
multiple prompts
multiple temperatures / seeds if available
multiple model tiers
named versus anonymized variants if available
sector-template versus generic-template variants
```

Store:

```text
mean_ai_irr
median_ai_irr
std_ai_irr
rank_std
interquartile range
model_disagreement
prompt_disagreement
parse_failure_rate
quality_flags
```

Support:

```text
sample_weight = 1 / label_uncertainty
exclude or downweight high-uncertainty labels
```

### 17. Add sector-specific DCF template support

Create a common schema with sector-specific prompt logic or templates for at least a scaffold of:

```text
Software
Semiconductors
Energy
Industrials
Healthcare
Utilities
Consumer / general fallback
```

The common output schema must stay stable:

```text
historical financials
forecast drivers
terminal assumptions
risk assumptions
implied IRR
quality flags
sources
```

Do not let templates produce incompatible label structures.

### 18. Add LLM text-feature path as a separate experiment

Do not treat this as the same experiment as DCF label residuals.

Add a separate, claim-safe path for LLM-extracted text features from date-limited filings when feasible:

```text
10-K MD&A
10-K risk factors
10-Q MD&A
earnings-call transcripts if available
```

Candidate structured features:

```text
tone change
uncertainty change
risk-factor novelty
management hedging
capital allocation discipline
competitive pressure
pricing power
demand weakness
supply chain stress
regulatory pressure
accounting aggressiveness
guidance credibility
```

This should be Experiment C, not quietly mixed into Experiment A or B.

## Research Framing To Preserve

The final project should separate three experiments:

### Experiment A: Current raw AI signal

```text
Given AI labels generated today, do they predict future realized returns?
```

Use:

```text
raw_ai_irr
ai_minus_mechanical_irr
ai_factor_residual
```

This is forward-looking from the label date. Do not claim historical AI alpha from one modern vintage.

### Experiment B: AI-implied factor portability

```text
Given a current AI label cross-section, is the feature map implied by that label historically portable?
```

This is the current distillation idea, but with better labels, controls, decomposition, and benchmarks.

### Experiment C: LLM text-feature alpha

```text
Can LLM-extracted filing text features predict returns beyond standard accounting and price factors?
```

This is distinct from asking the LLM to be a valuation oracle.

## Must Not

- Do not optimize for the easiest literal interpretation. Build the thing that the objective is trying to cause.
- Do not call the goal complete by changing definitions, weakening gates, hiding failures, or converting a substantive blocker into a wording issue.
- Do not claim alpha, tradability, institutional validity, or point-in-time purity unless the evidence supports it.
- Do not silently heal failed AI reports into usable labels.
- Do not use future returns to tune factor weights, feature blocks, residualization choices, costs, or model family decisions.
- Do not start by adding 100 random features or an unconstrained ML model.
- Do not use the XGBoost ranker unless there are real date-level query groups and the objective is appropriate.
- Do not run paid or rate-limited API calls in bulk unless explicitly configured and authorized.
- Do not overwrite prior result artifacts without preserving provenance or making the regeneration explicit.
- Do not leave stale absolute local paths in output summaries.
- Do not mark passing tests as sufficient if the tests do not protect the intended research claim.

## Acceptance Criteria

The goal is not complete unless all of these are true or honestly documented as blocked by missing data/API/paywalled sources:

- Existing saved reports can be processed into a label panel with explicit IRR labels, residual labels, uncertainty fields, and quality/failure flags.
- The code can produce mechanical IRR labels and AI-minus-mechanical residuals on a nontrivial sample or explain row-level blockers.
- Factor residualization produces both fitted factor-compressible AI scores and non-factor AI residuals.
- Rank IC diagnostics run before and independently of portfolio backtests.
- At least global and sector-neutral signal diagnostics work; across-sector decomposition is implemented or has a concrete blocker.
- The strong composite benchmark exists and is not fitted to future returns.
- Raw and adjusted prices are separated in valuation versus return/momentum paths.
- Backtests write holdings-level, turnover, costs, and monthly return audit artifacts.
- Freeze docs and scripts no longer contradict each other about primary XGBoost versus ElasticNet baseline artifacts.
- Tests cover parser/default behavior, DCF sanity checks, label panel construction, factor residualization, rank IC, sector-neutral ranking, adjusted/raw price separation, bucket assignment/T+1 execution, cost application, and freeze artifact consistency.
- Docs explain the three-experiment framing and the claim ceiling.
- A final completion report lists commands run, artifacts written, result summaries, limitations, and remaining blockers.

## Suggested Execution Order

Use judgment, but this order reflects the highest-return sequence:

1. Map current data and signal flow.
2. Build label panel: `raw_ai_irr`, `mechanical_irr`, `ai_minus_mechanical_irr`, `ai_factor_residual`.
3. Add label-quality and DCF sanity checks.
4. Implement rank IC diagnostics for raw AI, mechanical, residual, five-feature surrogate, and strong composite.
5. Add sector decomposition: global, within-sector, across-sector.
6. Build the strong composite benchmark.
7. Fix adjusted-price versus raw-price usage.
8. Add holdings, turnover, costs, and exposure outputs.
9. Fix freeze-doc/script/artifact inconsistency.
10. Add experiment registry metadata and path hygiene.
11. Add repeated-label uncertainty support.
12. Add quarterly fundamentals support where feasible.
13. Add point-in-time universe scaffolding or honest blocker output.
14. Add sector-specific DCF template support.
15. Add LLM text-feature experiment scaffolding.

## Critic / Verifier Loop

Use real subagents or equivalent independent review passes if the environment supports them. The main agent owns integration.

At minimum:

- Spawn or run an implementation critic after the first working pass. Give it this original objective, the changed files, and produced artifacts. Ask it to find places where the implementation satisfies words but misses the research intent.
- Spawn or run a verification critic focused on leakage, adjusted/raw price misuse, survivor-bias overclaims, stale artifacts, weak tests, and unsupported performance language.
- If docs or README claims changed, run a wording critic to catch overclaiming and claim drift.
- Fix all high and medium findings, then re-audit. If a finding cannot be fixed, document the blocker and why the project should not claim that level of evidence.

## Verification Requirements

Run targeted tests for all new behavior. Then run the full test suite if feasible.

Record exact commands and key outputs. Good candidates:

```text
python -m pytest
python -m pytest tests/test_parser.py tests/test_historical.py tests/test_distill.py tests/test_stability.py
```

Add any new test modules needed for:

```text
test_label_panel.py
test_ic.py
test_benchmarks.py
test_backtest_audit_outputs.py
test_experiment_registry.py
```

If tests cannot run because dependencies are missing or the environment is unpinned, fix the environment if in scope. Otherwise report the exact blocker and the smallest command needed to reproduce.

## Completion Report

The final answer must include:

- What changed, grouped by research layer.
- Which artifacts were generated or updated, with paths.
- Commands run and their pass/fail status.
- A concise summary of what the new evidence says, if the run generated evidence.
- Remaining blockers and whether they are data, API, scope, statistical, or engineering blockers.
- Clear claim ceiling: diagnostic/private research, trustworthy research, or institutional/production standard.
- Whether critic/verifier review happened and what it found.

Do not mark the goal complete unless the original objective is true under the original definition. If substantive blockers remain, implement every in-scope remediation you can, prove what still blocks completion, and state that the full objective remains incomplete.

## Prior Estimate

This is a broad, research-systems goal.

Expected runtime:

```text
shallow completion: 2-4 hours
decent completion: 1-2 days
deep completion: 3-7 days
median guess: 2 days for a useful v2 implementation, longer for quarterly/PIT/text features
```

Expected quality priors if run as one broad goal:

```text
literal checklist score: 3/5
spirit score: 3/5
adversarial robustness: 2/5 without a real critic loop, 4/5 with one
durability: 3/5 unless tests and artifacts are made first-class
```

After the run, compare actual behavior to these priors. If the repo has a prompting log or lessons-learned doc, add a short post-run entry with:

```text
goal file path
actual runtime
what was implemented
where the agent matched or missed the intended spirit
critic/verifier findings
remaining prompt adjustment for the next goal
```
