# Goal Brief: Honest 0.85 AI Label Reconstruction Attempt

Use this from the repo root with:

```text
/goal follow the objective in inputs/goal_prompts/2026-05-21_honest_085_reconstruction.md
```

## Objective

Try to raise Parallax's current-cross-section raw AI IRR label reconstruction from roughly `CV Spearman ~= 0.60` toward `>= 0.85`, but only under strict anti-overfitting and anti-leakage guardrails.

This is not a mandate to make a metric look good. It is a scientific attempt:

```text
Can ordinary public factors, richer engineered features, sector structure, and better label-quality handling honestly reconstruct the current raw AI implied IRR ranking at Spearman >= 0.85?
```

If the answer is no, prove that clearly and preserve the best honest result. Do not redefine success, tune on the final holdout, hide failed variants, or claim `.85` from in-sample fit.

## Current Context

Read the repo before editing. Important current artifacts and modules include:

- `inputs/goal_prompts/2026-05-19_ai_label_decomposition.md`: prior broad redesign objective.
- `label_panel.py`: builds the AI label panel and factor residuals.
- `experiment_b_factor_portability.py`: current factor-map reconstruction of `raw_ai_implied_irr`, `ai_minus_mechanical_irr`, and `ai_factor_residual`.
- `benchmarks.py`: VQMIA blocks.
- `quarterly_fundamentals.py`: quarterly/TTM scaffold.
- `price_model.py`, `historical.py`, `edgar.py`: raw/adjusted price and public-data feature paths.
- `results/experiment_b_factor_portability_summary.json`: baseline current-label reconstruction result.
- `docs/ai_label_decomposition_v2.md` and `docs/ai_label_decomposition_completion_report.md`: claim ceiling and blockers.

Current baseline to reproduce before changing anything:

```text
raw_ai_implied_irr current-label reconstruction:
  CV Spearman about 0.60

ai_minus_mechanical_irr current-label reconstruction:
  CV Spearman about 0.06

ai_factor_residual:
  blocked/degenerate map
```

The prior research conclusion was not "AI alpha." It was:

```text
raw AI IRR contains a factor-like component,
but the non-mechanical / non-factor residual has not shown usable predictive evidence.
```

Preserve that distinction.

## Environment Reality Check

At the time this goal was written, the shell environment did not expose:

```text
OPENROUTER_API_KEY
OPENAI_API_KEY
SEC_USER_AGENT_NAME
SEC_USER_AGENT_EMAIL
```

There was no `.env` or `.env.local`; only `.env.example` existed. `openrouter.py` can load `.env` and `.env.local` if present. SEC-related code expects real `SEC_USER_AGENT_NAME` and `SEC_USER_AGENT_EMAIL` for production downloads.

Therefore:

- The goal must be able to run fully offline on existing cached labels and public-data artifacts.
- Do not assume new AI labels can be collected.
- Do not assume SEC filing text can be downloaded.
- If credentials are present when the goal runs, detect them without printing secrets and use them only under explicit rate/budget caps.
- If credentials are missing, document the blocker and do not fabricate repeated-label or text-feature evidence.

## Primary Success Definition

The goal succeeds only if all are true:

```text
raw_ai_implied_irr reconstruction achieves Spearman >= 0.85
on a locked, ticker-grouped, untouched holdout
and in repeated/nested cross-validation,
with no target leakage,
no future-return optimization,
no duplicate-label leakage,
and no unsupported feature provenance.
```

Suggested minimum evidence:

- Out-of-fold CV Spearman.
- Repeated K-fold or bootstrap confidence interval.
- Locked final holdout Spearman.
- Grouped-by-ticker split so repeated labels for the same ticker cannot cross train/test.
- Sector holdout or leave-one-sector-out stress test.
- Permutation/null test.
- Feature/target leakage audit.
- Top error analysis and feature attribution.

If `.85` is not reached, the run is still valuable if it proves the ceiling honestly.

## Must Do

### 1. Reproduce and lock the baseline

- Re-run or validate the existing `experiment_b_factor_portability.py` baseline.
- Record exact baseline metrics and artifact hashes.
- Do not start feature/model search until the baseline is reproduced or the blocker is documented.
- Create a new experiment id, for example `honest_085_reconstruction`.

### 2. Estimate the label reliability ceiling

Before chasing `.85`, estimate whether `.85` is even plausible.

Use available data first:

- Detect repeated labels per ticker in the existing label panel.
- Compute pairwise and aggregate rank agreement when repeats exist.
- Report whether current duplicate coverage is enough to infer a ceiling.

If credentials and budget are available:

- Optionally collect a small repeated-label reliability sample, e.g. `30-75` tickers and `3` repeats each.
- Use strict budget/rate caps and record total cost.
- Do not launch a broad expensive run unless explicitly configured.
- Store report ids, model ids, prompt ids, dates, quality flags, and failure rates.

Interpretation rules:

```text
repeat-label reliability < 0.75:
  .85 reconstruction is probably impossible or scientifically suspicious.

repeat-label reliability 0.75-0.90:
  .85 may be possible but needs strong features and careful validation.

repeat-label reliability > 0.90:
  feature/model limitations, not label noise, are the likely bottleneck.
```

### 3. Pre-register allowed feature families

Use feature blocks, not random feature mining.

Allowed families:

- Existing VQMIA blocks from `benchmarks.py`.
- Richer accounting ratios derivable from existing EDGAR features.
- Quarterly/TTM features from `quarterly_fundamentals.py` if available.
- Price/risk/liquidity features from raw/adjusted price panels.
- Sector and sector-interaction terms, with explicit sector-holdout stress tests.
- Missingness indicators and data-quality flags.
- Label-quality/sample weights from `label_panel.py`.
- Text features only if an actual date-limited text corpus and extraction outputs exist; otherwise block them honestly.

Forbidden unless explicitly justified and audited:

- Raw AI label columns, target-derived ranks, residual columns derived from the target being predicted, or any direct target proxy.
- Future returns or backtest performance for choosing reconstruction features.
- Current realized return information not available at label time.
- Ticker identity one-hot encodings that memorize labels.
- Report path, report date quirks, filename order, or model output metadata that can act as accidental target leakage.
- Any feature selected solely because it improved the final holdout.

### 4. Build an honest reconstruction campaign

Implement a reusable reconstruction experiment, for example:

```text
reconstruct_ai_label.py
results/honest_085_reconstruction_summary.json
results/honest_085_reconstruction_predictions.parquet
results/honest_085_reconstruction_trials.parquet
results/honest_085_reconstruction_coefficients_or_importances.*
```

Evaluate a small, defensible model set:

- Elastic Net / Ridge / Huber linear baselines.
- Monotone or shallow XGBoost where constraints make sense.
- Random forest or gradient boosting only with tight complexity limits.
- Sector-specific or hierarchical models only if sample size per sector is adequate.
- Ensembling only if each constituent model is validated independently and the ensemble rule is fixed within nested CV.

Use nested validation:

- Outer repeated CV for honest metric estimates.
- Inner CV for hyperparameter selection.
- A locked final holdout created before model exploration.
- Ticker-grouped splitting.
- Optional sector-held-out stress test.

Log every trial:

```text
trial_id
target
feature_family
feature_columns
model_family
hyperparameters
split_id
train_n
test_n
cv_spearman
holdout_spearman
sector_holdout_spearman
permutation_p_value
notes
```

### 5. Improve the label target before the model

Try target-quality improvements before adding model flexibility:

- Exclude or downweight low-quality reports.
- Compare raw AI IRR versus median/mean repeated AI IRR where repeats exist.
- Compare cash-flow IRR versus annualized value gap.
- Test whether label uncertainty weighting improves reconstruction honestly.
- Quantify how much coverage is lost under stricter quality filters.

Do not cherry-pick the target after seeing final holdout results. If target variants are explored, report all of them.

### 6. Audit leakage and overfitting

Add an explicit leakage audit artifact.

At minimum, check:

- No target columns or target-derived residuals are in feature matrices.
- No exact duplicates leak across folds.
- Repeated labels for the same ticker stay in the same fold.
- Sector dummies are not the whole answer unless sector-holdout performance survives.
- Feature count is reasonable relative to `n`.
- Final holdout was touched only once after model selection.
- Permuted-target performance collapses near zero.
- In-sample fit is not reported as evidence of success.

If `.85` appears, treat it as suspicious until it passes this audit.

### 7. Re-run the downstream analysis only after reconstruction is honest

If and only if a reconstruction candidate survives the audit:

- Run the Experiment B historical portability screen with the new reconstruction map.
- Compare against old raw AI map, `FCF/EV`, composite VQMIA, and existing baselines.
- Report Rank IC, within-sector/across-sector decomposition, portfolios, turnover/cost artifacts, and claim ceiling.

Do not optimize reconstruction based on historical return performance. Reconstruction is judged against the AI label; returns are downstream evidence only.

## Must Not

- Do not optimize for the easiest literal interpretation of `.85`.
- Do not call `.85` achieved from in-sample fit, non-nested CV, a leaky split, or a tiny cherry-picked subset.
- Do not lower the threshold, change the target after seeing results, or remove hard cases to inflate the metric.
- Do not hide failed trials.
- Do not use future returns or backtest metrics to choose reconstruction features.
- Do not use APIs unless credentials are present and the run has explicit caps.
- Do not print secrets.
- Do not claim AI alpha even if reconstruction improves.
- Do not overwrite prior v2 artifacts without preserving provenance.

## Critic / Verifier Loop

Use real subagents or equivalent independent review passes if available.

Required review passes:

- Leakage critic: inspect feature matrices, splits, duplicate handling, and target provenance.
- Statistical critic: inspect validation design, sample size, confidence intervals, null tests, and `.85` claim.
- Wording critic: inspect docs and summaries for overclaiming.

Fix all high and medium findings. If a finding cannot be fixed, mark the run incomplete and document the blocker.

## Acceptance Criteria

The run may report "0.85 achieved" only if:

- Baseline was reproduced.
- Label reliability ceiling was estimated or explicitly blocked.
- Trial registry exists and includes failed trials.
- Locked holdout Spearman for `raw_ai_implied_irr` is `>= 0.85`.
- Repeated/nested CV mean or median Spearman is also `>= 0.85`, with reported dispersion.
- Sector holdout does not collapse in a way that reveals sector memorization.
- Permutation/null test collapses near zero.
- Leakage critic signs off.
- Full tests and artifact verification pass.

If these are not true, final status must be one of:

```text
not_reached_honestly
blocked_by_label_noise
blocked_by_missing_credentials
blocked_by_feature_provenance
blocked_by_sample_size
```

## Completion Report

Write:

```text
docs/honest_085_reconstruction_report.md
```

It must include:

- Baseline metrics.
- Best honest metrics.
- Whether `.85` was reached.
- Why the best model did or did not improve.
- Label reliability ceiling estimate.
- Trial count and trial registry path.
- Feature families used.
- Leakage audit result.
- Critic findings.
- Downstream Experiment B result if run.
- Remaining blockers.
- Claim ceiling.

## Prior Estimate

Expected runtime:

```text
offline shallow run: 1-3 hours
offline decent run: 4-8 hours
with repeated AI labels: 1-2 days plus API cost/rate limits
deep run with text/quarterly/fresh data: 2-5 days
```

Priors:

```text
honest .70-.78: plausible
honest .80-.85: possible but hard
honest >= .85: suspicious unless label reliability is high and leakage audit is clean
honest >= .90: very unlikely without the AI label being mostly deterministic conventional factor structure
```

After the run, compare actual behavior to these priors in `docs/lessons_learned.md` or a dedicated post-run note.
