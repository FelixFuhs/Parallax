# AI Label Decomposition Completion Report

Generated: 2026-05-19

Goal file: `inputs/goal_prompts/2026-05-19_ai_label_decomposition.md`

Full objective status: incomplete. The repo now has a working v2 decomposition workflow and verified artifacts, but the original research objective cannot be marked complete because core evidence still depends on missing historical AI label vintages, non-survivor universe data, historical sector membership, longer realized Experiment A returns, and a downloaded plus extracted Experiment C text corpus.

Claim ceiling: diagnostic/private research. The current evidence supports a reproducible prototype and a hypothesis-generating Experiment B portability screen. It does not support AI alpha, tradability, institutional point-in-time validity, or production deployment.

## Research Layers Changed

- Label layer: `label_panel.py` builds row-level labels from saved reports with `raw_ai_implied_irr`, `raw_ai_annualized_value_gap`, `mechanical_dcf_implied_irr`, `ai_minus_mechanical_irr`, `factor_compressible_ai_score`, `ai_factor_residual`, uncertainty fields, quality flags, source paths, sectors, raw close, adjusted close, and mechanical price source.
- DCF quality layer: label quality policies distinguish excluded, downweighted, and warning-only flags. The generated summary reports failure rates by sector, model tier, and market-cap bucket.
- Price layer: `price_model.py`, `historical.py`, and `edgar.py` separate raw close prices for valuation/market cap/EV from adjusted close prices for returns and momentum. The label panel uses cached raw close before legacy EDGAR current price and now refuses AI report prices for mechanical DCF controls.
- Diagnostics layer: `signal_diagnostics.py` writes global, within-sector, across-sector, by-year, by-sector, coverage, and horizon-specific Rank IC artifacts.
- Benchmark layer: `benchmarks.py` defines the equal-block VQMIA benchmark and its value, quality, momentum, investment, accruals, and balance-sheet blocks.
- Portfolio audit layer: `backtest.py` and `v2_experiment.py` write holdings, monthly returns, turnover, cost, and exposure artifacts with unconstrained and sector-neutral modes.
- Experiment layer: Experiment A is implemented as current-label forward-return diagnostics and is blocked by insufficient realized returns; Experiment B has current label-reconstruction fits and a historical-compatible portability screen; Experiment C is a separate schema/request-manifest/LLM-request/LLM-response-ingestion/downloader/extractor path blocked on missing downloaded filing text and LLM extraction.
- Freeze layer: XGBoost is documented and verified as the primary frozen model, with Elastic Net retained as a baseline artifact.
- Reproducibility layer: major generated artifacts have repo-relative metadata and snapshot hashes through `experiment_registry.py`; `verify_artifacts.py` checks path hygiene and core artifact contracts.

## Artifacts Written Or Updated

- Label panel: `results/label_panel.parquet`, `results/label_panel.csv`, `results/label_panel_summary.json`, `results/label_panel_experiment_metadata.json`
- Current forward returns and v2 diagnostics: `results/forward_returns.parquet`, `results/forward_returns_summary.json`, `results/forward_returns_metadata.json`, `results/v2_experiment_status.json`, `results/v2_experiment_metadata.json`, `results/rank_ic.parquet`, `results/rank_ic_summary.parquet`, `results/rank_ic_by_year.parquet`, `results/rank_ic_by_sector.parquet`, `results/rank_ic_coverage.parquet`, `results/signal_comparison.parquet`
- v2 audit outputs: `results/holdings.parquet`, `results/monthly_returns.parquet`, `results/turnover.parquet`, `results/exposures.parquet`
- Experiment B current maps: `results/experiment_b_factor_portability.parquet`, `results/experiment_b_factor_portability_coefficients.csv`, `results/experiment_b_factor_portability_summary.json`, `results/experiment_b_factor_portability_metadata.json`
- Experiment B historical screen: `results/experiment_b_historical_backcast_scores.parquet`, `results/experiment_b_historical_backcast_monthly_returns.parquet`, `results/experiment_b_historical_backcast_holdings.parquet`, `results/experiment_b_historical_backcast_turnover.parquet`, `results/experiment_b_historical_backcast_exposures.parquet`, `results/experiment_b_historical_backcast_rank_ic.parquet`, `results/experiment_b_historical_backcast_rank_ic_summary.parquet`, `results/experiment_b_historical_backcast_rank_ic_by_year.parquet`, `results/experiment_b_historical_backcast_rank_ic_by_sector.parquet`, `results/experiment_b_historical_backcast_rank_ic_coverage.parquet`, `results/experiment_b_historical_backcast_summary.json`, `results/experiment_b_historical_backcast_metadata.json`
- Data-boundary artifacts: `data/sector_map_wikipedia.csv`, `data/sp500_changes_wikipedia.csv`, `data/security_master.parquet`, `data/quarterly_fundamentals.parquet`, `data/forward_price_panels.parquet`
- Data-boundary metadata: `data/approx_sp500_membership.parquet`, `results/approx_sp500_membership_summary.json`, `results/approx_sp500_membership_metadata.json`, `results/sector_map_summary.json`, `results/sector_map_metadata.json`, `results/sp500_changes_summary.json`, `results/sp500_changes_metadata.json`, `results/universe_snapshot.json`, `results/security_master_metadata.json`, `results/quarterly_fundamentals_summary.json`, `results/quarterly_fundamentals_metadata.json`
- Experiment C: `results/experiment_c_text_features.parquet`, `results/experiment_c_text_corpus_requests.parquet`, `results/experiment_c_llm_extraction_requests.parquet`, `results/experiment_c_llm_extraction_responses.parquet`, `results/experiment_c_text_features_manifest.json`, `results/experiment_c_text_features_metadata.json`
- Freeze and review artifacts: `models/frozen_model_metadata.json`, `models/frozen_elasticnet_metadata.json`, `docs/freeze_xgb_regressor.md`, `docs/freeze_elasticnet_baseline.md`, `results/ai_label_decomposition_critic_report.json`
- Documentation: `README.md`, `docs/ai_label_decomposition_v2.md`, `docs/lessons_learned.md`, `docs/research_note.md`, `docs/ai_label_decomposition_completion_report.md`

## Result Summary

- Label panel: 272 saved report rows across 269 tickers. Raw AI IRR coverage is 95.6%, mechanical DCF IRR coverage is 81.3%, AI-minus-mechanical coverage is 79.0%, and raw-AI factor residual coverage is 92.3%. Clean-label count is 251.
- Price separation: raw/adjusted price coverage in the label panel is 71.7%. Mechanical price sources are 178 raw-close rows, 66 legacy EDGAR current-price rows, and 28 missing rows. No current mechanical DCF row uses AI report current price.
- Experiment A/v2: status is `blocked`. It has one realized one-month return cross-section with 195 non-null one-month adjusted-close returns and zero 3m/6m/12m coverage. The rank-IC, signal-comparison, and portfolio artifacts are audit/plumbing outputs, not robust evidence.
- v2 portfolios: generated 960 monthly return rows and 43,904 holding rows across unconstrained and sector-neutral modes with 0, 10, 25, and 50 bps one-way cost levels.
- Experiment B current maps: 457 usable target/ticker prediction rows across the two non-degenerate targets. Current 2026 label-reconstruction CV Spearman is 0.60 for raw AI IRR and 0.06 for AI-minus-mechanical IRR. AI-factor-residual is blocked as a degenerate fitted map.
- Experiment B historical screen: 347,956 score rows, 44,128 monthly return rows, and 2,660,544 holding rows across 167 rebalance months and both `unconstrained` and `sector_neutral` modes. The artifacts now carry historical benchmark controls alongside the AI-map signals in the score, Rank IC, monthly-return, and holding tables. The summary embeds the approximate membership gap over all 168 calendar-aligned backcast months and warns that removed historical names are still missing from the scored universe. The raw-AI-IRR current-label projection has positive global Rank IC at 1m, 3m, 6m, and 12m, with 1m mean IC 0.026 and 12m mean IC 0.068. Its unconstrained Q1-Q5 screen is +4.9% annualized with 55.7% monthly hit rate; the sector-neutral Q1-Q5 screen is +3.5% annualized with 56.3% monthly hit rate. Composite VQMIA is positive but slightly weaker than raw AI IRR, while pure FCF/EV is stronger in this survivor-universe screen. AI-minus-mechanical is negative in this screen. AI-factor-residual is excluded from historical scores and portfolios because the fitted historical-compatible score is degenerate.
- Universe membership gap: `data/approx_sp500_membership.parquet` walks public selected S&P 500 changes backward from the current security master for 2012-2025. It has 70,455 monthly member rows across 666 approximate tickers. 271 tickers are not in the current security master with CIK/features/returns; 110 of those have cached SEC company_tickers identifier hints, while 161 have no local SEC ticker match. The average monthly missing-current-master rate is 25.2%, with a maximum of 44.7%. `results/experiment_b_historical_backcast_summary.json` now imports the same gap and records the 168-month backcast overlap, but this quantifies the removed/delisted-name gap and does not resolve it.
- Experiment C: schema, request manifest, LLM extraction request manifest, LLM response validation/audit artifact, bounded downloader, deterministic baseline extractor, and blocker manifest exist. The request manifest has 1,809 date-limited SEC filing text requests across 396 tickers, using a two-filing lookback by ticker/form where available, with 10-K, 10-K/A, 10-Q, and 10-Q/A SEC Archives URLs plus download status, timestamp/status/error, extraction status, and source-hash fields. `results/experiment_c_llm_extraction_requests.parquet` maps all 1,809 requests into schema-bound LLM jobs with prompt id `experiment_c_sec_filing_text_features`, prompt version `v1_date_limited_sec_filings`, source controls, strict text-feature output schema, and `not_run` extraction status; all 1,809 are currently `blocked_missing_downloaded_text`. `results/experiment_c_llm_extraction_responses.parquet` is an empty audit schema until offline LLM JSONL responses are ingested and validated; the ingestion path rejects forbidden DCF/return fields and only writes validated rows to the text-feature panel. The regenerated artifact has zero downloaded rows and zero ingested LLM response rows. Feature extraction remains blocked on the missing downloaded filing text corpus; the deterministic extractor command ran and produced zero rows because no local filing text exists. LLM extraction is still not run.

## Commands Run

- `.venv_review/bin/ruff check label_panel.py price_model.py v2_experiment.py experiment_b_factor_portability.py experiment_b_historical_backcast.py verify_artifacts.py tests/test_label_panel.py tests/test_price_model.py tests/test_v2_experiment.py tests/test_verify_artifacts.py tests/test_experiment_b_factor_portability.py tests/test_experiment_b_historical_backcast.py`  
  Status: passed, `All checks passed!`
- `.venv_review/bin/ruff check signal_diagnostics.py v2_experiment.py verify_artifacts.py tests/test_signal_diagnostics.py tests/test_v2_experiment.py tests/test_verify_artifacts.py`  
  Status: passed, `All checks passed!`
- `.venv_review/bin/ruff check experiment_c_text_features.py verify_artifacts.py tests/test_experiment_c_text_features.py tests/test_verify_artifacts.py`  
  Status: passed, `All checks passed!`
- `.venv_review/bin/python -m pytest tests/test_experiment_c_text_features.py tests/test_verify_artifacts.py`  
  Status: passed, `29 passed`
- `.venv_review/bin/python -m pytest`  
  Status: passed, `118 passed`
- `.venv_review/bin/python verify_artifacts.py`  
  Status: passed, `Artifact verification passed.`

Regeneration commands were run for `label_panel.py`, `forward_returns.py`, `v2_experiment.py`, `experiment_b_factor_portability.py`, `experiment_b_historical_backcast.py`, `experiment_c_text_features.py`, and `universe_membership.py` against the saved reports and cached public/free data artifacts. The v2 regeneration writes `results/signal_comparison.parquet` for the section-7 global/sector-neutral/across-sector comparison table. The Experiment B regeneration writes both unconstrained and sector-neutral historical audit artifacts for AI-map signals and benchmark controls, and its summary now embeds the approximate selected-changes membership gap. The Experiment C regeneration did not download SEC filings; downloading is now an explicit bounded mode via `--download-corpus --download-limit N`, `--extract-features` is available for downloaded local text, and `--ingest-llm-responses --llm-responses-jsonl PATH` is available for schema-validating offline LLM responses before they populate the text-feature panel.

## Critic And Verifier Review

Critic/verifier review happened. The review artifact is `results/ai_label_decomposition_critic_report.json`.

- Implementation critic findings fixed: v2 status no longer claims complete; mechanical DCF no longer relies on AI report price as a fallback; degenerate AI-factor-residual historical scores no longer produce portfolio rows.
- Verification critic findings fixed: raw price separation is now checked; constant historical scores are rejected; stale freeze/backtest role metadata was corrected; the required critic report artifact exists.
- Wording critic findings fixed: README and v2 docs lead with no validated AI alpha; broad point-in-time wording was replaced with accepted-date filing snapshots plus price matrices; v2 portfolio artifacts are described as audit/plumbing outputs; current cross-section CV is described as label reconstruction, not predictive validation.

## Remaining Blockers

- Data blocker: no repeated historical AI label vintages. The saved reports are one modern label vintage, so Experiment B remains a current-label projection.
- Data blocker: universe membership is survivor-biased and based on current public constituents plus approximate selected public changes, not CRSP/Compustat-quality point-in-time membership. The approximate membership artifact quantifies 271 missing historical tickers and finds local SEC ticker-cache hints for 110, but still does not provide missing fundamentals, sectors, prices, or delisting returns.
- Data blocker: delisted-name return coverage is missing.
- Data blocker: sectors are current public sectors, not historical classifications.
- Statistical/time blocker: Experiment A has only one realized one-month cross-section and no 3m/6m/12m realized returns yet.
- Scope/data blocker: Experiment C text features remain empty until the requested SEC filing text corpus is downloaded; deterministic baseline extraction, schema-bound LLM extraction requests, and validated LLM response ingestion are implemented, but the requested LLM extraction has not run.
- Engineering/data blocker: the current annual EDGAR feature snapshot does not fully contain raw and adjusted price fields, though the v2 price path and label panel enforce the raw/adjusted split where cached prices exist.

## Acceptance Audit

- Label panel with IRR labels, residual labels, uncertainty fields, and quality/failure flags: achieved for saved reports.
- Mechanical IRR and AI-minus-mechanical labels: achieved on a nontrivial sample, with row-level missingness and price-source flags.
- Factor residualization: achieved for the current cross-section; residual historical portability is blocked when the fitted map is degenerate.
- Rank IC before portfolio backtests: achieved for v2 current diagnostics and Experiment B historical screen.
- Global, within-sector, and across-sector diagnostics: implemented in Rank IC artifacts and summarized for key signals in `results/signal_comparison.parquet`.
- Strong composite benchmark: implemented as equal-block VQMIA and included in v2 diagnostics.
- Raw/adjusted price separation: implemented and tested; residual coverage gaps are documented.
- Holdings, turnover, costs, and exposure audit artifacts: implemented for v2 diagnostics and Experiment B historical screen.
- Freeze docs/scripts/artifacts consistency: corrected and verified.
- Tests and verifier: ruff, full pytest, and artifact verification passed.
- Docs with three-experiment framing and claim ceiling: updated.
- Final completion report: this file.

The full research objective remains incomplete because the remaining blockers are substantive data, time, and scope limitations rather than wording or implementation issues.
