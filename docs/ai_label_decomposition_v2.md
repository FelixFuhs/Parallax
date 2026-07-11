# AI Label Decomposition v2

## Purpose

Parallax v2 separates the original AI DCF label into explicit research objects:

- `raw_ai_implied_irr`: cash-flow discount rate implied by saved AI FCFF forecasts and terminal value where feasible.
- `raw_ai_annualized_value_gap`: secondary annualized AI base-case value gap versus report price.
- `mechanical_dcf_implied_irr`: standardized public-data DCF cash-flow IRR baseline.
- `ai_minus_mechanical_irr`: raw AI IRR minus the mechanical DCF IRR.
- `factor_compressible_ai_score`: fitted value from accounting/price factor residualization.
- `ai_factor_residual`: non-factor residual from the same cross-section.

The central research question is now whether any of these separated signals predicts future returns. Portfolio backtests are not treated as the first evidence layer; rank IC diagnostics should run first.

## Three Experiments

Experiment A asks whether current AI labels generated today predict future realized returns from the label date. This can only be tested prospectively unless older AI label vintages exist.

Experiment B keeps the original portability idea: take a current AI-label cross-section, decompose it, and test whether its factor-compressible component is historically portable.

Experiment C is reserved for date-limited LLM-extracted filing text features. It is intentionally separate from DCF-label residuals and should not be mixed into Experiments A or B.

## Implemented Workflow

Run the current label panel builder with:

```bash
.venv_review/bin/python sector_map.py \
  --source-html /path/to/sp500_constituents.html \
  --output data/sector_map_wikipedia.csv \
  --summary-output results/sector_map_summary.json \
  --metadata-output results/sector_map_metadata.json

.venv_review/bin/python sp500_changes.py \
  --source-html /path/to/sp500_constituents.html \
  --output data/sp500_changes_wikipedia.csv \
  --summary-output results/sp500_changes_summary.json \
  --metadata-output results/sp500_changes_metadata.json

.venv_review/bin/python label_panel.py \
  --reports-dir reports \
  --edgar-file data/edgar_features_full.json \
  --sector-map data/sector_map_wikipedia.csv \
  --price-panel-cache data/forward_price_panels.parquet \
  --output results/label_panel.parquet \
  --csv-output results/label_panel.csv \
  --summary-output results/label_panel_summary.json \
  --metadata-output results/label_panel_experiment_metadata.json
```

Current generated artifacts:

- `results/label_panel.parquet`
- `results/label_panel.csv`
- `results/label_panel_summary.json`
- `results/label_panel_experiment_metadata.json`
- `results/forward_returns.parquet`
- `results/forward_returns_summary.json`
- `results/forward_returns_metadata.json`
- `data/sector_map_wikipedia.csv`
- `results/sector_map_summary.json`
- `results/sector_map_metadata.json`
- `data/sp500_changes_wikipedia.csv`
- `results/sp500_changes_summary.json`
- `results/sp500_changes_metadata.json`
- `data/security_master.parquet`
- `results/universe_snapshot.json`
- `results/security_master_metadata.json`
- `data/quarterly_fundamentals.parquet`
- `results/quarterly_fundamentals_summary.json`
- `results/quarterly_fundamentals_metadata.json`
- `results/experiment_c_text_features.parquet`
- `results/experiment_c_text_corpus_requests.parquet`
- `results/experiment_c_llm_extraction_requests.parquet`
- `results/experiment_c_llm_extraction_responses.parquet`
- `results/experiment_c_text_features_manifest.json`
- `results/experiment_c_text_features_metadata.json`
- `results/experiment_b_factor_portability.parquet`
- `results/experiment_b_factor_portability_coefficients.csv`
- `results/experiment_b_factor_portability_summary.json`
- `results/experiment_b_factor_portability_metadata.json`
- `results/experiment_b_historical_backcast_scores.parquet`
- `results/experiment_b_historical_backcast_monthly_returns.parquet`
- `results/experiment_b_historical_backcast_holdings.parquet`
- `results/experiment_b_historical_backcast_turnover.parquet`
- `results/experiment_b_historical_backcast_exposures.parquet`
- `results/experiment_b_historical_backcast_rank_ic.parquet`
- `results/experiment_b_historical_backcast_rank_ic_summary.parquet`
- `results/experiment_b_historical_backcast_rank_ic_by_year.parquet`
- `results/experiment_b_historical_backcast_rank_ic_by_sector.parquet`
- `results/experiment_b_historical_backcast_rank_ic_coverage.parquet`
- `results/experiment_b_historical_backcast_summary.json`
- `results/experiment_b_historical_backcast_metadata.json`
- `results/v2_experiment_status.json`
- `results/v2_experiment_metadata.json`
- `results/rank_ic.parquet`
- `results/rank_ic_summary.parquet`
- `results/rank_ic_by_year.parquet`
- `results/rank_ic_by_sector.parquet`
- `results/rank_ic_coverage.parquet`
- `results/signal_comparison.parquet`
- `results/holdings.parquet`
- `results/monthly_returns.parquet`
- `results/turnover.parquet`
- `results/exposures.parquet`

The 2026-05-19 run processed 272 saved report rows across 269 tickers. Raw AI cash-flow IRR coverage was 95.6%, mechanical DCF cash-flow IRR coverage was 81.3%, AI-minus-mechanical coverage was 79.0%, and raw-AI factor residual coverage was 92.3%. Raw/adjusted price coverage from the cached forward price panel is 71.7%; the mechanical DCF path uses raw close for 178 rows and legacy EDGAR current price for 66 rows, with no AI report-price fallback. There were 21 clean-label exclusions.

The sector map is built from the current S&P 500 constituent table at [Wikipedia's List of S&P 500 companies](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies), retrieved on 2026-05-19 into a local HTML snapshot before parsing. It provides current GICS sector/sub-industry coverage only. The generated map has 503 tickers, 11 sectors, and 127 sub-industries; the label panel maps 267 of 272 rows, or 98.2%.

The same public HTML snapshot is also used for `data/sp500_changes_wikipedia.csv`, parsed from Wikipedia's selected changes table. It currently has 395 selected-change rows from 1976-07-01 through 2026-05-07. This is recorded in the security-master metadata as approximate public membership-change provenance, not point-in-time index membership.

The approximate membership gap artifact `data/approx_sp500_membership.parquet` walks those selected changes backward from the current security master for 2012-2025. It is not used to claim point-in-time validity; it quantifies what remains missing. The generated panel has 70,455 monthly member rows across 666 approximate tickers. Of those, 271 tickers are not in the current security master with CIK/features/returns, and the average monthly missing-current-master rate is 25.2%, peaking at 44.7%. The cached SEC `company_tickers.json` can supply identifier hints for 110 of those missing tickers, while 161 have no local SEC ticker match. This is direct evidence that the current Experiment B screen is still survivor-biased and lacks removed/delisted-name coverage.

Label-quality summaries now include model-tier and market-cap-bucket cuts. In the current run, 271 rows are `cheap` tier and one row is `full` tier; the cheap-tier exclude rate is 7.7%. Market-cap buckets are also reported in `results/label_panel_summary.json`, with exclude rates of 3.3% for large, 11.2% for mid, and 8.9% for small names. These cuts are descriptive quality diagnostics, not performance evidence.

Repeated-label uncertainty is implemented at the ticker level. The panel stores `mean_ai_irr`, `median_ai_irr`, `std_ai_irr`, `ai_irr_iqr`, `ai_irr_rank_std`, `model_disagreement`, `tier_disagreement`, `prompt_disagreement`, `uncertainty_inverse_weight`, and `uncertainty_adjusted_label_weight`. In the saved artifacts only AAPL has repeated labels across cheap and full tiers; prompt disagreement remains null because the saved reports do not include prompt identifiers.

The v2 experiment status is currently `blocked` for research evidence, while artifact execution completed with warnings/blockers. After an approved public yfinance refresh, the forward-return artifact has 195 non-null one-month adjusted-close returns across 270 ticker/date rows. Three-, six-, and twelve-month horizons still have zero coverage because those exits are future-dated from the March 2026 label vintage. The status artifact carries blockers for a one-month diagnostic sample and zero-coverage longer horizons.

The rank-IC artifact is no longer AI-only. The 2026-05-19 v2 status registers 30 signal columns: the AI/mechanical decomposition columns, `composite_vqmia_score`, VQMIA block scores, and simple controls such as `fcf_to_ev`, `momentum_12_1`, `accruals`, and `debt_to_equity`. It writes global/within-sector/across-sector IC rows, by-year IC summaries, by-sector IC summaries, per-date coverage rows, and `results/signal_comparison.parquet`, which lines up AI IRR, AI residual, Mechanical IRR, Composite VQMIA, and FCF/EV across global, sector-neutral/within-sector, and across-sector mean IC columns. It now contains one realized one-month cross-section, so the output is useful for smoke-testing signal plumbing and audit artifacts, but it is not enough history for robust Newey-West statistics or alpha claims.

The portfolio audit artifacts now include both `unconstrained` and `sector_neutral` modes. Unconstrained buckets are equal-name weighted. Sector-neutral buckets rank names within sector and then weight sectors equally before equal-weighting names inside each sector/bucket. The current generated `monthly_returns` artifact has 960 rows across both modes and the four one-way transaction cost levels, while `holdings` has 43,904 holding rows with label and benchmark signal fields attached. These are one-month audit/plumbing artifacts, not robust strategy evidence.

Experiment B has two layers. `experiment_b_factor_portability.py` fits label-reconstruction CV within the current 2026 cross-section from public features to `raw_ai_implied_irr`, `ai_minus_mechanical_irr`, and `ai_factor_residual`, writing predictions and coefficients. The 2026-05-19 run fit 457 usable target/ticker rows across the two non-degenerate targets. Cross-validated Spearman was 0.60 for raw AI IRR and 0.06 for AI-minus-mechanical IRR; the AI-factor-residual map is blocked because its fitted score is degenerate.

`experiment_b_historical_backcast.py` then fits historical-compatible versions of those current-label maps and projects them through monthly accepted-date filing snapshots plus historical price matrices from 2012-2025. It writes Rank IC, scores, holdings, monthly returns, turnover, cost, exposure, summary, and metadata artifacts. The run covered 167 rebalance months and 347,956 historical score rows after excluding current constituents before their public S&P 500 add date when that add date appears in the selected-changes table. The regenerated audit artifacts now include both `unconstrained` and `sector_neutral` modes for the AI maps and historical benchmark controls, producing 44,128 monthly-return rows and 2,660,544 holding rows. The Experiment B summary now embeds `approximate_membership_gap` from `results/approx_sp500_membership_summary.json`; on a calendar-month alignment it overlaps all 168 backcast rebalance months and records an average 25.2% missing-current-master rate, a 44.7% maximum missing rate, and the `removed_names_missing_from_backcast_universe` warning. The raw-AI-IRR projection had positive global Rank IC at 1m (`0.026`, Newey-West t `3.03`), 3m (`0.039`, t `2.82`), 6m (`0.052`, t `2.98`), and 12m (`0.068`, t `3.23`). Within-sector IC stayed positive at the same horizons, while across-sector IC was weaker. The raw-AI-IRR projection also had a positive unconstrained Q1-Q5 screen (`+4.9%` annualized gap; Q1 beat Q5 in 55.7% of months) and a positive sector-neutral Q1-Q5 screen (`+3.5%` annualized gap; Q1 beat Q5 in 56.3% of months). Historical benchmark controls are now present in the same score, Rank IC, monthly-return, and holding artifacts: composite VQMIA was positive but slightly weaker than raw AI IRR, while pure FCF/EV was stronger in this survivor-universe screen. AI-minus-mechanical IC and portfolio spreads were negative, and AI-factor-residual is excluded from scores and portfolios because its fitted historical-compatible score is degenerate. This is still a current-label projection over a survivor-biased universe with current sector mapping, incomplete delisting coverage, and approximate membership provenance, not a true historical AI-label-vintage test.

The security-master artifact records ticker/CIK mappings and universe assumptions for the current public-data universe. The generated snapshot has 396 tickers, 100% CIK coverage, 100% current EDGAR-feature coverage, 98.5% current-sector coverage, 395 approximate public membership-change rows, and `point_in_time_membership: false`. It explicitly records that the universe is current S&P 500 ex-Financials/ex-Real-Estate membership rather than CRSP/Compustat-quality historical membership. `results/approx_sp500_membership_summary.json` now makes the survivor-bias gap measurable and separates SEC ticker-cache identifier hints from fully unmapped removed names, but it does not add missing fundamentals, sectors, prices, or delisting returns for the removed historical names.

The quarterly-fundamentals artifact extracts cached 10-Q facts for the same 396-ticker universe. The current run wrote 75,331 quarterly period rows, covering periods from 2007-06-30 through 2026-03-17. It includes quarter-length revenue, gross profit, operating cash flow, capex, balance-sheet fields, free cash flow, TTM rollups, QoQ changes, and YoY changes. The extraction intentionally does not infer fourth-quarter values from annual 10-K facts, so revenue TTM coverage is limited to rows with four explicit quarterly observations.

Experiment C has a separate schema artifact, SEC filing text request manifest, LLM extraction request manifest, LLM response validation/audit artifact, bounded SEC text downloader, deterministic baseline extractor, and blocker manifest. `results/experiment_c_text_corpus_requests.parquet` currently lists 1,809 date-limited SEC filing text requests across 396 tickers, with a two-filing lookback by ticker/form where available, SEC Archives URLs, accessions, forms, filed dates, target local text paths, download status, download timestamps/status codes/errors, extraction status, and source hashes once downloaded. `results/experiment_c_llm_extraction_requests.parquet` maps those same 1,809 filing requests into schema-bound LLM jobs with prompt id `experiment_c_sec_filing_text_features`, prompt version `v1_date_limited_sec_filings`, a strict output schema for the text features, source controls, and explicit `not_run` extraction status. All 1,809 LLM jobs are currently `blocked_missing_downloaded_text`, so no LLM extraction is claimed. `results/experiment_c_llm_extraction_responses.parquet` is an empty response-audit schema until offline LLM JSONL responses are ingested; the ingestion path validates feature names, numeric ranges, evidence shape, source identifiers, and forbidden DCF/return fields before writing any text-feature rows. The downloader is explicit and capped by `--download-corpus --download-limit N`; the default limit is zero, so regenerating the artifacts does not perform network downloads. `--extract-features` can populate the text-feature panel from downloaded local filing text using `deterministic_keyword_baseline_v1`; this is a pipeline baseline, not the requested LLM extraction. `--ingest-llm-responses --llm-responses-jsonl PATH` can populate the same panel from validated LLM responses once downloaded filing text and responses exist. `results/experiment_c_text_features.parquet` remains an empty schema placeholder because zero filing texts are downloaded and zero LLM responses are ingested. The manifest is blocked on `missing_date_limited_text_corpus`, `text_extraction_not_run`, `llm_extraction_requests_not_ready`, and `llm_text_extraction_not_run`; it explicitly forbids mixing DCF label columns into the text-feature panel, LLM output schema, or accepted response payloads.

Sector-specific DCF prompt support is now an implemented scaffold and prompt contract rather than an informal TODO. `research_scaffolds.py` defines stable Software, Semiconductors, Energy, Industrials, Healthcare, Utilities, Consumer, and General templates with shared output schema, forecast-driver focus, terminal-assumption focus, risk checks, and preferred source evidence. `openrouter.py` loads `data/sector_map_wikipedia.csv` by default and appends the matching sector context to generated prompts while instructing the model not to add sector-specific JSON fields. This supports future label collection without changing the saved report schema; it is not yet validated production label collection.

## Quality Policy

Rows are not silently rescued. Flags are assigned policies:

- Exclude: `missing_price`, `missing_ai_irr`, `missing_wacc`, `missing_scenario`, `scenario_order_fail`, `internal_inconsistency`, `nonpositive_ai_value`, `wacc_terminal_spread_too_small`, `share_count_outlier`, `ev_negative`.
- Downweight: `default_terminal_growth`, `suspiciously_round_forecast`, `terminal_value_dominates`, `forecast_discontinuity`, `stale_price`.
- Warning: missing mechanical DCF optional inputs, missing comps, raw-price unavailability, ignored AI-report price fallbacks if they ever occur, and legacy parser flags such as `margin_reversal`.

Mechanical labels prefer raw close from the cached public price panel. If raw close is unavailable, they can use legacy EDGAR `current_price` with `mechanical_price_asof_unverified`; AI report `current_price` is now refused for mechanical DCF controls and flagged as `report_price_ignored_for_mechanical_dcf` rather than used as a last-resort price. The current generated panel has no report-price mechanical rows.

## Diagnostics And Backtests

Implemented library support:

- `security_master.py`: file-based ticker/CIK security master and universe snapshot with survivor-bias metadata.
- `universe_membership.py`: approximate public selected-change membership timeline and survivor-bias gap report.
- `sector_map.py`: current S&P 500 sector/sub-industry map parser with source metadata.
- `sp500_changes.py`: approximate public selected-change parser for membership provenance.
- `quarterly_fundamentals.py`: cached 10-Q quarterly fundamentals, TTM rollups, QoQ changes, and YoY changes.
- `benchmarks.py`: equal-block VQMIA benchmark construction.
- `experiment_b_factor_portability.py`: current-label factor-map fitter for Experiment B with explicit no-historical-backcast warning.
- `experiment_b_historical_backcast.py`: historical-compatible Experiment B Rank IC, score, holdings, return, turnover, cost, exposure, and summary runner with explicit survivor/current-label caveats.
- `experiment_c_text_features.py`: separate Experiment C text-feature schema, request manifest, bounded SEC filing-text downloader, deterministic baseline extractor, and blocker manifest.
- `signal_diagnostics.py`: global, within-sector, across-sector, by-year, by-sector, and monthly coverage rank IC diagnostics.
- `v2_experiment.build_portfolio_audit_artifacts`: unconstrained and sector-neutral holdings-level, monthly return, turnover, cost, and exposure outputs for v2 label diagnostics.
- `backtest.run_audited_signal_backtest`: legacy frozen-surrogate holdings-level, monthly return, turnover, cost, and exposure outputs.
- `experiment_registry.py`: reproducibility metadata with repo-relative paths and snapshot hashes.
- `research_scaffolds.py`: explicit boundaries for quarterly fundamental rollups, survivor-only universe metadata, sector DCF prompt templates with a stable output schema, and Experiment C text-feature configuration.
- `forward_returns.py`: adjusted-close forward-return builder with an offline cache-only mode for reproducible blocked artifacts.
- `v2_experiment.py`: rank-IC runner for AI decomposition signals plus VQMIA/control benchmarks when usable forward returns exist; blocked-status artifact writer when they do not.
- `verify_artifacts.py`: label-panel, quality-summary, forward-return, v2-status, Experiment B/C, and generated path hygiene checks.

These pieces are tested on synthetic data. A full historical v2 run still needs repeated label vintages, historical sector history, full point-in-time universe membership, delisting coverage, and a non-survivor universe. The new Experiment B backcast is useful as a portability screen, but its claim ceiling remains below institutional point-in-time evidence.

## Claim Ceiling

Current status is diagnostic/private research. The code can build the decomposition artifacts from saved reports and public repo data, but the project cannot yet claim institutional point-in-time validity or AI alpha.

Open blockers are data and scope blockers:

- Sector history is absent; the attached sector map is current-only and cannot support historical sector-neutral claims.
- S&P 500 selected changes are attached as approximate provenance, not full point-in-time membership.
- The universe is still survivor-biased.
- The saved AI reports are one modern label vintage, not a historical panel of label vintages.
- The current EDGAR feature snapshot lacks raw close and adjusted close as separate stored fields, though new code enforces the split.
- The forward-return artifact currently has only one realized one-month cross-section; longer horizons are not observable yet.
- Experiment B now has a historical-compatible current-label projection, but not repeated historical AI label vintages.
- Sector-specific prompts, historical universe-membership data, bulk SEC filing text downloads, and actual LLM text-feature extraction are scaffold/blocker-level next steps, not completed production workflows.
