# Parallax Lessons Learned

This document captures the operational lessons from building and running Parallax through the first EDGAR validation pass, Nano smoke tests, and the first full 100-ticker Nano batch.

## OpenRouter Cheap Tier

- `response_format: {"type": "json_object"}` cannot be combined with `:online` on the cheap tier reliably. The working pattern in this repo is:
  - do not request `json_object` for `openai/gpt-5.4-nano:online`
  - enable the `response-healing` plugin
  - parse the first valid JSON object with `json.JSONDecoder().raw_decode()`
- The cheap-tier healing path in [openrouter.py](C:/Users/Anwender/OneDrive/Desktop/Parallax/openrouter.py) is materially better than naive brace slicing. It can recover fenced JSON and ignore trailing commentary.
- Cheap-tier failures are not just `429`s. Under load, OpenRouter also returned payloads with no `choices[0]`, or empty message content. Those need to be treated as operational failures even when token usage is zero.

## Rate Limits And Retries

- Raising the cheap tier to 50 req/min worked mechanically, but it was too aggressive for a 100-name bulk pass.
- On the first 100-ticker Nano run at 50 req/min, about 34% of tickers failed with `OpenRouter response did not include choices[0].`
- The total first-pass failure rate at that setting was worse than the raw 34% payload error rate because some additional names failed with empty responses or incomplete JSON.
- Recommended practical rate for cheap-tier bulk runs: 20 to 30 req/min.
- Always plan a retry pass. The retry recovered a meaningful number of names without any code changes.
- Do not assume the rate limiter is the only control surface. The provider can still degrade structurally before it starts returning clean `429`s.

## EDGAR Validation Method

- Spot-check 5 tickers against StockAnalysis annual pages:
  - revenue
  - net income
  - total debt
  - tolerance: within 10%
- Always check fiscal years. Three bad pulls stood out immediately:
  - `DUK` pulled FY2016
  - `NEE` pulled FY2012
  - `TJX` pulled FY2017 / period end 2018-02-03
- Always inspect derived-feature sanity:
  - `MCD` had broken `shares_outstanding`, which corrupted `market_cap` and produced nonsense `fcf_yield`
  - extreme ROIC values for `BKNG`, `FTNT`, `GE`, and `ORLY` were likely driven by negative or tiny book equity / invested capital, not necessarily extraction bugs
- Gross margin coverage was weak, but XGBoost-style tabular models can tolerate that if the rest of the feature set is sound and missingness is handled explicitly.

## Parser Bugs Found And Fixed

- Ratio parsing corruption for sub-1% values:
  - values like `0.5%` must become `0.005`
  - the fix was to distinguish percent-suffixed inputs from already-decimal ratios so sub-1% values are not divided twice
- Silent `or`-based defaults overwriting explicit zeros:
  - explicit `0.0` values for fields like tax rate, terminal growth, and NOL utilization must survive parsing
  - the fix was to use `value is not None` semantics instead of truthiness-based defaults
- First-brace / last-brace JSON healing was too brittle:
  - it failed when the model returned valid JSON followed by trailing explanation text
  - the fix was to walk the string and use `json.JSONDecoder().raw_decode()` to extract the first valid JSON object
- The parser test suite in [tests/test_parser.py](C:/Users/Anwender/OneDrive/Desktop/Parallax/tests/test_parser.py) and [tests/test_openrouter.py](C:/Users/Anwender/OneDrive/Desktop/Parallax/tests/test_openrouter.py) now covers these cases.

## Nano Vs Full Tier

- Nano is cheap enough for broad sweeps:
  - roughly `$0.02` per report in typical runs
- Full tier is much more expensive:
  - budget roughly `$0.87` per report
  - real observed cost can vary materially with prompt length, output length, and online context
- Nano has a systematic bearish bias in the saved base-case upsides, but the cross-sectional spread is still wide enough to use as a distillation target.
- Nano still fails to produce complete JSON for about 10% to 15% of names with unusual accounting or awkward output structure, and can fail much more often if the request rate is pushed too hard.

## Parallelism

- `--parallel 100` is acceptable from the script's point of view. The internal rate limiter is still the real throttle.
- The main scaling risk is provider behavior, not local thread count.
- On Windows, backgrounding the Python process opens a visible console window. That is normal for this setup.
- Same-day reruns can be skipped because report filenames are date-keyed. If you need a true rerun, move or rename existing `YYYY-MM-DD` report files first.

## BYOK

- When using your own OpenAI key through OpenRouter, billing lands on the OpenAI dashboard, not the OpenRouter dashboard.
- OpenRouter still handles routing and response shape, but cost attribution follows the underlying provider key.

## Additional Lessons

- A saved report is not automatically a usable training row:
  - `CMS` saved successfully but carried `stale_price`, so it had no usable upside target
- Bulk matching should define "clean EDGAR" explicitly:
  - no EDGAR `error`
  - fiscal year at least 2024
  - manually exclude known broken rows like `MCD`
- Retry-only reruns are important for both cost control and data hygiene. Limiting every ticker to at most two attempts kept the batch process predictable.
- The dominant production failures were operational and structural, not valuation-specific:
  - missing `choices[0]`
  - empty message content
  - missing `historical.revenue`
  - missing `assumptions.wacc`
  - terminal-method outputs that referenced exit multiples without providing the multiple
- Keep a separate backup folder when moving prior reports out of the way for reruns. That preserves the original artifacts without polluting the active training set scan.

## 2026-05-19 AI Label Decomposition Goal Pass

- Goal file: `inputs/goal_prompts/2026-05-19_ai_label_decomposition.md`.
- Implemented a v2 label-panel path rather than widening the old five-feature surrogate in place:
  - `label_panel.py` builds raw AI IRR, mechanical DCF IRR, AI-minus-mechanical IRR, factor-compressible fitted scores, residuals, uncertainty fields, and quality flags.
  - `sector_map.py` turns a local public S&P 500 constituent HTML snapshot into a current sector/sub-industry map with source metadata.
  - `sp500_changes.py` turns the same public HTML snapshot's selected changes table into approximate membership-change provenance without claiming point-in-time membership.
  - `security_master.py` records ticker/CIK mappings, filing provenance, current-universe membership assumptions, and survivor-bias caveats in file-based artifacts.
  - `quarterly_fundamentals.py` extracts cached 10-Q fundamentals, TTM rollups, QoQ changes, and YoY changes without inventing Q4 values from annual 10-K facts.
  - `price_model.py`, `historical.py`, and `edgar.py` now separate raw close for valuation from adjusted close for returns and momentum.
  - `benchmarks.py`, `signal_diagnostics.py`, `experiment_registry.py`, and audit helpers in `backtest.py` add the benchmark, rank-IC, metadata, and holdings-output foundations.
  - `forward_returns.py` builds adjusted-close forward-return artifacts, with an offline cache-only mode so blocked runs are reproducible without an implicit network dependency.
  - `v2_experiment.py` now includes AI decomposition signals, VQMIA block scores, and simple factor controls in the same rank-IC artifact instead of testing the AI labels without benchmark columns.
  - `signal_diagnostics.py` now writes global/within-sector/across-sector IC rows plus by-year, by-sector, and per-date coverage diagnostics.
  - v2 portfolio audit artifacts now include both unconstrained equal-name buckets and sector-neutral buckets with equal-sector-then-equal-name weights.
  - Repeated-label uncertainty now includes AI-IRR IQR, rank dispersion, model/tier disagreement, prompt-disagreement slots, and uncertainty-adjusted label weights.
  - Freeze artifacts now explicitly separate the primary XGBoost regressor from the Elastic Net transparency baseline; verifier checks reject absolute paths and baseline/primary role confusion.
  - `experiment_b_factor_portability.py` fits current AI-label-to-factor maps for raw AI IRR and AI-minus-mechanical IRR; AI factor residual is blocked when its fitted score is degenerate.
  - `experiment_b_historical_backcast.py` fits historical-compatible versions of those current-label maps and projects them through monthly accepted-date filing snapshots plus historical price matrices with Rank IC, holdings, turnover, cost, exposure, performance artifacts, and historical benchmark controls.
  - `experiment_c_text_features.py` writes a separate text-feature schema, SEC filing text request manifest, bounded SEC filing-text downloader, deterministic baseline extractor, and blocker manifest that is explicitly blocked until the requested date-limited filing text corpus is downloaded and LLM extraction runs.
  - `research_scaffolds.py` keeps quarterly fundamentals, universe assumptions, stable sector-specific DCF prompt templates, and Experiment C text features explicit but prototype-grade.
  - `openrouter.py` can now append sector-specific DCF prompt context from `data/sector_map_wikipedia.csv` without changing the common JSON output schema.
- Generated `results/label_panel.parquet`, `results/label_panel.csv`, `results/label_panel_summary.json`, and `results/label_panel_experiment_metadata.json`.
- Generated `data/sector_map_wikipedia.csv`, `results/sector_map_summary.json`, and `results/sector_map_metadata.json` from Wikipedia's current S&P 500 constituent table, retrieved 2026-05-19.
- Generated `data/sp500_changes_wikipedia.csv`, `results/sp500_changes_summary.json`, and `results/sp500_changes_metadata.json` from Wikipedia's selected S&P 500 changes table; it has 395 rows from 1976-07-01 through 2026-05-07 and is explicitly approximate.
- Generated `data/security_master.parquet`, `results/universe_snapshot.json`, and `results/security_master_metadata.json`; the snapshot confirms current-universe, non-point-in-time membership with 98.5% current-sector coverage and approximate selected-change provenance.
- Generated `data/approx_sp500_membership.parquet`, `results/approx_sp500_membership_summary.json`, and `results/approx_sp500_membership_metadata.json`; the selected-change walkback covers 70,455 monthly member rows across 666 approximate tickers for 2012-2025 and quantifies 271 tickers missing from the current security master with CIK/features/returns, including 110 with cached SEC ticker identifier hints and 161 with no local SEC ticker match.
- Generated `data/quarterly_fundamentals.parquet`, `results/quarterly_fundamentals_summary.json`, and `results/quarterly_fundamentals_metadata.json`; the cached 10-Q panel has 75,331 rows across 396 tickers.
- Generated `results/experiment_c_text_features.parquet`, `results/experiment_c_text_corpus_requests.parquet`, `results/experiment_c_text_features_manifest.json`, and `results/experiment_c_text_features_metadata.json`; Experiment C now has 1,809 SEC filing text requests across 396 tickers, a two-filing lookback by ticker/form where available, download/extraction bookkeeping columns, a bounded downloader, and a deterministic baseline extractor, but remains blocked on downloading the date-limited text corpus and running LLM extraction.
- Generated `results/experiment_b_factor_portability.parquet`, `results/experiment_b_factor_portability_coefficients.csv`, `results/experiment_b_factor_portability_summary.json`, and `results/experiment_b_factor_portability_metadata.json`; Experiment B now has current cross-section label-reconstruction fits for the two non-degenerate targets.
- Generated `results/experiment_b_historical_backcast_scores.parquet`, `results/experiment_b_historical_backcast_rank_ic.parquet`, `results/experiment_b_historical_backcast_rank_ic_summary.parquet`, `results/experiment_b_historical_backcast_rank_ic_by_year.parquet`, `results/experiment_b_historical_backcast_rank_ic_by_sector.parquet`, `results/experiment_b_historical_backcast_rank_ic_coverage.parquet`, `results/experiment_b_historical_backcast_monthly_returns.parquet`, `results/experiment_b_historical_backcast_holdings.parquet`, `results/experiment_b_historical_backcast_turnover.parquet`, `results/experiment_b_historical_backcast_exposures.parquet`, `results/experiment_b_historical_backcast_summary.json`, and `results/experiment_b_historical_backcast_metadata.json`. The 2012-2025 raw-AI-IRR current-label projection had positive global Rank IC at 1m/3m/6m/12m and positive unconstrained plus sector-neutral Q1-Q5 screens after an approximate public add-date filter; historical benchmark controls are now present in the same score, Rank IC, monthly-return, and holding artifacts; AI-minus-mechanical remained negative, and AI-factor-residual is excluded from historical scores and portfolio rows because its historical-compatible fit is degenerate.
- Generated `results/forward_returns.parquet`, `results/forward_returns_summary.json`, `results/v2_experiment_status.json`, rank-IC diagnostics, `results/signal_comparison.parquet`, and v2 audit artifacts. After an approved public yfinance refresh, the forward-return artifact has 195 one-month adjusted-close returns across 270 ticker/date rows; three-, six-, and twelve-month horizons are still zero-coverage because those exits are future-dated, so the v2 status is blocked for research evidence.
- The regenerated v2 monthly-return and holding artifacts include both `unconstrained` and `sector_neutral` portfolio modes. They remain one-month audit/plumbing artifacts rather than robust strategy evidence.
- The saved reports produced 272 rows across 269 tickers; cash-flow AI-minus-mechanical coverage was about 79.0%, raw/adjusted price coverage is 71.7%, no mechanical DCF row uses AI report price, and 21 rows were excluded from clean labels.
- Label-quality reporting now includes sector, model-tier, and market-cap-bucket cuts. Label-panel sector coverage is 98.2% using the current Wikipedia sector snapshot.
- The run matched the spirit on decomposition and quality separation, but it did not complete the full research end state:
  - sector history is absent; the attached sector map is current-only,
  - the universe is still survivor-biased, and the approximate membership artifact now quantifies rather than fixes the missing removed-name gap,
  - selected S&P 500 changes are not full point-in-time constituent history,
  - there is still only one modern AI-label vintage,
  - Experiment B has a historical-compatible current-label projection, not repeated label vintages generated at historical dates,
  - Experiment A rank-IC and portfolio outputs use only one realized one-month label cross-section,
  - Bulk SEC filing text download and LLM text-feature extraction remain future work; deterministic text-feature extraction is now implemented for downloaded local SEC text but has no rows until the corpus exists.
- Prompt adjustment for the next goal: ask for a concrete v2 historical run that rebuilds raw/adjusted price panels, replaces current-only sectors with sector history, runs IC diagnostics over many months, and writes audited backtest artifacts from non-survivor historical matrices.
