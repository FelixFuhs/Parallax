# Parallax

Distilling AI Equity Valuations into Backtestable Signals

## Overview

Parallax asks whether frontier LLM valuations contain a reusable cross-sectional signal. The original workflow generated AI-written DCF reports, converted base-case upside into one ranking target, and tested whether a five-feature surrogate survived historical falsification. The v2 workflow keeps that negative finding but adds an explicit AI label decomposition layer: raw AI IRR, a standardized mechanical DCF IRR, AI-minus-mechanical IRR, a factor-compressible fitted score, and a non-factor AI residual.

## Research Question

> Does any part of the AI-implied expected-return signal predict future returns after separating raw AI signal, mechanical DCF signal, factor-compressible AI signal, and non-factor AI residual signal?

## Current Answer

The project still has no validated AI alpha result and no true historical AI-label-vintage evidence. The original five-feature surrogate result remains negative: it did not outperform a simple `fcf_to_ev` value sort over 2012-2025 in either the broad or clean universe. The v2 workflow adds a historical-compatible Experiment B portability screen: the current raw-AI-IRR factor map has positive 2012-2025 Rank IC across 1/3/6/12-month horizons and a positive Q1-Q5 screen after an approximate public add-date filter, while AI-minus-mechanical does not and AI-factor-residual is blocked as degenerate. This is hypothesis-generating only because it projects one modern label vintage backward over a survivor-biased universe. See [docs/ai_label_decomposition_v2.md](docs/ai_label_decomposition_v2.md).

## Pipeline Overview

- `openrouter.py`: calls the GPT-5.4 family through OpenRouter, appends sector-specific DCF prompt context when a sector map is available, rate-limits requests, heals malformed cheap-tier JSON, runs validation plus DCF, and writes research reports.
- `parser.py`: normalizes raw research JSON into a structured valuation input with defaults, schema checks, and quality flags.
- `dcf.py`: runs the three-scenario FCFF DCF, computes enterprise and equity value, and writes reusable valuation artifacts.
- `edgar.py`: fetches latest annual SEC companyfacts, aligns tags and fallbacks, and computes the current cross-sectional feature set.
- `sector_map.py`: builds a provenance-tracked current S&P 500 sector/sub-industry map from a local public HTML snapshot.
- `sp500_changes.py`: builds an approximate selected S&P 500 membership-change artifact from the public changes table, without treating it as CRSP-quality history.
- `security_master.py`: writes a file-based security master and universe snapshot with ticker/CIK mapping, filing provenance, and explicit survivor-bias caveats.
- `quarterly_fundamentals.py`: extracts cached 10-Q quarterly fundamentals, TTM rollups, QoQ changes, and YoY changes for the security-master universe.
- `historical.py`: builds accepted-date filing snapshots and reconstructs monthly historical feature matrices from accepted SEC filings.
- `price_model.py`: separates raw close prices for valuation and market-cap/EV construction from adjusted close prices for returns and momentum.
- `label_panel.py`: builds the v2 AI label panel with raw AI IRR, mechanical DCF IRR, AI-minus-mechanical residuals, quality flags, uncertainty fields, and factor residualization.
- `forward_returns.py`: builds adjusted-close forward-return panels for v2 diagnostics from a cached public price panel or, when explicitly allowed, a fresh yfinance download.
- `benchmarks.py`: builds an equal-block VQMIA benchmark from value, quality, momentum, investment, accruals, and balance-sheet controls.
- `signal_diagnostics.py`: computes rank IC diagnostics globally, within sector, across sector, by sector, by year, and by monthly coverage before portfolio backtests.
- `experiment_registry.py`: writes repo-relative reproducibility metadata and snapshot hashes.
- `research_scaffolds.py`: records scaffold-level boundaries for quarterly fundamentals, survivor-only universe metadata, stable sector-specific DCF prompt templates, and the separate LLM text-feature experiment.
- `experiment_b_factor_portability.py`: fits claim-safe current AI-label-to-factor maps for Experiment B without treating them as historical alpha evidence.
- `experiment_b_historical_backcast.py`: projects historical-compatible current-label factor maps through accepted-date filing snapshots and historical price matrices, then writes Rank IC, scores, holdings, turnover, costs, exposures, and performance summaries.
- `experiment_c_text_features.py`: writes the separate Experiment C text-feature schema, SEC text request manifest, schema-bound LLM extraction request manifest, LLM response validation/audit artifact, bounded downloader, and blocker manifest for date-limited filing text extraction.
- `v2_experiment.py`: runs v2 rank-IC diagnostics for AI decomposition signals plus VQMIA/control benchmarks, with unconstrained and sector-neutral portfolio audit artifacts when forward returns exist.
- `verify_artifacts.py`: verifies label-panel, diagnostic, backtest, Experiment B/C, and generated path-hygiene artifacts.
- `distill.py`: matches AI labels to EDGAR features, compares Elastic Net, XGBoost regressor, and XGBoost ranker, and saves plots plus model metadata.
- `stability.py`: runs Elastic Net baseline stability diagnostics without overwriting the primary XGBoost freeze docs.
- `backtest.py`: scores the historical feature matrix with the frozen surrogate, forms monthly portfolios with T+1 execution, and includes holdings-level audit helpers for turnover, costs, and exposures.

## Key Results

![Backtest cumulative returns](plots/backtest_cumulative.png)

- Broad universe: the surrogate `Q1-Q5` spread was positive in 7 of 14 calendar years and delivered `-0.7%` annualized return.
- Clean universe: the surrogate `Q1-Q5` spread was positive in 5 of 14 calendar years and delivered `-2.4%` annualized return.
- Pure `FCF/EV` `Q1` beat surrogate `Q1` in both universes: `20.9%` vs `18.5%` annualized in broad, and `21.5%` vs `18.6%` in clean.
- The frozen XGBoost regressor fits the current matched sample well enough to look interesting ex ante, but that present-day fit does not translate into a robust historical long-short signal.

## Methodology Highlights

- Filing-date discipline: historical features are built from filings accepted before the rebalance date, not from fiscal period ends.
- T+1 execution: portfolio returns use the first available price after the rebalance date.
- Frozen model and feature spec: the primary backtest uses the pre-registered five-feature XGBoost regressor documented in [docs/model_freeze.md](docs/model_freeze.md).
- Raw/adjusted price separation: new price paths use raw close for valuation, market cap, and EV; adjusted close is used for returns and momentum.
- Label quality is first-class: failed, default-heavy, internally inconsistent, or scenario-order-failing reports are flagged instead of silently repaired.
- Honest benchmarking: the legacy surrogate is compared with equal-weight, the opposite quintile, and a pure `FCF/EV` value sort; the v2 diagnostics also register an equal-block VQMIA composite and its value, quality, momentum, investment, accrual, and balance-sheet blocks as control signals.
- Two universes: `clean` requires all five features to be present; `broad` allows up to two missing features and relies on XGBoost's native missing-value handling.

## Limitations

- The historical universe is survivor-biased because it starts from a current S&P 500 ex-Financials ex-REITs list in [tickers.txt](tickers.txt).
- The public selected-changes table is recorded as approximate membership provenance, not a replacement for point-in-time index constituents.
- The sector map is a current public S&P 500 snapshot, not historical point-in-time sector membership.
- The AI labels come from a single modern cross-section rather than repeated label vintages through time.
- The Experiment B historical screen projects that single modern label vintage backward; it is not evidence from labels generated at those historical dates.
- The primary labeling model is the cheapest GPT-5.4 Nano tier, not the stronger full model.
- There are no delisted stocks or CRSP-style return histories.
- Nano collection was operationally noisy: the preserved 100-name batch still ended with 33 failures after one retry, and broader logs imply roughly one-third to two-fifths failure rates depending on the pass.

## What Would Change With Better Resources

- Run the full GPT-5.4 model with `xhigh` reasoning across the entire cross-section, not just as a spot check.
- Replace the survivor-biased public-data setup with a proper point-in-time database that includes delistings, such as CRSP/Compustat.
- Generate multiple AI label vintages over time instead of distilling a single 2026 cross-section backward.
- Use sector-specific surrogate models and feature engineering rather than one pooled specification.

## Setup And Usage

This repo is a research workflow, not a packaged product. Install the dependencies into a virtual environment before running the scripts.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set the required environment variables:

```powershell
$env:OPENROUTER_API_KEY = "..."
$env:SEC_USER_AGENT_NAME = "Your Name"
$env:SEC_USER_AGENT_EMAIL = "you@example.com"
```

Optional OpenRouter attribution variables are shown in [.env.example](.env.example): `OPENROUTER_TITLE` and `OPENROUTER_REFERER`.

Core commands:

```powershell
# 1. Generate AI DCF reports
python openrouter.py --file tickers.txt --tier cheap --parallel 20

# 2. Pull current EDGAR feature data
python edgar.py --file tickers.txt --output data/edgar_features_full.json

# 3. Build current public S&P 500 sector and selected-change snapshots from a local Wikipedia HTML snapshot
curl.exe -L https://en.wikipedia.org/wiki/List_of_S%26P_500_companies -o sp500_constituents.html
python sector_map.py --source-html sp500_constituents.html
python sp500_changes.py --source-html sp500_constituents.html

# 4. Build the security master / universe snapshot
python security_master.py --tickers-file tickers.txt --sector-map data/sector_map_wikipedia.csv --membership-changes data/sp500_changes_wikipedia.csv

# 5. Build cached 10-Q quarterly fundamentals
python quarterly_fundamentals.py

# 6. Distill the AI cross-section into a surrogate
python distill.py --tickers-file tickers.txt --edgar-file data/edgar_features_full.json --reports-dir reports

# 7. Build the v2 AI label decomposition panel
python label_panel.py --reports-dir reports --edgar-file data/edgar_features_full.json --sector-map data/sector_map_wikipedia.csv

# 8. Build v2 forward-return labels from cached prices only
python forward_returns.py --offline

# 9. Run v2 diagnostics against the forward-return artifact
python v2_experiment.py --forward-returns results/forward_returns.parquet

# 10. Write the separate Experiment C text-feature schema/blockers
python experiment_c_text_features.py

# Optional bounded SEC text download for Experiment C requests; requires SEC contact env vars
python experiment_c_text_features.py --download-corpus --download-limit 25

# Optional offline LLM response ingestion for Experiment C after filing text has been downloaded
python experiment_c_text_features.py --ingest-llm-responses --llm-responses-jsonl path/to/responses.jsonl

# 11. Fit current-label factor-portability maps for Experiment B
python experiment_b_factor_portability.py

# 12. Run a historical-compatible Experiment B backcast screen
python experiment_b_historical_backcast.py

# 13. Run stability diagnostics / Elastic Net baseline artifacts
python stability.py --tickers-file tickers.txt --edgar-file data/edgar_features_full.json --reports-dir reports

# 14. Backtest the frozen surrogate
python backtest.py --start-year 2012 --end-year 2025 --tickers-file tickers.txt
```

Auxiliary usage:

```powershell
# Re-run the DCF engine on a normalized JSON payload or saved report
python dcf.py reports\AAPL_2026-03-20_cheap.json --output-dir valuations

# Run tests
python -m pytest
```

`parser.py` and `historical.py` are library modules used by the scripts above rather than standalone CLIs.

## Cost

The preserved cheap-tier batch logs in `tmp/` sum to about `$9.66`, so the all-in Nano spend is best thought of as roughly low double digits once exploratory runs are included. The saved full-tier AAPL comparison report in [`reports/AAPL_2026-03-20_full.json`](reports/AAPL_2026-03-20_full.json) records `$1.20` in report metadata. The earlier rough budget in [docs/lessons_learned.md](docs/lessons_learned.md) was about `$0.87` for a full report; realized cost varied with prompt length, output length, and reasoning-token usage.
