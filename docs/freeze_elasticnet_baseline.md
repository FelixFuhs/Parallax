# Parallax Elastic Net Baseline Freeze

Generated: 2026-05-19T15:41:44Z

## Status

This is a baseline freeze document, not the primary Parallax model freeze. The primary frozen model remains the XGBoost regressor documented in `docs/model_freeze.md` and `docs/freeze_xgb_regressor.md`.

## Baseline Artifacts

- Baseline coefficients: `models/frozen_elasticnet_coefficients.json`
- Baseline metadata: `models/frozen_elasticnet_metadata.json`
- Baseline training artifact: `models/distill_elasticnet_v2.pkl`
- Primary model reference: `models/frozen_xgb_regressor.json`

## Model

- Estimator: `ElasticNetCV(random_state=42, max_iter=100000)`
- Role: transparency baseline for the five-feature legacy distillation workflow
- Target: fractional percentile rank of AI DCF base-case upside
- Training set size: 264 matched tickers
- Feature order: `fcf_to_ev`, `gross_profitability_assets`, `asset_growth_1y`, `cash_earnings_gap`, `momentum_12_1`

The baseline pipeline median-imputes missing features, standardizes features, and fits Elastic Net on the percentile-ranked current AI DCF upside target. Its coefficients are useful for inspection, but this model family is not the declared primary frozen model.

## Current Metrics

| Metric | Value |
| --- | ---: |
| CV Spearman | 0.3692 |
| CV R2 | -0.2561 |
| CV MAE | 0.2556 |
| Repeated-CV Spearman mean | 0.4170 |
| Repeated-CV Spearman std | 0.2372 |
| Bootstrap OOB Spearman mean | 0.4957 |
| Bootstrap OOB Spearman upper CI | 0.6894 |

## Guardrail

`tests/test_freeze_artifacts.py` requires the default historical backtest model path to remain `models/frozen_xgb_regressor.json`, requires the primary metadata to identify an XGBoost regressor, and requires the Elastic Net metadata to identify itself as a non-primary baseline. This prevents future stability runs from silently replacing the primary freeze statement with Elastic Net artifacts.
