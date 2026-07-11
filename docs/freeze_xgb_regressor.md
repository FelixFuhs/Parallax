# XGBoost Regressor Freeze

## Status

The primary frozen model family is the XGBoost regressor:

- Model: `models/frozen_xgb_regressor.json`
- Metadata: `models/frozen_model_metadata.json`
- Feature spec: `models/feature_spec.json`

Elastic Net artifacts are retained only as a transparent baseline. Running `stability.py` now writes `docs/freeze_elasticnet_baseline.md` and `models/frozen_elasticnet_metadata.json` instead of overwriting the primary freeze statement.

## Guardrail

`tests/test_freeze_artifacts.py` verifies that the historical backtest default model path points at `models/frozen_xgb_regressor.json` and that the frozen metadata identifies an XGBoost regressor family. This is intended to catch accidental drift back to Elastic Net-centered freeze artifacts.

## Remaining Caveat

The frozen model is still a prototype research artifact trained on a single modern AI-label cross-section. It is not evidence of tradable alpha by itself.
