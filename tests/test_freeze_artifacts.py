import json
from pathlib import Path

from backtest import DEFAULT_MODEL_PATH, FROZEN_MODEL_PATH
from distill import ModelMetrics, determine_verdict

ROOT = Path(__file__).resolve().parents[1]


def test_backtest_model_matches_declared_freeze_doc():
    metadata = json.loads((FROZEN_MODEL_PATH.parent / "frozen_model_metadata.json").read_text(encoding="utf-8"))

    assert DEFAULT_MODEL_PATH == FROZEN_MODEL_PATH
    assert metadata["model_type"].lower() in {"xgbregressor", "xgboost_regressor"}
    assert metadata["artifact_role"] == "primary_model_freeze"
    assert FROZEN_MODEL_PATH.name == "frozen_xgb_regressor.json"
    assert metadata["source_artifacts"]["frozen_model"] == "models/frozen_xgb_regressor.json"


def test_elasticnet_freeze_artifacts_are_baseline_only():
    metadata_path = ROOT / "models" / "frozen_elasticnet_metadata.json"
    doc_path = ROOT / "docs" / "freeze_elasticnet_baseline.md"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert doc_path.exists()
    assert metadata["artifact_role"] == "baseline_model_freeze"
    assert metadata["primary_model"] is False
    assert metadata["primary_model_reference"] == "models/frozen_xgb_regressor.json"
    assert metadata["model_family"] == "ElasticNetCV"


def test_distill_verdict_names_primary_xgb_regressor_not_ranker():
    elasticnet_metrics = ModelMetrics(spearman=0.20, r2=0.05, mae=0.20)
    xgb_metrics = ModelMetrics(spearman=0.24, r2=-0.10, mae=0.30)

    assert determine_verdict(elasticnet_metrics, xgb_metrics) == "Verdict: XGBoost Regressor wins."
