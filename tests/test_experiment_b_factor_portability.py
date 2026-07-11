import json

import pandas as pd

from experiment_b_factor_portability import (
    aggregate_label_targets,
    build_training_frame,
    fit_target,
    run_experiment_b,
)


def _label_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["A", "A", "B", "C", "D", "E"],
            "company_name": ["A Inc.", "A Inc.", "B Inc.", "C Inc.", "D Inc.", "E Inc."],
            "sector": ["Tech", "Tech", "Tech", "Energy", "Energy", "Health Care"],
            "market_cap": [100.0, 100.0, 110.0, 90.0, 80.0, 70.0],
            "raw_ai_implied_irr": [0.10, 0.12, 0.20, 0.30, 0.40, 0.50],
            "ai_minus_mechanical_irr": [0.01, 0.02, 0.03, 0.01, 0.02, 0.03],
            "ai_factor_residual": [0.00, 0.01, -0.01, 0.02, -0.02, 0.03],
            "label_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "exclude_from_clean_label": [False, False, False, False, False, False],
        }
    )


def _edgar_payload() -> dict[str, dict[str, float]]:
    return {
        ticker: {
            "fcf_to_ev": index / 10,
            "gross_profitability_assets": index / 20,
            "momentum_12_1": index / 30,
            "asset_growth_1y": (6 - index) / 20,
            "cash_earnings_gap": index / 40,
            "accruals": -index / 50,
            "debt_to_equity": 6 - index,
            "current_ratio": 1 + index / 10,
            "market_cap": 100.0 * index,
        }
        for index, ticker in enumerate(["A", "B", "C", "D", "E"], start=1)
    }


def test_aggregate_label_targets_uses_weighted_ticker_level_labels():
    labels = aggregate_label_targets(_label_panel())
    a_row = labels[labels["ticker"] == "A"].iloc[0]

    assert a_row["raw_ai_implied_irr"] == 0.11
    assert a_row["label_observation_count"] == 2
    assert a_row["clean_label_rate"] == 1.0


def test_fit_target_outputs_predictions_and_coefficients():
    frame = build_training_frame(_label_panel(), _edgar_payload())
    predictions, coefficients, summary = fit_target(
        frame,
        "raw_ai_implied_irr",
        ["fcf_to_ev", "gross_profitability_assets", "log_market_cap"],
    )

    assert summary["status"] == "fit_current_cross_section_only"
    assert not predictions.empty
    assert not coefficients.empty
    assert "cv_predicted_label" in predictions.columns


def test_fit_target_blocks_degenerate_factor_map():
    panel = _label_panel()
    panel["raw_ai_implied_irr"] = 0.1
    frame = build_training_frame(panel, _edgar_payload())

    predictions, coefficients, summary = fit_target(
        frame,
        "raw_ai_implied_irr",
        ["fcf_to_ev", "gross_profitability_assets", "log_market_cap"],
    )

    assert summary["status"] == "blocked_degenerate_factor_map"
    assert summary["blockers"][0]["code"] == "degenerate_factor_map"
    assert predictions.empty
    assert coefficients.empty


def test_run_experiment_b_writes_claim_safe_artifacts(tmp_path):
    label_path = tmp_path / "labels.parquet"
    edgar_path = tmp_path / "edgar.json"
    output = tmp_path / "predictions.parquet"
    coefficients = tmp_path / "coefficients.csv"
    summary_path = tmp_path / "summary.json"
    metadata_path = tmp_path / "metadata.json"
    _label_panel().to_parquet(label_path, index=False)
    edgar_path.write_text(json.dumps(_edgar_payload()), encoding="utf-8")

    summary = run_experiment_b(
        label_panel_path=label_path,
        edgar_features_path=edgar_path,
        output_path=output,
        coefficients_path=coefficients,
        summary_path=summary_path,
        metadata_path=metadata_path,
    )

    assert summary["experiment_id"] == "experiment_b_ai_implied_factor_portability"
    assert summary["status"] == "fit_current_cross_section_only"
    assert output.exists()
    assert coefficients.exists()
    assert metadata_path.exists()
    assert "historical_backcast_not_run" in {warning["code"] for warning in summary["warnings"]}
