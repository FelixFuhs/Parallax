from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks import FEATURE_BLOCKS
from experiment_registry import build_experiment_metadata, repo_relative, write_experiment_metadata
from label_panel import DEFAULT_EDGAR_PATH, DEFAULT_OUTPUT_PATH, load_edgar_payload

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "experiment_b_factor_portability.parquet"
DEFAULT_COEFFICIENTS = RESULTS_DIR / "experiment_b_factor_portability_coefficients.csv"
DEFAULT_SUMMARY = RESULTS_DIR / "experiment_b_factor_portability_summary.json"
DEFAULT_METADATA = RESULTS_DIR / "experiment_b_factor_portability_metadata.json"
DEFAULT_TARGETS = (
    "raw_ai_implied_irr",
    "ai_minus_mechanical_irr",
    "ai_factor_residual",
)


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _safe_spearman(actual: pd.Series, predicted: pd.Series) -> float | None:
    aligned = pd.concat({"actual": actual, "predicted": predicted}, axis=1).dropna()
    if len(aligned) < 2 or aligned["actual"].nunique() < 2 or aligned["predicted"].nunique() < 2:
        return None
    statistic = spearmanr(aligned["actual"], aligned["predicted"]).statistic
    if statistic is None or not np.isfinite(statistic):
        return None
    return float(statistic)


def _percentile_rank(values: pd.Series) -> pd.Series:
    if len(values) == 1:
        return pd.Series(0.5, index=values.index, dtype=float)
    return (values.rank(method="average") - 1.0) / float(len(values) - 1)


def aggregate_label_targets(label_panel: pd.DataFrame, targets: Sequence[str] = DEFAULT_TARGETS) -> pd.DataFrame:
    if label_panel.empty:
        return pd.DataFrame()

    work = label_panel.copy()
    if "label_weight" not in work.columns:
        work["label_weight"] = 1.0
    work["label_weight"] = pd.to_numeric(work["label_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)

    rows: list[dict[str, Any]] = []
    for ticker, group in work.groupby("ticker", dropna=False):
        row: dict[str, Any] = {
            "ticker": ticker,
            "company_name": group["company_name"].dropna().iloc[0] if "company_name" in group and group["company_name"].notna().any() else None,
            "sector": group["sector"].dropna().iloc[0] if "sector" in group and group["sector"].notna().any() else None,
            "market_cap": pd.to_numeric(group.get("market_cap"), errors="coerce").dropna().iloc[0]
            if "market_cap" in group and pd.to_numeric(group["market_cap"], errors="coerce").notna().any()
            else None,
            "label_observation_count": int(len(group)),
            "clean_label_rate": float((~group["exclude_from_clean_label"]).mean())
            if "exclude_from_clean_label" in group
            else 1.0,
            "mean_label_weight": float(group["label_weight"].mean()),
        }
        weights = group["label_weight"]
        for target in targets:
            values = pd.to_numeric(group[target], errors="coerce") if target in group else pd.Series(dtype=float)
            valid = values.notna() & (weights > 0.0)
            if valid.any():
                row[target] = float(np.average(values.loc[valid], weights=weights.loc[valid]))
                row[f"{target}_coverage"] = float(valid.mean())
            else:
                row[target] = None
                row[f"{target}_coverage"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def feature_frame_from_edgar(edgar_payload: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    block_features = list(dict.fromkeys(feature for block in FEATURE_BLOCKS for feature in block.features))
    rows: list[dict[str, Any]] = []
    for ticker, record in edgar_payload.items():
        row: dict[str, Any] = {"ticker": ticker}
        for feature in block_features:
            if feature in record:
                row[feature] = record[feature]
        if "market_cap" in record:
            row["market_cap_feature"] = record["market_cap"]
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("ticker").set_index("ticker")


def build_training_frame(
    label_panel: pd.DataFrame,
    edgar_payload: Mapping[str, Mapping[str, Any]],
    *,
    targets: Sequence[str] = DEFAULT_TARGETS,
) -> pd.DataFrame:
    labels = aggregate_label_targets(label_panel, targets=targets)
    features = feature_frame_from_edgar(edgar_payload)
    if labels.empty:
        return labels
    frame = labels.merge(features, how="left", left_on="ticker", right_index=True, suffixes=("", "_feature"))
    market_cap = pd.to_numeric(frame["market_cap"], errors="coerce") if "market_cap" in frame else pd.Series(index=frame.index, dtype=float)
    if "market_cap_feature" in frame:
        market_cap = market_cap.combine_first(pd.to_numeric(frame["market_cap_feature"], errors="coerce"))
    frame["log_market_cap"] = np.where(market_cap > 0.0, np.log(market_cap), np.nan)
    return frame


def available_feature_columns(frame: pd.DataFrame) -> list[str]:
    candidate_columns = list(dict.fromkeys(feature for block in FEATURE_BLOCKS for feature in block.features))
    candidate_columns.append("log_market_cap")
    return [
        column
        for column in candidate_columns
        if column in frame.columns and pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]


def _design_matrix(frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    numeric = frame.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    if "sector" in frame.columns and frame["sector"].nunique(dropna=True) > 1:
        dummies = pd.get_dummies(frame["sector"].fillna("Unknown"), prefix="sector", drop_first=True, dtype=float)
        numeric = pd.concat([numeric, dummies], axis=1)
    return numeric


def _build_model(n_samples: int) -> Pipeline:
    cv = min(5, max(2, n_samples))
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNetCV(
                    cv=cv,
                    random_state=42,
                    max_iter=100000,
                    l1_ratio=[0.1, 0.5, 0.9],
                    alphas=np.logspace(-4, 1, 20),
                ),
            ),
        ]
    )


def _cross_validated_predictions(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    if len(x) < 3:
        return pd.Series(np.nan, index=x.index, dtype=float)
    splitter = KFold(n_splits=min(5, len(x)), shuffle=True, random_state=42)
    predictions = pd.Series(np.nan, index=x.index, dtype=float)
    for train_index, test_index in splitter.split(x):
        model = _build_model(len(train_index))
        model.fit(x.iloc[train_index], y.iloc[train_index])
        predictions.iloc[test_index] = model.predict(x.iloc[test_index])
    return predictions


def fit_target(frame: pd.DataFrame, target: str, feature_columns: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    target_values = pd.to_numeric(frame[target], errors="coerce") if target in frame else pd.Series(dtype=float)
    fit_mask = target_values.notna() & (pd.to_numeric(frame["mean_label_weight"], errors="coerce").fillna(0.0) > 0.0)
    fit_frame = frame.loc[fit_mask].copy()
    if len(fit_frame) < 3 or not feature_columns:
        summary = {
            "target": target,
            "status": "blocked",
            "blockers": [{"code": "insufficient_cross_section", "message": "Fewer than three usable labels or no features."}],
            "n": int(len(fit_frame)),
            "feature_count": int(len(feature_columns)),
        }
        return pd.DataFrame(), pd.DataFrame(), summary

    x = _design_matrix(fit_frame, feature_columns)
    y = pd.to_numeric(fit_frame[target], errors="coerce").astype(float)
    cv_predictions = _cross_validated_predictions(x, y)
    final_model = _build_model(len(x))
    final_model.fit(x, y)
    fit_predictions = pd.Series(final_model.predict(x), index=x.index, dtype=float)
    target_unique_count = int(y.nunique(dropna=True))
    prediction_unique_count = int(fit_predictions.nunique(dropna=True))
    if target_unique_count < 2 or prediction_unique_count < 2:
        summary = {
            "target": target,
            "status": "blocked_degenerate_factor_map",
            "blockers": [
                {
                    "code": "degenerate_factor_map",
                    "message": "The fitted current cross-section map produced no usable score variation.",
                }
            ],
            "n": int(len(fit_frame)),
            "feature_count": int(len(feature_columns)),
            "design_matrix_column_count": int(x.shape[1]),
            "target_unique_count": target_unique_count,
            "fit_prediction_unique_count": prediction_unique_count,
        }
        return pd.DataFrame(), pd.DataFrame(), summary

    output = fit_frame[
        [
            "ticker",
            "company_name",
            "sector",
            "label_observation_count",
            "clean_label_rate",
            "mean_label_weight",
            target,
        ]
    ].copy()
    output = output.rename(columns={target: "observed_label"})
    output["target"] = target
    output["cv_predicted_label"] = cv_predictions
    output["fit_predicted_label"] = fit_predictions
    output["observed_percentile"] = _percentile_rank(output["observed_label"])
    output["fit_predicted_percentile"] = _percentile_rank(output["fit_predicted_label"])
    output["cv_predicted_percentile"] = _percentile_rank(output["cv_predicted_label"].dropna()).reindex(output.index)
    output["fit_residual"] = output["observed_label"] - output["fit_predicted_label"]
    output["feature_null_count"] = x.loc[:, list(feature_columns)].isna().sum(axis=1)

    coefficients = final_model.named_steps["model"].coef_
    coefficient_frame = (
        pd.DataFrame({"target": target, "feature": x.columns, "coefficient": coefficients})
        .assign(abs_coefficient=lambda data: data["coefficient"].abs())
        .sort_values(["target", "abs_coefficient"], ascending=[True, False], kind="mergesort")
        .drop(columns=["abs_coefficient"])
        .reset_index(drop=True)
    )
    clean_cv = output.dropna(subset=["cv_predicted_label"])
    summary = {
        "target": target,
        "status": "fit_current_cross_section_only",
        "n": int(len(output)),
        "feature_count": int(len(feature_columns)),
        "design_matrix_column_count": int(x.shape[1]),
        "fit_spearman": _safe_spearman(output["observed_label"], output["fit_predicted_label"]),
        "cv_spearman": _safe_spearman(clean_cv["observed_label"], clean_cv["cv_predicted_label"]),
        "fit_percentile_r2": float(r2_score(output["observed_percentile"], output["fit_predicted_percentile"])),
        "fit_mae": float(mean_absolute_error(output["observed_label"], output["fit_predicted_label"])),
        "top_coefficients": coefficient_frame.head(10).to_dict(orient="records"),
        "blockers": [],
        "warnings": [
            {
                "code": "current_cross_section_only",
                "message": "This fit maps current AI labels to public factors but has not been historically backcast.",
            }
        ],
    }
    return output, coefficient_frame, summary


def run_experiment_b(
    *,
    label_panel_path: Path = DEFAULT_OUTPUT_PATH,
    edgar_features_path: Path = DEFAULT_EDGAR_PATH,
    output_path: Path = DEFAULT_OUTPUT,
    coefficients_path: Path = DEFAULT_COEFFICIENTS,
    summary_path: Path = DEFAULT_SUMMARY,
    metadata_path: Path = DEFAULT_METADATA,
    targets: Sequence[str] = DEFAULT_TARGETS,
) -> dict[str, Any]:
    label_panel = _read_frame(label_panel_path)
    edgar_payload = load_edgar_payload(edgar_features_path) if edgar_features_path.exists() else {}
    frame = build_training_frame(label_panel, edgar_payload, targets=targets)
    feature_columns = available_feature_columns(frame)

    prediction_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    target_summaries: list[dict[str, Any]] = []
    for target in targets:
        predictions, coefficients, target_summary = fit_target(frame, target, feature_columns)
        if not predictions.empty:
            prediction_frames.append(predictions)
        if not coefficients.empty:
            coefficient_frames.append(coefficients)
        target_summaries.append(target_summary)

    output = pd.concat(prediction_frames, axis=0, ignore_index=True) if prediction_frames else pd.DataFrame()
    coefficients = pd.concat(coefficient_frames, axis=0, ignore_index=True) if coefficient_frames else pd.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    coefficients.to_csv(coefficients_path, index=False)

    blockers = []
    if not target_summaries or all(str(summary["status"]).startswith("blocked") for summary in target_summaries):
        blockers.append({"code": "no_fit_targets", "message": "No Experiment B target had enough data to fit."})
    warnings = [
        {
            "code": "historical_backcast_not_run",
            "message": "Experiment B has fitted current AI-label factor maps; historical portability remains unproven.",
        },
        {
            "code": "survivor_universe",
            "message": "The fitted cross-section still comes from the current survivor-biased public-data universe.",
        },
    ]
    degenerate_targets = [
        str(summary["target"])
        for summary in target_summaries
        if any(blocker.get("code") == "degenerate_factor_map" for blocker in summary.get("blockers", []))
    ]
    if degenerate_targets:
        warnings.append(
            {
                "code": "degenerate_factor_map",
                "message": "One or more current-label factor maps were skipped because their fitted scores had no ranking variation.",
                "targets": degenerate_targets,
            }
        )
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment_id": "experiment_b_ai_implied_factor_portability",
        "status": "blocked" if blockers else "fit_current_cross_section_only",
        "claim_ceiling": "current_label_factor_map_no_historical_alpha_evidence",
        "row_count": int(len(output)),
        "target_count": int(len(target_summaries)),
        "targets": target_summaries,
        "feature_columns": list(feature_columns),
        "blockers": blockers,
        "warnings": warnings,
        "artifacts": {
            "predictions": repo_relative(output_path),
            "coefficients": repo_relative(coefficients_path),
            "summary": repo_relative(summary_path),
            "metadata": repo_relative(metadata_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    metadata = build_experiment_metadata(
        experiment_id="experiment_b_ai_implied_factor_portability",
        feature_config={"feature_columns": list(feature_columns), "targets": list(targets)},
        model_config={"model": "ElasticNetCV", "cv": "5-fold max, current cross-section only"},
        universe_config={"label_panel": repo_relative(label_panel_path), "survivor_bias_caveat": True},
        backtest_config={"historical_backcast": "not_run"},
        data_snapshot_paths={"edgar_features": edgar_features_path},
        label_snapshot_paths={"label_panel": label_panel_path},
        artifacts={"predictions": output_path, "coefficients": coefficients_path, "summary": summary_path},
    )
    write_experiment_metadata(metadata_path, metadata)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Experiment B current AI-label factor portability maps.")
    parser.add_argument("--label-panel", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--edgar-file", default=str(DEFAULT_EDGAR_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--coefficients-output", default=str(DEFAULT_COEFFICIENTS))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--metadata-output", default=str(DEFAULT_METADATA))
    parser.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_experiment_b(
        label_panel_path=Path(args.label_panel),
        edgar_features_path=Path(args.edgar_file),
        output_path=Path(args.output),
        coefficients_path=Path(args.coefficients_output),
        summary_path=Path(args.summary_output),
        metadata_path=Path(args.metadata_output),
        targets=tuple(args.targets),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
