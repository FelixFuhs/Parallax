from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker


ROOT = Path(__file__).resolve().parent
TICKERS_PATH = ROOT / "tickers_100.txt"
EDGAR_PATH = ROOT / "edgar_features_100_v3.json"
REPORTS_DIR = ROOT / "reports"
PLOTS_DIR = ROOT / "plots"
MODELS_DIR = ROOT / "models"
XGB_MODEL_PATH = MODELS_DIR / "distill_xgb_v2.json"
ELASTICNET_MODEL_PATH = MODELS_DIR / "distill_elasticnet_v2.pkl"
METADATA_PATH = MODELS_DIR / "distill_v2_metadata.json"
FEATURE_SPEC_PATH = MODELS_DIR / "feature_spec.json"
FEATURE_IMPORTANCE_PATH = PLOTS_DIR / "feature_importance_v2.png"
PREDICTED_VS_ACTUAL_PATH = PLOTS_DIR / "predicted_vs_actual_v2.png"
ELASTICNET_COEFFICIENTS_PATH = PLOTS_DIR / "elasticnet_coefficients.png"
BROKEN_EDGAR_TICKERS = {"MCD"}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    description: str
    monotonic_constraint: int
    economic_rationale: str


@dataclass(frozen=True)
class MatchSummary:
    matched_tickers: list[str]
    edgar_only: dict[str, str]
    nano_only: dict[str, str]
    edgar_coverage_count: int
    nano_coverage_count: int


@dataclass(frozen=True)
class ModelMetrics:
    spearman: float
    r2: float
    mae: float


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="fcf_to_ev",
        description="(operating cash flow - capex) / (market cap + total debt - cash), null when EV <= 0",
        monotonic_constraint=1,
        economic_rationale="Higher free cash flow relative to enterprise value indicates cheaper cash generation.",
    ),
    FeatureSpec(
        name="gross_profitability_assets",
        description="gross_profit / total_assets",
        monotonic_constraint=1,
        economic_rationale="Higher gross profits on the asset base are associated with better business quality.",
    ),
    FeatureSpec(
        name="asset_growth_1y",
        description="(current total assets - prior-year total assets) / prior-year total assets",
        monotonic_constraint=-1,
        economic_rationale="Aggressive balance-sheet expansion is often linked to weaker subsequent returns.",
    ),
    FeatureSpec(
        name="cash_earnings_gap",
        description="(operating cash flow - net income) / total_assets",
        monotonic_constraint=1,
        economic_rationale="Cash earnings above accounting earnings can signal stronger earnings quality.",
    ),
    FeatureSpec(
        name="momentum_12_1",
        description="(1 + price_return_12m) / (1 + price_return_1m) - 1",
        monotonic_constraint=0,
        economic_rationale="Intermediate-term momentum can capture trend persistence without the most recent month.",
    ),
)
CONSENSUS_FEATURES = tuple(spec.name for spec in FEATURE_SPECS)
MONOTONIC_CONSTRAINTS = tuple(spec.monotonic_constraint for spec in FEATURE_SPECS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Parallax distillation models on the fixed 5-feature consensus spec."
    )
    parser.add_argument("--tickers-file", default=str(TICKERS_PATH))
    parser.add_argument("--edgar-file", default=str(EDGAR_PATH))
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR))
    return parser.parse_args(argv)


def load_ticker_universe(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def load_edgar_payload(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a ticker-keyed JSON object.")
    return payload


def latest_successful_cheap_reports(reports_dir: Path, universe: set[str]) -> dict[str, tuple[str, Path]]:
    report_map: dict[str, tuple[str, Path]] = {}
    for path in reports_dir.glob("*_cheap.json"):
        stem_parts = path.stem.split("_")
        if len(stem_parts) < 3:
            continue
        ticker = stem_parts[0]
        report_date = stem_parts[1]
        if ticker not in universe:
            continue
        current = report_map.get(ticker)
        if current is None or report_date > current[0]:
            report_map[ticker] = (report_date, path)
    return report_map


def build_match_summary(
    universe: list[str],
    edgar_payload: dict[str, dict[str, Any]],
    report_map: dict[str, tuple[str, Path]],
) -> MatchSummary:
    clean_edgar: set[str] = set()
    edgar_exclusions: dict[str, str] = {}
    for ticker in universe:
        record = edgar_payload.get(ticker)
        if record is None:
            edgar_exclusions[ticker] = "missing from EDGAR payload"
            continue
        if record.get("error"):
            edgar_exclusions[ticker] = f"edgar error: {record['error']}"
            continue
        fiscal_year = record.get("fiscal_year")
        if fiscal_year is None or fiscal_year < 2024:
            edgar_exclusions[ticker] = f"stale edgar fiscal_year={fiscal_year}"
            continue
        if ticker in BROKEN_EDGAR_TICKERS:
            edgar_exclusions[ticker] = "broken shares_outstanding / derived features"
            continue
        clean_edgar.add(ticker)

    clean_nano: set[str] = set()
    nano_exclusions: dict[str, str] = {}
    for ticker in universe:
        report_ref = report_map.get(ticker)
        if report_ref is None:
            nano_exclusions[ticker] = "no successful cheap report found"
            continue
        _, path = report_ref
        payload = json.loads(path.read_text(encoding="utf-8"))
        quality_flags = payload.get("_meta", {}).get("quality_flags", [])
        if "stale_price" in quality_flags:
            nano_exclusions[ticker] = "stale_price quality flag"
            continue
        upside = payload.get("_valuation", {}).get("scenarios", {}).get("base", {}).get("upside_downside_pct")
        if upside is None:
            nano_exclusions[ticker] = "missing base-case upside"
            continue
        clean_nano.add(ticker)

    matched_tickers = sorted(clean_edgar & clean_nano)
    edgar_only = {ticker: nano_exclusions[ticker] for ticker in sorted(clean_edgar - clean_nano)}
    nano_only = {ticker: edgar_exclusions[ticker] for ticker in sorted(clean_nano - clean_edgar)}
    return MatchSummary(
        matched_tickers=matched_tickers,
        edgar_only=edgar_only,
        nano_only=nano_only,
        edgar_coverage_count=len(clean_edgar),
        nano_coverage_count=len(clean_nano),
    )


def feature_coverage(edgar_payload: dict[str, dict[str, Any]], matched_tickers: list[str]) -> dict[str, dict[str, float]]:
    denominator = float(len(matched_tickers))
    coverage: dict[str, dict[str, float]] = {}
    for feature_name in CONSENSUS_FEATURES:
        non_null = sum(edgar_payload[ticker].get(feature_name) is not None for ticker in matched_tickers)
        coverage[feature_name] = {
            "non_null": float(non_null),
            "coverage": (non_null / denominator) if denominator else 0.0,
        }
    return coverage


def build_training_frame(
    matched_tickers: list[str],
    edgar_payload: dict[str, dict[str, Any]],
    report_map: dict[str, tuple[str, Path]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker in matched_tickers:
        _, report_path = report_map[ticker]
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        base_case = report_payload["_valuation"]["scenarios"]["base"]
        row: dict[str, Any] = {
            "ticker": ticker,
            "company_name": edgar_payload[ticker].get("company_name"),
            "actual_upside": float(base_case["upside_downside_pct"]),
        }
        for feature_name in CONSENSUS_FEATURES:
            row[feature_name] = edgar_payload[ticker].get(feature_name)
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker")


def fractional_percentile_rank(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 1:
        return np.asarray([0.5], dtype=float)
    ranks = pd.Series(array).rank(method="average")
    return ((ranks - 1.0) / float(array.size - 1)).to_numpy(dtype=float)


def decile_labels_from_percentiles(percentiles: Sequence[float]) -> np.ndarray:
    percentiles_array = np.asarray(percentiles, dtype=float)
    return np.clip(np.floor(percentiles_array * 10.0), 0, 9).astype(int)


def safe_spearman(actual: Sequence[float], predicted: Sequence[float]) -> float:
    statistic = spearmanr(actual, predicted).statistic
    if statistic is None or np.isnan(statistic):
        return 0.0
    return float(statistic)


def evaluate_predictions(
    actual_upside: np.ndarray,
    actual_percentile: np.ndarray,
    predicted_scores: np.ndarray,
    predicted_percentile: np.ndarray,
) -> ModelMetrics:
    return ModelMetrics(
        spearman=safe_spearman(actual_upside, predicted_scores),
        r2=float(r2_score(actual_percentile, predicted_percentile)),
        mae=float(mean_absolute_error(actual_percentile, predicted_percentile)),
    )


def build_elasticnet_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNetCV(random_state=42, max_iter=100000)),
        ]
    )


def run_elasticnet_repeated_cv(
    features: pd.DataFrame,
    target_percentile: np.ndarray,
    actual_upside: np.ndarray,
    *,
    n_splits: int = 5,
    n_repeats: int = 10,
) -> tuple[np.ndarray, ModelMetrics, list[ModelMetrics]]:
    splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    split_iterator = iter(splitter.split(features))
    repeat_predicted_percentiles: list[np.ndarray] = []
    repeat_metrics: list[ModelMetrics] = []

    for _ in range(n_repeats):
        repeat_raw_predictions = np.empty(len(features), dtype=float)
        for _ in range(n_splits):
            train_index, test_index = next(split_iterator)
            pipeline = build_elasticnet_pipeline()
            pipeline.fit(features.iloc[train_index], target_percentile[train_index])
            repeat_raw_predictions[test_index] = pipeline.predict(features.iloc[test_index])

        repeat_percentile_predictions = fractional_percentile_rank(repeat_raw_predictions)
        repeat_predicted_percentiles.append(repeat_percentile_predictions)
        repeat_metrics.append(
            evaluate_predictions(
                actual_upside=actual_upside,
                actual_percentile=target_percentile,
                predicted_scores=repeat_raw_predictions,
                predicted_percentile=repeat_percentile_predictions,
            )
        )

    mean_predicted_percentile = np.mean(np.vstack(repeat_predicted_percentiles), axis=0)
    mean_metrics = ModelMetrics(
        spearman=float(np.mean([metric.spearman for metric in repeat_metrics])),
        r2=float(np.mean([metric.r2 for metric in repeat_metrics])),
        mae=float(np.mean([metric.mae for metric in repeat_metrics])),
    )
    return mean_predicted_percentile, mean_metrics, repeat_metrics


def build_xgb_ranker() -> XGBRanker:
    return XGBRanker(
        objective="rank:ndcg",
        max_depth=2,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        min_child_weight=5,
        monotone_constraints=MONOTONIC_CONSTRAINTS,
        lambdarank_pair_method="mean",
        random_state=42,
        n_jobs=1,
    )


def build_query_fold_assignments(features: pd.DataFrame, *, n_splits: int = 5) -> np.ndarray:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_assignments = np.empty(len(features), dtype=int)
    for fold_index, (_, test_index) in enumerate(splitter.split(features)):
        fold_assignments[test_index] = fold_index
    return fold_assignments


def prepare_ranker_training_data(
    features: pd.DataFrame,
    target: np.ndarray,
    qid: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    order = np.argsort(qid, kind="stable")
    return features.iloc[order], target[order], qid[order]


def run_xgb_grouped_cv(
    features: pd.DataFrame,
    target_decile: np.ndarray,
    actual_upside: np.ndarray,
    actual_percentile: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, ModelMetrics, list[int]]:
    fold_assignments = build_query_fold_assignments(features)
    raw_predictions = np.empty(len(features), dtype=float)

    for fold_index in range(5):
        train_index = np.flatnonzero(fold_assignments != fold_index)
        test_index = np.flatnonzero(fold_assignments == fold_index)
        x_train, y_train, qid_train = prepare_ranker_training_data(
            features.iloc[train_index],
            target_decile[train_index],
            fold_assignments[train_index],
        )
        model = build_xgb_ranker()
        model.fit(x_train, y_train, qid=qid_train)
        raw_predictions[test_index] = model.predict(features.iloc[test_index])

    predicted_percentile = fractional_percentile_rank(raw_predictions)
    metrics = evaluate_predictions(
        actual_upside=actual_upside,
        actual_percentile=actual_percentile,
        predicted_scores=raw_predictions,
        predicted_percentile=predicted_percentile,
    )
    return raw_predictions, predicted_percentile, metrics, fold_assignments.tolist()


def run_xgb_loo_cv(
    features: pd.DataFrame,
    target_decile: np.ndarray,
    actual_upside: np.ndarray,
    actual_percentile: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, ModelMetrics]:
    splitter = LeaveOneOut()
    fold_assignments = build_query_fold_assignments(features)
    raw_predictions = np.empty(len(features), dtype=float)

    for train_index, test_index in splitter.split(features):
        x_train, y_train, qid_train = prepare_ranker_training_data(
            features.iloc[train_index],
            target_decile[train_index],
            fold_assignments[train_index],
        )
        model = build_xgb_ranker()
        model.fit(x_train, y_train, qid=qid_train)
        raw_predictions[test_index[0]] = float(model.predict(features.iloc[test_index])[0])

    predicted_percentile = fractional_percentile_rank(raw_predictions)
    metrics = evaluate_predictions(
        actual_upside=actual_upside,
        actual_percentile=actual_percentile,
        predicted_scores=raw_predictions,
        predicted_percentile=predicted_percentile,
    )
    return raw_predictions, predicted_percentile, metrics


def fit_final_elasticnet(features: pd.DataFrame, target_percentile: np.ndarray) -> Pipeline:
    pipeline = build_elasticnet_pipeline()
    pipeline.fit(features, target_percentile)
    return pipeline


def fit_final_xgb(features: pd.DataFrame, target_decile: np.ndarray, fold_assignments: np.ndarray) -> XGBRanker:
    x_train, y_train, qid_train = prepare_ranker_training_data(features, target_decile, fold_assignments)
    model = build_xgb_ranker()
    model.fit(x_train, y_train, qid=qid_train)
    return model


def elasticnet_coefficients(pipeline: Pipeline) -> pd.Series:
    model = pipeline.named_steps["model"]
    return pd.Series(model.coef_, index=CONSENSUS_FEATURES, dtype=float)


def save_feature_importance_plot(model: XGBRanker) -> pd.Series:
    importance = pd.Series(model.feature_importances_, index=CONSENSUS_FEATURES, dtype=float).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    importance.plot(kind="barh", ax=ax, color="#2f6db3")
    ax.set_title("XGBoost Ranker Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_PATH, dpi=160)
    plt.close(fig)
    return importance.sort_values(ascending=False)


def save_elasticnet_coefficients_plot(coefficients: pd.Series) -> None:
    ordered = coefficients.sort_values()
    colors = ["#b22222" if value < 0 else "#2f6db3" for value in ordered]
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Elastic Net Coefficients (Standardized Features)")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(ELASTICNET_COEFFICIENTS_PATH, dpi=160)
    plt.close(fig)


def save_predicted_vs_actual_plot(
    actual_percentile: np.ndarray,
    elasticnet_percentile: np.ndarray,
    xgb_percentile: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(
        actual_percentile,
        elasticnet_percentile,
        alpha=0.8,
        color="#2f6db3",
        edgecolor="white",
        linewidth=0.6,
        label="Elastic Net CV",
    )
    ax.scatter(
        actual_percentile,
        xgb_percentile,
        alpha=0.8,
        color="#d97a00",
        edgecolor="white",
        linewidth=0.6,
        label="XGBoost Ranker CV",
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#666666", linewidth=1.0)
    ax.set_title("Predicted vs Actual Upside Percentile")
    ax.set_xlabel("Actual upside percentile")
    ax.set_ylabel("Predicted upside percentile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PREDICTED_VS_ACTUAL_PATH, dpi=160)
    plt.close(fig)


def print_comparison_table(elasticnet_metrics: ModelMetrics, xgb_metrics: ModelMetrics) -> None:
    print("Head-to-head comparison (Elastic Net values are 10x repeated 5-fold means; XGBoost values are 5-fold OOF):")
    print(f"{'Metric':<18}{'Elastic Net':>14}{'XGBoost Ranker':>18}")
    print(f"{'Spearman (CV)':<18}{elasticnet_metrics.spearman:>14.4f}{xgb_metrics.spearman:>18.4f}")
    print(f"{'R^2 (CV)':<18}{elasticnet_metrics.r2:>14.4f}{xgb_metrics.r2:>18.4f}")
    print(f"{'MAE (CV)':<18}{elasticnet_metrics.mae:>14.4f}{xgb_metrics.mae:>18.4f}")


def determine_verdict(elasticnet_metrics: ModelMetrics, xgb_metrics: ModelMetrics) -> str:
    spearman_gap = xgb_metrics.spearman - elasticnet_metrics.spearman
    if spearman_gap > 0.01:
        return "Verdict: XGBoost Ranker wins."
    if spearman_gap < -0.01:
        return "Verdict: Elastic Net wins."

    elasticnet_secondary = 0
    xgb_secondary = 0
    if elasticnet_metrics.r2 > xgb_metrics.r2 + 0.02:
        elasticnet_secondary += 1
    elif xgb_metrics.r2 > elasticnet_metrics.r2 + 0.02:
        xgb_secondary += 1
    if elasticnet_metrics.mae + 0.02 < xgb_metrics.mae:
        elasticnet_secondary += 1
    elif xgb_metrics.mae + 0.02 < elasticnet_metrics.mae:
        xgb_secondary += 1

    if xgb_secondary > elasticnet_secondary:
        return "Verdict: XGBoost Ranker wins."
    if elasticnet_secondary > xgb_secondary:
        return "Verdict: Elastic Net wins."
    return "Verdict: Models are tied."


def summarize_coefficient_strength(coefficients: pd.Series) -> list[str]:
    max_abs = float(coefficients.abs().max())
    if max_abs == 0.0:
        max_abs = 1.0

    notes: list[str] = []
    for feature_name, coefficient in coefficients.items():
        if np.isclose(coefficient, 0.0):
            notes.append(f"{feature_name}: effectively unused by the Elastic Net.")
            continue
        relative_strength = abs(coefficient) / max_abs
        if relative_strength >= 0.67:
            magnitude = "strong"
        elif relative_strength >= 0.33:
            magnitude = "moderate"
        else:
            magnitude = "light"
        direction = "higher predicted upside percentile" if coefficient > 0 else "lower predicted upside percentile"
        notes.append(f"{feature_name}: {magnitude} {direction} effect (coef={coefficient:.4f}).")
    return notes


def used_and_ignored_features(importance: pd.Series) -> tuple[list[str], list[str]]:
    used = importance[importance > 0.0].index.tolist()
    ignored = importance[importance <= 0.0].index.tolist()
    return used, ignored


def build_prediction_frame(
    frame: pd.DataFrame,
    predicted_percentile: np.ndarray,
) -> pd.DataFrame:
    output = frame[["company_name", "actual_upside", "actual_percentile"]].copy()
    output["predicted_percentile"] = predicted_percentile
    return output.sort_values("predicted_percentile", ascending=False)


def serialize_prediction_slice(predictions: pd.DataFrame, top: bool) -> list[dict[str, Any]]:
    ordered = predictions.head(10) if top else predictions.tail(10).sort_values("predicted_percentile")
    records: list[dict[str, Any]] = []
    for ticker, row in ordered.iterrows():
        records.append(
            {
                "ticker": ticker,
                "company_name": row["company_name"],
                "predicted_percentile": float(row["predicted_percentile"]),
                "actual_percentile": float(row["actual_percentile"]),
                "actual_upside": float(row["actual_upside"]),
            }
        )
    return records


def print_prediction_block(model_name: str, predictions: pd.DataFrame) -> None:
    print(f"{model_name} top 10 by predicted upside percentile:")
    print(f"{'Ticker':<8}{'Pred %':>10}{'Actual %':>10}{'Upside':>12}")
    for ticker, row in predictions.head(10).iterrows():
        print(
            f"{ticker:<8}"
            f"{row['predicted_percentile']:>10.3f}"
            f"{row['actual_percentile']:>10.3f}"
            f"{row['actual_upside']:>12.3f}"
        )
    print(f"{model_name} bottom 10 by predicted upside percentile:")
    print(f"{'Ticker':<8}{'Pred %':>10}{'Actual %':>10}{'Upside':>12}")
    for ticker, row in predictions.tail(10).sort_values("predicted_percentile").iterrows():
        print(
            f"{ticker:<8}"
            f"{row['predicted_percentile']:>10.3f}"
            f"{row['actual_percentile']:>10.3f}"
            f"{row['actual_upside']:>12.3f}"
        )


def find_surprises(
    frame: pd.DataFrame,
    elasticnet_predictions: pd.DataFrame,
    xgb_predictions: pd.DataFrame,
) -> list[str]:
    notes: list[str] = []

    for model_name, predictions in (("Elastic Net", elasticnet_predictions), ("XGBoost Ranker", xgb_predictions)):
        for ticker, row in predictions.head(10).iterrows():
            if row["actual_percentile"] <= 0.35:
                notes.append(
                    f"{model_name}: {ticker} is top-10 predicted despite only {row['actual_percentile']:.2f} actual upside percentile."
                )
        for ticker, row in predictions.tail(10).sort_values("predicted_percentile").iterrows():
            if row["actual_percentile"] >= 0.65:
                notes.append(
                    f"{model_name}: {ticker} is bottom-10 predicted despite {row['actual_percentile']:.2f} actual upside percentile."
                )

    disagreement = pd.Series(
        np.abs(elasticnet_predictions["predicted_percentile"] - xgb_predictions["predicted_percentile"]),
        index=frame.index,
        dtype=float,
    ).sort_values(ascending=False)
    for ticker, gap in disagreement.head(3).items():
        if gap >= 0.35:
            notes.append(
                f"Model disagreement: {ticker} differs by {gap:.2f} predicted percentile points between Elastic Net and XGBoost."
            )

    if not notes:
        return ["Top and bottom baskets broadly line up with the AI upside ranks; no large anomalies crossed the review thresholds."]
    return notes[:8]


def write_feature_spec(path: Path) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_order": list(CONSENSUS_FEATURES),
        "features": [asdict(spec) for spec in FEATURE_SPECS],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tickers_path = Path(args.tickers_file)
    edgar_path = Path(args.edgar_file)
    reports_dir = Path(args.reports_dir)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    write_feature_spec(FEATURE_SPEC_PATH)

    universe = load_ticker_universe(tickers_path)
    edgar_payload = load_edgar_payload(edgar_path)
    report_map = latest_successful_cheap_reports(reports_dir, set(universe))
    match_summary = build_match_summary(universe, edgar_payload, report_map)
    coverage = feature_coverage(edgar_payload, match_summary.matched_tickers)

    frame = build_training_frame(match_summary.matched_tickers, edgar_payload, report_map)
    actual_upside = frame["actual_upside"].to_numpy(dtype=float)
    actual_percentile = fractional_percentile_rank(actual_upside)
    target_decile = decile_labels_from_percentiles(actual_percentile)
    frame["actual_percentile"] = actual_percentile
    features = frame.loc[:, CONSENSUS_FEATURES]

    elasticnet_cv_percentile, elasticnet_metrics, elasticnet_repeat_metrics = run_elasticnet_repeated_cv(
        features=features,
        target_percentile=actual_percentile,
        actual_upside=actual_upside,
    )
    xgb_cv_raw, xgb_cv_percentile, xgb_metrics, xgb_fold_assignments = run_xgb_grouped_cv(
        features=features,
        target_decile=target_decile,
        actual_upside=actual_upside,
        actual_percentile=actual_percentile,
    )
    xgb_loo_raw, xgb_loo_percentile, xgb_loo_metrics = run_xgb_loo_cv(
        features=features,
        target_decile=target_decile,
        actual_upside=actual_upside,
        actual_percentile=actual_percentile,
    )

    elasticnet_model = fit_final_elasticnet(features, actual_percentile)
    xgb_model = fit_final_xgb(features, target_decile, np.asarray(xgb_fold_assignments, dtype=int))
    joblib.dump(elasticnet_model, ELASTICNET_MODEL_PATH)
    xgb_model.save_model(XGB_MODEL_PATH)

    coefficients = elasticnet_coefficients(elasticnet_model)
    importance = save_feature_importance_plot(xgb_model)
    save_elasticnet_coefficients_plot(coefficients)
    save_predicted_vs_actual_plot(actual_percentile, elasticnet_cv_percentile, xgb_cv_percentile)

    elasticnet_full_predictions = build_prediction_frame(
        frame,
        fractional_percentile_rank(elasticnet_model.predict(features)),
    )
    xgb_full_predictions = build_prediction_frame(
        frame,
        fractional_percentile_rank(xgb_model.predict(features)),
    )
    surprises = find_surprises(frame, elasticnet_full_predictions, xgb_full_predictions)
    used_features, ignored_features = used_and_ignored_features(importance)
    xgb_degenerate = len(used_features) == 0
    verdict = determine_verdict(elasticnet_metrics, xgb_metrics)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matched_tickers": match_summary.matched_tickers,
        "coverage": coverage,
        "feature_order": list(CONSENSUS_FEATURES),
        "monotonic_constraints": list(MONOTONIC_CONSTRAINTS),
        "elasticnet": {
            "cv_mean_metrics": asdict(elasticnet_metrics),
            "cv_repeat_metrics": [asdict(metric) for metric in elasticnet_repeat_metrics],
            "coefficients": {name: float(value) for name, value in coefficients.items()},
        },
        "xgboost_ranker": {
            "cv_metrics": asdict(xgb_metrics),
            "loo_metrics": asdict(xgb_loo_metrics),
            "degenerate_constant_scores": xgb_degenerate,
            "cv_fold_assignments": xgb_fold_assignments,
            "feature_importance": {name: float(value) for name, value in importance.items()},
            "used_features": used_features,
            "ignored_features": ignored_features,
        },
        "comparison": {
            "elasticnet_cv_mean": asdict(elasticnet_metrics),
            "xgboost_cv": asdict(xgb_metrics),
            "verdict": verdict,
        },
        "full_sample_predictions": {
            "elasticnet": {
                "top10": serialize_prediction_slice(elasticnet_full_predictions, top=True),
                "bottom10": serialize_prediction_slice(elasticnet_full_predictions, top=False),
            },
            "xgboost_ranker": {
                "top10": serialize_prediction_slice(xgb_full_predictions, top=True),
                "bottom10": serialize_prediction_slice(xgb_full_predictions, top=False),
            },
        },
        "surprises": surprises,
        "artifacts": {
            "elasticnet_model": str(ELASTICNET_MODEL_PATH),
            "xgboost_model": str(XGB_MODEL_PATH),
            "metadata": str(METADATA_PATH),
            "feature_spec": str(FEATURE_SPEC_PATH),
            "feature_importance_plot": str(FEATURE_IMPORTANCE_PATH),
            "predicted_vs_actual_plot": str(PREDICTED_VS_ACTUAL_PATH),
            "elasticnet_coefficients_plot": str(ELASTICNET_COEFFICIENTS_PATH),
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Matched clean EDGAR + AI set: {len(match_summary.matched_tickers)} tickers")
    print("Consensus feature coverage across matched tickers:")
    for feature_name in CONSENSUS_FEATURES:
        non_null = int(coverage[feature_name]["non_null"])
        coverage_ratio = coverage[feature_name]["coverage"]
        print(f"  {feature_name}: {non_null}/{len(match_summary.matched_tickers)} ({coverage_ratio:.1%})")

    print_comparison_table(elasticnet_metrics, xgb_metrics)
    print(f"XGBoost Ranker LOO-CV Spearman: {xgb_loo_metrics.spearman:.4f}")
    print(verdict)

    print("Elastic Net coefficients (standardized feature space):")
    for note in summarize_coefficient_strength(coefficients):
        print(f"  {note}")

    print("XGBoost used features:", ", ".join(used_features) if used_features else "none")
    print("XGBoost ignored features:", ", ".join(ignored_features) if ignored_features else "none")
    if xgb_degenerate:
        print("XGBoost diagnostic: the requested rank:ndcg configuration collapsed to constant OOF scores on this local stack/data slice.")

    print_prediction_block("Elastic Net", elasticnet_full_predictions)
    print_prediction_block("XGBoost Ranker", xgb_full_predictions)

    print("Potential surprises:")
    for note in surprises:
        print(f"  {note}")

    print(f"Saved XGBoost model: {XGB_MODEL_PATH}")
    print(f"Saved Elastic Net model: {ELASTICNET_MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Saved feature spec: {FEATURE_SPEC_PATH}")
    print(f"Saved feature importance plot: {FEATURE_IMPORTANCE_PATH}")
    print(f"Saved predicted-vs-actual plot: {PREDICTED_VS_ACTUAL_PATH}")
    print(f"Saved Elastic Net coefficients plot: {ELASTICNET_COEFFICIENTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
