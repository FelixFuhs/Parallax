from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker, XGBRegressor


ROOT = Path(__file__).resolve().parent
TICKERS_PATH = ROOT / "tickers_100.txt"
EDGAR_PATH = ROOT / "edgar_features_100_v3.json"
REPORTS_DIR = ROOT / "reports"
PLOTS_DIR = ROOT / "plots"
MODELS_DIR = ROOT / "models"
XGB_MODEL_PATH = MODELS_DIR / "distill_xgb_v2.json"
XGB_REGRESSOR_MODEL_PATH = MODELS_DIR / "distill_xgb_regressor_v2.json"
ELASTICNET_MODEL_PATH = MODELS_DIR / "distill_elasticnet_v2.pkl"
METADATA_PATH = MODELS_DIR / "distill_v2_metadata.json"
FEATURE_SPEC_PATH = MODELS_DIR / "feature_spec.json"
FEATURE_IMPORTANCE_PATH = PLOTS_DIR / "feature_importance_v2.png"
PREDICTED_VS_ACTUAL_PATH = PLOTS_DIR / "predicted_vs_actual_v2.png"
ELASTICNET_COEFFICIENTS_PATH = PLOTS_DIR / "elasticnet_coefficients.png"
BROKEN_EDGAR_TICKERS = {"MCD"}
STABILITY_REPEATS = 50
BOOTSTRAP_SAMPLES = 1000


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


@dataclass(frozen=True)
class SummaryStats:
    mean: float
    std: float
    lower: float
    upper: float


@dataclass(frozen=True)
class ModelStabilityDiagnostics:
    repeated_cv_spearman: SummaryStats
    bootstrap_oob_spearman: SummaryStats
    top_quartile_overlap_stability: float


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
PredictorFn = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray]


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        statistic = spearmanr(actual, predicted).statistic
    if statistic is None or np.isnan(statistic):
        return 0.0
    return float(statistic)


def summary_stats(
    values: Sequence[float],
    *,
    lower_percentile: float,
    upper_percentile: float,
) -> SummaryStats:
    array = np.asarray(values, dtype=float)
    return SummaryStats(
        mean=float(array.mean()),
        std=float(array.std(ddof=0)),
        lower=float(np.percentile(array, lower_percentile)),
        upper=float(np.percentile(array, upper_percentile)),
    )


def top_quartile_tickers(predictions: pd.Series) -> set[str]:
    top_k = max(1, math.ceil(len(predictions) / 4.0))
    ordered = predictions.sort_values(ascending=False, kind="mergesort")
    return set(ordered.head(top_k).index.tolist())


def aggregate_top_quartile_rates(ticker_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        ticker_rows.groupby(["ticker", "company_name"], as_index=False)
        .agg(
            appearances=("top_quartile_hit", "size"),
            top_quartile_hits=("top_quartile_hit", "sum"),
            average_heldout_rank=("heldout_predicted_pct_rank", "mean"),
        )
        .sort_values(["top_quartile_hits", "average_heldout_rank"], ascending=[False, False], kind="mergesort")
    )
    grouped["top_quartile_rate"] = grouped["top_quartile_hits"] / grouped["appearances"]
    return grouped


def top_quartile_overlap_stability(top_quartile_rates: pd.DataFrame) -> float:
    if top_quartile_rates.empty:
        return 0.0
    top_k = max(1, math.ceil(len(top_quartile_rates) / 4.0))
    return float(top_quartile_rates.head(top_k)["top_quartile_rate"].mean())


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


def run_repeated_cv_with_stability(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    target: np.ndarray,
    actual_upside: np.ndarray,
    actual_percentile: np.ndarray,
    predictor: PredictorFn,
    *,
    n_splits: int = 5,
    n_repeats: int = STABILITY_REPEATS,
) -> tuple[np.ndarray, ModelMetrics, list[ModelMetrics], np.ndarray, pd.DataFrame]:
    repeat_predicted_percentiles: list[np.ndarray] = []
    repeat_metrics: list[ModelMetrics] = []
    fold_scores: list[float] = []
    ticker_rows: list[dict[str, Any]] = []

    for seed in range(n_repeats):
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        repeat_raw_predictions = np.empty(len(features), dtype=float)
        for fold_index, (train_index, test_index) in enumerate(splitter.split(features)):
            predicted_scores = np.asarray(
                predictor(features.iloc[train_index], target[train_index], features.iloc[test_index]),
                dtype=float,
            )
            repeat_raw_predictions[test_index] = predicted_scores
            fold_scores.append(safe_spearman(actual_upside[test_index], predicted_scores))

            predicted_series = pd.Series(predicted_scores, index=features.index[test_index], dtype=float)
            constant_fold_scores = predicted_scores.size > 0 and bool(np.allclose(predicted_scores, predicted_scores[0]))
            top_names = set() if constant_fold_scores else top_quartile_tickers(predicted_series)
            local_rank = predicted_series.rank(method="average", pct=True)
            for ticker, score in predicted_series.items():
                ticker_rows.append(
                    {
                        "seed": seed,
                        "fold": fold_index,
                        "ticker": ticker,
                        "company_name": frame.loc[ticker, "company_name"],
                        "predicted_score": float(score),
                        "heldout_actual_upside": float(frame.loc[ticker, "actual_upside"]),
                        "heldout_actual_percentile": float(frame.loc[ticker, "actual_percentile"]),
                        "heldout_predicted_pct_rank": float(local_rank.loc[ticker]),
                        "top_quartile_hit": ticker in top_names,
                    }
                )

        repeat_percentile_predictions = fractional_percentile_rank(repeat_raw_predictions)
        repeat_predicted_percentiles.append(repeat_percentile_predictions)
        repeat_metrics.append(
            evaluate_predictions(
                actual_upside=actual_upside,
                actual_percentile=actual_percentile,
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
    top_quartile_rates = aggregate_top_quartile_rates(pd.DataFrame(ticker_rows))
    return (
        mean_predicted_percentile,
        mean_metrics,
        repeat_metrics,
        np.asarray(fold_scores, dtype=float),
        top_quartile_rates,
    )


def predict_with_elasticnet(
    features: pd.DataFrame,
    target_percentile: np.ndarray,
    test_features: pd.DataFrame,
) -> np.ndarray:
    pipeline = build_elasticnet_pipeline()
    pipeline.fit(features, target_percentile)
    return np.asarray(pipeline.predict(test_features), dtype=float)


def run_elasticnet_repeated_cv(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    target_percentile: np.ndarray,
    actual_upside: np.ndarray,
    *,
    n_splits: int = 5,
    n_repeats: int = STABILITY_REPEATS,
) -> tuple[np.ndarray, ModelMetrics, list[ModelMetrics], np.ndarray, pd.DataFrame]:
    return run_repeated_cv_with_stability(
        frame=frame,
        features=features,
        target=target_percentile,
        actual_upside=actual_upside,
        actual_percentile=target_percentile,
        predictor=predict_with_elasticnet,
        n_splits=n_splits,
        n_repeats=n_repeats,
    )


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


def build_xgb_regressor() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        max_depth=2,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        min_child_weight=5,
        monotone_constraints=MONOTONIC_CONSTRAINTS,
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


def predict_with_xgb_regressor(
    features: pd.DataFrame,
    target_percentile: np.ndarray,
    test_features: pd.DataFrame,
) -> np.ndarray:
    model = build_xgb_regressor()
    model.fit(features, target_percentile)
    return np.asarray(model.predict(test_features), dtype=float)


def run_xgb_regressor_repeated_cv(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    target_percentile: np.ndarray,
    actual_upside: np.ndarray,
    *,
    n_splits: int = 5,
    n_repeats: int = STABILITY_REPEATS,
) -> tuple[np.ndarray, ModelMetrics, list[ModelMetrics], np.ndarray, pd.DataFrame]:
    return run_repeated_cv_with_stability(
        frame=frame,
        features=features,
        target=target_percentile,
        actual_upside=actual_upside,
        actual_percentile=target_percentile,
        predictor=predict_with_xgb_regressor,
        n_splits=n_splits,
        n_repeats=n_repeats,
    )


def predict_with_xgb_ranker(
    features: pd.DataFrame,
    target_decile: np.ndarray,
    test_features: pd.DataFrame,
) -> np.ndarray:
    qid = build_query_fold_assignments(features)
    x_train, y_train, qid_train = prepare_ranker_training_data(features, target_decile, qid)
    model = build_xgb_ranker()
    model.fit(x_train, y_train, qid=qid_train)
    return np.asarray(model.predict(test_features), dtype=float)


def run_xgb_ranker_repeated_cv(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    target_decile: np.ndarray,
    actual_upside: np.ndarray,
    actual_percentile: np.ndarray,
    *,
    n_splits: int = 5,
    n_repeats: int = STABILITY_REPEATS,
) -> tuple[np.ndarray, ModelMetrics, list[ModelMetrics], np.ndarray, pd.DataFrame]:
    return run_repeated_cv_with_stability(
        frame=frame,
        features=features,
        target=target_decile,
        actual_upside=actual_upside,
        actual_percentile=actual_percentile,
        predictor=predict_with_xgb_ranker,
        n_splits=n_splits,
        n_repeats=n_repeats,
    )


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


def fit_final_xgb_regressor(features: pd.DataFrame, target_percentile: np.ndarray) -> XGBRegressor:
    model = build_xgb_regressor()
    model.fit(features, target_percentile)
    return model


def fit_final_xgb(features: pd.DataFrame, target_decile: np.ndarray, fold_assignments: np.ndarray) -> XGBRanker:
    x_train, y_train, qid_train = prepare_ranker_training_data(features, target_decile, fold_assignments)
    model = build_xgb_ranker()
    model.fit(x_train, y_train, qid=qid_train)
    return model


def elasticnet_coefficients(pipeline: Pipeline) -> pd.Series:
    model = pipeline.named_steps["model"]
    return pd.Series(model.coef_, index=CONSENSUS_FEATURES, dtype=float)


def xgb_feature_importance(model: XGBRanker | XGBRegressor) -> pd.Series:
    return pd.Series(model.feature_importances_, index=CONSENSUS_FEATURES, dtype=float).sort_values(ascending=False)


def save_feature_importance_plot(model: XGBRanker) -> pd.Series:
    importance = xgb_feature_importance(model).sort_values()
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
    xgb_regressor_percentile: np.ndarray,
    xgb_ranker_percentile: np.ndarray,
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
        xgb_regressor_percentile,
        alpha=0.8,
        color="#2e8b57",
        edgecolor="white",
        linewidth=0.6,
        label="XGBoost Regressor CV",
    )
    ax.scatter(
        actual_percentile,
        xgb_ranker_percentile,
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


def build_model_stability_diagnostics(
    repeated_cv_fold_scores: np.ndarray,
    bootstrap_scores: np.ndarray,
    top_quartile_rates: pd.DataFrame,
) -> ModelStabilityDiagnostics:
    return ModelStabilityDiagnostics(
        repeated_cv_spearman=summary_stats(repeated_cv_fold_scores, lower_percentile=5.0, upper_percentile=95.0),
        bootstrap_oob_spearman=summary_stats(bootstrap_scores, lower_percentile=2.5, upper_percentile=97.5),
        top_quartile_overlap_stability=top_quartile_overlap_stability(top_quartile_rates),
    )


def run_bootstrap_oob_spearman(
    features: pd.DataFrame,
    target: np.ndarray,
    actual_upside: np.ndarray,
    predictor: PredictorFn,
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> np.ndarray:
    rng = np.random.default_rng(42)
    sample_size = len(features)
    scores: list[float] = []

    while len(scores) < bootstrap_samples:
        inbag_index = rng.integers(0, sample_size, size=sample_size)
        oob_mask = np.ones(sample_size, dtype=bool)
        oob_mask[np.unique(inbag_index)] = False
        oob_index = np.flatnonzero(oob_mask)
        if oob_index.size < 2:
            continue

        predicted_scores = np.asarray(
            predictor(features.iloc[inbag_index], target[inbag_index], features.iloc[oob_index]),
            dtype=float,
        )
        scores.append(safe_spearman(actual_upside[oob_index], predicted_scores))

    return np.asarray(scores, dtype=float)


def format_interval(stats: SummaryStats) -> str:
    return f"[{stats.lower:.4f}, {stats.upper:.4f}]"


def print_comparison_table(
    elasticnet_stability: ModelStabilityDiagnostics,
    xgb_regressor_stability: ModelStabilityDiagnostics,
    xgb_ranker_stability: ModelStabilityDiagnostics,
) -> None:
    print(
        "Head-to-head stability comparison "
        f"(repeated 5-fold CV, {STABILITY_REPEATS} repeats; bootstrap OOB, {BOOTSTRAP_SAMPLES} samples):"
    )
    print(f"{'Metric':<26}{'Elastic Net':>20}{'XGB Regressor':>20}{'XGB Ranker':>20}")
    print(
        f"{'Spearman (CV mean)':<26}"
        f"{elasticnet_stability.repeated_cv_spearman.mean:>20.4f}"
        f"{xgb_regressor_stability.repeated_cv_spearman.mean:>20.4f}"
        f"{xgb_ranker_stability.repeated_cv_spearman.mean:>20.4f}"
    )
    print(
        f"{'Spearman (CV std)':<26}"
        f"{elasticnet_stability.repeated_cv_spearman.std:>20.4f}"
        f"{xgb_regressor_stability.repeated_cv_spearman.std:>20.4f}"
        f"{xgb_ranker_stability.repeated_cv_spearman.std:>20.4f}"
    )
    print(
        f"{'Spearman (CV p05)':<26}"
        f"{elasticnet_stability.repeated_cv_spearman.lower:>20.4f}"
        f"{xgb_regressor_stability.repeated_cv_spearman.lower:>20.4f}"
        f"{xgb_ranker_stability.repeated_cv_spearman.lower:>20.4f}"
    )
    print(
        f"{'Spearman (CV p95)':<26}"
        f"{elasticnet_stability.repeated_cv_spearman.upper:>20.4f}"
        f"{xgb_regressor_stability.repeated_cv_spearman.upper:>20.4f}"
        f"{xgb_ranker_stability.repeated_cv_spearman.upper:>20.4f}"
    )
    print(
        f"{'Bootstrap 95% CI':<26}"
        f"{format_interval(elasticnet_stability.bootstrap_oob_spearman):>20}"
        f"{format_interval(xgb_regressor_stability.bootstrap_oob_spearman):>20}"
        f"{format_interval(xgb_ranker_stability.bootstrap_oob_spearman):>20}"
    )
    print(
        f"{'Top-Q overlap stability':<26}"
        f"{elasticnet_stability.top_quartile_overlap_stability:>19.1%}"
        f"{xgb_regressor_stability.top_quartile_overlap_stability:>19.1%}"
        f"{xgb_ranker_stability.top_quartile_overlap_stability:>19.1%}"
    )


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
    xgb_regressor_predictions: pd.DataFrame,
    xgb_predictions: pd.DataFrame,
) -> list[str]:
    notes: list[str] = []

    for model_name, predictions in (
        ("Elastic Net", elasticnet_predictions),
        ("XGBoost Regressor", xgb_regressor_predictions),
        ("XGBoost Ranker", xgb_predictions),
    ):
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

    for other_name, other_predictions in (
        ("XGBoost Regressor", xgb_regressor_predictions),
        ("XGBoost Ranker", xgb_predictions),
    ):
        disagreement = pd.Series(
            np.abs(elasticnet_predictions["predicted_percentile"] - other_predictions["predicted_percentile"]),
            index=frame.index,
            dtype=float,
        ).sort_values(ascending=False)
        for ticker, gap in disagreement.head(2).items():
            if gap >= 0.35:
                notes.append(
                    f"Model disagreement: {ticker} differs by {gap:.2f} predicted percentile points between Elastic Net and {other_name}."
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

    elasticnet_cv_percentile, elasticnet_metrics, elasticnet_repeat_metrics, elasticnet_fold_scores, elasticnet_topq_rates = run_elasticnet_repeated_cv(
        frame=frame,
        features=features,
        target_percentile=actual_percentile,
        actual_upside=actual_upside,
        n_repeats=STABILITY_REPEATS,
    )
    xgb_regressor_cv_percentile, xgb_regressor_metrics, xgb_regressor_repeat_metrics, xgb_regressor_fold_scores, xgb_regressor_topq_rates = run_xgb_regressor_repeated_cv(
        frame=frame,
        features=features,
        target_percentile=actual_percentile,
        actual_upside=actual_upside,
        n_repeats=STABILITY_REPEATS,
    )
    xgb_ranker_repeated_cv_percentile, xgb_ranker_repeated_metrics, xgb_ranker_repeat_metrics, xgb_ranker_fold_scores, xgb_ranker_topq_rates = run_xgb_ranker_repeated_cv(
        frame=frame,
        features=features,
        target_decile=target_decile,
        actual_upside=actual_upside,
        actual_percentile=actual_percentile,
        n_repeats=STABILITY_REPEATS,
    )
    elasticnet_bootstrap_scores = run_bootstrap_oob_spearman(
        features,
        actual_percentile,
        actual_upside,
        predict_with_elasticnet,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
    )
    xgb_regressor_bootstrap_scores = run_bootstrap_oob_spearman(
        features,
        actual_percentile,
        actual_upside,
        predict_with_xgb_regressor,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
    )
    xgb_ranker_bootstrap_scores = run_bootstrap_oob_spearman(
        features,
        target_decile,
        actual_upside,
        predict_with_xgb_ranker,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
    )
    elasticnet_stability = build_model_stability_diagnostics(
        elasticnet_fold_scores,
        elasticnet_bootstrap_scores,
        elasticnet_topq_rates,
    )
    xgb_regressor_stability = build_model_stability_diagnostics(
        xgb_regressor_fold_scores,
        xgb_regressor_bootstrap_scores,
        xgb_regressor_topq_rates,
    )
    xgb_ranker_stability = build_model_stability_diagnostics(
        xgb_ranker_fold_scores,
        xgb_ranker_bootstrap_scores,
        xgb_ranker_topq_rates,
    )
    _, _, xgb_metrics, xgb_fold_assignments = run_xgb_grouped_cv(
        features=features,
        target_decile=target_decile,
        actual_upside=actual_upside,
        actual_percentile=actual_percentile,
    )
    _, _, xgb_loo_metrics = run_xgb_loo_cv(
        features=features,
        target_decile=target_decile,
        actual_upside=actual_upside,
        actual_percentile=actual_percentile,
    )

    elasticnet_model = fit_final_elasticnet(features, actual_percentile)
    xgb_regressor_model = fit_final_xgb_regressor(features, actual_percentile)
    xgb_model = fit_final_xgb(features, target_decile, np.asarray(xgb_fold_assignments, dtype=int))
    joblib.dump(elasticnet_model, ELASTICNET_MODEL_PATH)
    xgb_regressor_model.save_model(XGB_REGRESSOR_MODEL_PATH)
    xgb_model.save_model(XGB_MODEL_PATH)

    coefficients = elasticnet_coefficients(elasticnet_model)
    xgb_regressor_importance = xgb_feature_importance(xgb_regressor_model)
    importance = save_feature_importance_plot(xgb_model)
    save_elasticnet_coefficients_plot(coefficients)
    save_predicted_vs_actual_plot(
        actual_percentile,
        elasticnet_cv_percentile,
        xgb_regressor_cv_percentile,
        xgb_ranker_repeated_cv_percentile,
    )

    elasticnet_full_predictions = build_prediction_frame(
        frame,
        fractional_percentile_rank(elasticnet_model.predict(features)),
    )
    xgb_regressor_full_predictions = build_prediction_frame(
        frame,
        fractional_percentile_rank(xgb_regressor_model.predict(features)),
    )
    xgb_full_predictions = build_prediction_frame(
        frame,
        fractional_percentile_rank(xgb_model.predict(features)),
    )
    surprises = find_surprises(frame, elasticnet_full_predictions, xgb_regressor_full_predictions, xgb_full_predictions)
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
            "stability": {
                "repeats": STABILITY_REPEATS,
                "spearman_stats": asdict(elasticnet_stability.repeated_cv_spearman),
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_spearman_stats": asdict(elasticnet_stability.bootstrap_oob_spearman),
                "top_quartile_overlap_stability": elasticnet_stability.top_quartile_overlap_stability,
            },
            "coefficients": {name: float(value) for name, value in coefficients.items()},
        },
        "xgboost_regressor": {
            "cv_mean_metrics": asdict(xgb_regressor_metrics),
            "cv_repeat_metrics": [asdict(metric) for metric in xgb_regressor_repeat_metrics],
            "stability": {
                "repeats": STABILITY_REPEATS,
                "spearman_stats": asdict(xgb_regressor_stability.repeated_cv_spearman),
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_spearman_stats": asdict(xgb_regressor_stability.bootstrap_oob_spearman),
                "top_quartile_overlap_stability": xgb_regressor_stability.top_quartile_overlap_stability,
            },
            "feature_importance": {name: float(value) for name, value in xgb_regressor_importance.items()},
        },
        "xgboost_ranker": {
            "cv_metrics": asdict(xgb_metrics),
            "loo_metrics": asdict(xgb_loo_metrics),
            "repeated_cv_mean_metrics": asdict(xgb_ranker_repeated_metrics),
            "cv_repeat_metrics": [asdict(metric) for metric in xgb_ranker_repeat_metrics],
            "stability": {
                "repeats": STABILITY_REPEATS,
                "spearman_stats": asdict(xgb_ranker_stability.repeated_cv_spearman),
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_spearman_stats": asdict(xgb_ranker_stability.bootstrap_oob_spearman),
                "top_quartile_overlap_stability": xgb_ranker_stability.top_quartile_overlap_stability,
            },
            "degenerate_constant_scores": xgb_degenerate,
            "cv_fold_assignments": xgb_fold_assignments,
            "feature_importance": {name: float(value) for name, value in importance.items()},
            "used_features": used_features,
            "ignored_features": ignored_features,
        },
        "comparison": {
            "elasticnet_cv_mean": asdict(elasticnet_metrics),
            "xgboost_regressor_cv_mean": asdict(xgb_regressor_metrics),
            "xgboost_cv": asdict(xgb_metrics),
            "stability": {
                "top_quartile_overlap_definition": "Mean held-out top-quartile hit rate among the model's most consistently top-ranked quartile of tickers across repeated 5-fold CV.",
                "elasticnet": asdict(elasticnet_stability),
                "xgboost_regressor": asdict(xgb_regressor_stability),
                "xgboost_ranker": asdict(xgb_ranker_stability),
            },
            "verdict": verdict,
        },
        "full_sample_predictions": {
            "elasticnet": {
                "top10": serialize_prediction_slice(elasticnet_full_predictions, top=True),
                "bottom10": serialize_prediction_slice(elasticnet_full_predictions, top=False),
            },
            "xgboost_regressor": {
                "top10": serialize_prediction_slice(xgb_regressor_full_predictions, top=True),
                "bottom10": serialize_prediction_slice(xgb_regressor_full_predictions, top=False),
            },
            "xgboost_ranker": {
                "top10": serialize_prediction_slice(xgb_full_predictions, top=True),
                "bottom10": serialize_prediction_slice(xgb_full_predictions, top=False),
            },
        },
        "surprises": surprises,
        "artifacts": {
            "elasticnet_model": str(ELASTICNET_MODEL_PATH),
            "xgboost_regressor_model": str(XGB_REGRESSOR_MODEL_PATH),
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

    print_comparison_table(elasticnet_stability, xgb_regressor_stability, xgb_ranker_stability)
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
    print_prediction_block("XGBoost Regressor", xgb_regressor_full_predictions)
    print_prediction_block("XGBoost Ranker", xgb_full_predictions)

    print("Potential surprises:")
    for note in surprises:
        print(f"  {note}")

    print(f"Saved XGBoost regressor model: {XGB_REGRESSOR_MODEL_PATH}")
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
