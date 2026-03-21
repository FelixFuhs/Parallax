from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
TICKERS_PATH = ROOT / "tickers_100.txt"
EDGAR_PATH = ROOT / "edgar_features_100.json"
REPORTS_DIR = ROOT / "reports"
PLOTS_DIR = ROOT / "plots"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "distill_xgb.json"
METADATA_PATH = MODELS_DIR / "distill_xgb_metadata.json"
FEATURE_IMPORTANCE_PATH = PLOTS_DIR / "feature_importance.png"
PREDICTED_VS_ACTUAL_PATH = PLOTS_DIR / "predicted_vs_actual.png"
BROKEN_EDGAR_TICKERS = {"MCD"}
MOMENTUM_FIELDS = (
    "price_return_1m",
    "price_return_3m",
    "price_return_6m",
    "price_return_12m",
)
RAW_FIELDS = (
    "revenue",
    "gross_profit",
    "net_income",
    "total_assets",
    "total_equity",
    "operating_income",
    "total_debt",
    "cash",
    "operating_cash_flow",
    "capex",
    "da",
    "shares_outstanding",
    "current_assets",
    "current_liabilities",
)
DERIVED_FIELDS = (
    "current_price",
    "price_return_1m",
    "price_return_3m",
    "price_return_6m",
    "price_return_12m",
    "market_cap",
    "free_cash_flow",
    "fcf_yield",
    "roic",
    "roe",
    "gross_margin",
    "gross_profitability_assets",
    "operating_margin",
    "net_margin",
    "book_to_market",
    "debt_to_equity",
    "current_ratio",
    "asset_turnover",
    "asset_growth_1y",
    "accruals",
    "capex_intensity",
)
FEATURE_FIELDS = RAW_FIELDS + DERIVED_FIELDS


@dataclass(frozen=True)
class MatchSummary:
    matched_tickers: list[str]
    edgar_only: dict[str, str]
    nano_only: dict[str, str]
    edgar_coverage_count: int
    nano_coverage_count: int


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
    pattern = re.compile(r"^([A-Z0-9-]+)_(\d{4}-\d{2}-\d{2})_cheap\.json$")
    for path in reports_dir.glob("*_cheap.json"):
        match = pattern.match(path.name)
        if match is None:
            continue
        ticker, report_date = match.groups()
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
        record = edgar_payload[ticker]
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


def select_features(edgar_payload: dict[str, dict[str, Any]], matched_tickers: list[str]) -> tuple[list[str], dict[str, float]]:
    coverage: dict[str, float] = {}
    denominator = float(len(matched_tickers))
    for field_name in FEATURE_FIELDS:
        non_null_count = sum(edgar_payload[ticker].get(field_name) is not None for ticker in matched_tickers)
        coverage[field_name] = non_null_count / denominator
    selected = [
        field_name
        for field_name in FEATURE_FIELDS
        if coverage[field_name] > 0.80 or field_name in MOMENTUM_FIELDS
    ]
    return selected, coverage


def build_training_frame(
    matched_tickers: list[str],
    selected_features: list[str],
    edgar_payload: dict[str, dict[str, Any]],
    report_map: dict[str, tuple[str, Path]],
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    targets: list[float] = []
    for ticker in matched_tickers:
        _, report_path = report_map[ticker]
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        base_case = report_payload["_valuation"]["scenarios"]["base"]
        row = {"ticker": ticker}
        for field_name in selected_features:
            row[field_name] = edgar_payload[ticker].get(field_name)
        rows.append(row)
        targets.append(float(base_case["upside_downside_pct"]))
    frame = pd.DataFrame(rows).set_index("ticker")
    return frame, np.asarray(targets, dtype=float)


def winsorize_targets(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    lower, upper = np.quantile(values, [0.01, 0.99])
    return np.clip(values, lower, upper), float(lower), float(upper)


def build_model() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        max_depth=3,
        n_estimators=100,
        learning_rate=0.1,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
    )


def loo_predictions(frame: pd.DataFrame, target: np.ndarray) -> np.ndarray:
    predictions = np.empty(len(frame), dtype=float)
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(frame):
        x_train = frame.iloc[train_index]
        x_test = frame.iloc[test_index]
        y_train = target[train_index]

        imputer = SimpleImputer(strategy="median")
        x_train_imputed = imputer.fit_transform(x_train)
        x_test_imputed = imputer.transform(x_test)

        model = build_model()
        model.fit(x_train_imputed, y_train)
        predictions[test_index[0]] = float(model.predict(x_test_imputed)[0])
    return predictions


def fit_final_model(frame: pd.DataFrame, target: np.ndarray) -> tuple[SimpleImputer, XGBRegressor]:
    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(frame)
    model = build_model()
    model.fit(x_imputed, target)
    return imputer, model


def save_feature_importance_plot(model: XGBRegressor, selected_features: list[str]) -> None:
    importance = pd.Series(model.feature_importances_, index=selected_features).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    importance.sort_values().plot(kind="barh", ax=ax, color="#2f6db3")
    ax.set_title("Distillation Model Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_PATH, dpi=160)
    plt.close(fig)


def save_predicted_vs_actual_plot(actual: np.ndarray, predicted: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, predicted, alpha=0.75, color="#1f77b4", edgecolor="white", linewidth=0.6)
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#666666", linewidth=1.0)
    ax.set_title("LOO-CV Predicted vs Actual Nano Upside")
    ax.set_xlabel("Actual upside (winsorized)")
    ax.set_ylabel("Predicted upside")
    fig.tight_layout()
    fig.savefig(PREDICTED_VS_ACTUAL_PATH, dpi=160)
    plt.close(fig)


def verdict_from_metrics(r2_value: float, spearman_value: float) -> str:
    if r2_value > 0.0 and spearman_value >= 0.25:
        return "Yes, the distillation model captures enough of the AI signal to justify a backtest."
    return "No, the distillation model does not yet capture enough of the AI signal to justify backtesting."


def main() -> int:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    universe = load_ticker_universe(TICKERS_PATH)
    edgar_payload = load_edgar_payload(EDGAR_PATH)
    report_map = latest_successful_cheap_reports(REPORTS_DIR, set(universe))
    match_summary = build_match_summary(universe, edgar_payload, report_map)

    selected_features, coverage = select_features(edgar_payload, match_summary.matched_tickers)
    frame, raw_target = build_training_frame(
        matched_tickers=match_summary.matched_tickers,
        selected_features=selected_features,
        edgar_payload=edgar_payload,
        report_map=report_map,
    )

    missing_counts = frame.isna().sum(axis=1)
    complete_rows = int((missing_counts == 0).sum())
    imputed_rows = int((missing_counts > 0).sum())
    isolated_null_rows = int(((missing_counts > 0) & (missing_counts <= 3)).sum())

    winsorized_target, target_p01, target_p99 = winsorize_targets(raw_target)
    predictions = loo_predictions(frame, winsorized_target)
    r2_value = float(r2_score(winsorized_target, predictions))
    mae_value = float(mean_absolute_error(winsorized_target, predictions))
    spearman_value = float(spearmanr(winsorized_target, predictions).statistic)

    imputer, model = fit_final_model(frame, winsorized_target)
    model.save_model(MODEL_PATH)
    save_feature_importance_plot(model, selected_features)
    save_predicted_vs_actual_plot(winsorized_target, predictions)

    metadata = {
        "matched_tickers": match_summary.matched_tickers,
        "edgar_only": match_summary.edgar_only,
        "nano_only": match_summary.nano_only,
        "selected_features": selected_features,
        "feature_coverage": {field_name: coverage[field_name] for field_name in FEATURE_FIELDS},
        "complete_rows_without_imputation": complete_rows,
        "rows_recovered_by_median_imputation": imputed_rows,
        "isolated_null_rows": isolated_null_rows,
        "winsorization": {"p01": target_p01, "p99": target_p99},
        "metrics": {
            "r2": r2_value,
            "mae": mae_value,
            "spearman": spearman_value,
        },
        "imputer_statistics": {
            field_name: float(value)
            for field_name, value in zip(selected_features, imputer.statistics_, strict=True)
        },
        "verdict": verdict_from_metrics(r2_value, spearman_value),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Matched clean EDGAR + Nano set: {len(match_summary.matched_tickers)} tickers")
    print(f"EDGAR-only clean rows: {len(match_summary.edgar_only)}")
    print(f"Nano-only rows: {len(match_summary.nano_only)}")
    print("Selected features:")
    print(", ".join(selected_features))
    print(f"Complete rows without imputation: {complete_rows}")
    print(f"Rows recovered by median imputation: {imputed_rows}")
    print(f"Rows with isolated nulls (<=3 missing selected features): {isolated_null_rows}")
    print(f"LOO-CV R^2: {r2_value:.4f}")
    print(f"LOO-CV MAE: {mae_value:.4f}")
    print(f"LOO-CV Spearman: {spearman_value:.4f}")
    print(verdict_from_metrics(r2_value, spearman_value))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Saved feature importance plot: {FEATURE_IMPORTANCE_PATH}")
    print(f"Saved predicted-vs-actual plot: {PREDICTED_VS_ACTUAL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
