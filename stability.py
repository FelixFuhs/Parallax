from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from distill import (
    CONSENSUS_FEATURES,
    EDGAR_PATH,
    FEATURE_SPECS,
    METADATA_PATH,
    MODELS_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    TICKERS_PATH,
    build_elasticnet_pipeline,
    build_match_summary,
    build_training_frame,
    elasticnet_coefficients,
    fractional_percentile_rank,
    latest_successful_cheap_reports,
    load_edgar_payload,
    load_ticker_universe,
    safe_spearman,
)

SPEARMAN_STABILITY_PATH = PLOTS_DIR / "spearman_stability.png"
COEFFICIENT_STABILITY_PATH = PLOTS_DIR / "coefficient_stability.png"
BOOTSTRAP_SPEARMAN_PATH = PLOTS_DIR / "bootstrap_spearman.png"
STABILITY_METADATA_PATH = MODELS_DIR / "elasticnet_stability.json"
FROZEN_COEFFICIENTS_PATH = MODELS_DIR / "frozen_elasticnet_coefficients.json"
FROZEN_ELASTICNET_METADATA_PATH = MODELS_DIR / "frozen_elasticnet_metadata.json"
ELASTICNET_FREEZE_PATH = Path(__file__).resolve().parent / "docs" / "freeze_elasticnet_baseline.md"

EXPECTED_SIGNS: dict[str, int] = {
    "fcf_to_ev": 1,
    "gross_profitability_assets": 1,
    "asset_growth_1y": -1,
    "cash_earnings_gap": 1,
    "momentum_12_1": 1,
}
EXPECTED_SIGN_LABELS = {1: "positive", -1: "negative"}
FEATURE_DEFINITION_DETAILS: dict[str, dict[str, Any]] = {
    "fcf_to_ev": {
        "formula": "(operating_cash_flow - abs(capex)) / (market_cap + total_debt - cash)",
        "tags": [
            "operating_cash_flow: us-gaap/NetCashProvidedByOperatingActivities",
            "operating_cash_flow fallback: us-gaap/NetCashProvidedByUsedInOperatingActivities",
            "operating_cash_flow fallback: us-gaap/NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
            "capex: us-gaap/PaymentsToAcquirePropertyPlantAndEquipment",
            "capex fallback: us-gaap/PropertyPlantAndEquipmentAdditions",
            "capex fallback: us-gaap/PaymentsToAcquireProductiveAssets",
            "cash: us-gaap/CashAndCashEquivalentsAtCarryingValue",
            "shares_outstanding: us-gaap/CommonStockSharesOutstanding",
            "shares_outstanding fallback: dei/EntityCommonStockSharesOutstanding",
            "shares_outstanding fallback: us-gaap/WeightedAverageNumberOfShareOutstandingsBasic",
            "shares_outstanding fallback: us-gaap/WeightedAverageNumberOfSharesOutstandingBasic",
            "shares_outstanding fallback: us-gaap/WeightedAverageNumberOfShareOutstandingsBasicAndDiluted",
            "shares_outstanding fallback: us-gaap/WeightedAverageNumberOfSharesOutstandingBasicAndDiluted",
            "total_debt direct: us-gaap/LongTermDebtAndCapitalLeaseObligations",
            "total_debt direct fallback: us-gaap/LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
            "total_debt direct fallback: us-gaap/LongTermDebtAndFinanceLeaseObligations",
            "total_debt direct fallback: us-gaap/LongTermDebtAndFinanceLeaseObligationsIncludingCurrentMaturities",
            "total_debt component fallback: us-gaap/LongTermDebt",
            "total_debt component fallback: us-gaap/LongTermDebtNoncurrent",
            "total_debt component fallback: us-gaap/ShortTermBorrowings",
            "total_debt component fallback: us-gaap/LongTermDebtCurrent",
            "total_debt component fallback: us-gaap/ShortTermBankLoansAndNotesPayable",
            "total_debt component fallback: us-gaap/ShortTermDebt",
            "total_debt component fallback: us-gaap/CommercialPaper",
            "total_debt component fallback: us-gaap/LongTermDebtAndCapitalLeaseObligationsCurrent",
            "total_debt component fallback: us-gaap/LongTermDebtAndFinanceLeaseObligationsCurrent",
        ],
        "fallback_logic": [
            "Capex is forced positive with abs().",
            "Market cap is current yfinance auto-adjusted close times shares_outstanding.",
            "Total debt uses the direct total-debt tags first; if none are available, long-term and short-term debt components are summed.",
            "If enterprise value is zero or negative, the feature is set to null.",
        ],
    },
    "gross_profitability_assets": {
        "formula": "gross_profit / total_assets",
        "tags": [
            "gross_profit: us-gaap/GrossProfit",
            "gross_profit fallback cost tag: us-gaap/CostOfRevenue",
            "gross_profit fallback cost tag: us-gaap/CostOfGoodsAndServicesSold",
            "gross_profit fallback cost tag: us-gaap/CostOfGoodsSold",
            "revenue: us-gaap/RevenueFromContractWithCustomerExcludingAssessedTax",
            "revenue fallback: us-gaap/RevenueFromContractWithCustomerIncludingAssessedTax",
            "revenue fallback: us-gaap/Revenues",
            "revenue fallback: us-gaap/SalesRevenueNet",
            "total_assets: us-gaap/Assets",
        ],
        "fallback_logic": [
            "If GrossProfit is missing but revenue is present, gross profit is reconstructed as revenue minus the first aligned cost tag available.",
            "If neither direct nor fallback gross profit can be resolved, the feature is null.",
        ],
    },
    "asset_growth_1y": {
        "formula": "(current_total_assets - prior_total_assets) / prior_total_assets",
        "tags": [
            "current_total_assets: us-gaap/Assets",
            "prior_total_assets: us-gaap/Assets",
        ],
        "fallback_logic": [
            "Uses the two most recent distinct annual asset facts selected from EDGAR.",
            "If the prior-year asset value is unavailable, the feature is null.",
        ],
    },
    "cash_earnings_gap": {
        "formula": "(operating_cash_flow - net_income) / total_assets",
        "tags": [
            "operating_cash_flow: us-gaap/NetCashProvidedByOperatingActivities",
            "operating_cash_flow fallback: us-gaap/NetCashProvidedByUsedInOperatingActivities",
            "operating_cash_flow fallback: us-gaap/NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
            "net_income: us-gaap/NetIncomeLoss",
            "net_income fallback: us-gaap/ProfitLoss",
            "total_assets: us-gaap/Assets",
        ],
        "fallback_logic": [
            "If any of operating_cash_flow, net_income, or total_assets are missing, the feature is null.",
        ],
    },
    "momentum_12_1": {
        "formula": "(1 + price_return_12m) / (1 + price_return_1m) - 1",
        "tags": [
            "price source: yfinance auto-adjusted close history over the trailing ~400 calendar days",
            "price_return_1m: latest price versus price on or before latest_date - 1 month",
            "price_return_12m: latest price versus price on or before latest_date - 12 months",
        ],
        "fallback_logic": [
            "If either component return is missing, the feature is null.",
            "If 1-month gross return equals zero (a -100% return), the feature is null to avoid division by zero.",
        ],
    },
}


def repo_relative(path: Path) -> str:
    root = Path(__file__).resolve().parent
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


@dataclass(frozen=True)
class SummaryStats:
    mean: float
    std: float
    lower: float
    upper: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run elastic-net stability diagnostics and freeze the current Parallax model spec."
    )
    parser.add_argument("--tickers-file", default=str(TICKERS_PATH))
    parser.add_argument("--edgar-file", default=str(EDGAR_PATH))
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR))
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args(argv)


def summary_stats(
    values: Sequence[float],
    *,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
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


def coefficient_sign_summary(coefficient_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_name in CONSENSUS_FEATURES:
        series = coefficient_frame[feature_name]
        positive_share = float((series > 0.0).mean())
        negative_share = float((series < 0.0).mean())
        zero_share = float(np.isclose(series, 0.0).mean())
        expected_sign = EXPECTED_SIGNS[feature_name]
        expected_share = float((np.sign(series) == expected_sign).mean())
        rows.append(
            {
                "feature": feature_name,
                "expected_sign": EXPECTED_SIGN_LABELS[expected_sign],
                "expected_sign_share": expected_share,
                "positive_share": positive_share,
                "negative_share": negative_share,
                "zero_share": zero_share,
                "sign_flip": bool(positive_share > 0.0 and negative_share > 0.0),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def load_training_data(
    tickers_path: Path,
    edgar_path: Path,
    reports_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    universe = load_ticker_universe(tickers_path)
    edgar_payload = load_edgar_payload(edgar_path)
    report_map = latest_successful_cheap_reports(reports_dir, set(universe))
    match_summary = build_match_summary(universe, edgar_payload, report_map)
    frame = build_training_frame(match_summary.matched_tickers, edgar_payload, report_map)
    actual_upside = frame["actual_upside"].to_numpy(dtype=float)
    actual_percentile = fractional_percentile_rank(actual_upside)
    frame["actual_percentile"] = actual_percentile
    features = frame.loc[:, CONSENSUS_FEATURES]
    return frame, features, actual_upside, actual_percentile, match_summary.matched_tickers


def run_repeated_cv_diagnostics(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    target_percentile: np.ndarray,
    actual_upside: np.ndarray,
    *,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import KFold

    fold_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    ticker_rows: list[dict[str, Any]] = []

    for seed in range(repeats):
        kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
        for fold_index, (train_index, test_index) in enumerate(kfold.split(features)):
            pipeline = build_elasticnet_pipeline()
            pipeline.fit(features.iloc[train_index], target_percentile[train_index])
            predicted_scores = pipeline.predict(features.iloc[test_index])
            fold_spearman = safe_spearman(actual_upside[test_index], predicted_scores)
            fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold_index,
                    "n_test": int(len(test_index)),
                    "spearman": fold_spearman,
                }
            )

            coefficient_row = {"seed": seed, "fold": fold_index}
            coefficient_row.update(
                {
                    feature_name: float(coefficient)
                    for feature_name, coefficient in elasticnet_coefficients(pipeline).items()
                }
            )
            coefficient_rows.append(coefficient_row)

            predicted_series = pd.Series(predicted_scores, index=features.index[test_index], dtype=float)
            top_names = top_quartile_tickers(predicted_series)
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

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(ticker_rows),
    )


def run_bootstrap_oob_diagnostics(
    features: pd.DataFrame,
    target_percentile: np.ndarray,
    actual_upside: np.ndarray,
    *,
    bootstrap_samples: int,
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

        pipeline = build_elasticnet_pipeline()
        pipeline.fit(features.iloc[inbag_index], target_percentile[inbag_index])
        predicted_scores = pipeline.predict(features.iloc[oob_index])
        scores.append(safe_spearman(actual_upside[oob_index], predicted_scores))

    return np.asarray(scores, dtype=float)


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


def fit_full_model(features: pd.DataFrame, target_percentile: np.ndarray):
    pipeline = build_elasticnet_pipeline()
    pipeline.fit(features, target_percentile)
    return pipeline


def build_frozen_coefficients_payload(
    pipeline,
    matched_tickers: list[str],
    cv_metrics: dict[str, float],
) -> dict[str, Any]:
    coefficients = elasticnet_coefficients(pipeline)
    model = pipeline.named_steps["model"]
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_family": "ElasticNetCV",
        "feature_order": list(CONSENSUS_FEATURES),
        "matched_tickers": matched_tickers,
        "target_definition": "Fractional percentile rank of AI DCF base-case upside across the matched set (0.0 lowest, 1.0 highest).",
        "scoring_formula": "prediction = intercept + sum(((imputed_feature - scaler_mean) / scaler_scale) * coefficient)",
        "intercept": float(model.intercept_),
        "coefficients": {feature: float(coefficients[feature]) for feature in CONSENSUS_FEATURES},
        "imputer_medians": {
            feature: float(value) for feature, value in zip(CONSENSUS_FEATURES, imputer.statistics_, strict=True)
        },
        "scaler_mean": {
            feature: float(value) for feature, value in zip(CONSENSUS_FEATURES, scaler.mean_, strict=True)
        },
        "scaler_scale": {
            feature: float(value) for feature, value in zip(CONSENSUS_FEATURES, scaler.scale_, strict=True)
        },
        "selected_alpha": float(model.alpha_),
        "selected_l1_ratio": float(model.l1_ratio_),
        "final_cv_metrics": cv_metrics,
    }


def build_frozen_elasticnet_metadata(
    *,
    matched_count: int,
    final_cv_metrics: dict[str, float],
    fold_stats: SummaryStats,
    bootstrap_stats: SummaryStats,
    verdict: str,
    verdict_reasons: list[str],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_role": "baseline_model_freeze",
        "primary_model": False,
        "primary_model_reference": "models/frozen_xgb_regressor.json",
        "model_type": "elasticnet_baseline",
        "model_family": "ElasticNetCV",
        "feature_names": list(CONSENSUS_FEATURES),
        "training_set_size": int(matched_count),
        "final_cv_metrics": final_cv_metrics,
        "stability": {
            "spearman_cv_mean": fold_stats.mean,
            "spearman_cv_std": fold_stats.std,
            "spearman_cv_p05": fold_stats.lower,
            "spearman_cv_p95": fold_stats.upper,
            "bootstrap_oob_spearman_mean": bootstrap_stats.mean,
            "bootstrap_oob_spearman_ci": {
                "lower": bootstrap_stats.lower,
                "upper": bootstrap_stats.upper,
            },
        },
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "source_artifacts": {
            "frozen_coefficients": repo_relative(FROZEN_COEFFICIENTS_PATH),
            "baseline_freeze_doc": repo_relative(ELASTICNET_FREEZE_PATH),
            "source_metadata": repo_relative(METADATA_PATH),
        },
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def save_histogram(
    values: np.ndarray,
    path: Path,
    *,
    title: str,
    xlabel: str,
    stats: SummaryStats,
    lower_label: str,
    upper_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=24, color="#2f6db3", alpha=0.85, edgecolor="white", linewidth=0.8)
    ax.axvline(stats.mean, color="#b22222", linestyle="--", linewidth=1.5, label=f"Mean {stats.mean:.3f}")
    ax.axvline(stats.lower, color="#666666", linestyle=":", linewidth=1.2, label=f"{lower_label} {stats.lower:.3f}")
    ax.axvline(stats.upper, color="#666666", linestyle=":", linewidth=1.2, label=f"{upper_label} {stats.upper:.3f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_coefficient_boxplot(coefficient_frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    box = ax.boxplot(
        [coefficient_frame[feature].to_numpy(dtype=float) for feature in CONSENSUS_FEATURES],
        tick_labels=list(CONSENSUS_FEATURES),
        patch_artist=True,
    )
    for patch in box["boxes"]:
        patch.set(facecolor="#d7e5f4", edgecolor="#2f6db3")
    for median in box["medians"]:
        median.set(color="#b22222", linewidth=1.5)
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_title("Elastic Net Coefficient Stability Across 250 Folds")
    ax.set_ylabel("Coefficient (standardized feature space)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def determine_stability_verdict(
    fold_stats: SummaryStats,
    bootstrap_stats: SummaryStats,
    sign_summary: pd.DataFrame,
    top_quartile_rates: pd.DataFrame,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    stable = True

    if fold_stats.mean >= 0.5:
        reasons.append(f"Repeated-CV fold Spearman is materially positive on average ({fold_stats.mean:.3f}).")
    else:
        stable = False
        reasons.append(f"Repeated-CV fold Spearman mean is too weak ({fold_stats.mean:.3f}).")

    if fold_stats.lower > 0.0:
        reasons.append(f"The 5th percentile fold Spearman stays positive ({fold_stats.lower:.3f}).")
    else:
        stable = False
        reasons.append(f"The 5th percentile fold Spearman is non-positive ({fold_stats.lower:.3f}).")

    if bootstrap_stats.lower > 0.0:
        reasons.append(f"The bootstrap OOB Spearman 95% interval stays above zero ({bootstrap_stats.lower:.3f} to {bootstrap_stats.upper:.3f}).")
    else:
        stable = False
        reasons.append(f"The bootstrap OOB Spearman interval reaches zero or below ({bootstrap_stats.lower:.3f} to {bootstrap_stats.upper:.3f}).")

    consistent_expected_signs = int((sign_summary["expected_sign_share"] >= 0.75).sum())
    if consistent_expected_signs >= 3:
        reasons.append(f"{consistent_expected_signs}/5 coefficients keep the expected sign in at least 75% of folds.")
    else:
        stable = False
        reasons.append(f"Only {consistent_expected_signs}/5 coefficients keep the expected sign in at least 75% of folds.")

    mean_top10_rate = float(top_quartile_rates.head(10)["top_quartile_rate"].mean())
    if mean_top10_rate >= 0.5:
        reasons.append(f"The top-10 stable names land in the held-out top quartile {mean_top10_rate:.1%} of the time on average.")
    else:
        stable = False
        reasons.append(f"The top-10 stable names only land in the held-out top quartile {mean_top10_rate:.1%} of the time on average.")

    verdict = "Yes" if stable else "No"
    return verdict, reasons


def write_model_freeze_doc(
    path: Path,
    tickers_file_label: str,
    matched_count: int,
    final_cv_metrics: dict[str, float],
    fold_stats: SummaryStats,
    bootstrap_stats: SummaryStats,
    sign_summary: pd.DataFrame,
    top_quartile_rates: pd.DataFrame,
    frozen_coefficients: dict[str, Any],
    verdict: str,
    verdict_reasons: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    feature_rows = []
    for spec in FEATURE_SPECS:
        feature_rows.append(
            (
                spec.name,
                EXPECTED_SIGN_LABELS[EXPECTED_SIGNS[spec.name]],
                spec.economic_rationale,
            )
        )

    sign_rows = []
    for feature_name, row in sign_summary.iterrows():
        sign_rows.append(
            (
                feature_name,
                row["expected_sign"],
                f"{row['expected_sign_share']:.1%}",
                f"{row['positive_share']:.1%}",
                f"{row['negative_share']:.1%}",
                f"{row['zero_share']:.1%}",
                "Yes" if row["sign_flip"] else "No",
            )
        )

    top_rows = [
        (row.ticker, row.company_name, f"{row.top_quartile_rate:.1%}", int(row.appearances))
        for row in top_quartile_rates.head(10).itertuples(index=False)
    ]
    bottom_rows = [
        (row.ticker, row.company_name, f"{row.top_quartile_rate:.1%}", int(row.appearances))
        for row in top_quartile_rates.tail(10).sort_values("top_quartile_rate").itertuples(index=False)
    ]

    sections: list[str] = []
    sections.append("# Parallax Elastic Net Baseline Freeze")
    sections.append("")
    sections.append(f"Generated: {datetime.now(UTC).isoformat()}")
    sections.append("")
    sections.append("## Frozen Statement")
    sections.append("")
    sections.append("This baseline spec is frozen for transparency. It is not the primary Parallax frozen model; the primary model remains the XGBoost regressor documented separately.")
    sections.append("")
    sections.append("## Universe")
    sections.append("")
    sections.append("- Conceptual universe: S&P 500 ex-Financials ex-REITs.")
    sections.append(f"- Current frozen research slice: `{tickers_file_label}` in this repo.")
    sections.append(f"- Matched training sample used for the frozen model: {matched_count} tickers.")
    sections.append("- Entry requirements: latest successful cheap valuation report, no `stale_price` quality flag, base-case upside present, EDGAR row present with no extraction error, fiscal year >= 2024, and not in the hard-coded broken ticker list.")
    sections.append("- Missing feature values are allowed at the row level and are median-imputed inside each training fold and in the final full-sample fit.")
    sections.append("")
    sections.append("## Exclusion Rules")
    sections.append("")
    sections.append("- Exclude EDGAR rows with stale fiscal years (`fiscal_year < 2024`).")
    sections.append("- Exclude rows with explicit EDGAR extraction errors.")
    sections.append("- Exclude hard-coded broken data cases such as `MCD` (broken shares-outstanding / derived-feature path).")
    sections.append("- Exclude valuation reports with missing base-case upside or `stale_price` flags.")
    sections.append("")
    sections.append("## Target Variable")
    sections.append("")
    sections.append("- Target: fractional percentile rank of AI DCF base-case upside (`_valuation.scenarios.base.upside_downside_pct`) across the matched set.")
    sections.append("- Scale: `0.0` = lowest upside in the matched set, `1.0` = highest upside.")
    sections.append("- Ranking is computed directly from raw upside values; the target is not winsorized.")
    sections.append("")
    sections.append("## Winsorization")
    sections.append("")
    sections.append("- None.")
    sections.append("- No percentiles are clipped before ranking, after ranking, or before Elastic Net fitting.")
    sections.append("")
    sections.append("## Scaling And Normalization")
    sections.append("")
    sections.append("- Features are used in the fixed order: `" + "`, `".join(CONSENSUS_FEATURES) + "`.")
    sections.append("- Missing feature values are imputed with the training-fold median (`SimpleImputer(strategy='median')`).")
    sections.append("- Imputed features are standardized with `StandardScaler()` fit on the training fold only.")
    sections.append("- The target is left on percentile-rank scale; no target normalization is applied.")
    sections.append("- Final frozen coefficients come from refitting the same pipeline on the full matched sample.")
    sections.append("")
    sections.append("## ElasticNetCV Search Space")
    sections.append("")
    sections.append("- Estimator: `ElasticNetCV(random_state=42, max_iter=100000)`.")
    sections.append("- `l1_ratio` remains at the sklearn default of `0.5` (not searched).")
    sections.append("- Alpha search uses sklearn's default auto-generated log-spaced alpha path, from `alpha_max` down to `alpha_max * eps`, with `eps=1e-3` and the default number of alpha values.")
    sections.append("- Inner CV is sklearn's default 5-fold cross-validation.")
    sections.append(f"- On the frozen full-sample fit, the selected alpha is `{frozen_coefficients['selected_alpha']:.6f}` and the selected `l1_ratio` is `{frozen_coefficients['selected_l1_ratio']:.2f}`.")
    sections.append("")
    sections.append("## Feature Set")
    sections.append("")
    sections.append(markdown_table(["Feature", "Expected Sign", "Economic Rationale"], feature_rows))
    sections.append("")
    sections.append("## Exact Feature Definitions")
    sections.append("")
    sections.append("EDGAR values are taken from the latest selected annual filing context; when a primary tag is absent, the documented fallback chain is used.")
    sections.append("")
    for feature_name in CONSENSUS_FEATURES:
        detail = FEATURE_DEFINITION_DETAILS[feature_name]
        sections.append(f"### `{feature_name}`")
        sections.append("")
        sections.append(f"- Formula: `{detail['formula']}`")
        sections.append("- Tags used:")
        for tag in detail["tags"]:
            sections.append(f"  - {tag}")
        sections.append("- Fallback logic:")
        for item in detail["fallback_logic"]:
            sections.append(f"  - {item}")
        sections.append("")
    sections.append("## Frozen CV Metrics")
    sections.append("")
    sections.append(
        markdown_table(
            ["Metric", "Value"],
            [
                ("Spearman (CV)", f"{final_cv_metrics['spearman']:.4f}"),
                ("R^2 (CV)", f"{final_cv_metrics['r2']:.4f}"),
                ("MAE (CV)", f"{final_cv_metrics['mae']:.4f}"),
            ],
        )
    )
    sections.append("")
    sections.append("## Stability Checks")
    sections.append("")
    sections.append(
        markdown_table(
            ["Diagnostic", "Value"],
            [
                ("Repeated-CV fold Spearman mean", f"{fold_stats.mean:.4f}"),
                ("Repeated-CV fold Spearman std", f"{fold_stats.std:.4f}"),
                ("Repeated-CV fold Spearman 5th pct", f"{fold_stats.lower:.4f}"),
                ("Repeated-CV fold Spearman 95th pct", f"{fold_stats.upper:.4f}"),
                ("Bootstrap OOB Spearman mean", f"{bootstrap_stats.mean:.4f}"),
                ("Bootstrap OOB Spearman 95% CI", f"[{bootstrap_stats.lower:.4f}, {bootstrap_stats.upper:.4f}]"),
            ],
        )
    )
    sections.append("")
    sections.append("### Coefficient Sign Stability")
    sections.append("")
    sections.append(
        markdown_table(
            ["Feature", "Expected", "Expected Sign %", "Positive %", "Negative %", "Zero %", "Flip?"],
            sign_rows,
        )
    )
    sections.append("")
    sections.append("### Top-Quartile Stability")
    sections.append("")
    sections.append("Top-quartile means the top 25% of predicted names inside each held-out fold.")
    sections.append("")
    sections.append("Most consistently top-ranked:")
    sections.append("")
    sections.append(markdown_table(["Ticker", "Company", "Top Quartile %", "Hold-out Appearances"], top_rows))
    sections.append("")
    sections.append("Most consistently bottom-ranked:")
    sections.append("")
    sections.append(markdown_table(["Ticker", "Company", "Top Quartile %", "Hold-out Appearances"], bottom_rows))
    sections.append("")
    sections.append("## Baseline Backcasting Verdict")
    sections.append("")
    sections.append(f"Is the model stable enough to trust for backcasting? {verdict}.")
    sections.append("")
    for reason in verdict_reasons:
        sections.append(f"- {reason}")
    sections.append("")
    sections.append("## Frozen Linear Parameters")
    sections.append("")
    sections.append("- Frozen scoring payload: `models/frozen_elasticnet_coefficients.json`.")
    sections.append("- This payload includes the intercept, standardized-feature coefficients, imputer medians, scaler means/scales, and the selected alpha.")
    sections.append("")

    path.write_text("\n".join(sections), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    tickers_path = Path(args.tickers_file)
    edgar_path = Path(args.edgar_file)
    reports_dir = Path(args.reports_dir)

    frame, features, actual_upside, actual_percentile, matched_tickers = load_training_data(
        tickers_path,
        edgar_path,
        reports_dir,
    )

    fold_frame, coefficient_frame, ticker_frame = run_repeated_cv_diagnostics(
        frame,
        features,
        actual_percentile,
        actual_upside,
        repeats=args.repeats,
    )
    bootstrap_scores = run_bootstrap_oob_diagnostics(
        features,
        actual_percentile,
        actual_upside,
        bootstrap_samples=args.bootstrap_samples,
    )

    fold_stats = summary_stats(fold_frame["spearman"], lower_percentile=5.0, upper_percentile=95.0)
    bootstrap_stats = summary_stats(bootstrap_scores, lower_percentile=2.5, upper_percentile=97.5)
    coefficient_only = coefficient_frame.loc[:, CONSENSUS_FEATURES]
    sign_summary = coefficient_sign_summary(coefficient_only)
    top_quartile_rates = aggregate_top_quartile_rates(ticker_frame)

    save_histogram(
        fold_frame["spearman"].to_numpy(dtype=float),
        SPEARMAN_STABILITY_PATH,
        title="Elastic Net Per-Fold Spearman Stability",
        xlabel="Held-out fold Spearman correlation",
        stats=fold_stats,
        lower_label="5th pct",
        upper_label="95th pct",
    )
    save_coefficient_boxplot(coefficient_only, COEFFICIENT_STABILITY_PATH)
    save_histogram(
        bootstrap_scores,
        BOOTSTRAP_SPEARMAN_PATH,
        title="Elastic Net Bootstrap OOB Spearman",
        xlabel="Out-of-bag Spearman correlation",
        stats=bootstrap_stats,
        lower_label="2.5th pct",
        upper_label="97.5th pct",
    )

    final_model = fit_full_model(features, actual_percentile)
    final_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    final_cv_metrics = final_metadata["elasticnet"]["cv_mean_metrics"]
    frozen_coefficients = build_frozen_coefficients_payload(final_model, matched_tickers, final_cv_metrics)
    FROZEN_COEFFICIENTS_PATH.write_text(json.dumps(frozen_coefficients, indent=2, sort_keys=True), encoding="utf-8")

    verdict, verdict_reasons = determine_stability_verdict(
        fold_stats,
        bootstrap_stats,
        sign_summary,
        top_quartile_rates,
    )
    write_model_freeze_doc(
        ELASTICNET_FREEZE_PATH,
        tickers_file_label=tickers_path.name,
        matched_count=len(matched_tickers),
        final_cv_metrics=final_cv_metrics,
        fold_stats=fold_stats,
        bootstrap_stats=bootstrap_stats,
        sign_summary=sign_summary,
        top_quartile_rates=top_quartile_rates,
        frozen_coefficients=frozen_coefficients,
        verdict=verdict,
        verdict_reasons=verdict_reasons,
    )
    frozen_elasticnet_metadata = build_frozen_elasticnet_metadata(
        matched_count=len(matched_tickers),
        final_cv_metrics=final_cv_metrics,
        fold_stats=fold_stats,
        bootstrap_stats=bootstrap_stats,
        verdict=verdict,
        verdict_reasons=verdict_reasons,
    )
    FROZEN_ELASTICNET_METADATA_PATH.write_text(
        json.dumps(frozen_elasticnet_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    stability_metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "matched_tickers": matched_tickers,
        "repeated_cv": {
            "repeats": args.repeats,
            "folds": int(len(fold_frame)),
            "spearman_stats": asdict(fold_stats),
            "coefficient_sign_summary": sign_summary.reset_index().to_dict(orient="records"),
            "top_quartile_rates": top_quartile_rates.to_dict(orient="records"),
        },
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "spearman_stats": asdict(bootstrap_stats),
        },
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "artifacts": {
            "spearman_stability_plot": repo_relative(SPEARMAN_STABILITY_PATH),
            "coefficient_stability_plot": repo_relative(COEFFICIENT_STABILITY_PATH),
            "bootstrap_spearman_plot": repo_relative(BOOTSTRAP_SPEARMAN_PATH),
            "frozen_coefficients": repo_relative(FROZEN_COEFFICIENTS_PATH),
            "frozen_elasticnet_metadata": repo_relative(FROZEN_ELASTICNET_METADATA_PATH),
            "elasticnet_freeze_doc": repo_relative(ELASTICNET_FREEZE_PATH),
        },
    }
    STABILITY_METADATA_PATH.write_text(json.dumps(stability_metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Matched set: {len(matched_tickers)} tickers")
    print(
        "Repeated-CV fold Spearman: "
        f"mean={fold_stats.mean:.4f}, std={fold_stats.std:.4f}, "
        f"p05={fold_stats.lower:.4f}, p95={fold_stats.upper:.4f}"
    )
    print(
        "Bootstrap OOB Spearman: "
        f"mean={bootstrap_stats.mean:.4f}, 95% CI=[{bootstrap_stats.lower:.4f}, {bootstrap_stats.upper:.4f}]"
    )
    print("Coefficient sign stability:")
    for feature_name, row in sign_summary.iterrows():
        print(
            f"  {feature_name}: expected_sign={row['expected_sign']}, "
            f"expected_sign_pct={row['expected_sign_share']:.1%}, "
            f"flip={'yes' if row['sign_flip'] else 'no'}"
        )
    print("Most consistently top-ranked tickers:")
    for row in top_quartile_rates.head(10).itertuples(index=False):
        print(f"  {row.ticker}: {row.top_quartile_rate:.1%} top-quartile hit rate over {row.appearances} hold-out appearances")
    print("Most consistently bottom-ranked tickers:")
    for row in top_quartile_rates.tail(10).sort_values("top_quartile_rate").itertuples(index=False):
        print(f"  {row.ticker}: {row.top_quartile_rate:.1%} top-quartile hit rate over {row.appearances} hold-out appearances")
    print(f"Is the model stable enough to trust for backcasting? {verdict}.")
    for reason in verdict_reasons:
        print(f"  - {reason}")
    print(f"Saved spearman stability plot: {SPEARMAN_STABILITY_PATH}")
    print(f"Saved coefficient stability plot: {COEFFICIENT_STABILITY_PATH}")
    print(f"Saved bootstrap spearman plot: {BOOTSTRAP_SPEARMAN_PATH}")
    print(f"Saved stability metadata: {STABILITY_METADATA_PATH}")
    print(f"Saved frozen coefficients: {FROZEN_COEFFICIENTS_PATH}")
    print(f"Saved Elastic Net baseline metadata: {FROZEN_ELASTICNET_METADATA_PATH}")
    print(f"Saved Elastic Net baseline freeze doc: {ELASTICNET_FREEZE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
