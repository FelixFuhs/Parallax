from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

import historical
from backtest import (
    build_rebalance_dates,
    compute_performance_metrics,
    first_price_after,
    load_tickers,
    run_audited_signal_backtest,
)
from benchmarks import build_composite_vqmia
from edgar import DEFAULT_SEC_EMAIL, DEFAULT_SEC_NAME, SecClient, normalize_ticker
from experiment_b_factor_portability import (
    DEFAULT_TARGETS,
    _build_model,
    _design_matrix,
    _percentile_rank,
    _safe_spearman,
    available_feature_columns,
    build_training_frame,
)
from experiment_registry import build_experiment_metadata, repo_relative, write_experiment_metadata
from label_panel import DEFAULT_EDGAR_PATH, DEFAULT_OUTPUT_PATH, load_edgar_payload
from price_model import PricePanels
from signal_diagnostics import (
    rank_ic_by_sector,
    rank_ic_coverage,
    rank_ic_diagnostics,
    summarize_rank_ic,
    summarize_rank_ic_by_sector,
    summarize_rank_ic_by_year,
)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DEFAULT_TICKERS_PATH = ROOT / "tickers.txt"
DEFAULT_SECTOR_MAP = ROOT / "data" / "sector_map_wikipedia.csv"
DEFAULT_SP500_CHANGES = ROOT / "data" / "sp500_changes_wikipedia.csv"
DEFAULT_APPROX_MEMBERSHIP = ROOT / "data" / "approx_sp500_membership.parquet"
DEFAULT_APPROX_MEMBERSHIP_SUMMARY = RESULTS_DIR / "approx_sp500_membership_summary.json"
DEFAULT_SCORES = RESULTS_DIR / "experiment_b_historical_backcast_scores.parquet"
DEFAULT_MONTHLY_RETURNS = RESULTS_DIR / "experiment_b_historical_backcast_monthly_returns.parquet"
DEFAULT_HOLDINGS = RESULTS_DIR / "experiment_b_historical_backcast_holdings.parquet"
DEFAULT_TURNOVER = RESULTS_DIR / "experiment_b_historical_backcast_turnover.parquet"
DEFAULT_EXPOSURES = RESULTS_DIR / "experiment_b_historical_backcast_exposures.parquet"
DEFAULT_RANK_IC = RESULTS_DIR / "experiment_b_historical_backcast_rank_ic.parquet"
DEFAULT_RANK_IC_SUMMARY = RESULTS_DIR / "experiment_b_historical_backcast_rank_ic_summary.parquet"
DEFAULT_RANK_IC_BY_YEAR = RESULTS_DIR / "experiment_b_historical_backcast_rank_ic_by_year.parquet"
DEFAULT_RANK_IC_BY_SECTOR = RESULTS_DIR / "experiment_b_historical_backcast_rank_ic_by_sector.parquet"
DEFAULT_RANK_IC_COVERAGE = RESULTS_DIR / "experiment_b_historical_backcast_rank_ic_coverage.parquet"
DEFAULT_SUMMARY = RESULTS_DIR / "experiment_b_historical_backcast_summary.json"
DEFAULT_METADATA = RESULTS_DIR / "experiment_b_historical_backcast_metadata.json"
DEFAULT_COST_BPS = (0, 10, 25, 50)
DEFAULT_RANK_IC_HORIZONS = (1, 3, 6, 12)
HISTORICAL_BENCHMARK_CONTROLS = {
    "composite_vqmia_score": {
        "target": "benchmark_composite_vqmia",
        "signal_name": "benchmark_composite_vqmia_score",
    },
    "fcf_to_ev": {
        "target": "benchmark_fcf_to_ev",
        "signal_name": "benchmark_fcf_to_ev",
    },
    "value_block_score": {
        "target": "benchmark_value_block",
        "signal_name": "benchmark_value_block_score",
    },
    "quality_block_score": {
        "target": "benchmark_quality_block",
        "signal_name": "benchmark_quality_block_score",
    },
    "momentum_block_score": {
        "target": "benchmark_momentum_block",
        "signal_name": "benchmark_momentum_block_score",
    },
}
HISTORICAL_BACKCAST_FEATURES = (
    "fcf_to_ev",
    "book_to_market",
    "fcf_yield",
    "gross_profitability_assets",
    "roic",
    "roe",
    "operating_margin",
    "cash_earnings_gap",
    "momentum_12_1",
    "price_return_6m",
    "price_return_1m",
    "asset_growth_1y",
    "accruals",
    "current_ratio",
    "debt_to_equity",
    "log_market_cap",
)


def historical_signal_name_for_target(target: str) -> str:
    for benchmark_spec in HISTORICAL_BENCHMARK_CONTROLS.values():
        if str(benchmark_spec["target"]) == str(target):
            return str(benchmark_spec["signal_name"])
    return f"experiment_b_{target}"


SCORE_COLUMNS = [
    "date",
    "ticker",
    "target",
    "signal_name",
    "score",
    "sector",
    "market_cap",
    "feature_null_count",
    "public_sp500_add_date",
    "approximate_membership_eligible",
]
AUDIT_COLUMNS = {
    "holdings": [
        "date",
        "next_rebalance_date",
        "ticker",
        "sector",
        "market_cap",
        "score",
        "signal_name",
        "portfolio_mode",
        "weighting_method",
        "bucket",
        "weight",
        "feature_null_count",
        "entry_price",
        "exit_price",
        "raw_close_entry_price",
        "raw_return",
        "cost_bps_one_way",
        "transaction_cost",
        "net_return",
    ],
    "monthly_returns": [
        "date",
        "next_rebalance_date",
        "signal_name",
        "portfolio_mode",
        "weighting_method",
        "bucket",
        "cost_bps_one_way",
        "gross_return",
        "transaction_cost_drag",
        "net_return",
        "name_count",
    ],
    "turnover": [
        "date",
        "signal_name",
        "portfolio_mode",
        "weighting_method",
        "bucket",
        "cost_bps_one_way",
        "turnover",
        "transaction_cost_drag",
    ],
    "exposures": [
        "date",
        "signal_name",
        "portfolio_mode",
        "weighting_method",
        "bucket",
        "sector_weights",
        "average_market_cap",
        "name_count",
    ],
}
RANK_IC_COLUMNS = {
    "rank_ic": ["date", "signal", "horizon", "decomposition", "n", "rank_ic", "sector_count", "sector_status"],
    "rank_ic_summary": [
        "signal",
        "horizon",
        "decomposition",
        "months",
        "mean_ic",
        "median_ic",
        "ic_std",
        "newey_west_tstat",
        "positive_ic_hit_rate",
    ],
    "rank_ic_by_year": [
        "year",
        "signal",
        "horizon",
        "decomposition",
        "months",
        "mean_ic",
        "median_ic",
        "positive_ic_hit_rate",
    ],
    "rank_ic_by_sector": [
        "sector",
        "signal",
        "horizon",
        "months",
        "mean_ic",
        "median_ic",
        "positive_ic_hit_rate",
        "mean_n",
    ],
    "rank_ic_coverage": [
        "date",
        "signal",
        "horizon",
        "universe_n",
        "score_non_null",
        "return_non_null",
        "paired_n",
        "paired_coverage",
        "sector_count",
    ],
}


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def load_sector_map(path: Path = DEFAULT_SECTOR_MAP) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype="object")
    frame = pd.read_csv(path)
    if not {"ticker", "sector"}.issubset(frame.columns):
        return pd.Series(dtype="object")
    tickers = frame["ticker"].astype(str).map(normalize_ticker)
    return pd.Series(frame["sector"].values, index=tickers, dtype="object")


def load_public_sp500_add_dates(path: Path = DEFAULT_SP500_CHANGES) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype="datetime64[ns]")
    frame = pd.read_csv(path)
    if not {"effective_date", "added_ticker"}.issubset(frame.columns):
        return pd.Series(dtype="datetime64[ns]")
    work = frame[["effective_date", "added_ticker"]].copy()
    work["ticker"] = work["added_ticker"].fillna("").astype(str).map(normalize_ticker)
    work["effective_date"] = pd.to_datetime(work["effective_date"], errors="coerce")
    work = work[(work["ticker"].str.len() > 0) & work["effective_date"].notna()]
    if work.empty:
        return pd.Series(dtype="datetime64[ns]")
    return work.groupby("ticker")["effective_date"].min().sort_index()


def add_backcast_context(frame: pd.DataFrame, sector_map: pd.Series) -> pd.DataFrame:
    output = frame.copy()
    output.index = [normalize_ticker(str(value)) for value in output.index]
    output.index.name = "ticker"
    mapped_sector = pd.Series(output.index, index=output.index).map(sector_map) if not sector_map.empty else pd.Series(index=output.index, dtype="object")
    if "sector" in output.columns:
        output["sector"] = output["sector"].combine_first(mapped_sector)
    else:
        output["sector"] = mapped_sector

    market_cap = pd.to_numeric(output["market_cap"], errors="coerce") if "market_cap" in output else pd.Series(index=output.index, dtype=float)
    output["log_market_cap"] = np.where(market_cap > 0.0, np.log(market_cap), np.nan)
    return output


def apply_public_add_date_filter(
    frame: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    add_dates: pd.Series,
) -> tuple[pd.DataFrame, int]:
    output = frame.copy()
    if add_dates.empty:
        output["public_sp500_add_date"] = pd.NaT
        output["approximate_membership_eligible"] = True
        return output, 0
    output["public_sp500_add_date"] = pd.Series(output.index, index=output.index).map(add_dates)
    eligible = output["public_sp500_add_date"].isna() | (output["public_sp500_add_date"] <= pd.Timestamp(rebalance_date))
    output["approximate_membership_eligible"] = eligible
    return output.loc[eligible].copy(), int((~eligible).sum())


def historical_feature_columns_from_matrices(
    matrices: Mapping[pd.Timestamp, pd.DataFrame] | None,
) -> set[str]:
    available = set(HISTORICAL_BACKCAST_FEATURES)
    if matrices:
        matrix_columns: set[str] = set()
        for matrix in matrices.values():
            matrix_columns.update(str(column) for column in matrix.columns)
        if "market_cap" in matrix_columns:
            matrix_columns.add("log_market_cap")
        available &= matrix_columns
    return available


def select_historical_compatible_features(
    training_frame: pd.DataFrame,
    matrices: Mapping[pd.Timestamp, pd.DataFrame] | None = None,
) -> list[str]:
    current_features = available_feature_columns(training_frame)
    historical_features = historical_feature_columns_from_matrices(matrices)
    return [feature for feature in current_features if feature in historical_features]


def fit_historical_compatible_models(
    training_frame: pd.DataFrame,
    *,
    targets: Sequence[str] = DEFAULT_TARGETS,
    feature_columns: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    model_specs: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for target in targets:
        target_values = pd.to_numeric(training_frame[target], errors="coerce") if target in training_frame else pd.Series(dtype=float)
        weights = pd.to_numeric(training_frame.get("mean_label_weight"), errors="coerce").fillna(0.0)
        fit_mask = target_values.notna() & (weights > 0.0)
        fit_frame = training_frame.loc[fit_mask].copy()
        if len(fit_frame) < 3 or not feature_columns:
            summaries.append(
                {
                    "target": target,
                    "status": "blocked",
                    "n": int(len(fit_frame)),
                    "feature_count": int(len(feature_columns)),
                    "blockers": [
                        {
                            "code": "insufficient_historical_compatible_training_data",
                            "message": "Fewer than three current labels or no historical-compatible features were available.",
                        }
                    ],
                }
            )
            continue

        x = _design_matrix(fit_frame, feature_columns)
        y = pd.to_numeric(fit_frame[target], errors="coerce").astype(float)
        model = _build_model(len(x))
        model.fit(x, y)
        fit_predictions = pd.Series(model.predict(x), index=x.index, dtype=float)
        target_unique_count = int(y.nunique(dropna=True))
        prediction_unique_count = int(fit_predictions.nunique(dropna=True))
        if target_unique_count < 2 or prediction_unique_count < 2:
            summaries.append(
                {
                    "target": target,
                    "status": "blocked_degenerate_historical_compatible_factor_map",
                    "n": int(len(fit_frame)),
                    "feature_count": int(len(feature_columns)),
                    "design_matrix_column_count": int(x.shape[1]),
                    "feature_columns": list(feature_columns),
                    "design_matrix_columns": list(x.columns),
                    "target_unique_count": target_unique_count,
                    "fit_prediction_unique_count": prediction_unique_count,
                    "blockers": [
                        {
                            "code": "degenerate_historical_compatible_factor_map",
                            "message": "The current-label factor map produced no usable cross-sectional score variation.",
                        }
                    ],
                }
            )
            continue
        observed_percentile = _percentile_rank(y)
        predicted_percentile = _percentile_rank(fit_predictions)
        summaries.append(
            {
                "target": target,
                "status": "fit_historical_compatible_current_cross_section",
                "n": int(len(fit_frame)),
                "feature_count": int(len(feature_columns)),
                "design_matrix_column_count": int(x.shape[1]),
                "feature_columns": list(feature_columns),
                "design_matrix_columns": list(x.columns),
                "fit_spearman": _safe_spearman(y, fit_predictions),
                "fit_percentile_r2": float(r2_score(observed_percentile, predicted_percentile)),
                "fit_mae": float(mean_absolute_error(y, fit_predictions)),
                "blockers": [],
            }
        )
        model_specs[target] = {
            "target": target,
            "model": model,
            "feature_columns": list(feature_columns),
            "design_columns": list(x.columns),
        }
    return model_specs, summaries


def score_historical_matrices(
    matrices: Mapping[pd.Timestamp, pd.DataFrame],
    model_specs: Mapping[str, Mapping[str, Any]],
    *,
    sector_map: pd.Series,
    public_add_dates: pd.Series | None = None,
    max_feature_null_count: int = 2,
) -> tuple[pd.DataFrame, dict[str, dict[pd.Timestamp, pd.DataFrame]], dict[str, Any]]:
    score_rows: list[dict[str, Any]] = []
    scored_matrices: dict[str, dict[pd.Timestamp, pd.DataFrame]] = {
        target: {} for target in [*model_specs, *(spec["target"] for spec in HISTORICAL_BENCHMARK_CONTROLS.values())]
    }
    add_dates = public_add_dates if public_add_dates is not None else pd.Series(dtype="datetime64[ns]")
    membership_excluded_rows = 0
    degenerate_score_months: dict[str, int] = {target: 0 for target in model_specs}
    degenerate_score_rows = 0

    for rebalance_date, matrix in sorted(matrices.items()):
        if matrix.empty:
            continue
        base = add_backcast_context(matrix, sector_map)
        base, excluded_count = apply_public_add_date_filter(base, pd.Timestamp(rebalance_date), add_dates)
        membership_excluded_rows += excluded_count
        if "feature_null_count" in base:
            base = base[pd.to_numeric(base["feature_null_count"], errors="coerce") <= max_feature_null_count].copy()
        if base.empty:
            continue

        for target, spec in model_specs.items():
            x = _design_matrix(base, spec["feature_columns"]).reindex(
                columns=spec["design_columns"],
                fill_value=0.0,
            )
            scores = pd.Series(spec["model"].predict(x), index=base.index, dtype=float)
            valid_scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid_scores) < 2 or valid_scores.nunique(dropna=True) < 2:
                degenerate_score_months[target] = degenerate_score_months.get(target, 0) + 1
                degenerate_score_rows += int(len(base))
                continue
            signal_name = historical_signal_name_for_target(target)
            scored = base.loc[scores.notna()].copy()
            scored["score"] = scores.reindex(scored.index)
            scored_matrices[target][pd.Timestamp(rebalance_date)] = scored
            for ticker, row in scored.iterrows():
                score_rows.append(
                    {
                        "date": pd.Timestamp(rebalance_date).normalize(),
                        "ticker": ticker,
                        "target": target,
                        "signal_name": signal_name,
                        "score": float(row["score"]),
                        "sector": row.get("sector"),
                        "market_cap": row.get("market_cap"),
                        "feature_null_count": row.get("feature_null_count"),
                        "public_sp500_add_date": row.get("public_sp500_add_date"),
                        "approximate_membership_eligible": row.get("approximate_membership_eligible"),
                    }
                )

        benchmark_scores = build_composite_vqmia(base)
        for source_column, benchmark_spec in HISTORICAL_BENCHMARK_CONTROLS.items():
            target = str(benchmark_spec["target"])
            signal_name = str(benchmark_spec["signal_name"])
            score_source = benchmark_scores if source_column in benchmark_scores.columns else base
            if source_column not in score_source.columns:
                continue
            scores = pd.to_numeric(score_source[source_column], errors="coerce").astype(float)
            valid_scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid_scores) < 2 or valid_scores.nunique(dropna=True) < 2:
                degenerate_score_months[target] = degenerate_score_months.get(target, 0) + 1
                degenerate_score_rows += int(len(base))
                continue
            scored = base.copy()
            scored["score"] = scores
            scored_matrices[target][pd.Timestamp(rebalance_date)] = scored
            for ticker, row in scored.iterrows():
                score_rows.append(
                    {
                        "date": pd.Timestamp(rebalance_date).normalize(),
                        "ticker": ticker,
                        "target": target,
                        "signal_name": signal_name,
                        "score": float(row["score"]),
                        "sector": row.get("sector"),
                        "market_cap": row.get("market_cap"),
                        "feature_null_count": row.get("feature_null_count"),
                        "public_sp500_add_date": row.get("public_sp500_add_date"),
                        "approximate_membership_eligible": row.get("approximate_membership_eligible"),
                    }
                )

    scores = pd.DataFrame(score_rows)
    if scores.empty:
        scores = _empty_frame(SCORE_COLUMNS)
    membership_summary = {
        "filter": "current_tickers_after_public_add_date",
        "source": repo_relative(DEFAULT_SP500_CHANGES),
        "excluded_matrix_rows_before_feature_filter": int(membership_excluded_rows),
        "degenerate_score_months": {target: int(count) for target, count in degenerate_score_months.items() if count},
        "degenerate_score_rows": int(degenerate_score_rows),
        "unique_tickers_with_public_add_date": int(len(add_dates.dropna())) if not add_dates.empty else 0,
        "point_in_time_membership": False,
    }
    return scores, scored_matrices, membership_summary


def summarize_historical_feature_coverage(
    matrices: Mapping[pd.Timestamp, pd.DataFrame],
    feature_columns: Sequence[str],
    *,
    public_add_dates: pd.Series | None = None,
    max_feature_null_count: int = 2,
) -> list[dict[str, Any]]:
    if not matrices or not feature_columns:
        return []
    frames: list[pd.DataFrame] = []
    add_dates = public_add_dates if public_add_dates is not None else pd.Series(dtype="datetime64[ns]")
    for rebalance_date, matrix in matrices.items():
        if matrix.empty:
            continue
        work = matrix.copy()
        if "market_cap" in work.columns and "log_market_cap" in feature_columns:
            market_cap = pd.to_numeric(work["market_cap"], errors="coerce")
            work["log_market_cap"] = np.where(market_cap > 0.0, np.log(market_cap), np.nan)
        work, _ = apply_public_add_date_filter(work, pd.Timestamp(rebalance_date), add_dates)
        if "feature_null_count" in work.columns:
            work = work[pd.to_numeric(work["feature_null_count"], errors="coerce") <= max_feature_null_count]
        frames.append(work.reindex(columns=list(feature_columns)))
    if not frames:
        return []
    combined = pd.concat(frames, axis=0, ignore_index=True)
    row_count = int(len(combined))
    return [
        {
            "feature": feature,
            "row_count": row_count,
            "non_null": int(pd.to_numeric(combined[feature], errors="coerce").notna().sum()) if feature in combined else 0,
            "coverage": float(pd.to_numeric(combined[feature], errors="coerce").notna().mean()) if feature in combined and row_count else 0.0,
        }
        for feature in feature_columns
    ]


def build_point_in_time_matrices(
    tickers: Sequence[str],
    *,
    start_year: int,
    end_year: int,
    refresh_sec_cache: bool = False,
    refresh_price_cache: bool = False,
    user_agent_name: str = DEFAULT_SEC_NAME,
    user_agent_email: str = DEFAULT_SEC_EMAIL,
) -> tuple[dict[pd.Timestamp, pd.DataFrame], list[pd.Timestamp], PricePanels]:
    sec_client = SecClient(user_agent_name=user_agent_name, user_agent_email=user_agent_email)
    histories = historical.load_ticker_histories(
        tickers,
        sec_client=sec_client,
        refresh_sec_cache=refresh_sec_cache,
    )
    price_start_year = max(start_year - 2, 1990)
    price_panels = historical.load_price_panel_history(
        tickers,
        start=f"{price_start_year}-01-01",
        end=f"{end_year + 1}-01-31",
        refresh=refresh_price_cache,
    )
    rebalance_dates = build_rebalance_dates(
        price_panels.adjusted_close,
        start_year=start_year,
        end_year=end_year,
    )
    matrices = {
        rebalance_date: historical.build_point_in_time_feature_matrix(histories, price_panels, rebalance_date)
        for rebalance_date in rebalance_dates
    }
    return matrices, rebalance_dates, price_panels


def _concat_artifacts(artifact_sets: Sequence[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for name, columns in AUDIT_COLUMNS.items():
        frames = [artifacts[name] for artifacts in artifact_sets if name in artifacts and not artifacts[name].empty]
        output[name] = pd.concat(frames, ignore_index=True) if frames else _empty_frame(columns)
    return output


def summarize_backcast_performance(monthly_returns: pd.DataFrame) -> list[dict[str, Any]]:
    if monthly_returns.empty:
        return []
    work = monthly_returns[pd.to_numeric(monthly_returns["cost_bps_one_way"], errors="coerce") == 0].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    summaries: list[dict[str, Any]] = []
    group_columns = ["signal_name"]
    if "portfolio_mode" in work.columns:
        group_columns.append("portfolio_mode")
    for keys, group in work.groupby(group_columns, dropna=False):
        if isinstance(keys, tuple):
            signal_name = keys[0]
            portfolio_mode = keys[1] if len(keys) > 1 else None
        else:
            signal_name = keys
            portfolio_mode = None
        pivot = group.pivot_table(index="date", columns="bucket", values="net_return", aggfunc="mean").sort_index()
        q1 = pivot["Q1"] if "Q1" in pivot else pd.Series(dtype=float)
        q5 = pivot["Q5"] if "Q5" in pivot else pd.Series(dtype=float)
        spread = (q1 - q5).dropna()
        q1_vs_q5 = {
            "overlap_months": 0,
            "monthly_hit_rate": None,
            "annualized_return_gap": None,
            "cumulative_return_gap": None,
            "beats": False,
        }
        if not q1.empty and not q5.empty:
            from backtest import aligned_outperformance

            q1_vs_q5 = aligned_outperformance(q1, q5)
        summaries.append(
            {
                "signal_name": signal_name,
                "portfolio_mode": portfolio_mode,
                "months": int(len(pivot)),
                "q1_metrics": compute_performance_metrics(q1),
                "q5_metrics": compute_performance_metrics(q5),
                "q1_q5_spread_metrics": compute_performance_metrics(spread),
                "q1_vs_q5": q1_vs_q5,
                "median_name_count": float(group["name_count"].median()) if "name_count" in group and group["name_count"].notna().any() else None,
            }
        )
    return summaries


def _rank_ic_return_columns(horizons: Sequence[int] = DEFAULT_RANK_IC_HORIZONS) -> list[str]:
    return [f"return_{int(horizon)}m" for horizon in horizons]


def _forward_return_from_series(
    series: pd.Series,
    rebalance_date: pd.Timestamp,
    exit_date: pd.Timestamp,
) -> float | None:
    if series.empty:
        return None
    entry_price = first_price_after(series, rebalance_date)
    exit_price = first_price_after(series, exit_date)
    if entry_price in (None, 0.0) or exit_price is None:
        return None
    return float((exit_price / entry_price) - 1.0)


def _historical_forward_return_frame(
    pairs: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp],
    price_panels: PricePanels,
    *,
    horizons: Sequence[int] = DEFAULT_RANK_IC_HORIZONS,
) -> pd.DataFrame:
    return_columns = _rank_ic_return_columns(horizons)
    output = pairs[["date", "ticker"]].drop_duplicates().copy()
    for column in return_columns:
        output[column] = np.nan
    if output.empty:
        return output

    rebalance_list = sorted({pd.Timestamp(date).normalize() for date in rebalance_dates})
    rebalance_index = {date: index for index, date in enumerate(rebalance_list)}
    adjusted_prices = price_panels.adjusted_close
    if adjusted_prices.empty:
        return output

    ticker_set = {normalize_ticker(str(ticker)) for ticker in output["ticker"].dropna()}
    series_cache = {
        ticker: adjusted_prices[ticker].dropna()
        for ticker in ticker_set
        if ticker in adjusted_prices.columns and not adjusted_prices[ticker].dropna().empty
    }
    for row in output.itertuples(index=True):
        date_value = pd.Timestamp(row.date).normalize()
        ticker = normalize_ticker(str(row.ticker))
        series = series_cache.get(ticker)
        start_index = rebalance_index.get(date_value)
        if series is None or start_index is None:
            continue
        for horizon in horizons:
            horizon = int(horizon)
            exit_index = start_index + horizon
            if exit_index >= len(rebalance_list):
                continue
            output.at[row.Index, f"return_{horizon}m"] = _forward_return_from_series(
                series,
                date_value,
                rebalance_list[exit_index],
            )
    return output


def build_historical_rank_ic_input(
    scores: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp],
    price_panels: PricePanels,
    *,
    horizons: Sequence[int] = DEFAULT_RANK_IC_HORIZONS,
) -> pd.DataFrame:
    return_columns = _rank_ic_return_columns(horizons)
    if scores.empty:
        return pd.DataFrame(columns=["date", "ticker", "sector", *return_columns])

    work = scores.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work["ticker"] = work["ticker"].fillna("").astype(str).map(normalize_ticker)
    work["signal_name"] = work["signal_name"].fillna("").astype(str)
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    work = work[(work["date"].notna()) & (work["ticker"].str.len() > 0) & (work["signal_name"].str.len() > 0)]
    if work.empty:
        return pd.DataFrame(columns=["date", "ticker", "sector", *return_columns])

    signal_columns = sorted(work["signal_name"].dropna().unique().tolist())
    score_wide = work.pivot_table(
        index=["date", "ticker"],
        columns="signal_name",
        values="score",
        aggfunc="first",
    ).reset_index()
    score_wide.columns.name = None
    score_wide = score_wide.reindex(columns=["date", "ticker", *signal_columns])

    sectors = work.groupby(["date", "ticker"], as_index=False)["sector"].first()
    returns = _historical_forward_return_frame(
        score_wide[["date", "ticker"]],
        rebalance_dates,
        price_panels,
        horizons=horizons,
    )
    diagnostic_input = score_wide.merge(sectors, on=["date", "ticker"], how="left")
    diagnostic_input = diagnostic_input.merge(returns, on=["date", "ticker"], how="left")
    return diagnostic_input[["date", "ticker", "sector", *signal_columns, *return_columns]]


def _ensure_artifact_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=list(columns))
    return frame.reindex(columns=list(columns))


def build_historical_rank_ic_artifacts(
    scores: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp],
    price_panels: PricePanels,
    *,
    horizons: Sequence[int] = DEFAULT_RANK_IC_HORIZONS,
) -> dict[str, pd.DataFrame]:
    diagnostic_input = build_historical_rank_ic_input(
        scores,
        rebalance_dates,
        price_panels,
        horizons=horizons,
    )
    return_columns = _rank_ic_return_columns(horizons)
    signal_columns = [
        column
        for column in diagnostic_input.columns
        if str(column).startswith("experiment_b_") or str(column).startswith("benchmark_")
    ]
    if diagnostic_input.empty or not signal_columns:
        return {name: _empty_frame(columns) for name, columns in RANK_IC_COLUMNS.items()}

    rank_ic = rank_ic_diagnostics(
        diagnostic_input,
        date_column="date",
        signal_columns=signal_columns,
        return_columns=return_columns,
    )
    sector_ic = rank_ic_by_sector(
        diagnostic_input,
        date_column="date",
        signal_columns=signal_columns,
        return_columns=return_columns,
    )
    artifacts = {
        "rank_ic": rank_ic,
        "rank_ic_summary": summarize_rank_ic(rank_ic),
        "rank_ic_by_year": summarize_rank_ic_by_year(rank_ic),
        "rank_ic_by_sector": summarize_rank_ic_by_sector(sector_ic),
        "rank_ic_coverage": rank_ic_coverage(
            diagnostic_input,
            date_column="date",
            signal_columns=signal_columns,
            return_columns=return_columns,
        ),
    }
    return {
        name: _ensure_artifact_columns(frame, RANK_IC_COLUMNS[name])
        for name, frame in artifacts.items()
    }


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(output):
        return None
    return output


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def summarize_approximate_membership_gap(
    panel_path: Path,
    summary_path: Path,
    rebalance_dates: Sequence[pd.Timestamp],
) -> dict[str, Any]:
    artifacts = {
        "approximate_membership": repo_relative(panel_path),
        "approximate_membership_summary": repo_relative(summary_path),
    }
    output: dict[str, Any] = {
        "status": "missing_approximate_membership_artifact",
        "point_in_time_membership": False,
        "artifacts": artifacts,
        "blockers": [
            {
                "code": "approximate_membership_artifact_missing",
                "message": "The approximate selected-changes membership gap artifact was not available for this backcast run.",
            }
        ],
    }
    if not panel_path.exists() or not summary_path.exists():
        output["missing_paths"] = [
            repo_relative(path) for path in (panel_path, summary_path) if not path.exists()
        ]
        return output

    try:
        source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        panel = pd.read_parquet(panel_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        output["status"] = "unreadable_approximate_membership_artifact"
        output["blockers"] = [
            {
                "code": "approximate_membership_artifact_unreadable",
                "message": f"The approximate selected-changes membership gap artifact could not be read: {exc}",
            }
        ]
        return output

    output.update(
        {
            "status": source_summary.get("status", "approximate_gap_analysis_not_point_in_time_membership"),
            "row_count": _optional_int(source_summary.get("row_count")),
            "ticker_count": _optional_int(source_summary.get("ticker_count")),
            "date_count": _optional_int(source_summary.get("date_count")),
            "current_security_master_ticker_count": _optional_int(
                source_summary.get("current_security_master_ticker_count")
            ),
            "missing_from_current_security_master_ticker_count": _optional_int(
                source_summary.get("missing_from_current_security_master_ticker_count")
            ),
            "missing_with_sec_company_ticker_match_count": _optional_int(
                source_summary.get("missing_with_sec_company_ticker_match_count")
            ),
            "missing_without_sec_company_ticker_match_count": _optional_int(
                source_summary.get("missing_without_sec_company_ticker_match_count")
            ),
            "average_monthly_missing_from_security_master_rate": _optional_float(
                source_summary.get("average_monthly_missing_from_security_master_rate")
            ),
            "max_monthly_missing_from_security_master_rate": _optional_float(
                source_summary.get("max_monthly_missing_from_security_master_rate")
            ),
            "claim_limit": source_summary.get("claim_limit"),
            "blockers": source_summary.get("blockers", []),
        }
    )
    if source_summary.get("point_in_time_membership") is not False:
        output["point_in_time_membership"] = False

    if panel.empty or "date" not in panel or "in_current_security_master" not in panel:
        output["backcast_rebalance_overlap"] = {
            "rebalance_date_count": int(len(rebalance_dates)),
            "overlap_month_count": 0,
            "average_missing_from_current_security_master_rate": None,
            "max_missing_from_current_security_master_rate": None,
        }
        return output

    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work = work[work["date"].notna()].copy()
    work["rebalance_month"] = work["date"].dt.to_period("M")
    if "approximate_member" in work:
        work = work[work["approximate_member"].fillna(False).astype(bool)].copy()
    rebalance_months = {pd.Timestamp(date).to_period("M") for date in rebalance_dates}
    overlap = work[work["rebalance_month"].isin(rebalance_months)].copy()
    if overlap.empty:
        output["backcast_rebalance_overlap"] = {
            "rebalance_date_count": int(len(rebalance_months)),
            "alignment": "calendar_month",
            "overlap_month_count": 0,
            "average_missing_from_current_security_master_rate": None,
            "max_missing_from_current_security_master_rate": None,
        }
        return output

    overlap["in_current_security_master"] = overlap["in_current_security_master"].fillna(False).astype(bool)
    overlap["missing_from_current_security_master"] = ~overlap["in_current_security_master"]
    if "company_tickers_match" in overlap:
        overlap["company_tickers_match"] = overlap["company_tickers_match"].fillna(False).astype(bool)
    else:
        overlap["company_tickers_match"] = False
    overlap["missing_with_sec_company_ticker_match"] = (
        overlap["missing_from_current_security_master"] & overlap["company_tickers_match"]
    )
    monthly = (
        overlap.groupby("rebalance_month")
        .agg(
            approximate_member_count=("ticker", "nunique"),
            current_security_master_member_count=("in_current_security_master", "sum"),
            missing_from_current_security_master_count=("missing_from_current_security_master", "sum"),
            missing_with_sec_company_ticker_match_count=("missing_with_sec_company_ticker_match", "sum"),
        )
        .reset_index()
    )
    monthly["date"] = monthly["rebalance_month"].dt.to_timestamp(how="end").dt.normalize()
    monthly["missing_from_current_security_master_rate"] = np.where(
        monthly["approximate_member_count"] > 0,
        monthly["missing_from_current_security_master_count"] / monthly["approximate_member_count"],
        np.nan,
    )
    missing_rate = pd.to_numeric(monthly["missing_from_current_security_master_rate"], errors="coerce")
    output["backcast_rebalance_overlap"] = {
        "rebalance_date_count": int(len(rebalance_months)),
        "alignment": "calendar_month",
        "overlap_month_count": int(len(monthly)),
        "start_date": monthly["date"].min().date().isoformat(),
        "end_date": monthly["date"].max().date().isoformat(),
        "average_approximate_member_count": _optional_float(monthly["approximate_member_count"].mean()),
        "average_missing_from_current_security_master_count": _optional_float(
            monthly["missing_from_current_security_master_count"].mean()
        ),
        "max_missing_from_current_security_master_count": _optional_int(
            monthly["missing_from_current_security_master_count"].max()
        ),
        "average_missing_from_current_security_master_rate": _optional_float(missing_rate.mean()),
        "max_missing_from_current_security_master_rate": _optional_float(missing_rate.max()),
    }
    return output


def _write_artifacts(
    *,
    scores: pd.DataFrame,
    artifacts: Mapping[str, pd.DataFrame],
    scores_path: Path,
    monthly_returns_path: Path,
    holdings_path: Path,
    turnover_path: Path,
    exposures_path: Path,
) -> None:
    for path in (scores_path, monthly_returns_path, holdings_path, turnover_path, exposures_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(scores_path, index=False)
    artifacts["monthly_returns"].to_parquet(monthly_returns_path, index=False)
    artifacts["holdings"].to_parquet(holdings_path, index=False)
    artifacts["turnover"].to_parquet(turnover_path, index=False)
    artifacts["exposures"].to_parquet(exposures_path, index=False)


def _write_rank_ic_artifacts(
    *,
    artifacts: Mapping[str, pd.DataFrame],
    rank_ic_path: Path,
    rank_ic_summary_path: Path,
    rank_ic_by_year_path: Path,
    rank_ic_by_sector_path: Path,
    rank_ic_coverage_path: Path,
) -> None:
    for path in (
        rank_ic_path,
        rank_ic_summary_path,
        rank_ic_by_year_path,
        rank_ic_by_sector_path,
        rank_ic_coverage_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    artifacts["rank_ic"].to_parquet(rank_ic_path, index=False)
    artifacts["rank_ic_summary"].to_parquet(rank_ic_summary_path, index=False)
    artifacts["rank_ic_by_year"].to_parquet(rank_ic_by_year_path, index=False)
    artifacts["rank_ic_by_sector"].to_parquet(rank_ic_by_sector_path, index=False)
    artifacts["rank_ic_coverage"].to_parquet(rank_ic_coverage_path, index=False)


def run_experiment_b_historical_backcast(
    *,
    label_panel_path: Path = DEFAULT_OUTPUT_PATH,
    edgar_features_path: Path = DEFAULT_EDGAR_PATH,
    tickers_path: Path = DEFAULT_TICKERS_PATH,
    sector_map_path: Path = DEFAULT_SECTOR_MAP,
    sp500_changes_path: Path = DEFAULT_SP500_CHANGES,
    approximate_membership_path: Path = DEFAULT_APPROX_MEMBERSHIP,
    approximate_membership_summary_path: Path = DEFAULT_APPROX_MEMBERSHIP_SUMMARY,
    scores_path: Path = DEFAULT_SCORES,
    monthly_returns_path: Path = DEFAULT_MONTHLY_RETURNS,
    holdings_path: Path = DEFAULT_HOLDINGS,
    turnover_path: Path = DEFAULT_TURNOVER,
    exposures_path: Path = DEFAULT_EXPOSURES,
    rank_ic_path: Path = DEFAULT_RANK_IC,
    rank_ic_summary_path: Path = DEFAULT_RANK_IC_SUMMARY,
    rank_ic_by_year_path: Path = DEFAULT_RANK_IC_BY_YEAR,
    rank_ic_by_sector_path: Path = DEFAULT_RANK_IC_BY_SECTOR,
    rank_ic_coverage_path: Path = DEFAULT_RANK_IC_COVERAGE,
    summary_path: Path = DEFAULT_SUMMARY,
    metadata_path: Path = DEFAULT_METADATA,
    targets: Sequence[str] = DEFAULT_TARGETS,
    start_year: int = 2012,
    end_year: int = 2025,
    cost_bps_levels: Sequence[int] = DEFAULT_COST_BPS,
    refresh_sec_cache: bool = False,
    refresh_price_cache: bool = False,
    user_agent_name: str = DEFAULT_SEC_NAME,
    user_agent_email: str = DEFAULT_SEC_EMAIL,
    matrices: Mapping[pd.Timestamp, pd.DataFrame] | None = None,
    rebalance_dates: Sequence[pd.Timestamp] | None = None,
    price_panels: PricePanels | None = None,
) -> dict[str, Any]:
    label_panel = _read_frame(label_panel_path)
    edgar_payload = load_edgar_payload(edgar_features_path) if edgar_features_path.exists() else {}
    training_frame = build_training_frame(label_panel, edgar_payload, targets=targets)
    sector_map = load_sector_map(sector_map_path)
    public_add_dates = load_public_sp500_add_dates(sp500_changes_path)

    if matrices is None or rebalance_dates is None or price_panels is None:
        tickers = load_tickers(tickers_path)
        matrices, rebalance_dates, price_panels = build_point_in_time_matrices(
            tickers,
            start_year=start_year,
            end_year=end_year,
            refresh_sec_cache=refresh_sec_cache,
            refresh_price_cache=refresh_price_cache,
            user_agent_name=user_agent_name,
            user_agent_email=user_agent_email,
        )
    else:
        matrices = {pd.Timestamp(date): frame for date, frame in matrices.items()}
        rebalance_dates = [pd.Timestamp(date) for date in rebalance_dates]

    feature_columns = select_historical_compatible_features(training_frame, matrices)
    model_specs, fit_summaries = fit_historical_compatible_models(
        training_frame,
        targets=targets,
        feature_columns=feature_columns,
    )
    blockers: list[dict[str, str]] = []
    if not model_specs:
        blockers.append(
            {
                "code": "no_historical_compatible_factor_maps",
                "message": "No target had enough current labels and historical-compatible public features to fit.",
            }
        )

    scores, scored_matrices, membership_summary = score_historical_matrices(
        matrices,
        model_specs,
        sector_map=sector_map,
        public_add_dates=public_add_dates,
    )
    approximate_membership_gap = summarize_approximate_membership_gap(
        approximate_membership_path,
        approximate_membership_summary_path,
        rebalance_dates,
    )
    artifact_sets: list[dict[str, pd.DataFrame]] = []
    if not scores.empty and len(rebalance_dates) >= 2:
        for target, target_matrices in scored_matrices.items():
            if not target_matrices:
                continue
            artifact_sets.append(
                run_audited_signal_backtest(
                    signal_name=historical_signal_name_for_target(str(target)),
                    score_column="score",
                    matrices=target_matrices,
                    rebalance_dates=list(rebalance_dates),
                    price_frame=price_panels,
                    cost_bps_levels=tuple(int(value) for value in cost_bps_levels),
                )
            )
    artifacts = _concat_artifacts(artifact_sets)
    if artifacts["monthly_returns"].empty and not blockers:
        blockers.append(
            {
                "code": "no_historical_backcast_returns",
                "message": "Historical scores were produced, but no one-month return windows survived the audit backtest.",
            }
        )
    rank_ic_artifacts = build_historical_rank_ic_artifacts(scores, rebalance_dates, price_panels)
    rank_ic_frame = rank_ic_artifacts["rank_ic"]
    rank_ic_usable_month_count = (
        int(rank_ic_frame.loc[pd.to_numeric(rank_ic_frame["rank_ic"], errors="coerce").notna(), "date"].nunique())
        if not rank_ic_frame.empty and "rank_ic" in rank_ic_frame.columns
        else 0
    )
    if not scores.empty and rank_ic_usable_month_count == 0 and not any(
        blocker["code"] == "no_historical_backcast_returns" for blocker in blockers
    ):
        blockers.append(
            {
                "code": "no_historical_rank_ic",
                "message": "Historical scores were produced, but no Rank IC horizon had enough paired returns and score variation.",
            }
        )

    _write_artifacts(
        scores=scores,
        artifacts=artifacts,
        scores_path=scores_path,
        monthly_returns_path=monthly_returns_path,
        holdings_path=holdings_path,
        turnover_path=turnover_path,
        exposures_path=exposures_path,
    )
    _write_rank_ic_artifacts(
        artifacts=rank_ic_artifacts,
        rank_ic_path=rank_ic_path,
        rank_ic_summary_path=rank_ic_summary_path,
        rank_ic_by_year_path=rank_ic_by_year_path,
        rank_ic_by_sector_path=rank_ic_by_sector_path,
        rank_ic_coverage_path=rank_ic_coverage_path,
    )

    missing_historical_features = sorted(set(available_feature_columns(training_frame)) - set(feature_columns))
    warnings = [
        {
            "code": "current_label_projection",
            "message": "The backcast projects one current AI-label vintage backward; it is not a repeated historical label-vintage panel.",
        },
        {
            "code": "survivor_universe",
            "message": "The ticker universe is still based on the current public-data universe and selected S&P changes, not CRSP/Compustat-quality point-in-time membership.",
        },
        {
            "code": "approximate_public_add_date_filter",
            "message": "Current constituents are excluded before their public S&P 500 add date when present in the selected-changes table, but removed/delisted names are still absent.",
        },
        {
            "code": "current_sector_map_used",
            "message": "Sector controls use the current public sector map; historical sector classifications are not available.",
        },
        {
            "code": "yfinance_price_history",
            "message": "Historical prices are sourced through Yahoo Finance via yfinance and use adjusted close for returns.",
        },
    ]
    degenerate_fit_targets = [
        str(summary["target"])
        for summary in fit_summaries
        if any(
            blocker.get("code") == "degenerate_historical_compatible_factor_map"
            for blocker in summary.get("blockers", [])
        )
    ]
    if degenerate_fit_targets:
        warnings.append(
            {
                "code": "degenerate_historical_compatible_factor_map",
                "message": "One or more target factor maps were skipped because they produced no cross-sectional score variation.",
                "targets": degenerate_fit_targets,
            }
        )
    if membership_summary.get("degenerate_score_months"):
        warnings.append(
            {
                "code": "degenerate_historical_scores_skipped",
                "message": "Some target/date score panels were skipped before backtesting because all scores were tied.",
                "degenerate_score_months": membership_summary.get("degenerate_score_months"),
            }
        )
    if missing_historical_features:
        warnings.append(
            {
                "code": "partial_historical_feature_coverage",
                "message": "Some current factor-map features were unavailable in the historical point-in-time matrix.",
                "features": missing_historical_features,
            }
        )
    if _optional_int(approximate_membership_gap.get("missing_from_current_security_master_ticker_count")):
        warnings.append(
            {
                "code": "removed_names_missing_from_backcast_universe",
                "message": "The approximate selected-changes membership artifact identifies removed historical tickers absent from the current security master; the backcast still cannot score those names.",
                "missing_from_current_security_master_ticker_count": approximate_membership_gap.get(
                    "missing_from_current_security_master_ticker_count"
                ),
            }
        )

    status = "blocked" if blockers else "historical_backcast_screen"
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment_id": "experiment_b_ai_implied_factor_portability_backcast",
        "status": status,
        "claim_ceiling": "survivor_universe_historical_portability_screen_not_production_alpha_evidence",
        "start_year": int(start_year),
        "end_year": int(end_year),
        "target_count": int(len(targets)),
        "fit_targets": fit_summaries,
        "feature_columns": list(feature_columns),
        "membership_filter": membership_summary,
        "approximate_membership_gap": approximate_membership_gap,
        "historical_feature_coverage": summarize_historical_feature_coverage(
            matrices,
            feature_columns,
            public_add_dates=public_add_dates,
        ),
        "score_row_count": int(len(scores)),
        "monthly_return_row_count": int(len(artifacts["monthly_returns"])),
        "holding_row_count": int(len(artifacts["holdings"])),
        "portfolio_modes": (
            sorted(str(mode) for mode in artifacts["monthly_returns"]["portfolio_mode"].dropna().unique())
            if "portfolio_mode" in artifacts["monthly_returns"].columns
            else []
        ),
        "rank_ic_row_count": int(len(rank_ic_artifacts["rank_ic"])),
        "rank_ic_usable_month_count": int(rank_ic_usable_month_count),
        "rank_ic_return_horizons": _rank_ic_return_columns(),
        "rank_ic_summary": _frame_records(rank_ic_artifacts["rank_ic_summary"]),
        "benchmark_signals": [str(spec["signal_name"]) for spec in HISTORICAL_BENCHMARK_CONTROLS.values()],
        "rebalance_month_count": int(max(0, len(rebalance_dates) - 1)),
        "performance": summarize_backcast_performance(artifacts["monthly_returns"]),
        "blockers": blockers,
        "warnings": warnings,
        "data_sources": [
            {
                "name": "SEC EDGAR companyfacts",
                "url": "https://data.sec.gov/api/xbrl/companyfacts/",
                "usage": "Point-in-time public accounting features from cached companyfacts JSON.",
            },
            {
                "name": "Yahoo Finance via yfinance",
                "url": "https://finance.yahoo.com",
                "usage": "Raw and adjusted close history for returns, momentum, market capitalization, and EV features.",
            },
            {
                "name": "Wikipedia S&P 500 constituents",
                "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                "usage": "Current sector map and selected public index-change caveat artifacts.",
            },
        ],
        "artifacts": {
            "scores": repo_relative(scores_path),
            "monthly_returns": repo_relative(monthly_returns_path),
            "holdings": repo_relative(holdings_path),
            "turnover": repo_relative(turnover_path),
            "exposures": repo_relative(exposures_path),
            "rank_ic": repo_relative(rank_ic_path),
            "rank_ic_summary": repo_relative(rank_ic_summary_path),
            "rank_ic_by_year": repo_relative(rank_ic_by_year_path),
            "rank_ic_by_sector": repo_relative(rank_ic_by_sector_path),
            "rank_ic_coverage": repo_relative(rank_ic_coverage_path),
            "approximate_membership": repo_relative(approximate_membership_path),
            "approximate_membership_summary": repo_relative(approximate_membership_summary_path),
            "summary": repo_relative(summary_path),
            "metadata": repo_relative(metadata_path),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    metadata = build_experiment_metadata(
        experiment_id="experiment_b_ai_implied_factor_portability_backcast",
        feature_config={
            "feature_columns": list(feature_columns),
            "targets": list(targets),
            "benchmark_signals": [str(spec["signal_name"]) for spec in HISTORICAL_BENCHMARK_CONTROLS.values()],
            "historical_compatible_features": list(HISTORICAL_BACKCAST_FEATURES),
        },
        model_config={"model": "ElasticNetCV", "fit_scope": "current labels, historical-compatible features"},
        universe_config={
            "tickers_file": repo_relative(tickers_path),
            "survivor_bias_caveat": True,
            "point_in_time_membership": False,
            "sector_map": repo_relative(sector_map_path),
            "sp500_changes": repo_relative(sp500_changes_path),
            "approximate_membership_gap": {
                "panel": repo_relative(approximate_membership_path),
                "summary": repo_relative(approximate_membership_summary_path),
                "missing_from_current_security_master_ticker_count": approximate_membership_gap.get(
                    "missing_from_current_security_master_ticker_count"
                ),
                "point_in_time_membership": False,
            },
        },
        backtest_config={
            "start_year": start_year,
            "end_year": end_year,
            "cost_bps_levels": [int(value) for value in cost_bps_levels],
            "t_plus_one_trading": True,
            "portfolio_modes": ["unconstrained", "sector_neutral"],
            "sector_neutral_weighting": "equal_sector_then_equal_name",
        },
        data_snapshot_paths={
            "edgar_features": edgar_features_path,
            "sector_map": sector_map_path,
            "sp500_changes": sp500_changes_path,
            "approximate_membership": approximate_membership_path,
            "approximate_membership_summary": approximate_membership_summary_path,
            "price_panel_cache": historical.PRICE_PANEL_CACHE_PATH,
        },
        label_snapshot_paths={"label_panel": label_panel_path},
        artifacts={
            "scores": scores_path,
            "monthly_returns": monthly_returns_path,
            "holdings": holdings_path,
            "turnover": turnover_path,
            "exposures": exposures_path,
            "rank_ic": rank_ic_path,
            "rank_ic_summary": rank_ic_summary_path,
            "rank_ic_by_year": rank_ic_by_year_path,
            "rank_ic_by_sector": rank_ic_by_sector_path,
            "rank_ic_coverage": rank_ic_coverage_path,
            "summary": summary_path,
        },
    )
    write_experiment_metadata(metadata_path, metadata)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment B historical-compatible AI factor-map backcast.")
    parser.add_argument("--label-panel", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--edgar-file", default=str(DEFAULT_EDGAR_PATH))
    parser.add_argument("--tickers-file", default=str(DEFAULT_TICKERS_PATH))
    parser.add_argument("--sector-map", default=str(DEFAULT_SECTOR_MAP))
    parser.add_argument("--sp500-changes", default=str(DEFAULT_SP500_CHANGES))
    parser.add_argument("--approximate-membership", default=str(DEFAULT_APPROX_MEMBERSHIP))
    parser.add_argument("--approximate-membership-summary", default=str(DEFAULT_APPROX_MEMBERSHIP_SUMMARY))
    parser.add_argument("--scores-output", default=str(DEFAULT_SCORES))
    parser.add_argument("--monthly-returns-output", default=str(DEFAULT_MONTHLY_RETURNS))
    parser.add_argument("--holdings-output", default=str(DEFAULT_HOLDINGS))
    parser.add_argument("--turnover-output", default=str(DEFAULT_TURNOVER))
    parser.add_argument("--exposures-output", default=str(DEFAULT_EXPOSURES))
    parser.add_argument("--rank-ic-output", default=str(DEFAULT_RANK_IC))
    parser.add_argument("--rank-ic-summary-output", default=str(DEFAULT_RANK_IC_SUMMARY))
    parser.add_argument("--rank-ic-by-year-output", default=str(DEFAULT_RANK_IC_BY_YEAR))
    parser.add_argument("--rank-ic-by-sector-output", default=str(DEFAULT_RANK_IC_BY_SECTOR))
    parser.add_argument("--rank-ic-coverage-output", default=str(DEFAULT_RANK_IC_COVERAGE))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--metadata-output", default=str(DEFAULT_METADATA))
    parser.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS))
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--cost-bps", nargs="*", type=int, default=list(DEFAULT_COST_BPS))
    parser.add_argument("--refresh-sec-cache", action="store_true")
    parser.add_argument("--refresh-price-cache", action="store_true")
    parser.add_argument("--user-agent-name", default=DEFAULT_SEC_NAME)
    parser.add_argument("--user-agent-email", default=DEFAULT_SEC_EMAIL)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_experiment_b_historical_backcast(
        label_panel_path=Path(args.label_panel),
        edgar_features_path=Path(args.edgar_file),
        tickers_path=Path(args.tickers_file),
        sector_map_path=Path(args.sector_map),
        sp500_changes_path=Path(args.sp500_changes),
        approximate_membership_path=Path(args.approximate_membership),
        approximate_membership_summary_path=Path(args.approximate_membership_summary),
        scores_path=Path(args.scores_output),
        monthly_returns_path=Path(args.monthly_returns_output),
        holdings_path=Path(args.holdings_output),
        turnover_path=Path(args.turnover_output),
        exposures_path=Path(args.exposures_output),
        rank_ic_path=Path(args.rank_ic_output),
        rank_ic_summary_path=Path(args.rank_ic_summary_output),
        rank_ic_by_year_path=Path(args.rank_ic_by_year_output),
        rank_ic_by_sector_path=Path(args.rank_ic_by_sector_output),
        rank_ic_coverage_path=Path(args.rank_ic_coverage_output),
        summary_path=Path(args.summary_output),
        metadata_path=Path(args.metadata_output),
        targets=tuple(args.targets),
        start_year=args.start_year,
        end_year=args.end_year,
        cost_bps_levels=tuple(args.cost_bps),
        refresh_sec_cache=args.refresh_sec_cache,
        refresh_price_cache=args.refresh_price_cache,
        user_agent_name=args.user_agent_name,
        user_agent_email=args.user_agent_email,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
