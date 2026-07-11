from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks import FEATURE_BLOCKS, build_composite_vqmia
from experiment_registry import build_experiment_metadata, repo_relative, write_experiment_metadata
from label_panel import load_edgar_payload
from signal_diagnostics import (
    SIGNAL_COMPARISON_COLUMNS,
    build_signal_comparison_table,
    rank_ic_by_sector,
    rank_ic_coverage,
    rank_ic_diagnostics,
    summarize_rank_ic,
    summarize_rank_ic_by_sector,
    summarize_rank_ic_by_year,
)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DEFAULT_LABEL_PANEL = RESULTS_DIR / "label_panel.parquet"
DEFAULT_EDGAR_FEATURES = ROOT / "data" / "edgar_features_full.json"
DEFAULT_STATUS = RESULTS_DIR / "v2_experiment_status.json"
DEFAULT_RANK_IC = RESULTS_DIR / "rank_ic.parquet"
DEFAULT_RANK_IC_SUMMARY = RESULTS_DIR / "rank_ic_summary.parquet"
DEFAULT_RANK_IC_BY_YEAR = RESULTS_DIR / "rank_ic_by_year.parquet"
DEFAULT_RANK_IC_BY_SECTOR = RESULTS_DIR / "rank_ic_by_sector.parquet"
DEFAULT_RANK_IC_COVERAGE = RESULTS_DIR / "rank_ic_coverage.parquet"
DEFAULT_SIGNAL_COMPARISON = RESULTS_DIR / "signal_comparison.parquet"
DEFAULT_METADATA = RESULTS_DIR / "v2_experiment_metadata.json"
SIGNAL_COLUMNS = (
    "raw_ai_implied_irr",
    "mechanical_dcf_implied_irr",
    "ai_minus_mechanical_irr",
    "factor_compressible_ai_score",
    "ai_factor_residual",
    "mechanical_adjusted_factor_score",
    "mechanical_adjusted_factor_residual",
)
BENCHMARK_SIGNAL_COLUMNS = (
    "composite_vqmia_score",
    "composite_vqmia_within_sector_score",
    "value_block_score",
    "quality_block_score",
    "momentum_block_score",
    "investment_block_score",
    "accruals_block_score",
    "balance_sheet_block_score",
    "fcf_to_ev",
    "fcf_yield",
    "book_to_market",
    "gross_profitability_assets",
    "roic",
    "roe",
    "operating_margin",
    "momentum_12_1",
    "price_return_6m",
    "price_return_1m",
    "asset_growth_1y",
    "cash_earnings_gap",
    "accruals",
    "debt_to_equity",
    "current_ratio",
)
ALL_SIGNAL_COLUMNS = (*SIGNAL_COLUMNS, *BENCHMARK_SIGNAL_COLUMNS)
KEY_SIGNAL_COMPARISON_LABELS = {
    "raw_ai_implied_irr": "AI IRR",
    "ai_factor_residual": "AI residual",
    "mechanical_dcf_implied_irr": "Mechanical IRR",
    "composite_vqmia_score": "Composite VQMIA",
    "fcf_to_ev": "FCF/EV",
}
COST_BPS_LEVELS = (0, 10, 25, 50)
RETURN_COLUMN_RE = re.compile(r"^return_\d+m$")


def assign_buckets(scores: pd.Series) -> pd.Series:
    ordered = scores.dropna().sort_values(ascending=False, kind="mergesort")
    if ordered.empty:
        return pd.Series(index=scores.index, dtype="object")
    bucket_count = 5 if len(ordered) >= 50 else 3 if len(ordered) >= 3 else 1
    labels = ["Q1", "Q2", "Q3", "Q4", "Q5"] if bucket_count == 5 else ["Q1", "Q3", "Q5"] if bucket_count == 3 else ["Q1"]
    boundaries = (pd.Series(range(len(ordered)), index=ordered.index) * bucket_count // len(ordered)).astype(int)
    buckets = pd.Series([labels[index] for index in boundaries], index=ordered.index, dtype="object")
    return buckets.reindex(scores.index)


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def is_forward_return_column(column: str) -> bool:
    return bool(RETURN_COLUMN_RE.match(column))


def load_forward_returns(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    frame = _read_frame(path)
    required = {"ticker", "date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Forward return file is missing required columns: {sorted(missing)}")
    return frame


def edgar_feature_frame(edgar_features_path: Path | None) -> pd.DataFrame:
    if edgar_features_path is None or not edgar_features_path.exists():
        return pd.DataFrame()
    payload = load_edgar_payload(edgar_features_path)
    rows: list[dict[str, Any]] = []
    block_features = {feature for block in FEATURE_BLOCKS for feature in block.features}
    passthrough_columns = {*block_features, "sector", "market_cap"}
    for ticker, record in payload.items():
        row: dict[str, Any] = {"ticker": ticker}
        for column in passthrough_columns:
            if column in record:
                row[column] = record[column]
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("ticker").set_index("ticker")


def add_benchmark_signals(label_panel: pd.DataFrame, feature_frame: pd.DataFrame) -> pd.DataFrame:
    if label_panel.empty or feature_frame.empty:
        return label_panel.copy()

    output = label_panel.copy()
    output = output.merge(
        feature_frame,
        how="left",
        left_on="ticker",
        right_index=True,
        suffixes=("", "_feature"),
    )
    for column in feature_frame.columns:
        feature_column = f"{column}_feature"
        if feature_column not in output.columns:
            continue
        if column in output.columns:
            output[column] = output[column].combine_first(output[feature_column])
        else:
            output[column] = output[feature_column]
        output = output.drop(columns=[feature_column])

    source_columns = [column for column in feature_frame.columns if column in output.columns and column != "sector"]
    if source_columns:
        output["feature_null_count"] = output[source_columns].isna().sum(axis=1)
    scores = build_composite_vqmia(output)
    for column in scores.columns:
        output[column] = scores[column]
    return output


def build_rank_ic_input(label_panel: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.DataFrame:
    label_frame = label_panel.copy()
    label_frame["date"] = pd.to_datetime(label_frame["report_date"], errors="coerce").dt.date.astype("string")
    returns = forward_returns.copy()
    returns["date"] = pd.to_datetime(returns["date"], errors="coerce").dt.date.astype("string")
    return label_frame.merge(returns, how="inner", on=["ticker", "date"], suffixes=("", "_return"))


def empty_artifact(path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=list(columns)).to_parquet(path, index=False)


def _entry_exit_prices(row_dict: dict[str, Any], return_column: str) -> tuple[Any, Any]:
    entry_price = row_dict.get("entry_price_1m")
    exit_price = row_dict.get("exit_price_1m")
    suffix = return_column.removeprefix("return_")
    if entry_price is None:
        entry_price = row_dict.get(f"entry_price_{suffix}")
    if exit_price is None:
        exit_price = row_dict.get(f"exit_price_{suffix}")
    return entry_price, exit_price


def _append_bucket_rows(
    *,
    bucket_frame: pd.DataFrame,
    bucket: str,
    date_value: Any,
    signal_name: str,
    portfolio_mode: str,
    weighting_method: str,
    return_column: str,
    cost_bps_levels: Sequence[int],
    holding_rows: list[dict[str, Any]],
    monthly_rows: list[dict[str, Any]],
    turnover_rows: list[dict[str, Any]],
    exposure_rows: list[dict[str, Any]],
    signal_columns: Sequence[str],
) -> None:
    if bucket_frame.empty:
        return
    frame = bucket_frame.copy()
    if "weight" not in frame.columns:
        frame["weight"] = 1.0 / len(frame)
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").astype(float)
    frame = frame[frame["weight"].notna() & (frame["weight"] > 0.0)].copy()
    if frame.empty:
        return

    total_weight = float(frame["weight"].sum())
    if total_weight <= 0.0:
        return
    frame["weight"] = frame["weight"] / total_weight
    gross_return = float((frame["raw_return"].astype(float) * frame["weight"]).sum())
    turnover = 1.0
    sector_weights = (
        frame.groupby("sector", dropna=False)["weight"].sum().to_dict()
        if "sector" in frame.columns
        else {}
    )
    average_market_cap = (
        float((frame["market_cap"].astype(float) * frame["weight"]).sum())
        if "market_cap" in frame.columns and frame["market_cap"].notna().any()
        else None
    )
    exposure_rows.append(
        {
            "date": date_value,
            "signal_name": signal_name,
            "portfolio_mode": portfolio_mode,
            "weighting_method": weighting_method,
            "bucket": bucket,
            "sector_weights": json.dumps({str(key): float(value) for key, value in sector_weights.items()}),
            "average_market_cap": average_market_cap,
            "name_count": int(len(frame)),
        }
    )
    for cost_bps in cost_bps_levels:
        cost_rate = float(cost_bps) / 10000.0
        transaction_cost_drag = turnover * cost_rate
        net_return = gross_return - transaction_cost_drag
        monthly_rows.append(
            {
                "date": date_value,
                "signal_name": signal_name,
                "portfolio_mode": portfolio_mode,
                "weighting_method": weighting_method,
                "bucket": bucket,
                "cost_bps_one_way": int(cost_bps),
                "gross_return": gross_return,
                "transaction_cost_drag": transaction_cost_drag,
                "net_return": net_return,
                "name_count": int(len(frame)),
                "return_column": return_column,
            }
        )
        turnover_rows.append(
            {
                "date": date_value,
                "signal_name": signal_name,
                "portfolio_mode": portfolio_mode,
                "weighting_method": weighting_method,
                "bucket": bucket,
                "cost_bps_one_way": int(cost_bps),
                "turnover": turnover,
                "transaction_cost_drag": transaction_cost_drag,
            }
        )
        for row in frame.itertuples(index=False):
            row_dict = row._asdict()
            entry_price, exit_price = _entry_exit_prices(row_dict, return_column)
            raw_return = float(row_dict["raw_return"])
            weight = float(row_dict["weight"])
            holding_transaction_cost = cost_rate * weight
            holding = {
                "date": date_value,
                "ticker": row_dict["ticker"],
                "sector": row_dict.get("sector"),
                "market_cap": row_dict.get("market_cap"),
                "score": row_dict["score"],
                "signal_name": signal_name,
                "portfolio_mode": portfolio_mode,
                "weighting_method": weighting_method,
                "bucket": bucket,
                "weight": weight,
                "feature_null_count": row_dict.get("feature_null_count"),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "raw_return": raw_return,
                "cost_bps_one_way": int(cost_bps),
                "transaction_cost": holding_transaction_cost,
                "net_return": raw_return - holding_transaction_cost,
                "return_column": return_column,
            }
            for signal_column in dict.fromkeys((*SIGNAL_COLUMNS, *signal_columns)):
                if signal_column in row_dict:
                    holding[signal_column] = row_dict[signal_column]
            holding_rows.append(holding)


def _sector_neutral_bucket_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "sector" not in frame.columns or frame["sector"].nunique(dropna=True) < 2:
        return pd.DataFrame(columns=[*frame.columns, "bucket", "weight"])

    pieces: list[pd.DataFrame] = []
    for sector, sector_frame in frame.dropna(subset=["sector"]).groupby("sector", dropna=False):
        if len(sector_frame) < 3:
            continue
        sector_bucketed = sector_frame.copy()
        sector_bucketed["bucket"] = assign_buckets(sector_bucketed["score"])
        sector_bucketed = sector_bucketed[sector_bucketed["bucket"].notna()].copy()
        if sector_bucketed.empty:
            continue
        sector_bucketed["_sector_key"] = str(sector)
        pieces.append(sector_bucketed)
    if not pieces:
        return pd.DataFrame(columns=[*frame.columns, "bucket", "weight"])

    bucketed = pd.concat(pieces, axis=0, ignore_index=False)
    weight_parts: list[pd.Series] = []
    for bucket, bucket_frame in bucketed.groupby("bucket", dropna=False):
        sectors = sorted(bucket_frame["_sector_key"].dropna().unique().tolist())
        if not sectors:
            continue
        sector_allocation = 1.0 / len(sectors)
        weights = pd.Series(index=bucket_frame.index, dtype=float)
        for sector in sectors:
            sector_index = bucket_frame.index[bucket_frame["_sector_key"] == sector]
            weights.loc[sector_index] = sector_allocation / len(sector_index)
        weight_parts.append(weights)
    if not weight_parts:
        return pd.DataFrame(columns=[*frame.columns, "bucket", "weight"])
    bucketed["weight"] = pd.concat(weight_parts).sort_index()
    return bucketed.drop(columns=["_sector_key"])


def build_portfolio_audit_artifacts(
    diagnostic_input: pd.DataFrame,
    *,
    signal_columns: Sequence[str],
    return_column: str,
    cost_bps_levels: Sequence[int] = COST_BPS_LEVELS,
) -> dict[str, pd.DataFrame]:
    holding_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    turnover_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []

    for date_value, date_frame in diagnostic_input.groupby("date", dropna=False):
        for signal_name in signal_columns:
            if signal_name not in date_frame.columns:
                continue
            frame = date_frame.copy()
            frame["score"] = pd.to_numeric(frame[signal_name], errors="coerce")
            frame["raw_return"] = pd.to_numeric(frame[return_column], errors="coerce")
            frame = frame[frame["score"].notna() & frame["raw_return"].notna()].copy()
            if frame.empty:
                continue
            unconstrained = frame.copy()
            unconstrained["bucket"] = assign_buckets(unconstrained["score"])
            unconstrained = unconstrained[unconstrained["bucket"].notna()].copy()
            unconstrained["weight"] = unconstrained.groupby("bucket")["ticker"].transform(lambda values: 1.0 / len(values))
            for bucket, bucket_frame in unconstrained.groupby("bucket", dropna=False):
                _append_bucket_rows(
                    bucket_frame=bucket_frame,
                    bucket=str(bucket),
                    date_value=date_value,
                    signal_name=signal_name,
                    portfolio_mode="unconstrained",
                    weighting_method="equal_name",
                    return_column=return_column,
                    cost_bps_levels=cost_bps_levels,
                    holding_rows=holding_rows,
                    monthly_rows=monthly_rows,
                    turnover_rows=turnover_rows,
                    exposure_rows=exposure_rows,
                    signal_columns=signal_columns,
                )

            sector_neutral = _sector_neutral_bucket_frame(frame)
            for bucket, bucket_frame in sector_neutral.groupby("bucket", dropna=False):
                _append_bucket_rows(
                    bucket_frame=bucket_frame,
                    bucket=str(bucket),
                    date_value=date_value,
                    signal_name=signal_name,
                    portfolio_mode="sector_neutral",
                    weighting_method="equal_sector_then_equal_name",
                    return_column=return_column,
                    cost_bps_levels=cost_bps_levels,
                    holding_rows=holding_rows,
                    monthly_rows=monthly_rows,
                    turnover_rows=turnover_rows,
                    exposure_rows=exposure_rows,
                    signal_columns=signal_columns,
                )

    return {
        "holdings": pd.DataFrame(holding_rows),
        "monthly_returns": pd.DataFrame(monthly_rows),
        "turnover": pd.DataFrame(turnover_rows),
        "exposures": pd.DataFrame(exposure_rows),
    }


def run_v2_experiment(
    *,
    label_panel_path: Path = DEFAULT_LABEL_PANEL,
    forward_returns_path: Path | None = None,
    edgar_features_path: Path | None = DEFAULT_EDGAR_FEATURES,
    output_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    label_panel = _read_frame(label_panel_path)
    feature_frame = edgar_feature_frame(edgar_features_path)
    label_panel = add_benchmark_signals(label_panel, feature_frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_ic_path = output_dir / "rank_ic.parquet"
    rank_ic_summary_path = output_dir / "rank_ic_summary.parquet"
    rank_ic_by_year_path = output_dir / "rank_ic_by_year.parquet"
    rank_ic_by_sector_path = output_dir / "rank_ic_by_sector.parquet"
    rank_ic_coverage_path = output_dir / "rank_ic_coverage.parquet"
    signal_comparison_path = output_dir / "signal_comparison.parquet"
    status_path = output_dir / "v2_experiment_status.json"
    metadata_path = output_dir / "v2_experiment_metadata.json"

    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    portfolio_return_column: str | None = None
    portfolio_modes_available: list[str] = []
    diagnostic_month_count = 0
    forward_return_coverage: dict[str, dict[str, float | int]] = {}
    sector_coverage = float(label_panel["sector"].notna().mean()) if "sector" in label_panel.columns and len(label_panel) else 0.0
    benchmark_signals = [column for column in BENCHMARK_SIGNAL_COLUMNS if column in label_panel.columns]
    if not benchmark_signals:
        blockers.append(
            {
                "code": "missing_benchmark_feature_source",
                "message": "No EDGAR benchmark feature source was available for VQMIA and control-signal diagnostics.",
            }
        )
    forward_returns = load_forward_returns(forward_returns_path)
    if forward_returns is None:
        blockers.append(
            {
                "code": "missing_forward_returns",
                "message": "No forward returns file was supplied; rank IC and portfolio evidence are blocked.",
            }
        )
        empty_artifact(rank_ic_path, ["date", "signal", "horizon", "decomposition", "n", "rank_ic", "sector_status"])
        empty_artifact(
            rank_ic_summary_path,
            [
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
        )
        empty_artifact(
            rank_ic_by_year_path,
            ["year", "signal", "horizon", "decomposition", "months", "mean_ic", "median_ic", "positive_ic_hit_rate"],
        )
        empty_artifact(
            rank_ic_by_sector_path,
            ["sector", "signal", "horizon", "months", "mean_ic", "median_ic", "positive_ic_hit_rate", "mean_n"],
        )
        empty_artifact(
            rank_ic_coverage_path,
            [
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
        )
        empty_artifact(signal_comparison_path, SIGNAL_COMPARISON_COLUMNS)
    else:
        return_columns = [column for column in forward_returns.columns if is_forward_return_column(column)]
        if not return_columns:
            raise ValueError("Forward return file must include at least one column named return_*.")
        diagnostic_input = build_rank_ic_input(label_panel, forward_returns)
        forward_return_coverage = {
            column: {
                "non_null": int(diagnostic_input[column].notna().sum()) if column in diagnostic_input.columns else 0,
                "coverage": (
                    float(diagnostic_input[column].notna().mean())
                    if column in diagnostic_input.columns and len(diagnostic_input)
                    else 0.0
                ),
            }
            for column in return_columns
        }
        if diagnostic_input.empty:
            diagnostic_month_count = 0
        else:
            usable_any_return = diagnostic_input[return_columns].notna().any(axis=1)
            diagnostic_month_count = int(diagnostic_input.loc[usable_any_return, "date"].nunique())
        sector_coverage = (
            float(diagnostic_input["sector"].notna().mean())
            if "sector" in diagnostic_input.columns and len(diagnostic_input)
            else 0.0
        )
        if diagnostic_input.empty:
            blockers.append(
                {
                    "code": "no_label_return_overlap",
                    "message": "Forward returns had no ticker/date overlap with the label panel.",
                }
            )
        elif not diagnostic_input[return_columns].notna().any().any():
            blockers.append(
                {
                    "code": "insufficient_forward_return_coverage",
                    "message": "Forward returns overlap the label panel, but all requested return horizons are null.",
                }
            )
        signal_columns = [column for column in ALL_SIGNAL_COLUMNS if column in diagnostic_input.columns]
        rank_ic = rank_ic_diagnostics(
            diagnostic_input,
            date_column="date",
            signal_columns=signal_columns,
            return_columns=return_columns,
        )
        rank_ic_summary = summarize_rank_ic(rank_ic)
        rank_ic_year = summarize_rank_ic_by_year(rank_ic)
        sector_ic = rank_ic_by_sector(
            diagnostic_input,
            date_column="date",
            signal_columns=signal_columns,
            return_columns=return_columns,
        )
        rank_ic_sector = summarize_rank_ic_by_sector(sector_ic)
        coverage = rank_ic_coverage(
            diagnostic_input,
            date_column="date",
            signal_columns=signal_columns,
            return_columns=return_columns,
        )
        signal_comparison = build_signal_comparison_table(
            rank_ic_summary,
            signal_labels=KEY_SIGNAL_COMPARISON_LABELS,
            horizons=("return_1m",),
        )
        rank_ic.to_parquet(rank_ic_path, index=False)
        rank_ic_summary.to_parquet(rank_ic_summary_path, index=False)
        rank_ic_year.to_parquet(rank_ic_by_year_path, index=False)
        rank_ic_sector.to_parquet(rank_ic_by_sector_path, index=False)
        coverage.to_parquet(rank_ic_coverage_path, index=False)
        signal_comparison.to_parquet(signal_comparison_path, index=False)
        if "sector" not in diagnostic_input.columns or diagnostic_input["sector"].nunique(dropna=True) < 2:
            blockers.append(
                {
                    "code": "missing_sector_coverage",
                    "message": "Sector-neutral and across-sector diagnostics are unavailable without sector coverage.",
                }
            )
        usable_return_columns = [
            column for column in return_columns if column in diagnostic_input.columns and diagnostic_input[column].notna().any()
        ]
        primary_return_column = (
            "return_1m" if "return_1m" in usable_return_columns else usable_return_columns[0] if usable_return_columns else None
        )
        portfolio_return_column = primary_return_column
        if primary_return_column is not None:
            primary_month_count = int(diagnostic_input.loc[diagnostic_input[primary_return_column].notna(), "date"].nunique())
            if primary_month_count < 12:
                blockers.append(
                    {
                        "code": "insufficient_rank_ic_history",
                        "message": (
                            f"Only {primary_month_count} month(s) have usable {primary_return_column} returns; "
                            "rank-IC and portfolio outputs are smoke-test diagnostics, not robust evidence."
                        ),
                    }
                )
        zero_coverage_horizons = [
            column for column, coverage in forward_return_coverage.items() if int(coverage["non_null"]) == 0
        ]
        if zero_coverage_horizons:
            blockers.append(
                {
                    "code": "zero_coverage_horizons",
                    "message": f"No non-null returns were available for: {', '.join(zero_coverage_horizons)}.",
                }
            )
        portfolio_artifacts = build_portfolio_audit_artifacts(
            diagnostic_input,
            signal_columns=signal_columns,
            return_column=primary_return_column,
        ) if primary_return_column is not None else {}
        if "monthly_returns" in portfolio_artifacts and not portfolio_artifacts["monthly_returns"].empty:
            portfolio_modes_available = sorted(
                str(mode) for mode in portfolio_artifacts["monthly_returns"]["portfolio_mode"].dropna().unique()
            )
        for artifact_name, frame in portfolio_artifacts.items():
            frame.to_parquet(output_dir / f"{artifact_name}.parquet", index=False)

    for name, columns in {
        "holdings.parquet": [
            "date",
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
            "raw_return",
            "transaction_cost",
            "net_return",
        ],
        "monthly_returns.parquet": [
            "date",
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
        "turnover.parquet": [
            "date",
            "signal_name",
            "portfolio_mode",
            "weighting_method",
            "bucket",
            "cost_bps_one_way",
            "turnover",
            "transaction_cost_drag",
        ],
        "exposures.parquet": [
            "date",
            "signal_name",
            "portfolio_mode",
            "weighting_method",
            "bucket",
            "sector_weights",
            "average_market_cap",
            "name_count",
        ],
    }.items():
        artifact_path = output_dir / name
        if forward_returns is None or portfolio_return_column is None or not artifact_path.exists():
            empty_artifact(artifact_path, columns)

    if forward_returns is None:
        blockers.append(
            {
                "code": "portfolio_backtest_not_run",
                "message": "Holdings, monthly return, turnover, and exposure artifacts are schema placeholders until point-in-time matrices and price panels are supplied to an audited portfolio run.",
            }
        )
    elif portfolio_return_column is None:
        blockers.append(
            {
                "code": "portfolio_backtest_not_run",
                "message": "Holdings, monthly return, turnover, and exposure artifacts are schema placeholders because no supplied forward-return horizon had usable returns.",
            }
        )

    status = {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "blocked" if blockers else "complete",
        "label_panel": repo_relative(label_panel_path),
        "forward_returns": repo_relative(forward_returns_path) if forward_returns_path else None,
        "edgar_features": repo_relative(edgar_features_path) if edgar_features_path else None,
        "portfolio_return_column": portfolio_return_column,
        "sector_coverage": sector_coverage,
        "diagnostic_month_count": diagnostic_month_count,
        "forward_return_coverage": forward_return_coverage,
        "portfolio_modes": portfolio_modes_available,
        "signals": [column for column in ALL_SIGNAL_COLUMNS if column in label_panel.columns],
        "artifacts": {
            "rank_ic": repo_relative(rank_ic_path),
            "rank_ic_summary": repo_relative(rank_ic_summary_path),
            "rank_ic_by_year": repo_relative(rank_ic_by_year_path),
            "rank_ic_by_sector": repo_relative(rank_ic_by_sector_path),
            "rank_ic_coverage": repo_relative(rank_ic_coverage_path),
            "signal_comparison": repo_relative(signal_comparison_path),
            "holdings": "results/holdings.parquet",
            "monthly_returns": "results/monthly_returns.parquet",
            "turnover": "results/turnover.parquet",
            "exposures": "results/exposures.parquet",
        },
        "blockers": blockers,
        "warnings": warnings,
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    metadata = build_experiment_metadata(
        experiment_id="ai_label_decomposition_v2_diagnostics",
        feature_config={
            "ai_decomposition_signals": list(SIGNAL_COLUMNS),
            "benchmark_signals": list(BENCHMARK_SIGNAL_COLUMNS),
        },
        model_config={"rank_ic_first": True},
        universe_config={"label_panel": repo_relative(label_panel_path), "survivor_bias_caveat": True},
        backtest_config={
            "portfolio_modes": ["unconstrained", "sector_neutral"],
            "sector_neutral_weighting": "equal_sector_then_equal_name",
            "portfolio_artifacts": "blocked_without_point_in_time_price_panels",
        },
        data_snapshot_paths={"edgar_features": edgar_features_path} if edgar_features_path else {},
        label_snapshot_paths={"label_panel": label_panel_path},
        artifacts={
            "rank_ic": rank_ic_path,
            "rank_ic_summary": rank_ic_summary_path,
            "rank_ic_by_year": rank_ic_by_year_path,
            "rank_ic_by_sector": rank_ic_by_sector_path,
            "rank_ic_coverage": rank_ic_coverage_path,
            "signal_comparison": signal_comparison_path,
            "status": status_path,
        },
    )
    write_experiment_metadata(metadata_path, metadata)
    return status


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v2 AI label decomposition diagnostics.")
    parser.add_argument("--label-panel", default=str(DEFAULT_LABEL_PANEL))
    parser.add_argument("--forward-returns")
    parser.add_argument("--edgar-file", default=str(DEFAULT_EDGAR_FEATURES))
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    status = run_v2_experiment(
        label_panel_path=Path(args.label_panel),
        forward_returns_path=Path(args.forward_returns) if args.forward_returns else None,
        edgar_features_path=Path(args.edgar_file) if args.edgar_file else None,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
