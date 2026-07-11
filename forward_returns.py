from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from experiment_registry import build_experiment_metadata, repo_relative, write_experiment_metadata
from price_model import PricePanels, load_cached_price_panels, load_price_panels, normalize_ticker, price_on_or_after

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DEFAULT_LABEL_PANEL = RESULTS_DIR / "label_panel.parquet"
DEFAULT_PRICE_PANEL_CACHE = ROOT / "data" / "forward_price_panels.parquet"
DEFAULT_OUTPUT = RESULTS_DIR / "forward_returns.parquet"
DEFAULT_SUMMARY = RESULTS_DIR / "forward_returns_summary.json"
DEFAULT_METADATA = RESULTS_DIR / "forward_returns_metadata.json"
DEFAULT_HORIZONS = (1, 3, 6, 12)


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def label_dates_and_tickers(label_panel: pd.DataFrame) -> tuple[list[str], pd.Timestamp, pd.Timestamp]:
    tickers = sorted({normalize_ticker(str(ticker)) for ticker in label_panel["ticker"].dropna()})
    report_dates = pd.to_datetime(label_panel["report_date"], errors="coerce").dropna()
    if not tickers:
        raise ValueError("Label panel does not contain any tickers.")
    if report_dates.empty:
        raise ValueError("Label panel does not contain any valid report dates.")
    start = pd.Timestamp(report_dates.min()).normalize() - pd.DateOffset(days=10)
    end = pd.Timestamp(report_dates.max()).normalize() + pd.DateOffset(months=max(DEFAULT_HORIZONS), days=10)
    return tickers, start, end


def forward_return_for_horizon(
    adjusted_series: pd.Series,
    label_date: pd.Timestamp,
    *,
    months: int,
) -> tuple[float | None, pd.Timestamp | None, pd.Timestamp | None, float | None, float | None]:
    series = adjusted_series.dropna().sort_index()
    if series.empty:
        return None, None, None, None, None
    entry_date = pd.Timestamp(label_date).normalize()
    exit_target = entry_date + pd.DateOffset(months=months)
    entry_price = price_on_or_after(series, entry_date)
    exit_price = price_on_or_after(series, exit_target)
    if entry_price in (None, 0.0) or exit_price is None:
        return None, None, None, entry_price, exit_price
    actual_entry_date = series[series.index > entry_date].index[0]
    actual_exit_date = series[series.index > exit_target].index[0]
    return (
        float(exit_price / entry_price - 1.0),
        pd.Timestamp(actual_entry_date),
        pd.Timestamp(actual_exit_date),
        float(entry_price),
        float(exit_price),
    )


def build_forward_returns(
    label_panel: pd.DataFrame,
    price_panels: PricePanels,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    label_dates = (
        label_panel[["ticker", "report_date"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["report_date", "ticker"], kind="mergesort")
    )
    adjusted = price_panels.adjusted_close
    for row in label_dates.itertuples(index=False):
        ticker = normalize_ticker(str(row.ticker))
        report_date = pd.Timestamp(row.report_date).normalize()
        output_row: dict[str, object] = {
            "ticker": ticker,
            "date": report_date.date().isoformat(),
            "return_source": "adjusted_close",
        }
        if ticker not in adjusted.columns:
            for months in horizons:
                output_row[f"return_{months}m"] = None
            rows.append(output_row)
            continue
        series = adjusted[ticker]
        for months in horizons:
            forward_return, entry_date, exit_date, entry_price, exit_price = forward_return_for_horizon(
                series,
                report_date,
                months=months,
            )
            output_row[f"return_{months}m"] = forward_return
            output_row[f"entry_date_{months}m"] = entry_date.date().isoformat() if entry_date is not None else None
            output_row[f"exit_date_{months}m"] = exit_date.date().isoformat() if exit_date is not None else None
            output_row[f"entry_price_{months}m"] = entry_price
            output_row[f"exit_price_{months}m"] = exit_price
        rows.append(output_row)
    return pd.DataFrame(rows)


def summarize_forward_returns(frame: pd.DataFrame, *, horizons: Sequence[int] = DEFAULT_HORIZONS) -> dict[str, object]:
    coverage = {}
    total_non_null = 0
    zero_coverage_horizons: list[str] = []
    for months in horizons:
        column = f"return_{months}m"
        non_null = int(frame[column].notna().sum()) if column in frame.columns else 0
        total_non_null += non_null
        if non_null == 0:
            zero_coverage_horizons.append(column)
        coverage[column] = {
            "non_null": non_null,
            "coverage": float(frame[column].notna().mean()) if column in frame.columns and len(frame) else 0.0,
        }
    blockers = []
    if total_non_null == 0:
        blockers.append(
            {
                "code": "no_usable_forward_returns",
                "message": "No requested forward-return horizon had non-null adjusted-close returns.",
            }
        )
    warnings = []
    if total_non_null > 0 and zero_coverage_horizons:
        warnings.append(
            {
                "code": "zero_coverage_horizons",
                "message": f"No non-null returns were available for: {', '.join(zero_coverage_horizons)}.",
            }
        )
    if total_non_null > 0 and any(value["coverage"] < 1.0 for value in coverage.values()):
        warnings.append(
            {
                "code": "partial_forward_return_coverage",
                "message": "At least one requested horizon has partial or zero adjusted-close return coverage.",
            }
        )
    return {
        "row_count": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()) if "ticker" in frame.columns else 0,
        "coverage": coverage,
        "blockers": blockers,
        "warnings": warnings,
    }


def write_forward_returns(
    *,
    label_panel_path: Path = DEFAULT_LABEL_PANEL,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    metadata_path: Path = DEFAULT_METADATA,
    price_cache_path: Path = DEFAULT_PRICE_PANEL_CACHE,
    refresh_prices: bool = False,
    offline: bool = False,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> dict[str, object]:
    label_panel = _read_frame(label_panel_path)
    tickers, start, end = label_dates_and_tickers(label_panel)
    if offline:
        price_panels = load_cached_price_panels(tickers, price_cache_path)
    else:
        price_panels = load_price_panels(
            tickers,
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            cache_path=price_cache_path,
            refresh=refresh_prices,
        )
    frame = build_forward_returns(label_panel, price_panels, horizons=horizons)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    summary = summarize_forward_returns(frame, horizons=horizons)
    summary["price_window"] = {
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "cache": repo_relative(price_cache_path),
        "offline": offline,
    }
    if offline and summary["blockers"]:
        summary["blockers"].append(
            {
                "code": "price_cache_missing_or_incomplete",
                "message": "Offline mode used the local price cache only; no usable adjusted-close forward returns were available.",
            }
        )
    summary["artifacts"] = {
        "forward_returns": repo_relative(output_path),
        "summary": repo_relative(summary_path),
        "metadata": repo_relative(metadata_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    metadata = build_experiment_metadata(
        experiment_id="forward_returns_from_adjusted_close",
        feature_config={"horizons_months": list(horizons), "offline": offline},
        model_config={},
        universe_config={"label_panel": repo_relative(label_panel_path)},
        backtest_config={"entry_execution": "first adjusted close after label date"},
        data_snapshot_paths={"price_panel_cache": price_cache_path},
        label_snapshot_paths={"label_panel": label_panel_path},
        artifacts={"forward_returns": output_path, "summary": summary_path},
    )
    write_experiment_metadata(metadata_path, metadata)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build forward returns for v2 AI label diagnostics.")
    parser.add_argument("--label-panel", default=str(DEFAULT_LABEL_PANEL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--metadata-output", default=str(DEFAULT_METADATA))
    parser.add_argument("--price-cache", default=str(DEFAULT_PRICE_PANEL_CACHE))
    parser.add_argument("--refresh-prices", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use the local price cache only and skip yfinance downloads.")
    parser.add_argument("--horizons", nargs="*", type=int, default=list(DEFAULT_HORIZONS))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = write_forward_returns(
        label_panel_path=Path(args.label_panel),
        output_path=Path(args.output),
        summary_path=Path(args.summary_output),
        metadata_path=Path(args.metadata_output),
        price_cache_path=Path(args.price_cache),
        refresh_prices=args.refresh_prices,
        offline=args.offline,
        horizons=tuple(args.horizons),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
