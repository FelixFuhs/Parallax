from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from edgar import normalize_ticker
from experiment_registry import build_experiment_metadata, repo_relative, write_experiment_metadata
from security_master import (
    DEFAULT_COMPANY_TICKERS_CACHE,
    DEFAULT_SECURITY_MASTER,
    DEFAULT_UNIVERSE_SNAPSHOT,
    load_company_ticker_map,
)
from sp500_changes import DEFAULT_OUTPUT as DEFAULT_SP500_CHANGES
from sp500_changes import load_sp500_changes

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "data" / "approx_sp500_membership.parquet"
DEFAULT_SUMMARY = ROOT / "results" / "approx_sp500_membership_summary.json"
DEFAULT_METADATA = ROOT / "results" / "approx_sp500_membership_metadata.json"
DEFAULT_START_YEAR = 2012
DEFAULT_END_YEAR = 2025


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _coerce_date(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _monthly_dates(start_year: int, end_year: int) -> list[pd.Timestamp]:
    periods = pd.period_range(start=f"{int(start_year)}-01", end=f"{int(end_year)}-12", freq="M")
    return [period.to_timestamp(how="end").normalize() for period in periods]


def infer_universe_as_of_date(
    *,
    universe_snapshot_path: Path = DEFAULT_UNIVERSE_SNAPSHOT,
    changes: pd.DataFrame | None = None,
) -> pd.Timestamp:
    snapshot = _read_json(universe_snapshot_path)
    for key in ("ticker_source_retrieved_date", "generated_at"):
        parsed = _coerce_date(snapshot.get(key))
        if parsed is not None:
            return parsed
    if changes is not None and not changes.empty and "effective_date" in changes.columns:
        effective_dates = pd.to_datetime(changes["effective_date"], errors="coerce").dropna()
        if not effective_dates.empty:
            return pd.Timestamp(effective_dates.max()).normalize()
    return pd.Timestamp(datetime.now(UTC)).normalize()


def _prepare_security_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "cik", "sector", "sector_source"])
    frame = pd.read_parquet(path).copy()
    if "ticker" not in frame.columns:
        raise ValueError("security master must contain ticker")
    frame["ticker"] = frame["ticker"].fillna("").astype(str).map(normalize_ticker)
    frame = frame[frame["ticker"].str.len() > 0].drop_duplicates("ticker", keep="last")
    return frame


def _prepare_company_ticker_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    return {ticker: dict(record) for ticker, record in load_company_ticker_map(path).items()}


def _prepare_changes(path: Path) -> pd.DataFrame:
    changes = load_sp500_changes(path).copy()
    if changes.empty:
        return pd.DataFrame(columns=["effective_date", "added_ticker", "removed_ticker"])
    changes["effective_date"] = pd.to_datetime(changes["effective_date"], errors="coerce").dt.normalize()
    for column in ("added_ticker", "removed_ticker"):
        if column in changes.columns:
            changes[column] = changes[column].where(changes[column].notna(), "").astype(str).map(normalize_ticker)
    return changes[changes["effective_date"].notna()].copy()


def membership_on_date(
    *,
    current_tickers: set[str],
    changes: pd.DataFrame,
    membership_date: pd.Timestamp,
    as_of_date: pd.Timestamp,
) -> set[str]:
    members = set(current_tickers)
    if changes.empty:
        return members
    future_changes = changes[
        (changes["effective_date"] > pd.Timestamp(membership_date).normalize())
        & (changes["effective_date"] <= pd.Timestamp(as_of_date).normalize())
    ].sort_values("effective_date", ascending=False)
    for event in future_changes.itertuples(index=False):
        added = normalize_ticker(str(getattr(event, "added_ticker", "") or ""))
        removed = normalize_ticker(str(getattr(event, "removed_ticker", "") or ""))
        if added:
            members.discard(added)
        if removed:
            members.add(removed)
    return members


def build_approximate_membership_panel(
    *,
    security_master_path: Path = DEFAULT_SECURITY_MASTER,
    changes_path: Path = DEFAULT_SP500_CHANGES,
    universe_snapshot_path: Path = DEFAULT_UNIVERSE_SNAPSHOT,
    company_tickers_cache_path: Path | None = DEFAULT_COMPANY_TICKERS_CACHE,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    security = _prepare_security_master(security_master_path)
    changes = _prepare_changes(changes_path)
    company_ticker_map = _prepare_company_ticker_map(company_tickers_cache_path)
    as_of_date = infer_universe_as_of_date(universe_snapshot_path=universe_snapshot_path, changes=changes)
    current_tickers = set(security["ticker"].dropna().astype(str))
    security_by_ticker = security.set_index("ticker").to_dict(orient="index") if not security.empty else {}

    rows: list[dict[str, Any]] = []
    dates = [date for date in _monthly_dates(start_year, end_year) if date <= as_of_date]
    for date_value in dates:
        members = membership_on_date(
            current_tickers=current_tickers,
            changes=changes,
            membership_date=date_value,
            as_of_date=as_of_date,
        )
        for ticker in sorted(members):
            security_record = security_by_ticker.get(ticker, {})
            in_security_master = ticker in security_by_ticker
            company_record = company_ticker_map.get(ticker, {})
            fallback_cik = company_record.get("cik_str")
            cik = security_record.get("cik") if in_security_master else fallback_cik
            company_tickers_match = bool(company_record)
            rows.append(
                {
                    "date": date_value.date().isoformat(),
                    "ticker": ticker,
                    "approximate_member": True,
                    "in_current_security_master": bool(in_security_master),
                    "company_tickers_match": company_tickers_match,
                    "has_cik": bool(pd.notna(cik)),
                    "cik": cik,
                    "cik_source": (
                        "current_security_master"
                        if in_security_master and pd.notna(security_record.get("cik"))
                        else "sec_company_tickers_cache"
                        if company_tickers_match and pd.notna(fallback_cik)
                        else None
                    ),
                    "company_name": security_record.get("company_name") or company_record.get("title"),
                    "sector": security_record.get("sector"),
                    "sector_source": security_record.get("sector_source"),
                    "membership_basis": (
                        "current_security_master"
                        if in_security_master
                        else "selected_changes_removed_ticker_backfill"
                    ),
                    "point_in_time_membership": False,
                    "membership_history_quality": "approximate_public_selected_changes_not_full_constituent_history",
                }
            )

    panel = pd.DataFrame(rows)
    if panel.empty:
        panel = pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "approximate_member",
                "in_current_security_master",
                "company_tickers_match",
                "has_cik",
                "cik",
                "cik_source",
                "company_name",
                "sector",
                "sector_source",
                "membership_basis",
                "point_in_time_membership",
                "membership_history_quality",
            ]
        )

    missing = panel.loc[~panel["in_current_security_master"], "ticker"].dropna().unique().tolist()
    missing_with_company_match = (
        panel.loc[(~panel["in_current_security_master"]) & (panel["company_tickers_match"]), "ticker"]
        .dropna()
        .unique()
        .tolist()
    )
    monthly_missing_rate = (
        panel.groupby("date")["in_current_security_master"].apply(lambda values: float((~values).mean()))
        if not panel.empty
        else pd.Series(dtype=float)
    )
    change_dates = pd.to_datetime(changes["effective_date"], errors="coerce").dropna() if not changes.empty else pd.Series(dtype="datetime64[ns]")
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_role": "approximate_public_membership_gap_analysis",
        "status": "approximate_gap_analysis_not_point_in_time_membership",
        "start_year": int(start_year),
        "end_year": int(end_year),
        "as_of_date": as_of_date.date().isoformat(),
        "row_count": int(len(panel)),
        "date_count": int(panel["date"].nunique()) if not panel.empty else 0,
        "ticker_count": int(panel["ticker"].nunique()) if not panel.empty else 0,
        "current_security_master_ticker_count": int(len(current_tickers)),
        "selected_change_event_count": int(len(changes)),
        "selected_change_date_start": change_dates.min().date().isoformat() if not change_dates.empty else None,
        "selected_change_date_end": change_dates.max().date().isoformat() if not change_dates.empty else None,
        "missing_from_current_security_master_ticker_count": int(len(missing)),
        "missing_from_current_security_master_ticker_examples": sorted(missing)[:25],
        "missing_with_sec_company_ticker_match_count": int(len(missing_with_company_match)),
        "missing_without_sec_company_ticker_match_count": int(len(set(missing) - set(missing_with_company_match))),
        "missing_with_sec_company_ticker_match_examples": sorted(missing_with_company_match)[:25],
        "average_monthly_missing_from_security_master_rate": (
            float(monthly_missing_rate.mean()) if not monthly_missing_rate.empty else 0.0
        ),
        "max_monthly_missing_from_security_master_rate": (
            float(monthly_missing_rate.max()) if not monthly_missing_rate.empty else 0.0
        ),
        "point_in_time_membership": False,
        "claim_limit": (
            "This artifact walks the public selected-changes table backward from the current security master. "
            "It quantifies survivor-bias gaps, but it is not CRSP/Compustat-quality membership and does not add "
            "fundamentals, sectors, or returns for removed/delisted names. SEC company_tickers matches are only "
            "identifier hints for some missing tickers."
        ),
        "blockers": [
            {
                "code": "removed_names_missing_security_master_rows",
                "message": "Removed historical constituents identified from public selected changes are not present in the current security master with CIK/features/returns.",
            },
            {
                "code": "selected_changes_not_full_point_in_time_membership",
                "message": "The public selected-changes table is incomplete provenance, not a full historical constituent database.",
            },
        ],
        "artifacts": {},
    }
    return panel, summary


def write_approximate_membership_artifacts(
    *,
    security_master_path: Path = DEFAULT_SECURITY_MASTER,
    changes_path: Path = DEFAULT_SP500_CHANGES,
    universe_snapshot_path: Path = DEFAULT_UNIVERSE_SNAPSHOT,
    company_tickers_cache_path: Path | None = DEFAULT_COMPANY_TICKERS_CACHE,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    metadata_path: Path = DEFAULT_METADATA,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
) -> dict[str, Any]:
    panel, summary = build_approximate_membership_panel(
        security_master_path=security_master_path,
        changes_path=changes_path,
        universe_snapshot_path=universe_snapshot_path,
        company_tickers_cache_path=company_tickers_cache_path,
        start_year=start_year,
        end_year=end_year,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)
    summary["artifacts"] = {
        "approximate_membership": repo_relative(output_path),
        "summary": repo_relative(summary_path),
        "metadata": repo_relative(metadata_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    metadata = build_experiment_metadata(
        experiment_id="approximate_sp500_membership_gap_analysis",
        feature_config={"fields": list(panel.columns)},
        model_config={},
        universe_config={
            "point_in_time_membership": False,
            "membership_history_quality": summary["status"],
            "survivor_bias_gap_quantified": True,
        },
        backtest_config={},
        data_snapshot_paths={
            "security_master": security_master_path,
            **({"company_tickers_cache": company_tickers_cache_path} if company_tickers_cache_path else {}),
            "sp500_changes": changes_path,
            "universe_snapshot": universe_snapshot_path,
        },
        artifacts={"approximate_membership": output_path, "summary": summary_path},
    )
    write_experiment_metadata(metadata_path, metadata)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build approximate public S&P 500 membership gap artifacts.")
    parser.add_argument("--security-master", default=str(DEFAULT_SECURITY_MASTER))
    parser.add_argument("--changes", default=str(DEFAULT_SP500_CHANGES))
    parser.add_argument("--universe-snapshot", default=str(DEFAULT_UNIVERSE_SNAPSHOT))
    parser.add_argument("--company-tickers-cache", default=str(DEFAULT_COMPANY_TICKERS_CACHE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--metadata-output", default=str(DEFAULT_METADATA))
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = write_approximate_membership_artifacts(
        security_master_path=Path(args.security_master),
        changes_path=Path(args.changes),
        universe_snapshot_path=Path(args.universe_snapshot),
        company_tickers_cache_path=Path(args.company_tickers_cache) if args.company_tickers_cache else None,
        output_path=Path(args.output),
        summary_path=Path(args.summary_output),
        metadata_path=Path(args.metadata_output),
        start_year=args.start_year,
        end_year=args.end_year,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
