from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from edgar import normalize_ticker
from experiment_registry import build_experiment_metadata, repo_relative, write_experiment_metadata
from sector_map import DEFAULT_OUTPUT as DEFAULT_SECTOR_MAP
from sector_map import load_sector_map
from sp500_changes import DEFAULT_OUTPUT as DEFAULT_SP500_CHANGES
from sp500_changes import load_sp500_changes

ROOT = Path(__file__).resolve().parent
DEFAULT_TICKERS_FILE = ROOT / "tickers.txt"
DEFAULT_COMPANY_TICKERS_CACHE = ROOT / "data" / "edgar_cache" / "company_tickers.json"
DEFAULT_EDGAR_FEATURES = ROOT / "data" / "edgar_features_full.json"
DEFAULT_SECURITY_MASTER = ROOT / "data" / "security_master.parquet"
DEFAULT_UNIVERSE_SNAPSHOT = ROOT / "results" / "universe_snapshot.json"
DEFAULT_METADATA = ROOT / "results" / "security_master_metadata.json"
DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object.")
    return payload


def parse_ticker_file(path: Path) -> tuple[list[str], list[str], dict[str, str]]:
    tickers: list[str] = []
    comments: list[str] = []
    sector_groups: dict[str, str] = {}
    current_group: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            comment = line.lstrip("#").strip()
            comments.append(comment)
            if comment and ":" not in comment and len(comment.split()) <= 4:
                current_group = comment
            continue
        ticker = normalize_ticker(line)
        tickers.append(ticker)
        if current_group is not None:
            sector_groups[ticker] = current_group
    return list(dict.fromkeys(tickers)), comments, sector_groups


def load_company_ticker_map(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = _load_json(path)
    by_ticker: dict[str, Mapping[str, Any]] = {}
    for value in payload.values():
        if not isinstance(value, Mapping):
            continue
        ticker = normalize_ticker(str(value.get("ticker", "")))
        if ticker:
            by_ticker[ticker] = value
    return by_ticker


def load_edgar_features(path: Path | None) -> dict[str, Mapping[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = _load_json(path)
    return {normalize_ticker(str(ticker)): value for ticker, value in payload.items() if isinstance(value, Mapping)}


def _retrieved_date(comments: Sequence[str]) -> str | None:
    for comment in comments:
        match = DATE_RE.search(comment)
        if match:
            return match.group(1)
    return None


def build_security_master(
    *,
    tickers_file: Path = DEFAULT_TICKERS_FILE,
    company_tickers_cache: Path = DEFAULT_COMPANY_TICKERS_CACHE,
    edgar_features_path: Path | None = DEFAULT_EDGAR_FEATURES,
    sector_map_path: Path | None = DEFAULT_SECTOR_MAP,
    membership_changes_path: Path | None = DEFAULT_SP500_CHANGES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tickers, comments, sector_groups = parse_ticker_file(tickers_file)
    company_map = load_company_ticker_map(company_tickers_cache)
    edgar_features = load_edgar_features(edgar_features_path)
    sector_map = load_sector_map(sector_map_path)
    membership_changes = load_sp500_changes(membership_changes_path)
    rows: list[dict[str, Any]] = []

    for ticker in tickers:
        company = company_map.get(ticker, {})
        features = edgar_features.get(ticker, {})
        sector_record = sector_map.get(ticker, {})
        sector_group = sector_groups.get(ticker)
        sector = features.get("sector") or sector_record.get("sector") or sector_group
        sub_industry = features.get("sub_industry") or sector_record.get("sub_industry")
        if features.get("sector"):
            sector_source = "edgar_features"
        elif sector_record.get("sector"):
            sector_source = sector_record.get("sector_source") or "sector_map"
        elif sector_group is not None:
            sector_source = "ticker_file_group"
        else:
            sector_source = None
        rows.append(
            {
                "ticker": ticker,
                "cik": int(company["cik_str"]) if company.get("cik_str") is not None else features.get("cik"),
                "company_name": features.get("company_name") or company.get("title"),
                "ticker_source": repo_relative(tickers_file),
                "company_tickers_source": repo_relative(company_tickers_cache),
                "edgar_feature_available": bool(features),
                "filing_accession": features.get("filing_accession"),
                "filing_form": features.get("filing_form"),
                "period_end": features.get("period_end"),
                "sector": sector,
                "sub_industry": sub_industry,
                "sector_source": sector_source,
                "sector_map_source_url": sector_record.get("source_url"),
                "point_in_time_membership": False,
                "membership_source": "current_ticker_file",
                "membership_history_source": (
                    repo_relative(membership_changes_path)
                    if membership_changes_path is not None and membership_changes_path.exists()
                    else None
                ),
                "membership_history_quality": (
                    "selected_public_changes_not_full_point_in_time_membership"
                    if membership_changes_path is not None and membership_changes_path.exists()
                    else None
                ),
                "survivor_bias_warning": (
                    "Universe is derived from a current ticker file, not historical index membership."
                ),
            }
        )

    frame = pd.DataFrame(rows)
    snapshot = {
        "generated_at": datetime.now(UTC).isoformat(),
        "universe_id": tickers_file.stem,
        "ticker_source": repo_relative(tickers_file),
        "ticker_source_retrieved_date": _retrieved_date(comments),
        "ticker_count": int(len(frame)),
        "cik_coverage": float(frame["cik"].notna().mean()) if len(frame) else 0.0,
        "edgar_feature_coverage": float(frame["edgar_feature_available"].mean()) if len(frame) else 0.0,
        "sector_coverage": float(frame["sector"].notna().mean()) if len(frame) else 0.0,
        "sector_source": (
            frame["sector_source"].dropna().mode().iloc[0]
            if len(frame) and frame["sector_source"].notna().any()
            else None
        ),
        "sector_source_counts": {
            str(key): int(value) for key, value in frame["sector_source"].fillna("missing").value_counts().items()
        } if len(frame) else {},
        "sector_map": repo_relative(sector_map_path) if sector_map_path is not None and sector_map_path.exists() else None,
        "sector_map_current_snapshot_only": bool(sector_map),
        "point_in_time_membership": False,
        "membership_history_source": (
            repo_relative(membership_changes_path)
            if membership_changes_path is not None and membership_changes_path.exists()
            else None
        ),
        "membership_history_quality": (
            "selected_public_changes_not_full_point_in_time_membership"
            if membership_changes_path is not None and membership_changes_path.exists()
            else None
        ),
        "membership_change_event_count": int(len(membership_changes)),
        "membership_change_date_start": (
            str(pd.to_datetime(membership_changes["effective_date"], errors="coerce").min().date())
            if not membership_changes.empty
            else None
        ),
        "membership_change_date_end": (
            str(pd.to_datetime(membership_changes["effective_date"], errors="coerce").max().date())
            if not membership_changes.empty
            else None
        ),
        "survivor_bias_warning": (
            "Universe is derived from a current S&P 500 ex-Financials/ex-Real-Estate ticker file. "
            "It is not CRSP/Compustat-quality point-in-time membership and excludes delisted names."
        ),
        "comments": comments,
        "unmatched_tickers": sorted(frame.loc[frame["cik"].isna(), "ticker"].tolist()) if len(frame) else [],
    }
    return frame, snapshot


def write_security_master(
    *,
    tickers_file: Path = DEFAULT_TICKERS_FILE,
    company_tickers_cache: Path = DEFAULT_COMPANY_TICKERS_CACHE,
    edgar_features_path: Path | None = DEFAULT_EDGAR_FEATURES,
    sector_map_path: Path | None = DEFAULT_SECTOR_MAP,
    membership_changes_path: Path | None = DEFAULT_SP500_CHANGES,
    output_path: Path = DEFAULT_SECURITY_MASTER,
    snapshot_path: Path = DEFAULT_UNIVERSE_SNAPSHOT,
    metadata_path: Path = DEFAULT_METADATA,
) -> dict[str, Any]:
    frame, snapshot = build_security_master(
        tickers_file=tickers_file,
        company_tickers_cache=company_tickers_cache,
        edgar_features_path=edgar_features_path,
        sector_map_path=sector_map_path,
        membership_changes_path=membership_changes_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    snapshot["artifacts"] = {
        "security_master": repo_relative(output_path),
        "universe_snapshot": repo_relative(snapshot_path),
        "metadata": repo_relative(metadata_path),
    }
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    metadata = build_experiment_metadata(
        experiment_id="security_master_universe_snapshot",
        feature_config={"fields": list(frame.columns)},
        model_config={},
        universe_config={
            "universe_id": tickers_file.stem,
            "point_in_time_membership": False,
            "survivor_bias_caveat": True,
            "membership_history_quality": snapshot["membership_history_quality"],
        },
        backtest_config={},
        data_snapshot_paths={
            "tickers_file": tickers_file,
            "company_tickers_cache": company_tickers_cache,
            **({"edgar_features": edgar_features_path} if edgar_features_path else {}),
            **({"sector_map": sector_map_path} if sector_map_path is not None and sector_map_path.exists() else {}),
            **(
                {"membership_changes": membership_changes_path}
                if membership_changes_path is not None and membership_changes_path.exists()
                else {}
            ),
        },
        artifacts={"security_master": output_path, "universe_snapshot": snapshot_path},
    )
    write_experiment_metadata(metadata_path, metadata)
    return snapshot


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Parallax security master and universe snapshot.")
    parser.add_argument("--tickers-file", default=str(DEFAULT_TICKERS_FILE))
    parser.add_argument("--company-tickers-cache", default=str(DEFAULT_COMPANY_TICKERS_CACHE))
    parser.add_argument("--edgar-features", default=str(DEFAULT_EDGAR_FEATURES))
    parser.add_argument("--sector-map", default=str(DEFAULT_SECTOR_MAP))
    parser.add_argument("--membership-changes", default=str(DEFAULT_SP500_CHANGES))
    parser.add_argument("--output", default=str(DEFAULT_SECURITY_MASTER))
    parser.add_argument("--snapshot-output", default=str(DEFAULT_UNIVERSE_SNAPSHOT))
    parser.add_argument("--metadata-output", default=str(DEFAULT_METADATA))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    snapshot = write_security_master(
        tickers_file=Path(args.tickers_file),
        company_tickers_cache=Path(args.company_tickers_cache),
        edgar_features_path=Path(args.edgar_features) if args.edgar_features else None,
        sector_map_path=Path(args.sector_map) if args.sector_map else None,
        membership_changes_path=Path(args.membership_changes) if args.membership_changes else None,
        output_path=Path(args.output),
        snapshot_path=Path(args.snapshot_output),
        metadata_path=Path(args.metadata_output),
    )
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
