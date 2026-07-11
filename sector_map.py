from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd

from edgar import normalize_ticker
from experiment_registry import build_experiment_metadata, repo_relative, write_experiment_metadata

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_SOURCE_NAME = "wikipedia_sp500_constituents"
DEFAULT_OUTPUT = ROOT / "data" / "sector_map_wikipedia.csv"
DEFAULT_SUMMARY = ROOT / "results" / "sector_map_summary.json"
DEFAULT_METADATA = ROOT / "results" / "sector_map_metadata.json"
FOOTNOTE_RE = re.compile(r"\[\d+\]")


def _clean_text(value: str) -> str:
    return FOOTNOTE_RE.sub("", " ".join(value.split())).strip()


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


class _HtmlTableParser(HTMLParser):
    def __init__(self, *, table_id: str) -> None:
        super().__init__(convert_charrefs=True)
        self.table_id = table_id
        self.rows: list[list[str]] = []
        self._in_table = False
        self._table_depth = 0
        self._in_row = False
        self._in_cell = False
        self._current_row: list[str] = []
        self._current_cell: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_by_name = {name: value for name, value in attrs}
        if tag == "table":
            if self._in_table:
                self._table_depth += 1
            elif attrs_by_name.get("id") == self.table_id:
                self._in_table = True
                self._table_depth = 1
            return
        if not self._in_table:
            return
        if tag == "tr":
            self._in_row = True
            self._current_row = []
        elif tag in {"td", "th"} and self._in_row:
            self._in_cell = True
            self._current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if not self._in_table:
            return
        if tag in {"td", "th"} and self._in_cell:
            self._current_row.append(_clean_text(" ".join(self._current_cell)))
            self._current_cell = []
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = []
            self._in_row = False
        elif tag == "table":
            self._table_depth -= 1
            if self._table_depth <= 0:
                self._in_table = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell.append(data)


def _table_rows(source_html: Path, *, table_id: str = "constituents") -> list[dict[str, str]]:
    parser = _HtmlTableParser(table_id=table_id)
    parser.feed(source_html.read_text(encoding="utf-8"))
    if len(parser.rows) < 2:
        raise ValueError(f"No table with id={table_id!r} and body rows was found in {source_html}.")
    headers = [_normalize_header(header) for header in parser.rows[0]]
    rows: list[dict[str, str]] = []
    for row in parser.rows[1:]:
        if len(row) < len(headers):
            row = [*row, *([""] * (len(headers) - len(row)))]
        rows.append({header: value for header, value in zip(headers, row)})
    return rows


def _pick(row: Mapping[str, str], *candidates: str) -> str | None:
    for candidate in candidates:
        value = row.get(_normalize_header(candidate))
        if value:
            return value
    return None


def build_sector_map(
    source_html: Path,
    *,
    source_url: str = DEFAULT_SOURCE_URL,
    source_name: str = DEFAULT_SOURCE_NAME,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_row in _table_rows(source_html):
        symbol = _pick(source_row, "Symbol", "Ticker")
        sector = _pick(source_row, "GICS Sector", "Sector")
        if not symbol or not sector:
            continue
        cik = _pick(source_row, "CIK")
        rows.append(
            {
                "ticker": normalize_ticker(symbol),
                "symbol_raw": symbol,
                "company_name": _pick(source_row, "Security", "Company", "Name"),
                "sector": sector,
                "sub_industry": _pick(source_row, "GICS Sub-Industry", "Sub-Industry"),
                "cik": int(cik) if cik and cik.isdigit() else None,
                "headquarters": _pick(source_row, "Headquarters Location", "Headquarters"),
                "date_added": _pick(source_row, "Date added", "Date Added"),
                "founded": _pick(source_row, "Founded"),
                "sector_source": source_name,
                "source_url": source_url,
            }
        )

    if not rows:
        raise ValueError(f"No sector rows could be parsed from {source_html}.")
    frame = pd.DataFrame(rows).drop_duplicates("ticker", keep="first").sort_values("ticker")
    return frame.reset_index(drop=True)


def load_sector_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    frame = pd.read_csv(path)
    required_columns = {"ticker", "sector"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Sector map is missing required columns: {sorted(missing)}")

    records: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        ticker = normalize_ticker(str(row.get("ticker", "")))
        if not ticker:
            continue
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if pd.isna(value):
                clean[key] = None
            else:
                clean[key] = value
        records[ticker] = clean
    return records


def summarize_sector_map(frame: pd.DataFrame, *, source_url: str = DEFAULT_SOURCE_URL) -> dict[str, Any]:
    sector_counts = frame.groupby("sector", dropna=False).size().sort_values(ascending=False)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_name": DEFAULT_SOURCE_NAME,
        "source_url": source_url,
        "source_license_note": (
            "Wikipedia page content is used as a current public constituent-sector snapshot; "
            "consult the source page for license and revision history."
        ),
        "current_snapshot_only": True,
        "point_in_time_sector_membership": False,
        "row_count": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()),
        "sector_count": int(frame["sector"].nunique(dropna=True)),
        "sub_industry_count": int(frame["sub_industry"].nunique(dropna=True)) if "sub_industry" in frame else 0,
        "sector_counts": {str(key): int(value) for key, value in sector_counts.items()},
    }


def write_sector_map(
    *,
    source_html: Path,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    metadata_path: Path = DEFAULT_METADATA,
    source_url: str = DEFAULT_SOURCE_URL,
) -> dict[str, Any]:
    frame = build_sector_map(source_html, source_url=source_url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    summary = summarize_sector_map(frame, source_url=source_url)
    summary["artifacts"] = {
        "sector_map": repo_relative(output_path),
        "summary": repo_relative(summary_path),
        "metadata": repo_relative(metadata_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    metadata = build_experiment_metadata(
        experiment_id="sector_map_wikipedia_current_snapshot",
        feature_config={
            "fields": list(frame.columns),
            "sector_source": DEFAULT_SOURCE_NAME,
            "current_snapshot_only": True,
        },
        model_config={},
        universe_config={
            "source_url": source_url,
            "point_in_time_sector_membership": False,
            "historical_sector_caveat": True,
        },
        backtest_config={},
        data_snapshot_paths={"source_html_snapshot": source_html},
        artifacts={"sector_map": output_path, "summary": summary_path},
    )
    write_experiment_metadata(metadata_path, metadata)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a current S&P 500 sector map from a local HTML snapshot.")
    parser.add_argument("--source-html", required=True)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--metadata-output", default=str(DEFAULT_METADATA))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = write_sector_map(
        source_html=Path(args.source_html),
        output_path=Path(args.output),
        summary_path=Path(args.summary_output),
        metadata_path=Path(args.metadata_output),
        source_url=args.source_url,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
