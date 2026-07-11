from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd

from edgar import normalize_ticker
from experiment_registry import build_experiment_metadata, repo_relative, write_experiment_metadata
from sector_map import DEFAULT_SOURCE_URL

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_NAME = "wikipedia_sp500_selected_changes"
DEFAULT_OUTPUT = ROOT / "data" / "sp500_changes_wikipedia.csv"
DEFAULT_SUMMARY = ROOT / "results" / "sp500_changes_summary.json"
DEFAULT_METADATA = ROOT / "results" / "sp500_changes_metadata.json"
FOOTNOTE_RE = re.compile(r"\[\d+\]")


def _clean_text(value: str) -> str:
    return FOOTNOTE_RE.sub("", " ".join(value.split())).strip()


class _TableParser(HTMLParser):
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


def _parse_effective_date(value: str) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).date().isoformat()


def build_sp500_changes(
    source_html: Path,
    *,
    source_url: str = DEFAULT_SOURCE_URL,
    source_name: str = DEFAULT_SOURCE_NAME,
) -> pd.DataFrame:
    parser = _TableParser(table_id="changes")
    parser.feed(source_html.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for row in parser.rows:
        if len(row) < 6:
            continue
        if row[0] in {"Effective Date", "Date"} or row[0] == "Ticker":
            continue
        effective_date = _parse_effective_date(row[0])
        if effective_date is None:
            continue
        added_ticker = normalize_ticker(row[1]) if row[1] else None
        removed_ticker = normalize_ticker(row[3]) if row[3] else None
        rows.append(
            {
                "effective_date": effective_date,
                "added_ticker": added_ticker,
                "added_security": row[2] or None,
                "removed_ticker": removed_ticker,
                "removed_security": row[4] or None,
                "reason": row[5] or None,
                "source_name": source_name,
                "source_url": source_url,
                "approximate_membership_history": True,
                "point_in_time_membership": False,
            }
        )
    if not rows:
        raise ValueError(f"No S&P 500 change rows could be parsed from {source_html}.")
    frame = pd.DataFrame(rows).sort_values(["effective_date", "added_ticker", "removed_ticker"], na_position="last")
    return frame.reset_index(drop=True)


def load_sp500_changes(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required_columns = {"effective_date", "added_ticker", "removed_ticker", "reason"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"S&P 500 changes file is missing required columns: {sorted(missing)}")
    return frame


def summarize_sp500_changes(frame: pd.DataFrame, *, source_url: str = DEFAULT_SOURCE_URL) -> dict[str, Any]:
    effective_dates = pd.to_datetime(frame["effective_date"], errors="coerce").dropna()
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_name": DEFAULT_SOURCE_NAME,
        "source_url": source_url,
        "source_license_note": (
            "Wikipedia page content is used as a public selected-changes snapshot; consult the source page for "
            "license and revision history."
        ),
        "approximate_membership_history": True,
        "point_in_time_membership": False,
        "claim_limit": (
            "Selected public changes are useful provenance, but they are not CRSP/Compustat-quality "
            "point-in-time index membership."
        ),
        "row_count": int(len(frame)),
        "date_start": effective_dates.min().date().isoformat() if not effective_dates.empty else None,
        "date_end": effective_dates.max().date().isoformat() if not effective_dates.empty else None,
        "added_ticker_count": int(frame["added_ticker"].dropna().nunique()),
        "removed_ticker_count": int(frame["removed_ticker"].dropna().nunique()),
        "rows_with_added": int(frame["added_ticker"].notna().sum()),
        "rows_with_removed": int(frame["removed_ticker"].notna().sum()),
    }


def write_sp500_changes(
    *,
    source_html: Path,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    metadata_path: Path = DEFAULT_METADATA,
    source_url: str = DEFAULT_SOURCE_URL,
) -> dict[str, Any]:
    frame = build_sp500_changes(source_html, source_url=source_url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    summary = summarize_sp500_changes(frame, source_url=source_url)
    summary["artifacts"] = {
        "sp500_changes": repo_relative(output_path),
        "summary": repo_relative(summary_path),
        "metadata": repo_relative(metadata_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    metadata = build_experiment_metadata(
        experiment_id="sp500_selected_changes_public_snapshot",
        feature_config={
            "fields": list(frame.columns),
            "source_name": DEFAULT_SOURCE_NAME,
            "approximate_membership_history": True,
        },
        model_config={},
        universe_config={
            "source_url": source_url,
            "point_in_time_membership": False,
            "historical_membership_caveat": True,
        },
        backtest_config={},
        data_snapshot_paths={"source_html_snapshot": source_html},
        artifacts={"sp500_changes": output_path, "summary": summary_path},
    )
    write_experiment_metadata(metadata_path, metadata)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an approximate S&P 500 selected-changes artifact.")
    parser.add_argument("--source-html", required=True)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--metadata-output", default=str(DEFAULT_METADATA))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = write_sp500_changes(
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
