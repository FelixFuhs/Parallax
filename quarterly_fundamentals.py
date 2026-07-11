from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from edgar import (
    FIELD_SPECS,
    LONG_TERM_DEBT_SPEC,
    SHORT_TERM_DEBT_SPEC,
    TOTAL_DEBT_SPEC,
    FieldSpec,
    coerce_float,
    coerce_int,
    normalize_ticker,
    parse_date,
    preferred_units,
)
from experiment_registry import build_experiment_metadata, repo_relative, write_experiment_metadata

ROOT = Path(__file__).resolve().parent
DEFAULT_SECURITY_MASTER = ROOT / "data" / "security_master.parquet"
DEFAULT_COMPANYFACTS_DIR = ROOT / "data" / "edgar_cache"
DEFAULT_OUTPUT = ROOT / "data" / "quarterly_fundamentals.parquet"
DEFAULT_SUMMARY = ROOT / "results" / "quarterly_fundamentals_summary.json"
DEFAULT_METADATA = ROOT / "results" / "quarterly_fundamentals_metadata.json"
QUARTERLY_FORMS = {"10-Q", "10-Q/A"}
QUARTERLY_DURATION_FIELDS = {
    "revenue": FIELD_SPECS["revenue"],
    "gross_profit": FIELD_SPECS["gross_profit"],
    "operating_cash_flow": FIELD_SPECS["operating_cash_flow"],
    "capex": FIELD_SPECS["capex"],
}
QUARTERLY_INSTANT_FIELDS = {
    "total_assets": FIELD_SPECS["total_assets"],
    "cash": FIELD_SPECS["cash"],
    "shares": FIELD_SPECS["shares_outstanding"],
    "total_debt": TOTAL_DEBT_SPEC,
}
TTM_FIELDS = ("revenue", "gross_profit", "operating_cash_flow", "capex", "free_cash_flow")


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object.")
    return payload


def _is_quarterly_fact(fact: Mapping[str, Any], *, is_duration: bool) -> bool:
    if fact.get("form") not in QUARTERLY_FORMS:
        return False
    fp = str(fact.get("fp") or "")
    if fp not in {"Q1", "Q2", "Q3"}:
        return False
    end = parse_date(str(fact.get("end"))) if fact.get("end") else None
    if end is None:
        return False
    if not is_duration:
        return True
    start = parse_date(str(fact.get("start"))) if fact.get("start") else None
    if start is None:
        return False
    duration_days = (end - start).days
    return 45 <= duration_days <= 130


def _iter_quarterly_field_facts(
    company_facts: Mapping[str, Any],
    field_name: str,
    spec: FieldSpec,
) -> list[dict[str, Any]]:
    facts_root = company_facts.get("facts", {})
    if not isinstance(facts_root, Mapping):
        return []

    rows: list[dict[str, Any]] = []
    for candidate_rank, candidate in enumerate(spec.candidates):
        taxonomy_facts = facts_root.get(candidate.taxonomy, {})
        if not isinstance(taxonomy_facts, Mapping):
            continue
        tag_payload = taxonomy_facts.get(candidate.tag)
        if not isinstance(tag_payload, Mapping):
            continue
        units = tag_payload.get("units", {})
        if not isinstance(units, Mapping):
            continue

        for unit_name in preferred_units(dict(units), candidate.units):
            values = units.get(unit_name)
            if not isinstance(values, list):
                continue
            for fact in values:
                if not isinstance(fact, Mapping) or not _is_quarterly_fact(fact, is_duration=candidate.is_duration):
                    continue
                value = coerce_float(fact.get("val"))
                if value is None:
                    continue
                rows.append(
                    {
                        "field_name": field_name,
                        "taxonomy": candidate.taxonomy,
                        "tag": candidate.tag,
                        "candidate_rank": candidate_rank,
                        "value": abs(value) if field_name == "capex" else value,
                        "start": parse_date(str(fact.get("start"))) if fact.get("start") else None,
                        "end": parse_date(str(fact.get("end"))) if fact.get("end") else None,
                        "filed": parse_date(str(fact.get("filed"))) if fact.get("filed") else None,
                        "fy": coerce_int(fact.get("fy")),
                        "fp": str(fact.get("fp")) if fact.get("fp") else None,
                        "form": str(fact.get("form")) if fact.get("form") else None,
                        "accn": str(fact.get("accn")) if fact.get("accn") else None,
                    }
                )
            if unit_name in candidate.units:
                break
    return rows


def _period_key(fact: Mapping[str, Any]) -> tuple[int | None, str | None, date | None]:
    return fact.get("fy"), fact.get("fp"), fact.get("end")


def _select_fact(facts: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not facts:
        return None
    return max(
        facts,
        key=lambda fact: (
            fact.get("filed") or date.min,
            fact.get("end") or date.min,
            -int(fact.get("candidate_rank", 999)),
        ),
    )


def _field_values_by_period(facts: Sequence[Mapping[str, Any]]) -> dict[tuple[int | None, str | None, date | None], Mapping[str, Any]]:
    grouped: dict[tuple[int | None, str | None, date | None], list[Mapping[str, Any]]] = defaultdict(list)
    for fact in facts:
        grouped[_period_key(fact)].append(fact)
    return {key: selected for key, values in grouped.items() if (selected := _select_fact(values)) is not None}


def extract_quarterly_company_fundamentals(ticker: str, cik: int | str, company_facts: Mapping[str, Any]) -> pd.DataFrame:
    field_maps: dict[str, dict[tuple[int | None, str | None, date | None], Mapping[str, Any]]] = {}
    for field_name, spec in {**QUARTERLY_DURATION_FIELDS, **QUARTERLY_INSTANT_FIELDS}.items():
        field_maps[field_name] = _field_values_by_period(_iter_quarterly_field_facts(company_facts, field_name, spec))

    if not field_maps["total_debt"]:
        long_debt = _field_values_by_period(_iter_quarterly_field_facts(company_facts, "long_term_debt", LONG_TERM_DEBT_SPEC))
        short_debt = _field_values_by_period(_iter_quarterly_field_facts(company_facts, "short_term_debt", SHORT_TERM_DEBT_SPEC))
        for key in sorted(set(long_debt) | set(short_debt), key=lambda item: (item[2] or date.min, item[1] or "")):
            total = 0.0
            found = False
            for source in (long_debt, short_debt):
                fact = source.get(key)
                if fact is not None:
                    total += float(fact["value"])
                    found = True
            if found:
                base = long_debt.get(key) or short_debt.get(key)
                field_maps["total_debt"][key] = {**base, "field_name": "total_debt", "value": total}

    period_keys = sorted(set().union(*[set(values) for values in field_maps.values()]), key=lambda item: (item[2] or date.min, item[1] or ""))
    rows: list[dict[str, Any]] = []
    for key in period_keys:
        fy, fp, period_end = key
        field_facts = {field: values.get(key) for field, values in field_maps.items()}
        facts_for_meta = [fact for fact in field_facts.values() if fact is not None]
        meta_fact = _select_fact(facts_for_meta)
        row: dict[str, Any] = {
            "ticker": normalize_ticker(ticker),
            "cik": f"{int(cik):010d}",
            "company_name": company_facts.get("entityName"),
            "fiscal_year": fy,
            "fiscal_period": fp,
            "period_end": period_end.isoformat() if period_end else None,
            "filed": meta_fact.get("filed").isoformat() if meta_fact and meta_fact.get("filed") else None,
            "filing_accession": meta_fact.get("accn") if meta_fact else None,
            "filing_form": meta_fact.get("form") if meta_fact else None,
        }
        for field_name in (*QUARTERLY_DURATION_FIELDS.keys(), *QUARTERLY_INSTANT_FIELDS.keys()):
            fact = field_facts.get(field_name)
            row[field_name] = fact.get("value") if fact is not None else None
        if row["operating_cash_flow"] is not None and row["capex"] is not None:
            row["free_cash_flow"] = float(row["operating_cash_flow"]) - float(row["capex"])
        else:
            row["free_cash_flow"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def add_quarterly_rollups(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    output["period_end_ts"] = pd.to_datetime(output["period_end"], errors="coerce")
    output = output.sort_values(["ticker", "period_end_ts", "filed"], kind="mergesort")
    for field_name in TTM_FIELDS:
        if field_name not in output.columns:
            continue
        numeric = pd.to_numeric(output[field_name], errors="coerce")
        output[f"{field_name}_ttm"] = numeric.groupby(output["ticker"]).transform(
            lambda values: values.rolling(4, min_periods=4).sum()
        )
        output[f"{field_name}_qoq_change"] = numeric.groupby(output["ticker"]).transform(
            lambda values: values / values.shift(1) - 1.0
        )
        output[f"{field_name}_yoy_change"] = numeric.groupby(output["ticker"]).transform(
            lambda values: values / values.shift(4) - 1.0
        )
    return output.drop(columns=["period_end_ts"])


def build_quarterly_fundamentals(
    *,
    security_master_path: Path = DEFAULT_SECURITY_MASTER,
    companyfacts_dir: Path = DEFAULT_COMPANYFACTS_DIR,
) -> pd.DataFrame:
    security_master = pd.read_parquet(security_master_path)
    frames: list[pd.DataFrame] = []
    for row in security_master.itertuples(index=False):
        ticker = getattr(row, "ticker")
        cik = getattr(row, "cik")
        if pd.isna(cik):
            continue
        cache_path = companyfacts_dir / f"{int(cik):010d}.json"
        if not cache_path.exists():
            continue
        company_facts = _load_json(cache_path)
        frame = extract_quarterly_company_fundamentals(str(ticker), int(cik), company_facts)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return add_quarterly_rollups(pd.concat(frames, ignore_index=True))


def summarize_quarterly_fundamentals(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "row_count": 0,
            "ticker_count": 0,
            "blockers": [{"code": "no_quarterly_fundamentals", "message": "No cached 10-Q facts were available."}],
        }
    coverage = {}
    for field_name in (
        "revenue",
        "gross_profit",
        "operating_cash_flow",
        "capex",
        "free_cash_flow",
        "total_assets",
        "total_debt",
        "cash",
        "shares",
    ):
        coverage[field_name] = float(frame[field_name].notna().mean()) if field_name in frame.columns else 0.0
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "row_count": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()),
        "period_start": str(pd.to_datetime(frame["period_end"]).min().date()),
        "period_end": str(pd.to_datetime(frame["period_end"]).max().date()),
        "coverage": coverage,
        "ttm_row_count": int(frame["revenue_ttm"].notna().sum()) if "revenue_ttm" in frame.columns else 0,
        "blockers": [],
        "method_notes": [
            "Duration fields use 10-Q/10-Q/A facts with quarter-length durations only.",
            "TTM rollups require four available quarterly observations and do not infer Q4 from 10-K annual facts.",
        ],
    }


def write_quarterly_fundamentals(
    *,
    security_master_path: Path = DEFAULT_SECURITY_MASTER,
    companyfacts_dir: Path = DEFAULT_COMPANYFACTS_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    metadata_path: Path = DEFAULT_METADATA,
) -> dict[str, Any]:
    frame = build_quarterly_fundamentals(
        security_master_path=security_master_path,
        companyfacts_dir=companyfacts_dir,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    summary = summarize_quarterly_fundamentals(frame)
    summary["artifacts"] = {
        "quarterly_fundamentals": repo_relative(output_path),
        "summary": repo_relative(summary_path),
        "metadata": repo_relative(metadata_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    metadata = build_experiment_metadata(
        experiment_id="quarterly_fundamentals_10q_panel",
        feature_config={"fields": list(frame.columns), "ttm_fields": list(TTM_FIELDS)},
        model_config={},
        universe_config={"security_master": repo_relative(security_master_path), "survivor_bias_caveat": True},
        backtest_config={},
        data_snapshot_paths={"security_master": security_master_path},
        artifacts={"quarterly_fundamentals": output_path, "summary": summary_path},
    )
    write_experiment_metadata(metadata_path, metadata)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached 10-Q quarterly fundamentals for Parallax.")
    parser.add_argument("--security-master", default=str(DEFAULT_SECURITY_MASTER))
    parser.add_argument("--companyfacts-dir", default=str(DEFAULT_COMPANYFACTS_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--metadata-output", default=str(DEFAULT_METADATA))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = write_quarterly_fundamentals(
        security_master_path=Path(args.security_master),
        companyfacts_dir=Path(args.companyfacts_dir),
        output_path=Path(args.output),
        summary_path=Path(args.summary_output),
        metadata_path=Path(args.metadata_output),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
