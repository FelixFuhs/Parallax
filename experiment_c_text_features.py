from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from edgar import (
    DEFAULT_SEC_EMAIL,
    DEFAULT_SEC_NAME,
    PLACEHOLDER_SEC_EMAIL,
    PLACEHOLDER_SEC_NAME,
    SEC_MAX_REQUESTS_PER_SECOND,
    SEC_TIMEOUT_SECONDS,
    RateLimiter,
)
from experiment_registry import build_experiment_metadata, repo_relative, write_experiment_metadata
from research_scaffolds import TextFeatureExperimentConfig

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DEFAULT_FEATURE_PANEL = RESULTS_DIR / "experiment_c_text_features.parquet"
DEFAULT_TEXT_CORPUS_REQUESTS = RESULTS_DIR / "experiment_c_text_corpus_requests.parquet"
DEFAULT_LLM_EXTRACTION_REQUESTS = RESULTS_DIR / "experiment_c_llm_extraction_requests.parquet"
DEFAULT_LLM_EXTRACTION_RESPONSES = RESULTS_DIR / "experiment_c_llm_extraction_responses.parquet"
DEFAULT_MANIFEST = RESULTS_DIR / "experiment_c_text_features_manifest.json"
DEFAULT_METADATA = RESULTS_DIR / "experiment_c_text_features_metadata.json"
DEFAULT_SECURITY_MASTER = ROOT / "data" / "security_master.parquet"
DEFAULT_QUARTERLY_FUNDAMENTALS = ROOT / "data" / "quarterly_fundamentals.parquet"
DEFAULT_COMPANYFACTS_DIR = ROOT / "data" / "edgar_cache"
SEC_TEXT_DEFAULT_DELAY_SECONDS = 0.2
DEFAULT_TEXT_CORPUS_LOOKBACK_FILINGS_PER_FORM = 2
DETERMINISTIC_EXTRACTION_MODEL_ID = "deterministic_keyword_baseline_v1"
BASE_COLUMNS = (
    "ticker",
    "cik",
    "filing_accession",
    "filing_form",
    "filing_lookback_rank",
    "filed",
    "period_end",
    "source_type",
    "source_path",
    "source_hash",
    "extraction_model_id",
    "prompt_id",
    "prompt_version",
    "extraction_generated_at",
)
QUALITY_COLUMNS = (
    "text_window_start",
    "text_window_end",
    "date_limited_source",
    "quality_flags",
    "parse_failure_rate",
)
TEXT_CORPUS_REQUEST_COLUMNS = (
    "ticker",
    "cik",
    "company_name",
    "filing_accession",
    "filing_form",
    "filing_lookback_rank",
    "filed",
    "period_end",
    "source_type",
    "source_url",
    "source_path",
    "source_hash",
    "date_limited_source",
    "download_status",
    "downloaded_at",
    "http_status",
    "download_error",
    "text_extraction_status",
    "quality_flags",
)
LLM_EXTRACTION_REQUEST_COLUMNS = (
    "ticker",
    "cik",
    "company_name",
    "filing_accession",
    "filing_form",
    "filing_lookback_rank",
    "filed",
    "period_end",
    "source_type",
    "source_path",
    "source_hash",
    "date_limited_source",
    "download_status",
    "llm_request_status",
    "llm_extraction_status",
    "prompt_id",
    "prompt_version",
    "prompt_model_role",
    "system_prompt",
    "user_prompt",
    "output_schema_json",
    "quality_flags",
)
LLM_EXTRACTION_RESPONSE_COLUMNS = (
    "ticker",
    "cik",
    "company_name",
    "filing_accession",
    "filing_form",
    "filing_lookback_rank",
    "filed",
    "period_end",
    "source_path",
    "source_hash",
    "prompt_id",
    "prompt_version",
    "llm_model_id",
    "llm_response_id",
    "response_json",
    "validation_status",
    "validation_errors",
    "extraction_generated_at",
    "quality_flags",
)
LLM_EXTRACTION_PROMPT_ID = "experiment_c_sec_filing_text_features"
LLM_EXTRACTION_PROMPT_VERSION = "v1_date_limited_sec_filings"
LLM_EXTRACTION_IMPORT_MODEL_ID = "external_llm_response_import_v1"
FORBIDDEN_EXPERIMENT_C_RESPONSE_KEYS = {
    "raw_ai_implied_irr",
    "mechanical_dcf_implied_irr",
    "ai_minus_mechanical_irr",
    "ai_factor_residual",
    "return",
    "future_return",
    "price_target",
}
KEYWORD_GROUPS: Mapping[str, tuple[str, ...]] = {
    "positive_tone": (
        "improved",
        "increase",
        "growth",
        "strong demand",
        "expanded margin",
        "record",
        "favorable",
        "resilient",
    ),
    "negative_tone": (
        "decline",
        "decrease",
        "weakness",
        "adverse",
        "impairment",
        "challenging",
        "deterioration",
        "unfavorable",
    ),
    "uncertainty": (
        "uncertain",
        "uncertainty",
        "may adversely",
        "could adversely",
        "unable to predict",
        "volatile",
        "risk",
        "risks",
    ),
    "management_hedging": (
        "may",
        "could",
        "might",
        "believe",
        "expect",
        "intend",
        "estimate",
        "subject to",
    ),
    "capital_allocation_discipline": (
        "return capital",
        "share repurchase",
        "dividend",
        "capital allocation",
        "disciplined investment",
        "free cash flow",
    ),
    "competitive_pressure": (
        "competitive",
        "competition",
        "pricing pressure",
        "market share",
        "new entrants",
        "substitute products",
    ),
    "pricing_power": (
        "price increase",
        "pricing power",
        "favorable pricing",
        "price realization",
        "premium",
        "mix shift",
    ),
    "demand_weakness": (
        "demand weakness",
        "lower demand",
        "soft demand",
        "reduced demand",
        "volume decline",
        "customer destocking",
    ),
    "supply_chain_stress": (
        "supply chain",
        "supplier disruption",
        "shortage",
        "logistics",
        "inventory constraints",
        "lead times",
    ),
    "regulatory_pressure": (
        "regulatory",
        "regulation",
        "litigation",
        "compliance",
        "antitrust",
        "government investigation",
    ),
    "accounting_aggressiveness": (
        "material weakness",
        "restatement",
        "internal control",
        "non-gaap",
        "adjusted ebitda",
        "critical accounting estimate",
    ),
    "guidance_credibility": (
        "guidance",
        "outlook",
        "forecast",
        "expects",
        "reaffirm",
        "withdrew guidance",
        "updated guidance",
    ),
}
TEXT_CORPUS_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A"}


@dataclass(frozen=True)
class TextDownloadResponse:
    status_code: int
    content: bytes


class SecTextFetcher:
    def __init__(
        self,
        *,
        user_agent_name: str,
        user_agent_email: str,
        delay_seconds: float = SEC_TEXT_DEFAULT_DELAY_SECONDS,
    ) -> None:
        user_agent = f"{user_agent_name} ({user_agent_email})"
        self.delay_seconds = max(0.0, delay_seconds)
        self.rate_limiter = RateLimiter(SEC_MAX_REQUESTS_PER_SECOND)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/plain, */*",
                "From": user_agent_email,
            }
        )

    def __call__(self, url: str) -> TextDownloadResponse:
        self.rate_limiter.wait()
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        response = self.session.get(url, timeout=SEC_TIMEOUT_SECONDS)
        return TextDownloadResponse(status_code=int(response.status_code), content=response.content)


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object.")
    return payload


def _coerce_date(value: Any) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).date().isoformat()


def _normalize_cik(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    try:
        return f"{int(value):010d}"
    except (TypeError, ValueError):
        digits = "".join(character for character in str(value) if character.isdigit())
        return f"{int(digits):010d}" if digits else None


def _sec_archive_url(cik: Any, accession: Any) -> str | None:
    normalized_cik = _normalize_cik(cik)
    if normalized_cik is None or not accession:
        return None
    accession_text = str(accession)
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{int(normalized_cik)}/{accession_text.replace('-', '')}/{accession_text}.txt"
    )


def _source_path(cik: Any, accession: Any) -> str | None:
    normalized_cik = _normalize_cik(cik)
    if normalized_cik is None or not accession:
        return None
    return f"data/sec_filing_text/{normalized_cik}/{accession}.txt"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _resolve_source_path(source_path: Any, *, root: Path = ROOT) -> Path | None:
    if _is_missing(source_path):
        return None
    path = Path(str(source_path))
    return path if path.is_absolute() else root / path


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalize_quality_flags(value: Any) -> list[str]:
    if _is_missing(value):
        return []
    if isinstance(value, str):
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if not _is_missing(item)]
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return [str(item) for item in value if not _is_missing(item)]
    return [str(value)]


def _quality_flags_with(value: Any, flag: str) -> list[str]:
    flags = _normalize_quality_flags(value)
    if flag not in flags:
        flags.append(flag)
    return flags


def _uses_placeholder_sec_contact(user_agent_name: str, user_agent_email: str) -> bool:
    return user_agent_name == PLACEHOLDER_SEC_NAME or user_agent_email == PLACEHOLDER_SEC_EMAIL


def _request_row(
    *,
    ticker: Any,
    cik: Any,
    company_name: Any,
    filing_accession: Any,
    filing_form: Any,
    filing_lookback_rank: int | None,
    filed: Any,
    period_end: Any,
) -> dict[str, Any]:
    filed_date = _coerce_date(filed)
    accession = str(filing_accession) if filing_accession else None
    flags: list[str] = []
    if filed_date is None:
        flags.append("missing_filed_date")
    if not accession:
        flags.append("missing_filing_accession")
    return {
        "ticker": str(ticker) if ticker is not None else None,
        "cik": _normalize_cik(cik),
        "company_name": str(company_name) if company_name is not None else None,
        "filing_accession": accession,
        "filing_form": str(filing_form) if filing_form is not None else None,
        "filing_lookback_rank": int(filing_lookback_rank) if filing_lookback_rank is not None else None,
        "filed": filed_date,
        "period_end": _coerce_date(period_end),
        "source_type": "sec_filing_text_request",
        "source_url": _sec_archive_url(cik, accession),
        "source_path": _source_path(cik, accession),
        "source_hash": None,
        "date_limited_source": bool(filed_date and accession),
        "download_status": "not_downloaded",
        "downloaded_at": None,
        "http_status": None,
        "download_error": None,
        "text_extraction_status": "not_run",
        "quality_flags": flags,
    }


def _iter_companyfacts_filing_refs(company_facts: Mapping[str, Any], forms: set[str]) -> list[dict[str, Any]]:
    facts_root = company_facts.get("facts", {})
    if not isinstance(facts_root, Mapping):
        return []
    by_accession: dict[str, dict[str, Any]] = {}
    taxonomy_payloads: list[Mapping[str, Any]] = []
    dei_payload = facts_root.get("dei")
    if isinstance(dei_payload, Mapping):
        taxonomy_payloads.append(dei_payload)
    if not taxonomy_payloads:
        taxonomy_payloads = [payload for payload in facts_root.values() if isinstance(payload, Mapping)]
    for taxonomy_payload in taxonomy_payloads:
        if not isinstance(taxonomy_payload, Mapping):
            continue
        for tag_payload in taxonomy_payload.values():
            if not isinstance(tag_payload, Mapping):
                continue
            units = tag_payload.get("units", {})
            if not isinstance(units, Mapping):
                continue
            for values in units.values():
                if not isinstance(values, list):
                    continue
                for fact in values:
                    if not isinstance(fact, Mapping) or fact.get("form") not in forms or not fact.get("accn"):
                        continue
                    accession = str(fact["accn"])
                    current = by_accession.get(accession, {})
                    filed = _coerce_date(fact.get("filed")) or current.get("filed")
                    period_end = _coerce_date(fact.get("end")) or current.get("period_end")
                    by_accession[accession] = {
                        "filing_accession": accession,
                        "filing_form": str(fact.get("form")),
                        "filed": filed,
                        "period_end": period_end,
                    }
    return list(by_accession.values())


def _latest_by_ticker_and_form(frame: pd.DataFrame, *, limit_per_form: int = 1) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    output["filed_ts"] = pd.to_datetime(output["filed"], errors="coerce")
    output["period_end_ts"] = pd.to_datetime(output["period_end"], errors="coerce")
    output = output.sort_values(["ticker", "filing_form", "filed_ts", "period_end_ts"], ascending=True, kind="mergesort")
    output["_reverse_rank"] = output.groupby(["ticker", "filing_form"]).cumcount(ascending=False) + 1
    output = output[output["_reverse_rank"] <= max(1, int(limit_per_form))].copy()
    output["filing_lookback_rank"] = output["_reverse_rank"].astype(int)
    return output.drop(columns=["filed_ts", "period_end_ts", "_reverse_rank"]).reset_index(drop=True)


def text_feature_columns(config: TextFeatureExperimentConfig | None = None) -> list[str]:
    cfg = config or TextFeatureExperimentConfig()
    return [*BASE_COLUMNS, *cfg.feature_names, *QUALITY_COLUMNS]


def build_empty_text_feature_panel(config: TextFeatureExperimentConfig | None = None) -> pd.DataFrame:
    return pd.DataFrame(columns=text_feature_columns(config))


def build_text_corpus_requests(
    *,
    security_master_path: Path = DEFAULT_SECURITY_MASTER,
    quarterly_fundamentals_path: Path = DEFAULT_QUARTERLY_FUNDAMENTALS,
    companyfacts_dir: Path = DEFAULT_COMPANYFACTS_DIR,
    lookback_filings_per_form: int = DEFAULT_TEXT_CORPUS_LOOKBACK_FILINGS_PER_FORM,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    security = pd.read_parquet(security_master_path) if security_master_path.exists() else pd.DataFrame()
    by_ticker = (
        security.drop_duplicates("ticker").set_index("ticker").to_dict(orient="index")
        if not security.empty and "ticker" in security.columns
        else {}
    )

    if not security.empty:
        for security_row in security.itertuples(index=False):
            cik = getattr(security_row, "cik", None)
            if cik is None or pd.isna(cik):
                continue
            cache_path = companyfacts_dir / f"{int(cik):010d}.json"
            if not cache_path.exists():
                continue
            company_facts = _load_json(cache_path)
            filing_refs = _iter_companyfacts_filing_refs(company_facts, {"10-K", "10-K/A"})
            if not filing_refs:
                continue
            refs = pd.DataFrame(filing_refs)
            latest = _latest_by_ticker_and_form(
                refs.assign(
                    ticker=getattr(security_row, "ticker", None),
                    cik=getattr(security_row, "cik", None),
                    company_name=getattr(security_row, "company_name", None),
                ),
                limit_per_form=lookback_filings_per_form,
            )
            for ref in latest.itertuples(index=False):
                rows.append(
                    _request_row(
                        ticker=getattr(ref, "ticker", None),
                        cik=getattr(ref, "cik", None),
                        company_name=getattr(ref, "company_name", None),
                        filing_accession=getattr(ref, "filing_accession", None),
                        filing_form=getattr(ref, "filing_form", None),
                        filing_lookback_rank=getattr(ref, "filing_lookback_rank", None),
                        filed=getattr(ref, "filed", None),
                        period_end=getattr(ref, "period_end", None),
                    )
                )

    if quarterly_fundamentals_path.exists():
        quarterly = pd.read_parquet(quarterly_fundamentals_path)
        required_columns = {"ticker", "cik", "filing_accession", "filing_form", "filed", "period_end"}
        if required_columns.issubset(quarterly.columns):
            quarterly = quarterly.loc[quarterly["filing_form"].isin({"10-Q", "10-Q/A"})].copy()
            quarterly = quarterly.dropna(subset=["ticker", "cik", "filing_accession", "filed"])
            quarterly = quarterly.drop_duplicates(["ticker", "filing_accession", "filing_form"])
            latest_quarterly = _latest_by_ticker_and_form(quarterly, limit_per_form=lookback_filings_per_form)
            for ref in latest_quarterly.itertuples(index=False):
                security_record = by_ticker.get(getattr(ref, "ticker", None), {})
                rows.append(
                    _request_row(
                        ticker=getattr(ref, "ticker", None),
                        cik=getattr(ref, "cik", None),
                        company_name=getattr(ref, "company_name", None) or security_record.get("company_name"),
                        filing_accession=getattr(ref, "filing_accession", None),
                        filing_form=getattr(ref, "filing_form", None),
                        filing_lookback_rank=getattr(ref, "filing_lookback_rank", None),
                        filed=getattr(ref, "filed", None),
                        period_end=getattr(ref, "period_end", None),
                    )
                )

    if not rows:
        return pd.DataFrame(columns=TEXT_CORPUS_REQUEST_COLUMNS)
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(["ticker", "filing_accession", "filing_form"], keep="last")
    frame = frame.reindex(columns=TEXT_CORPUS_REQUEST_COLUMNS)
    return frame.sort_values(["ticker", "filing_form", "filed"], na_position="last").reset_index(drop=True)


def _prepare_text_corpus_request_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in TEXT_CORPUS_REQUEST_COLUMNS:
        if column not in output.columns:
            output[column] = None
    extra_columns = [column for column in output.columns if column not in TEXT_CORPUS_REQUEST_COLUMNS]
    return output.reindex(columns=[*TEXT_CORPUS_REQUEST_COLUMNS, *extra_columns])


def _read_existing_download(path: Path) -> tuple[bytes, str] | None:
    if not path.exists() or not path.is_file():
        return None
    payload = path.read_bytes()
    return payload, _sha256_bytes(payload)


def download_text_corpus_requests(
    *,
    requests_path: Path = DEFAULT_TEXT_CORPUS_REQUESTS,
    output_path: Path | None = None,
    root: Path = ROOT,
    limit: int | None = 0,
    force: bool = False,
    fetcher: Callable[[str], TextDownloadResponse] | None = None,
    user_agent_name: str = DEFAULT_SEC_NAME,
    user_agent_email: str = DEFAULT_SEC_EMAIL,
    delay_seconds: float = SEC_TEXT_DEFAULT_DELAY_SECONDS,
) -> dict[str, Any]:
    if not requests_path.exists():
        raise FileNotFoundError(f"missing text corpus request artifact: {requests_path}")
    frame = _prepare_text_corpus_request_frame(pd.read_parquet(requests_path))
    output_path = output_path or requests_path
    max_downloads = None if limit is None or limit < 0 else int(limit)
    active_fetcher = fetcher
    stats = {
        "row_count": int(len(frame)),
        "eligible_download_count": 0,
        "attempted_download_count": 0,
        "downloaded_count": 0,
        "cached_file_count": 0,
        "failed_count": 0,
        "skipped_missing_source_count": 0,
        "remaining_not_downloaded_count": 0,
        "output_path": repo_relative(output_path),
    }

    for index, row in frame.iterrows():
        source_url = row.get("source_url")
        destination = _resolve_source_path(row.get("source_path"), root=root)
        if _is_missing(source_url) or destination is None:
            frame.at[index, "download_status"] = "missing_source"
            frame.at[index, "download_error"] = "missing_source_url_or_path"
            frame.at[index, "quality_flags"] = _quality_flags_with(row.get("quality_flags"), "missing_download_source")
            stats["skipped_missing_source_count"] += 1
            continue

        existing_download = None if force else _read_existing_download(destination)
        if existing_download is not None:
            _payload, digest = existing_download
            frame.at[index, "source_hash"] = digest
            frame.at[index, "download_status"] = "downloaded"
            frame.at[index, "download_error"] = None
            frame.at[index, "text_extraction_status"] = row.get("text_extraction_status") or "not_run"
            stats["cached_file_count"] += 1
            continue

        stats["eligible_download_count"] += 1
        if max_downloads is not None and stats["attempted_download_count"] >= max_downloads:
            continue

        if active_fetcher is None:
            active_fetcher = SecTextFetcher(
                user_agent_name=user_agent_name,
                user_agent_email=user_agent_email,
                delay_seconds=delay_seconds,
            )
        stats["attempted_download_count"] += 1
        try:
            response = active_fetcher(str(source_url))
        except Exception as exc:
            frame.at[index, "download_status"] = "download_failed"
            frame.at[index, "download_error"] = f"{exc.__class__.__name__}: {exc}"
            frame.at[index, "quality_flags"] = _quality_flags_with(row.get("quality_flags"), "download_failed")
            stats["failed_count"] += 1
            continue

        frame.at[index, "http_status"] = int(response.status_code)
        if response.status_code != 200:
            frame.at[index, "download_status"] = "download_failed"
            frame.at[index, "download_error"] = f"http_status_{response.status_code}"
            frame.at[index, "quality_flags"] = _quality_flags_with(row.get("quality_flags"), "download_failed")
            stats["failed_count"] += 1
            continue
        if not response.content:
            frame.at[index, "download_status"] = "download_failed"
            frame.at[index, "download_error"] = "empty_response"
            frame.at[index, "quality_flags"] = _quality_flags_with(row.get("quality_flags"), "empty_download")
            stats["failed_count"] += 1
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        frame.at[index, "source_hash"] = _sha256_bytes(response.content)
        frame.at[index, "download_status"] = "downloaded"
        frame.at[index, "downloaded_at"] = datetime.now(UTC).isoformat()
        frame.at[index, "download_error"] = None
        frame.at[index, "text_extraction_status"] = row.get("text_extraction_status") or "not_run"
        stats["downloaded_count"] += 1

    stats["remaining_not_downloaded_count"] = int((frame["download_status"] != "downloaded").sum())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return stats


def summarize_text_corpus_requests(frame: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    frame = _prepare_text_corpus_request_frame(frame)
    if frame.empty:
        return {
            "row_count": 0,
            "ticker_count": 0,
            "form_counts": {},
            "date_limited_source_rate": 0.0,
            "download_status_counts": {},
            "text_extraction_status_counts": {},
            "downloaded_count": 0,
            "source_hash_coverage": 0.0,
            "artifacts": {"text_corpus_requests": repo_relative(output_path)},
        }
    source_hash_coverage = float(frame["source_hash"].notna().mean()) if "source_hash" in frame else 0.0
    return {
        "row_count": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()),
        "form_counts": {str(key): int(value) for key, value in frame["filing_form"].value_counts().items()},
        "date_limited_source_rate": float(frame["date_limited_source"].mean()),
        "download_status_counts": {
            str(key): int(value) for key, value in frame["download_status"].fillna("missing").value_counts().items()
        },
        "text_extraction_status_counts": {
            str(key): int(value)
            for key, value in frame["text_extraction_status"].fillna("missing").value_counts().items()
        },
        "downloaded_count": int((frame["download_status"] == "downloaded").sum()),
        "source_hash_coverage": source_hash_coverage,
        "artifacts": {"text_corpus_requests": repo_relative(output_path)},
    }


def llm_text_feature_output_schema(config: TextFeatureExperimentConfig | None = None) -> dict[str, Any]:
    cfg = config or TextFeatureExperimentConfig()
    feature_properties = {
        feature_name: {
            "type": ["number", "null"],
            "minimum": -1.0 if feature_name.endswith("_change") or feature_name == "tone_change" else 0.0,
            "maximum": 1.0,
        }
        for feature_name in cfg.feature_names
    }
    evidence_properties = {
        feature_name: {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["section", "rationale"],
                "properties": {
                    "section": {"type": "string"},
                    "rationale": {"type": "string"},
                    "short_quote": {
                        "type": ["string", "null"],
                        "description": "Optional source quote. Keep this to 25 words or fewer.",
                    },
                },
            },
        }
        for feature_name in cfg.feature_names
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "ticker",
            "filing_accession",
            "filing_form",
            "feature_values",
            "feature_evidence",
            "quality_flags",
            "parse_failure_rate",
        ],
        "properties": {
            "ticker": {"type": "string"},
            "filing_accession": {"type": "string"},
            "filing_form": {"type": "string"},
            "feature_values": {
                "type": "object",
                "additionalProperties": False,
                "required": list(cfg.feature_names),
                "properties": feature_properties,
            },
            "feature_evidence": {
                "type": "object",
                "additionalProperties": False,
                "required": list(cfg.feature_names),
                "properties": evidence_properties,
            },
            "quality_flags": {"type": "array", "items": {"type": "string"}},
            "parse_failure_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    }


def _llm_system_prompt(config: TextFeatureExperimentConfig | None = None) -> str:
    cfg = config or TextFeatureExperimentConfig()
    return "\n".join(
        [
            "You extract structured, date-limited SEC filing text features for Experiment C.",
            "Use only the supplied filing text and metadata. Do not use DCF labels, model forecasts, prices, returns, or future information.",
            "Return only JSON that validates against the provided schema.",
            "Score features on the requested numeric scale and use null when the filing does not contain enough evidence.",
            "Keep any direct source quote to 25 words or fewer.",
            f"Allowed source sections: {', '.join(cfg.allowed_sources)}.",
        ]
    )


def _llm_user_prompt(row: Mapping[str, Any], config: TextFeatureExperimentConfig | None = None) -> str:
    cfg = config or TextFeatureExperimentConfig()
    metadata = {
        "ticker": row.get("ticker"),
        "cik": row.get("cik"),
        "company_name": row.get("company_name"),
        "filing_accession": row.get("filing_accession"),
        "filing_form": row.get("filing_form"),
        "filing_lookback_rank": row.get("filing_lookback_rank"),
        "filed": row.get("filed"),
        "period_end": row.get("period_end"),
        "source_path": row.get("source_path"),
        "source_hash": row.get("source_hash"),
        "date_limited_source": bool(row.get("date_limited_source")),
    }
    lines = [
        "Extract the Experiment C filing-text features from this date-limited SEC filing.",
        "",
        "Filing metadata:",
        json.dumps(metadata, indent=2, sort_keys=True, default=str),
        "",
        "Feature definitions:",
    ]
    lines.extend(f"- {feature_name}" for feature_name in cfg.feature_names)
    lines.extend(
        [
            "",
            "Output rules:",
            "- Return the exact schema keys only.",
            "- `feature_values` must contain every feature name.",
            "- Use values from 0.0 to 1.0 for level/intensity features.",
            "- Use values from -1.0 to 1.0 for change features.",
            "- Set `quality_flags` for short text, missing section evidence, ambiguous evidence, or parsing limits.",
            "- Do not include raw AI IRR, mechanical DCF IRR, AI-minus-mechanical IRR, AI residual, returns, or price data.",
            "",
            "The orchestrator must inject the filing text after this marker before sending the request:",
            "<FILING_TEXT>",
        ]
    )
    return "\n".join(lines)


def _llm_request_status(row: Mapping[str, Any]) -> tuple[str, list[str]]:
    flags = _normalize_quality_flags(row.get("quality_flags"))
    if not bool(row.get("date_limited_source")):
        return "blocked_not_date_limited_source", _quality_flags_with(flags, "not_date_limited_source")
    if row.get("download_status") != "downloaded":
        return "blocked_missing_downloaded_text", _quality_flags_with(flags, "missing_downloaded_text_for_llm")
    if _is_missing(row.get("source_hash")):
        return "blocked_missing_source_hash", _quality_flags_with(flags, "missing_source_hash_for_llm")
    if _is_missing(row.get("source_path")):
        return "blocked_missing_source_path", _quality_flags_with(flags, "missing_source_path_for_llm")
    return "ready_for_llm_extraction", flags


def build_llm_extraction_requests(
    *,
    text_corpus_requests_path: Path = DEFAULT_TEXT_CORPUS_REQUESTS,
    output_path: Path = DEFAULT_LLM_EXTRACTION_REQUESTS,
    config: TextFeatureExperimentConfig | None = None,
) -> pd.DataFrame:
    cfg = config or TextFeatureExperimentConfig()
    if not text_corpus_requests_path.exists():
        frame = pd.DataFrame(columns=LLM_EXTRACTION_REQUEST_COLUMNS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, index=False)
        return frame
    corpus_requests = _prepare_text_corpus_request_frame(pd.read_parquet(text_corpus_requests_path))
    schema_json = json.dumps(llm_text_feature_output_schema(cfg), sort_keys=True, separators=(",", ":"))
    system_prompt = _llm_system_prompt(cfg)
    rows: list[dict[str, Any]] = []
    for request in corpus_requests.to_dict(orient="records"):
        status, quality_flags = _llm_request_status(request)
        rows.append(
            {
                "ticker": request.get("ticker"),
                "cik": request.get("cik"),
                "company_name": request.get("company_name"),
                "filing_accession": request.get("filing_accession"),
                "filing_form": request.get("filing_form"),
                "filing_lookback_rank": request.get("filing_lookback_rank"),
                "filed": _coerce_date(request.get("filed")),
                "period_end": _coerce_date(request.get("period_end")),
                "source_type": "sec_filing_text_llm_extraction_request",
                "source_path": request.get("source_path"),
                "source_hash": request.get("source_hash"),
                "date_limited_source": bool(request.get("date_limited_source")),
                "download_status": request.get("download_status"),
                "llm_request_status": status,
                "llm_extraction_status": "not_run",
                "prompt_id": LLM_EXTRACTION_PROMPT_ID,
                "prompt_version": LLM_EXTRACTION_PROMPT_VERSION,
                "prompt_model_role": "schema_bound_extractor",
                "system_prompt": system_prompt,
                "user_prompt": _llm_user_prompt(request, cfg),
                "output_schema_json": schema_json,
                "quality_flags": quality_flags,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=LLM_EXTRACTION_REQUEST_COLUMNS)
    else:
        frame = frame.reindex(columns=LLM_EXTRACTION_REQUEST_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return frame


def summarize_llm_extraction_requests(frame: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    if frame.empty:
        return {
            "row_count": 0,
            "ready_for_llm_extraction_count": 0,
            "status_counts": {},
            "prompt_id": LLM_EXTRACTION_PROMPT_ID,
            "prompt_version": LLM_EXTRACTION_PROMPT_VERSION,
            "artifacts": {"llm_extraction_requests": repo_relative(output_path)},
        }
    status_counts = {
        str(key): int(value) for key, value in frame["llm_request_status"].fillna("missing").value_counts().items()
    }
    return {
        "row_count": int(len(frame)),
        "ready_for_llm_extraction_count": int(status_counts.get("ready_for_llm_extraction", 0)),
        "status_counts": status_counts,
        "prompt_id": LLM_EXTRACTION_PROMPT_ID,
        "prompt_version": LLM_EXTRACTION_PROMPT_VERSION,
        "schema_hash": _sha256_bytes(
            json.dumps(llm_text_feature_output_schema(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "artifacts": {"llm_extraction_requests": repo_relative(output_path)},
    }


def build_empty_llm_response_audit() -> pd.DataFrame:
    return pd.DataFrame(columns=LLM_EXTRACTION_RESPONSE_COLUMNS)


def _iter_mapping_keys(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        keys: list[str] = [str(key) for key in value]
        for item in value.values():
            keys.extend(_iter_mapping_keys(item))
        return keys
    if isinstance(value, list):
        keys = []
        for item in value:
            keys.extend(_iter_mapping_keys(item))
        return keys
    return []


def _is_number_or_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def validate_llm_text_feature_response(
    response: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    config: TextFeatureExperimentConfig | None = None,
) -> list[str]:
    cfg = config or TextFeatureExperimentConfig()
    errors: list[str] = []
    required_keys = {
        "ticker",
        "filing_accession",
        "filing_form",
        "feature_values",
        "feature_evidence",
        "quality_flags",
        "parse_failure_rate",
    }
    extra_keys = sorted(set(response) - required_keys)
    missing_keys = sorted(required_keys - set(response))
    if extra_keys:
        errors.append(f"unexpected_top_level_keys:{','.join(extra_keys)}")
    if missing_keys:
        errors.append(f"missing_top_level_keys:{','.join(missing_keys)}")
    forbidden_keys = sorted(FORBIDDEN_EXPERIMENT_C_RESPONSE_KEYS & set(_iter_mapping_keys(response)))
    if forbidden_keys:
        errors.append(f"forbidden_experiment_c_keys:{','.join(forbidden_keys)}")

    for key in ("ticker", "filing_accession", "filing_form"):
        if str(response.get(key)) != str(request.get(key)):
            errors.append(f"{key}_mismatch")

    parse_failure_rate = response.get("parse_failure_rate")
    if not _is_number_or_null(parse_failure_rate) or parse_failure_rate is None:
        errors.append("parse_failure_rate_must_be_number")
    elif not 0.0 <= float(parse_failure_rate) <= 1.0:
        errors.append("parse_failure_rate_out_of_range")

    quality_flags = response.get("quality_flags")
    if not isinstance(quality_flags, list) or any(not isinstance(flag, str) for flag in quality_flags):
        errors.append("quality_flags_must_be_string_list")

    feature_names = set(cfg.feature_names)
    feature_values = response.get("feature_values")
    if not isinstance(feature_values, Mapping):
        errors.append("feature_values_must_be_object")
    else:
        missing_features = sorted(feature_names - set(feature_values))
        extra_features = sorted(set(feature_values) - feature_names)
        if missing_features:
            errors.append(f"feature_values_missing:{','.join(missing_features)}")
        if extra_features:
            errors.append(f"feature_values_extra:{','.join(extra_features)}")
        for feature_name in cfg.feature_names:
            value = feature_values.get(feature_name)
            if not _is_number_or_null(value):
                errors.append(f"{feature_name}_not_numeric_or_null")
                continue
            if value is None:
                continue
            numeric_value = float(value)
            minimum = -1.0 if feature_name.endswith("_change") or feature_name == "tone_change" else 0.0
            if numeric_value < minimum or numeric_value > 1.0:
                errors.append(f"{feature_name}_out_of_range")

    feature_evidence = response.get("feature_evidence")
    if not isinstance(feature_evidence, Mapping):
        errors.append("feature_evidence_must_be_object")
    else:
        missing_evidence = sorted(feature_names - set(feature_evidence))
        extra_evidence = sorted(set(feature_evidence) - feature_names)
        if missing_evidence:
            errors.append(f"feature_evidence_missing:{','.join(missing_evidence)}")
        if extra_evidence:
            errors.append(f"feature_evidence_extra:{','.join(extra_evidence)}")
        for feature_name in cfg.feature_names:
            evidence_items = feature_evidence.get(feature_name)
            if not isinstance(evidence_items, list):
                errors.append(f"{feature_name}_evidence_must_be_list")
                continue
            for item in evidence_items:
                if not isinstance(item, Mapping):
                    errors.append(f"{feature_name}_evidence_item_must_be_object")
                    continue
                item_extra_keys = sorted(set(item) - {"section", "rationale", "short_quote"})
                if item_extra_keys:
                    errors.append(f"{feature_name}_evidence_unexpected_keys:{','.join(item_extra_keys)}")
                if not isinstance(item.get("section"), str) or not item.get("section"):
                    errors.append(f"{feature_name}_evidence_missing_section")
                if not isinstance(item.get("rationale"), str) or not item.get("rationale"):
                    errors.append(f"{feature_name}_evidence_missing_rationale")
                short_quote = item.get("short_quote")
                if short_quote is not None:
                    if not isinstance(short_quote, str):
                        errors.append(f"{feature_name}_evidence_quote_not_string")
                    elif len(short_quote.split()) > 25:
                        errors.append(f"{feature_name}_evidence_quote_too_long")
    return errors


def _load_llm_response_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}:{line_number} did not contain a JSON object.")
        rows.append(dict(payload))
    return rows


def _response_payload_from_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    if "response_json" in record:
        value = record["response_json"]
    elif "response" in record:
        value = record["response"]
    else:
        value = record
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    if not isinstance(parsed, Mapping):
        raise ValueError("LLM response payload must be a JSON object.")
    return parsed


def ingest_llm_extraction_responses(
    *,
    llm_responses_jsonl_path: Path,
    llm_extraction_requests_path: Path = DEFAULT_LLM_EXTRACTION_REQUESTS,
    feature_panel_path: Path = DEFAULT_FEATURE_PANEL,
    response_audit_path: Path = DEFAULT_LLM_EXTRACTION_RESPONSES,
    config: TextFeatureExperimentConfig | None = None,
) -> dict[str, Any]:
    cfg = config or TextFeatureExperimentConfig()
    if not llm_extraction_requests_path.exists():
        raise FileNotFoundError(f"missing LLM extraction requests: {llm_extraction_requests_path}")
    requests = pd.read_parquet(llm_extraction_requests_path)
    if requests.empty:
        response_audit = build_empty_llm_response_audit()
        response_audit_path.parent.mkdir(parents=True, exist_ok=True)
        response_audit.to_parquet(response_audit_path, index=False)
        panel = build_empty_text_feature_panel(cfg)
        feature_panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(feature_panel_path, index=False)
        return summarize_text_feature_panel(panel, feature_panel_path) | {
            "response_row_count": 0,
            "valid_response_count": 0,
            "invalid_response_count": 0,
            "response_audit_path": repo_relative(response_audit_path),
        }

    requests = requests.copy()
    request_by_accession = {
        str(row.filing_accession): row._asdict()
        for row in requests.itertuples(index=False)
        if not _is_missing(getattr(row, "filing_accession", None))
    }
    response_records = _load_llm_response_jsonl(llm_responses_jsonl_path)
    panel_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    now = datetime.now(UTC).isoformat()

    for record in response_records:
        payload = _response_payload_from_record(record)
        accession = str(record.get("filing_accession") or payload.get("filing_accession"))
        request = request_by_accession.get(accession)
        validation_errors: list[str] = []
        if request is None:
            validation_errors.append("unknown_filing_accession")
            request = {
                "ticker": payload.get("ticker"),
                "cik": record.get("cik"),
                "company_name": record.get("company_name"),
                "filing_accession": accession,
                "filing_form": payload.get("filing_form"),
                "filing_lookback_rank": record.get("filing_lookback_rank"),
                "filed": record.get("filed"),
                "period_end": record.get("period_end"),
                "source_path": record.get("source_path"),
                "source_hash": record.get("source_hash"),
                "prompt_id": record.get("prompt_id") or LLM_EXTRACTION_PROMPT_ID,
                "prompt_version": record.get("prompt_version") or LLM_EXTRACTION_PROMPT_VERSION,
                "quality_flags": record.get("quality_flags") or [],
            }
        else:
            validation_errors.extend(validate_llm_text_feature_response(payload, request, config=cfg))

        status = "valid" if not validation_errors else "invalid"
        response_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        llm_model_id = str(record.get("llm_model_id") or record.get("model_id") or LLM_EXTRACTION_IMPORT_MODEL_ID)
        llm_response_id = record.get("llm_response_id") or record.get("response_id")
        quality_flags = _normalize_quality_flags(record.get("quality_flags") or request.get("quality_flags"))
        if status == "invalid":
            quality_flags = [*quality_flags, "llm_response_validation_failed"]

        audit_rows.append(
            {
                "ticker": request.get("ticker"),
                "cik": request.get("cik"),
                "company_name": request.get("company_name"),
                "filing_accession": request.get("filing_accession"),
                "filing_form": request.get("filing_form"),
                "filing_lookback_rank": request.get("filing_lookback_rank"),
                "filed": _coerce_date(request.get("filed")),
                "period_end": _coerce_date(request.get("period_end")),
                "source_path": request.get("source_path"),
                "source_hash": request.get("source_hash"),
                "prompt_id": request.get("prompt_id") or LLM_EXTRACTION_PROMPT_ID,
                "prompt_version": request.get("prompt_version") or LLM_EXTRACTION_PROMPT_VERSION,
                "llm_model_id": llm_model_id,
                "llm_response_id": llm_response_id,
                "response_json": response_json,
                "validation_status": status,
                "validation_errors": validation_errors,
                "extraction_generated_at": now,
                "quality_flags": quality_flags,
            }
        )

        request_mask = requests["filing_accession"].astype(str) == accession
        if not request_mask.any():
            continue
        if status == "invalid":
            requests.loc[request_mask, "llm_extraction_status"] = "validation_failed"
            requests.loc[request_mask, "quality_flags"] = requests.loc[request_mask, "quality_flags"].map(
                lambda value: _quality_flags_with(value, "llm_response_validation_failed")
            )
            continue

        feature_values = dict(payload["feature_values"])
        response_quality_flags = [str(flag) for flag in payload.get("quality_flags", [])]
        panel_rows.append(
            {
                "ticker": request.get("ticker"),
                "cik": request.get("cik"),
                "filing_accession": request.get("filing_accession"),
                "filing_form": request.get("filing_form"),
                "filing_lookback_rank": request.get("filing_lookback_rank"),
                "filed": _coerce_date(request.get("filed")),
                "period_end": _coerce_date(request.get("period_end")),
                "source_type": "sec_filing_text_llm",
                "source_path": request.get("source_path"),
                "source_hash": request.get("source_hash"),
                "extraction_model_id": llm_model_id,
                "prompt_id": request.get("prompt_id") or LLM_EXTRACTION_PROMPT_ID,
                "prompt_version": request.get("prompt_version") or LLM_EXTRACTION_PROMPT_VERSION,
                "extraction_generated_at": now,
                **{feature_name: feature_values.get(feature_name) for feature_name in cfg.feature_names},
                "text_window_start": _coerce_date(request.get("period_end")),
                "text_window_end": _coerce_date(request.get("filed")),
                "date_limited_source": bool(request.get("date_limited_source")),
                "quality_flags": [*_normalize_quality_flags(request.get("quality_flags")), *response_quality_flags],
                "parse_failure_rate": float(payload.get("parse_failure_rate", 0.0)),
            }
        )
        requests.loc[request_mask, "llm_extraction_status"] = "extracted"

    response_audit = pd.DataFrame(audit_rows)
    response_audit = (
        response_audit.reindex(columns=LLM_EXTRACTION_RESPONSE_COLUMNS)
        if not response_audit.empty
        else build_empty_llm_response_audit()
    )
    panel = pd.DataFrame(panel_rows)
    panel = panel.reindex(columns=text_feature_columns(cfg)) if not panel.empty else build_empty_text_feature_panel(cfg)
    response_audit_path.parent.mkdir(parents=True, exist_ok=True)
    response_audit.to_parquet(response_audit_path, index=False)
    feature_panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(feature_panel_path, index=False)
    requests.to_parquet(llm_extraction_requests_path, index=False)
    summary = summarize_text_feature_panel(panel, feature_panel_path)
    summary.update(
        {
            "response_row_count": int(len(response_audit)),
            "valid_response_count": int((response_audit["validation_status"] == "valid").sum())
            if "validation_status" in response_audit
            else 0,
            "invalid_response_count": int((response_audit["validation_status"] == "invalid").sum())
            if "validation_status" in response_audit
            else 0,
            "response_audit_path": repo_relative(response_audit_path),
        }
    )
    return summary


def summarize_llm_response_audit(frame: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    if frame.empty:
        return {
            "row_count": 0,
            "valid_response_count": 0,
            "invalid_response_count": 0,
            "validation_status_counts": {},
            "artifacts": {"llm_extraction_responses": repo_relative(output_path)},
        }
    status_counts = {
        str(key): int(value) for key, value in frame["validation_status"].fillna("missing").value_counts().items()
    }
    return {
        "row_count": int(len(frame)),
        "valid_response_count": int(status_counts.get("valid", 0)),
        "invalid_response_count": int(status_counts.get("invalid", 0)),
        "validation_status_counts": status_counts,
        "artifacts": {"llm_extraction_responses": repo_relative(output_path)},
    }


def _keyword_count(text: str, terms: Sequence[str]) -> int:
    normalized = text.lower()
    count = 0
    for term in terms:
        pattern = re.escape(term.lower())
        count += len(re.findall(rf"\b{pattern}\b", normalized))
    return int(count)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9_]{2,}", text.lower()))


def _density_score(text: str, terms: Sequence[str], word_count: int) -> float:
    if word_count <= 0:
        return 0.0
    # A density of ten matched phrases per thousand words maps to a full score.
    return float(min(1.0, (_keyword_count(text, terms) / word_count * 1000.0) / 10.0))


def _tone_level(text: str) -> float:
    positive = _keyword_count(text, KEYWORD_GROUPS["positive_tone"])
    negative = _keyword_count(text, KEYWORD_GROUPS["negative_tone"])
    total = positive + negative
    if total == 0:
        return 0.0
    return float((positive - negative) / total)


def _jaccard_novelty(current_text: str, previous_text: str | None) -> float | None:
    if not previous_text:
        return None
    current_tokens = _tokens(current_text)
    previous_tokens = _tokens(previous_text)
    union = current_tokens | previous_tokens
    if not union:
        return None
    return float(1.0 - (len(current_tokens & previous_tokens) / len(union)))


def extract_text_features_from_text(text: str, *, previous_text: str | None = None) -> tuple[dict[str, float | None], list[str]]:
    word_count = len(re.findall(r"[A-Za-z][A-Za-z0-9_'-]*", text))
    flags = ["deterministic_keyword_baseline_not_llm"]
    if word_count < 200:
        flags.append("short_text")

    tone = _tone_level(text)
    uncertainty = _density_score(text, KEYWORD_GROUPS["uncertainty"], word_count)
    if previous_text:
        previous_tone = _tone_level(previous_text)
        previous_uncertainty = _density_score(
            previous_text,
            KEYWORD_GROUPS["uncertainty"],
            len(re.findall(r"[A-Za-z][A-Za-z0-9_'-]*", previous_text)),
        )
        tone_change: float | None = float(tone - previous_tone)
        uncertainty_change: float | None = float(uncertainty - previous_uncertainty)
    else:
        tone_change = None
        uncertainty_change = None
        flags.append("missing_prior_text_for_change_features")

    features: dict[str, float | None] = {
        "tone_change": tone_change,
        "uncertainty_change": uncertainty_change,
        "risk_factor_novelty": _jaccard_novelty(text, previous_text),
        "management_hedging": _density_score(text, KEYWORD_GROUPS["management_hedging"], word_count),
        "capital_allocation_discipline": _density_score(text, KEYWORD_GROUPS["capital_allocation_discipline"], word_count),
        "competitive_pressure": _density_score(text, KEYWORD_GROUPS["competitive_pressure"], word_count),
        "pricing_power": _density_score(text, KEYWORD_GROUPS["pricing_power"], word_count),
        "demand_weakness": _density_score(text, KEYWORD_GROUPS["demand_weakness"], word_count),
        "supply_chain_stress": _density_score(text, KEYWORD_GROUPS["supply_chain_stress"], word_count),
        "regulatory_pressure": _density_score(text, KEYWORD_GROUPS["regulatory_pressure"], word_count),
        "accounting_aggressiveness": _density_score(text, KEYWORD_GROUPS["accounting_aggressiveness"], word_count),
        "guidance_credibility": _density_score(text, KEYWORD_GROUPS["guidance_credibility"], word_count),
    }
    return features, flags


def summarize_text_feature_panel(panel: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    panel = panel.copy()
    config = TextFeatureExperimentConfig()
    feature_non_null_counts = {
        feature_name: int(panel[feature_name].notna().sum()) if feature_name in panel else 0
        for feature_name in config.feature_names
    }
    quality_counter: Counter[str] = Counter()
    if "quality_flags" in panel:
        for value in panel["quality_flags"]:
            quality_counter.update(_normalize_quality_flags(value))
    extraction_models = (
        sorted(str(value) for value in panel["extraction_model_id"].dropna().unique())
        if "extraction_model_id" in panel and not panel.empty
        else []
    )
    return {
        "row_count": int(len(panel)),
        "ticker_count": int(panel["ticker"].nunique()) if "ticker" in panel and not panel.empty else 0,
        "extraction_models": extraction_models,
        "feature_non_null_counts": feature_non_null_counts,
        "quality_flag_counts": dict(sorted(quality_counter.items())),
        "artifacts": {"feature_panel": repo_relative(output_path)},
    }


def extract_text_feature_panel_from_requests(
    *,
    requests_path: Path = DEFAULT_TEXT_CORPUS_REQUESTS,
    feature_panel_path: Path = DEFAULT_FEATURE_PANEL,
    root: Path = ROOT,
    config: TextFeatureExperimentConfig | None = None,
) -> dict[str, Any]:
    cfg = config or TextFeatureExperimentConfig()
    requests = _prepare_text_corpus_request_frame(pd.read_parquet(requests_path))
    if "filing_lookback_rank" not in requests.columns:
        requests["filing_lookback_rank"] = None
    requests["filed_ts"] = pd.to_datetime(requests["filed"], errors="coerce")
    requests = requests.sort_values(["ticker", "filing_form", "filed_ts"], kind="mergesort").reset_index(drop=True)

    panel_rows: list[dict[str, Any]] = []
    previous_text_by_key: dict[tuple[str, str], str] = {}
    extracted_count = 0
    failed_count = 0

    for index, row in requests.iterrows():
        if row.get("download_status") != "downloaded":
            continue
        source_path = _resolve_source_path(row.get("source_path"), root=root)
        if source_path is None or not source_path.exists():
            requests.at[index, "text_extraction_status"] = "extraction_failed"
            requests.at[index, "quality_flags"] = _quality_flags_with(row.get("quality_flags"), "missing_downloaded_text_file")
            failed_count += 1
            continue
        payload = source_path.read_bytes()
        text = payload.decode("utf-8", errors="replace")
        source_hash = str(row.get("source_hash") or _sha256_bytes(payload))
        if _is_missing(row.get("source_hash")):
            requests.at[index, "source_hash"] = source_hash
        key = (str(row.get("ticker")), str(row.get("filing_form")))
        previous_text = previous_text_by_key.get(key)
        features, extraction_flags = extract_text_features_from_text(text, previous_text=previous_text)
        quality_flags = [*_normalize_quality_flags(row.get("quality_flags")), *extraction_flags]
        panel_rows.append(
            {
                "ticker": row.get("ticker"),
                "cik": row.get("cik"),
                "filing_accession": row.get("filing_accession"),
                "filing_form": row.get("filing_form"),
                "filing_lookback_rank": row.get("filing_lookback_rank"),
                "filed": _coerce_date(row.get("filed")),
                "period_end": _coerce_date(row.get("period_end")),
                "source_type": "sec_filing_text",
                "source_path": row.get("source_path"),
                "source_hash": source_hash,
                "extraction_model_id": DETERMINISTIC_EXTRACTION_MODEL_ID,
                "prompt_id": "keyword_baseline_no_prompt",
                "prompt_version": "v1",
                "extraction_generated_at": datetime.now(UTC).isoformat(),
                **features,
                "text_window_start": _coerce_date(row.get("period_end")),
                "text_window_end": _coerce_date(row.get("filed")),
                "date_limited_source": bool(row.get("date_limited_source")),
                "quality_flags": quality_flags,
                "parse_failure_rate": 0.0,
            }
        )
        requests.at[index, "text_extraction_status"] = "extracted"
        previous_text_by_key[key] = text
        extracted_count += 1

    panel = pd.DataFrame(panel_rows)
    if panel.empty:
        panel = build_empty_text_feature_panel(cfg)
    else:
        panel = panel.reindex(columns=text_feature_columns(cfg))
    feature_panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(feature_panel_path, index=False)
    requests = requests.drop(columns=["filed_ts"])
    requests.to_parquet(requests_path, index=False)
    summary = summarize_text_feature_panel(panel, feature_panel_path)
    summary.update(
        {
            "extracted_count": int(extracted_count),
            "failed_extraction_count": int(failed_count),
            "requests_path": repo_relative(requests_path),
        }
    )
    return summary


def write_experiment_c_manifest_from_requests(
    *,
    feature_panel_path: Path = DEFAULT_FEATURE_PANEL,
    text_corpus_requests_path: Path = DEFAULT_TEXT_CORPUS_REQUESTS,
    llm_extraction_requests_path: Path = DEFAULT_LLM_EXTRACTION_REQUESTS,
    llm_extraction_responses_path: Path = DEFAULT_LLM_EXTRACTION_RESPONSES,
    manifest_path: Path = DEFAULT_MANIFEST,
    metadata_path: Path = DEFAULT_METADATA,
    security_master_path: Path = DEFAULT_SECURITY_MASTER,
    text_corpus_path: Path | None = None,
    text_feature_panel_summary: Mapping[str, Any] | None = None,
    rebuild_llm_requests: bool = True,
) -> dict[str, Any]:
    requests = _prepare_text_corpus_request_frame(pd.read_parquet(text_corpus_requests_path))
    request_summary = summarize_text_corpus_requests(requests, text_corpus_requests_path)
    if rebuild_llm_requests or not llm_extraction_requests_path.exists():
        llm_requests = build_llm_extraction_requests(
            text_corpus_requests_path=text_corpus_requests_path,
            output_path=llm_extraction_requests_path,
        )
    else:
        llm_requests = pd.read_parquet(llm_extraction_requests_path)
    llm_request_summary = summarize_llm_extraction_requests(llm_requests, llm_extraction_requests_path)
    llm_responses = (
        pd.read_parquet(llm_extraction_responses_path)
        if llm_extraction_responses_path.exists()
        else build_empty_llm_response_audit()
    )
    if not llm_extraction_responses_path.exists():
        llm_extraction_responses_path.parent.mkdir(parents=True, exist_ok=True)
        llm_responses.to_parquet(llm_extraction_responses_path, index=False)
    llm_response_summary = summarize_llm_response_audit(llm_responses, llm_extraction_responses_path)
    manifest = build_experiment_c_manifest(
        feature_panel_path=feature_panel_path,
        text_corpus_requests_path=text_corpus_requests_path,
        llm_extraction_requests_path=llm_extraction_requests_path,
        llm_extraction_responses_path=llm_extraction_responses_path,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        security_master_path=security_master_path,
        text_corpus_path=text_corpus_path,
        text_corpus_request_summary=request_summary,
        llm_extraction_request_summary=llm_request_summary,
        llm_extraction_response_summary=llm_response_summary,
        text_feature_panel_summary=text_feature_panel_summary,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    config = TextFeatureExperimentConfig()
    metadata = build_experiment_metadata(
        experiment_id=config.experiment_id,
        feature_config={
            "feature_names": list(config.feature_names),
            "allowed_sources": list(config.allowed_sources),
            "text_corpus_request_summary": request_summary,
            "llm_extraction_request_summary": llm_request_summary,
            "llm_extraction_response_summary": llm_response_summary,
            "text_feature_panel_summary": dict(text_feature_panel_summary or {}),
            "deterministic_extraction_model_id": DETERMINISTIC_EXTRACTION_MODEL_ID,
            "llm_prompt_id": LLM_EXTRACTION_PROMPT_ID,
            "llm_prompt_version": LLM_EXTRACTION_PROMPT_VERSION,
            "llm_output_schema": llm_text_feature_output_schema(),
        },
        model_config={
            "extraction_model_id": DETERMINISTIC_EXTRACTION_MODEL_ID,
            "llm_extraction_not_run": True,
        },
        universe_config={"security_master": repo_relative(security_master_path), "survivor_bias_caveat": True},
        backtest_config={"separate_from_dcf_label_experiments": True},
        data_snapshot_paths={"security_master": security_master_path},
        artifacts={
            "feature_panel": feature_panel_path,
            "text_corpus_requests": text_corpus_requests_path,
            "llm_extraction_requests": llm_extraction_requests_path,
            "llm_extraction_responses": llm_extraction_responses_path,
            "manifest": manifest_path,
        },
    )
    write_experiment_metadata(metadata_path, metadata)
    return manifest


def build_experiment_c_manifest(
    *,
    feature_panel_path: Path = DEFAULT_FEATURE_PANEL,
    text_corpus_requests_path: Path = DEFAULT_TEXT_CORPUS_REQUESTS,
    llm_extraction_requests_path: Path = DEFAULT_LLM_EXTRACTION_REQUESTS,
    llm_extraction_responses_path: Path = DEFAULT_LLM_EXTRACTION_RESPONSES,
    manifest_path: Path = DEFAULT_MANIFEST,
    metadata_path: Path = DEFAULT_METADATA,
    security_master_path: Path = DEFAULT_SECURITY_MASTER,
    text_corpus_path: Path | None = None,
    text_corpus_request_summary: Mapping[str, Any] | None = None,
    llm_extraction_request_summary: Mapping[str, Any] | None = None,
    llm_extraction_response_summary: Mapping[str, Any] | None = None,
    text_feature_panel_summary: Mapping[str, Any] | None = None,
    config: TextFeatureExperimentConfig | None = None,
) -> dict[str, Any]:
    cfg = config or TextFeatureExperimentConfig()
    blockers: list[dict[str, str]] = []
    request_summary = dict(text_corpus_request_summary or {})
    llm_request_summary = dict(llm_extraction_request_summary or {})
    llm_response_summary = dict(llm_extraction_response_summary or {})
    feature_panel_summary = dict(text_feature_panel_summary or {})
    request_count = int(request_summary.get("row_count") or 0)
    downloaded_count = int(request_summary.get("downloaded_count") or 0)
    ready_llm_request_count = int(llm_request_summary.get("ready_for_llm_extraction_count") or 0)
    valid_llm_response_count = int(llm_response_summary.get("valid_response_count") or 0)
    extracted_count = int(feature_panel_summary.get("row_count") or feature_panel_summary.get("extracted_count") or 0)
    extraction_models = {str(value) for value in feature_panel_summary.get("extraction_models", [])}
    complete_request_download = request_count > 0 and downloaded_count == request_count
    if text_corpus_path is None or not text_corpus_path.exists() or not complete_request_download:
        blockers.append(
            {
                "code": "missing_date_limited_text_corpus",
                "message": "The date-limited 10-K/10-Q filing text corpus is missing or incomplete for Experiment C extraction.",
            }
        )
    if text_corpus_requests_path is None or not text_corpus_requests_path.exists():
        blockers.append(
            {
                "code": "missing_text_corpus_request_manifest",
                "message": "No SEC filing text request manifest was written for Experiment C.",
            }
        )
    if extracted_count == 0:
        blockers.append(
            {
                "code": "text_extraction_not_run",
                "message": "Text-feature extraction has not been run; the feature parquet artifact is a schema placeholder.",
            }
        )
    if request_count > 0 and ready_llm_request_count == 0:
        blockers.append(
            {
                "code": "llm_extraction_requests_not_ready",
                "message": "LLM extraction requests are schema-bound, but no request has a downloaded date-limited source hash yet.",
            }
        )
    if valid_llm_response_count == 0:
        blockers.append(
            {
                "code": "llm_text_extraction_not_run",
                "message": "No validated LLM extraction response has been ingested for Experiment C.",
            }
        )
    if DETERMINISTIC_EXTRACTION_MODEL_ID in extraction_models:
        blockers.append(
            {
                "code": "llm_text_extraction_not_run",
                "message": "A deterministic keyword baseline populated the text-feature schema, but LLM extraction has not been run.",
            }
        )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment_id": cfg.experiment_id,
        "status": "blocked" if blockers else "feature_panel_ready",
        "separate_from_dcf_labels": cfg.separate_from_dcf_labels,
        "allowed_sources": list(cfg.allowed_sources),
        "feature_names": list(cfg.feature_names),
        "deterministic_extraction_model_id": DETERMINISTIC_EXTRACTION_MODEL_ID,
        "llm_prompt_contract": {
            "prompt_id": LLM_EXTRACTION_PROMPT_ID,
            "prompt_version": LLM_EXTRACTION_PROMPT_VERSION,
            "output_schema": llm_text_feature_output_schema(cfg),
            "llm_extraction_not_run": True,
        },
        "required_source_controls": {
            "date_limited_source": True,
            "accepted_timestamp_required": True,
            "filing_accession_required": True,
            "source_hash_required": True,
            "no_dcf_label_columns": True,
        },
        "artifacts": {
            "feature_panel": repo_relative(feature_panel_path),
            "text_corpus_requests": repo_relative(text_corpus_requests_path),
            "llm_extraction_requests": repo_relative(llm_extraction_requests_path),
            "llm_extraction_responses": repo_relative(llm_extraction_responses_path),
            "manifest": repo_relative(manifest_path),
            "metadata": repo_relative(metadata_path),
        },
        "inputs": {
            "security_master": repo_relative(security_master_path),
            "text_corpus": repo_relative(text_corpus_path) if text_corpus_path else None,
        },
        "text_corpus_request_summary": request_summary,
        "llm_extraction_request_summary": llm_request_summary,
        "llm_extraction_response_summary": llm_response_summary,
        "text_feature_panel_summary": feature_panel_summary,
        "blockers": blockers,
        "claim_ceiling": (
            "deterministic_text_feature_panel_no_alpha_evidence"
            if extracted_count
            else "corpus_request_manifest_only_no_alpha_evidence"
        ),
    }


def write_experiment_c_artifacts(
    *,
    feature_panel_path: Path = DEFAULT_FEATURE_PANEL,
    text_corpus_requests_path: Path = DEFAULT_TEXT_CORPUS_REQUESTS,
    llm_extraction_requests_path: Path = DEFAULT_LLM_EXTRACTION_REQUESTS,
    llm_extraction_responses_path: Path = DEFAULT_LLM_EXTRACTION_RESPONSES,
    manifest_path: Path = DEFAULT_MANIFEST,
    metadata_path: Path = DEFAULT_METADATA,
    security_master_path: Path = DEFAULT_SECURITY_MASTER,
    quarterly_fundamentals_path: Path = DEFAULT_QUARTERLY_FUNDAMENTALS,
    companyfacts_dir: Path = DEFAULT_COMPANYFACTS_DIR,
    text_corpus_path: Path | None = None,
    lookback_filings_per_form: int = DEFAULT_TEXT_CORPUS_LOOKBACK_FILINGS_PER_FORM,
) -> dict[str, Any]:
    config = TextFeatureExperimentConfig()
    panel = build_empty_text_feature_panel(config)
    feature_panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(feature_panel_path, index=False)
    feature_panel_summary = summarize_text_feature_panel(panel, feature_panel_path)
    requests = build_text_corpus_requests(
        security_master_path=security_master_path,
        quarterly_fundamentals_path=quarterly_fundamentals_path,
        companyfacts_dir=companyfacts_dir,
        lookback_filings_per_form=lookback_filings_per_form,
    )
    text_corpus_requests_path.parent.mkdir(parents=True, exist_ok=True)
    requests.to_parquet(text_corpus_requests_path, index=False)
    request_summary = summarize_text_corpus_requests(requests, text_corpus_requests_path)
    llm_requests = build_llm_extraction_requests(
        text_corpus_requests_path=text_corpus_requests_path,
        output_path=llm_extraction_requests_path,
        config=config,
    )
    llm_request_summary = summarize_llm_extraction_requests(llm_requests, llm_extraction_requests_path)
    llm_responses = build_empty_llm_response_audit()
    llm_extraction_responses_path.parent.mkdir(parents=True, exist_ok=True)
    llm_responses.to_parquet(llm_extraction_responses_path, index=False)
    llm_response_summary = summarize_llm_response_audit(llm_responses, llm_extraction_responses_path)
    manifest = build_experiment_c_manifest(
        feature_panel_path=feature_panel_path,
        text_corpus_requests_path=text_corpus_requests_path,
        llm_extraction_requests_path=llm_extraction_requests_path,
        llm_extraction_responses_path=llm_extraction_responses_path,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        security_master_path=security_master_path,
        text_corpus_path=text_corpus_path,
        text_corpus_request_summary=request_summary,
        llm_extraction_request_summary=llm_request_summary,
        llm_extraction_response_summary=llm_response_summary,
        text_feature_panel_summary=feature_panel_summary,
        config=config,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    metadata = build_experiment_metadata(
        experiment_id=config.experiment_id,
        feature_config={
            "feature_names": list(config.feature_names),
            "allowed_sources": list(config.allowed_sources),
            "output_columns": list(panel.columns),
            "text_corpus_request_columns": list(requests.columns),
            "llm_extraction_request_columns": list(llm_requests.columns),
            "llm_extraction_request_summary": llm_request_summary,
            "llm_extraction_response_columns": list(llm_responses.columns),
            "llm_extraction_response_summary": llm_response_summary,
            "lookback_filings_per_form": int(lookback_filings_per_form),
            "deterministic_extraction_model_id": DETERMINISTIC_EXTRACTION_MODEL_ID,
            "llm_prompt_id": LLM_EXTRACTION_PROMPT_ID,
            "llm_prompt_version": LLM_EXTRACTION_PROMPT_VERSION,
            "llm_output_schema": llm_text_feature_output_schema(config),
        },
        model_config={"extraction_not_run": True},
        universe_config={"security_master": repo_relative(security_master_path), "survivor_bias_caveat": True},
        backtest_config={"separate_from_dcf_label_experiments": True},
        data_snapshot_paths={
            "security_master": security_master_path,
            **({"quarterly_fundamentals": quarterly_fundamentals_path} if quarterly_fundamentals_path.exists() else {}),
        },
        artifacts={
            "feature_panel": feature_panel_path,
            "text_corpus_requests": text_corpus_requests_path,
            "llm_extraction_requests": llm_extraction_requests_path,
            "llm_extraction_responses": llm_extraction_responses_path,
            "manifest": manifest_path,
        },
    )
    write_experiment_metadata(metadata_path, metadata)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Experiment C text-feature schema and blocker artifacts.")
    parser.add_argument("--feature-panel-output", default=str(DEFAULT_FEATURE_PANEL))
    parser.add_argument("--text-corpus-requests-output", default=str(DEFAULT_TEXT_CORPUS_REQUESTS))
    parser.add_argument("--llm-extraction-requests-output", default=str(DEFAULT_LLM_EXTRACTION_REQUESTS))
    parser.add_argument("--llm-extraction-responses-output", default=str(DEFAULT_LLM_EXTRACTION_RESPONSES))
    parser.add_argument("--manifest-output", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--metadata-output", default=str(DEFAULT_METADATA))
    parser.add_argument("--security-master", default=str(DEFAULT_SECURITY_MASTER))
    parser.add_argument("--quarterly-fundamentals", default=str(DEFAULT_QUARTERLY_FUNDAMENTALS))
    parser.add_argument("--companyfacts-dir", default=str(DEFAULT_COMPANYFACTS_DIR))
    parser.add_argument("--text-corpus")
    parser.add_argument("--lookback-filings-per-form", type=int, default=DEFAULT_TEXT_CORPUS_LOOKBACK_FILINGS_PER_FORM)
    parser.add_argument(
        "--download-corpus",
        action="store_true",
        help="Download SEC filing text for existing request rows instead of rebuilding Experiment C artifacts.",
    )
    parser.add_argument(
        "--extract-features",
        action="store_true",
        help="Extract deterministic baseline text features from downloaded SEC filing text.",
    )
    parser.add_argument(
        "--ingest-llm-responses",
        action="store_true",
        help="Validate offline LLM JSONL responses and write Experiment C LLM text-feature rows.",
    )
    parser.add_argument(
        "--llm-responses-jsonl",
        help="JSONL file containing one LLM response wrapper per filing accession.",
    )
    parser.add_argument(
        "--download-limit",
        type=int,
        default=0,
        help="Maximum SEC filing texts to download in this run. Default 0 performs no network downloads; use -1 for all.",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--download-delay-seconds", type=float, default=SEC_TEXT_DEFAULT_DELAY_SECONDS)
    parser.add_argument("--user-agent-name", default=DEFAULT_SEC_NAME)
    parser.add_argument("--user-agent-email", default=DEFAULT_SEC_EMAIL)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ingest_llm_responses:
        if not args.llm_responses_jsonl:
            raise SystemExit("--ingest-llm-responses requires --llm-responses-jsonl")
        feature_summary = ingest_llm_extraction_responses(
            llm_responses_jsonl_path=Path(args.llm_responses_jsonl),
            llm_extraction_requests_path=Path(args.llm_extraction_requests_output),
            feature_panel_path=Path(args.feature_panel_output),
            response_audit_path=Path(args.llm_extraction_responses_output),
        )
        manifest = write_experiment_c_manifest_from_requests(
            feature_panel_path=Path(args.feature_panel_output),
            text_corpus_requests_path=Path(args.text_corpus_requests_output),
            llm_extraction_requests_path=Path(args.llm_extraction_requests_output),
            llm_extraction_responses_path=Path(args.llm_extraction_responses_output),
            manifest_path=Path(args.manifest_output),
            metadata_path=Path(args.metadata_output),
            security_master_path=Path(args.security_master),
            text_corpus_path=Path(args.text_corpus) if args.text_corpus else ROOT / "data" / "sec_filing_text",
            text_feature_panel_summary=feature_summary,
            rebuild_llm_requests=False,
        )
        feature_summary["manifest_status"] = manifest["status"]
        feature_summary["manifest_path"] = repo_relative(args.manifest_output)
        print(json.dumps(feature_summary, indent=2, sort_keys=True))
        return 0
    if args.extract_features:
        feature_summary = extract_text_feature_panel_from_requests(
            requests_path=Path(args.text_corpus_requests_output),
            feature_panel_path=Path(args.feature_panel_output),
        )
        manifest = write_experiment_c_manifest_from_requests(
            feature_panel_path=Path(args.feature_panel_output),
            text_corpus_requests_path=Path(args.text_corpus_requests_output),
            llm_extraction_requests_path=Path(args.llm_extraction_requests_output),
            llm_extraction_responses_path=Path(args.llm_extraction_responses_output),
            manifest_path=Path(args.manifest_output),
            metadata_path=Path(args.metadata_output),
            security_master_path=Path(args.security_master),
            text_corpus_path=Path(args.text_corpus) if args.text_corpus else ROOT / "data" / "sec_filing_text",
            text_feature_panel_summary=feature_summary,
        )
        feature_summary["manifest_status"] = manifest["status"]
        feature_summary["manifest_path"] = repo_relative(args.manifest_output)
        print(json.dumps(feature_summary, indent=2, sort_keys=True))
        return 0
    if args.download_corpus:
        if args.download_limit != 0 and _uses_placeholder_sec_contact(args.user_agent_name, args.user_agent_email):
            raise SystemExit(
                "SEC text downloads require --user-agent-name and --user-agent-email, "
                "or SEC_USER_AGENT_NAME/SEC_USER_AGENT_EMAIL environment variables."
            )
        stats = download_text_corpus_requests(
            requests_path=Path(args.text_corpus_requests_output),
            output_path=Path(args.text_corpus_requests_output),
            limit=args.download_limit,
            force=args.force_download,
            user_agent_name=args.user_agent_name,
            user_agent_email=args.user_agent_email,
            delay_seconds=args.download_delay_seconds,
        )
        manifest = write_experiment_c_manifest_from_requests(
            feature_panel_path=Path(args.feature_panel_output),
            text_corpus_requests_path=Path(args.text_corpus_requests_output),
            llm_extraction_requests_path=Path(args.llm_extraction_requests_output),
            llm_extraction_responses_path=Path(args.llm_extraction_responses_output),
            manifest_path=Path(args.manifest_output),
            metadata_path=Path(args.metadata_output),
            security_master_path=Path(args.security_master),
            text_corpus_path=Path(args.text_corpus) if args.text_corpus else ROOT / "data" / "sec_filing_text",
            text_feature_panel_summary=summarize_text_feature_panel(
                pd.read_parquet(Path(args.feature_panel_output)) if Path(args.feature_panel_output).exists() else build_empty_text_feature_panel(),
                Path(args.feature_panel_output),
            ),
        )
        stats["manifest_status"] = manifest["status"]
        stats["manifest_path"] = repo_relative(args.manifest_output)
        print(json.dumps(stats, indent=2, sort_keys=True))
        return 0
    manifest = write_experiment_c_artifacts(
        feature_panel_path=Path(args.feature_panel_output),
        text_corpus_requests_path=Path(args.text_corpus_requests_output),
        llm_extraction_requests_path=Path(args.llm_extraction_requests_output),
        llm_extraction_responses_path=Path(args.llm_extraction_responses_output),
        manifest_path=Path(args.manifest_output),
        metadata_path=Path(args.metadata_output),
        security_master_path=Path(args.security_master),
        quarterly_fundamentals_path=Path(args.quarterly_fundamentals),
        companyfacts_dir=Path(args.companyfacts_dir),
        text_corpus_path=Path(args.text_corpus) if args.text_corpus else None,
        lookback_filings_per_form=args.lookback_filings_per_form,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
