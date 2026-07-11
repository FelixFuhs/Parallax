import hashlib
import json

import pandas as pd

from experiment_c_text_features import (
    DETERMINISTIC_EXTRACTION_MODEL_ID,
    LLM_EXTRACTION_PROMPT_ID,
    LLM_EXTRACTION_PROMPT_VERSION,
    TextDownloadResponse,
    build_empty_text_feature_panel,
    build_experiment_c_manifest,
    build_llm_extraction_requests,
    build_text_corpus_requests,
    download_text_corpus_requests,
    extract_text_feature_panel_from_requests,
    ingest_llm_extraction_responses,
    llm_text_feature_output_schema,
    validate_llm_text_feature_response,
    write_experiment_c_artifacts,
)


def test_empty_text_feature_panel_has_expected_schema():
    panel = build_empty_text_feature_panel()

    assert panel.empty
    assert "risk_factor_novelty" in panel.columns
    assert "filing_accession" in panel.columns
    assert "raw_ai_implied_irr" not in panel.columns


def test_experiment_c_manifest_blocks_without_date_limited_text_corpus(tmp_path):
    manifest = build_experiment_c_manifest(
        feature_panel_path=tmp_path / "panel.parquet",
        text_corpus_requests_path=tmp_path / "requests.parquet",
        manifest_path=tmp_path / "manifest.json",
        metadata_path=tmp_path / "metadata.json",
        security_master_path=tmp_path / "security_master.parquet",
    )

    assert manifest["experiment_id"] == "experiment_c_llm_text_features"
    assert manifest["separate_from_dcf_labels"] is True
    assert manifest["status"] == "blocked"
    assert {blocker["code"] for blocker in manifest["blockers"]} == {
        "missing_date_limited_text_corpus",
        "missing_text_corpus_request_manifest",
        "text_extraction_not_run",
        "llm_text_extraction_not_run",
    }


def test_text_corpus_requests_build_sec_archives_urls(tmp_path):
    security_master = tmp_path / "security_master.parquet"
    quarterly = tmp_path / "quarterly.parquet"
    companyfacts_dir = tmp_path / "companyfacts"
    companyfacts_dir.mkdir()
    pd.DataFrame({"ticker": ["ABC"], "cik": [1234], "company_name": ["ABC Inc."]}).to_parquet(
        security_master,
        index=False,
    )
    pd.DataFrame(
        {
            "ticker": ["ABC", "ABC"],
            "cik": ["0000001234", "0000001234"],
            "company_name": ["ABC Inc.", "ABC Inc."],
            "filing_accession": ["0000001234-26-000010", "0000001234-25-000009"],
            "filing_form": ["10-Q", "10-Q"],
            "filed": ["2026-05-01", "2025-11-01"],
            "period_end": ["2026-03-31", "2025-09-30"],
        }
    ).to_parquet(quarterly, index=False)
    (companyfacts_dir / "0000001234.json").write_text(
        """
        {
          "entityName": "ABC Inc.",
          "facts": {
            "us-gaap": {
              "Revenue": {
                "units": {
                  "USD": [
                    {
                      "accn": "0000001234-26-000001",
                      "form": "10-K",
                      "filed": "2026-02-15",
                      "end": "2025-12-31",
                      "val": 1
                    }
                  ]
                }
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )

    requests = build_text_corpus_requests(
        security_master_path=security_master,
        quarterly_fundamentals_path=quarterly,
        companyfacts_dir=companyfacts_dir,
    )

    assert set(requests["filing_form"]) == {"10-K", "10-Q"}
    assert len(requests.loc[requests["filing_form"] == "10-Q"]) == 2
    assert requests["date_limited_source"].all()
    assert requests["source_url"].str.contains("sec.gov/Archives/edgar/data/1234").all()
    assert set(requests["download_status"]) == {"not_downloaded"}
    assert {"filing_lookback_rank", "downloaded_at", "http_status", "download_error"}.issubset(requests.columns)
    assert set(requests.loc[requests["filing_form"] == "10-Q", "filing_lookback_rank"]) == {1, 2}


def test_llm_extraction_requests_are_schema_bound_and_claim_safe(tmp_path):
    requests_path = tmp_path / "requests.parquet"
    llm_requests_path = tmp_path / "llm_requests.parquet"
    pd.DataFrame(
        {
            "ticker": ["ABC", "XYZ"],
            "cik": ["0000001234", "0000005678"],
            "company_name": ["ABC Inc.", "XYZ Inc."],
            "filing_accession": ["0000001234-26-000001", "0000005678-26-000001"],
            "filing_form": ["10-K", "10-Q"],
            "filing_lookback_rank": [1, 1],
            "filed": ["2026-02-15", "2026-05-01"],
            "period_end": ["2025-12-31", "2026-03-31"],
            "source_type": ["sec_filing_text_request", "sec_filing_text_request"],
            "source_url": ["https://www.sec.gov/abc.txt", "https://www.sec.gov/xyz.txt"],
            "source_path": [
                "data/sec_filing_text/0000001234/0000001234-26-000001.txt",
                "data/sec_filing_text/0000005678/0000005678-26-000001.txt",
            ],
            "source_hash": ["a" * 64, None],
            "date_limited_source": [True, True],
            "download_status": ["downloaded", "not_downloaded"],
            "downloaded_at": ["2026-05-19T00:00:00+00:00", None],
            "http_status": [200, None],
            "download_error": [None, None],
            "text_extraction_status": ["not_run", "not_run"],
            "quality_flags": [[], []],
        }
    ).to_parquet(requests_path, index=False)

    llm_requests = build_llm_extraction_requests(
        text_corpus_requests_path=requests_path,
        output_path=llm_requests_path,
    )
    schema = llm_text_feature_output_schema()

    assert llm_requests_path.exists()
    assert set(llm_requests["llm_request_status"]) == {
        "ready_for_llm_extraction",
        "blocked_missing_downloaded_text",
    }
    assert set(llm_requests["llm_extraction_status"]) == {"not_run"}
    assert llm_requests["system_prompt"].str.contains("Do not use DCF labels").all()
    assert llm_requests["user_prompt"].str.contains("<FILING_TEXT>").all()
    assert "raw_ai_implied_irr" not in " ".join(llm_requests["output_schema_json"])
    assert llm_requests["user_prompt"].str.contains("Do not include raw AI IRR").all()
    assert "risk_factor_novelty" in schema["properties"]["feature_values"]["properties"]
    assert json.loads(llm_requests.loc[0, "output_schema_json"])["additionalProperties"] is False


def _valid_llm_response(*, ticker: str = "ABC", accession: str = "0000001234-26-000001", form: str = "10-K") -> dict:
    schema = llm_text_feature_output_schema()
    feature_names = list(schema["properties"]["feature_values"]["properties"])
    feature_values = {
        feature_name: 0.1 if feature_name.endswith("_change") or feature_name == "tone_change" else 0.2
        for feature_name in feature_names
    }
    feature_evidence = {
        feature_name: [
            {
                "section": "MD&A",
                "rationale": f"Evidence supports {feature_name}.",
                "short_quote": "Demand remained soft.",
            }
        ]
        for feature_name in feature_names
    }
    return {
        "ticker": ticker,
        "filing_accession": accession,
        "filing_form": form,
        "feature_values": feature_values,
        "feature_evidence": feature_evidence,
        "quality_flags": ["validated_test_response"],
        "parse_failure_rate": 0.0,
    }


def test_validate_llm_text_feature_response_rejects_forbidden_and_out_of_range_fields():
    request = {"ticker": "ABC", "filing_accession": "0000001234-26-000001", "filing_form": "10-K"}
    response = _valid_llm_response()

    assert validate_llm_text_feature_response(response, request) == []

    response["feature_values"]["risk_factor_novelty"] = 1.5
    response["raw_ai_implied_irr"] = 0.2
    errors = validate_llm_text_feature_response(response, request)

    assert "risk_factor_novelty_out_of_range" in errors
    assert any(error.startswith("forbidden_experiment_c_keys") for error in errors)


def test_ingest_llm_extraction_responses_writes_validated_feature_panel(tmp_path):
    requests_path = tmp_path / "requests.parquet"
    llm_requests_path = tmp_path / "llm_requests.parquet"
    responses_jsonl = tmp_path / "responses.jsonl"
    feature_panel_path = tmp_path / "panel.parquet"
    response_audit_path = tmp_path / "response_audit.parquet"
    pd.DataFrame(
        {
            "ticker": ["ABC"],
            "cik": ["0000001234"],
            "company_name": ["ABC Inc."],
            "filing_accession": ["0000001234-26-000001"],
            "filing_form": ["10-K"],
            "filing_lookback_rank": [1],
            "filed": ["2026-02-15"],
            "period_end": ["2025-12-31"],
            "source_type": ["sec_filing_text_request"],
            "source_url": ["https://www.sec.gov/abc.txt"],
            "source_path": ["data/sec_filing_text/0000001234/0000001234-26-000001.txt"],
            "source_hash": ["a" * 64],
            "date_limited_source": [True],
            "download_status": ["downloaded"],
            "downloaded_at": ["2026-05-19T00:00:00+00:00"],
            "http_status": [200],
            "download_error": [None],
            "text_extraction_status": ["not_run"],
            "quality_flags": [[]],
        }
    ).to_parquet(requests_path, index=False)
    build_llm_extraction_requests(text_corpus_requests_path=requests_path, output_path=llm_requests_path)
    responses_jsonl.write_text(
        json.dumps(
            {
                "filing_accession": "0000001234-26-000001",
                "llm_model_id": "test-llm",
                "llm_response_id": "resp-1",
                "response_json": _valid_llm_response(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = ingest_llm_extraction_responses(
        llm_responses_jsonl_path=responses_jsonl,
        llm_extraction_requests_path=llm_requests_path,
        feature_panel_path=feature_panel_path,
        response_audit_path=response_audit_path,
    )
    panel = pd.read_parquet(feature_panel_path)
    audit = pd.read_parquet(response_audit_path)
    llm_requests = pd.read_parquet(llm_requests_path)

    assert summary["row_count"] == 1
    assert summary["valid_response_count"] == 1
    assert set(panel["extraction_model_id"]) == {"test-llm"}
    assert panel.loc[0, "prompt_id"] == LLM_EXTRACTION_PROMPT_ID
    assert panel.loc[0, "prompt_version"] == LLM_EXTRACTION_PROMPT_VERSION
    assert panel.loc[0, "risk_factor_novelty"] == 0.2
    assert "raw_ai_implied_irr" not in panel.columns
    assert audit.loc[0, "validation_status"] == "valid"
    assert llm_requests.loc[0, "llm_extraction_status"] == "extracted"


def test_download_text_corpus_requests_respects_zero_default_limit(tmp_path):
    requests_path = tmp_path / "requests.parquet"
    pd.DataFrame(
        {
            "ticker": ["ABC"],
            "cik": ["0000001234"],
            "company_name": ["ABC Inc."],
            "filing_accession": ["0000001234-26-000001"],
            "filing_form": ["10-Q"],
            "filed": ["2026-05-01"],
            "period_end": ["2026-03-31"],
            "source_type": ["sec_filing_text_request"],
            "source_url": ["https://www.sec.gov/Archives/edgar/data/1234/a/a.txt"],
            "source_path": ["data/sec_filing_text/0000001234/0000001234-26-000001.txt"],
            "source_hash": [None],
            "date_limited_source": [True],
            "download_status": ["not_downloaded"],
            "downloaded_at": [None],
            "http_status": [None],
            "download_error": [None],
            "text_extraction_status": ["not_run"],
            "quality_flags": [[]],
        }
    ).to_parquet(requests_path, index=False)

    def fail_fetch(_url: str) -> TextDownloadResponse:
        raise AssertionError("fetcher should not be called with download limit 0")

    stats = download_text_corpus_requests(
        requests_path=requests_path,
        root=tmp_path,
        limit=0,
        fetcher=fail_fetch,
    )
    updated = pd.read_parquet(requests_path)

    assert stats["attempted_download_count"] == 0
    assert stats["eligible_download_count"] == 1
    assert updated.loc[0, "download_status"] == "not_downloaded"


def test_download_text_corpus_requests_writes_payload_and_hash(tmp_path):
    requests_path = tmp_path / "requests.parquet"
    payload = b"<SEC-DOCUMENT>test filing text</SEC-DOCUMENT>"
    pd.DataFrame(
        {
            "ticker": ["ABC"],
            "cik": ["0000001234"],
            "company_name": ["ABC Inc."],
            "filing_accession": ["0000001234-26-000001"],
            "filing_form": ["10-Q"],
            "filed": ["2026-05-01"],
            "period_end": ["2026-03-31"],
            "source_type": ["sec_filing_text_request"],
            "source_url": ["https://www.sec.gov/Archives/edgar/data/1234/a/a.txt"],
            "source_path": ["data/sec_filing_text/0000001234/0000001234-26-000001.txt"],
            "source_hash": [None],
            "date_limited_source": [True],
            "download_status": ["not_downloaded"],
            "downloaded_at": [None],
            "http_status": [None],
            "download_error": [None],
            "text_extraction_status": ["not_run"],
            "quality_flags": [[]],
        }
    ).to_parquet(requests_path, index=False)

    stats = download_text_corpus_requests(
        requests_path=requests_path,
        root=tmp_path,
        limit=1,
        fetcher=lambda _url: TextDownloadResponse(status_code=200, content=payload),
    )
    updated = pd.read_parquet(requests_path)
    target_path = tmp_path / "data/sec_filing_text/0000001234/0000001234-26-000001.txt"

    assert stats["attempted_download_count"] == 1
    assert stats["downloaded_count"] == 1
    assert target_path.read_bytes() == payload
    assert updated.loc[0, "download_status"] == "downloaded"
    assert updated.loc[0, "source_hash"] == hashlib.sha256(payload).hexdigest()
    assert int(updated.loc[0, "http_status"]) == 200
    assert pd.notna(updated.loc[0, "downloaded_at"])


def test_extract_text_feature_panel_from_downloaded_requests(tmp_path):
    requests_path = tmp_path / "requests.parquet"
    filing_dir = tmp_path / "data/sec_filing_text/0000001234"
    filing_dir.mkdir(parents=True)
    old_payload = (
        b"We face competitive pressure and regulatory risk. "
        b"Demand may be volatile and uncertain. "
        b"Management expects growth but results could adversely change. "
    )
    new_payload = (
        b"We face competitive pressure, supply chain disruption, and regulatory pressure. "
        b"Demand weakness and customer destocking may adversely affect results. "
        b"Management expects updated guidance and disciplined capital allocation. "
    )
    old_path = filing_dir / "0000001234-25-000001.txt"
    new_path = filing_dir / "0000001234-26-000001.txt"
    old_path.write_bytes(old_payload)
    new_path.write_bytes(new_payload)
    pd.DataFrame(
        {
            "ticker": ["ABC", "ABC"],
            "cik": ["0000001234", "0000001234"],
            "company_name": ["ABC Inc.", "ABC Inc."],
            "filing_accession": ["0000001234-25-000001", "0000001234-26-000001"],
            "filing_form": ["10-K", "10-K"],
            "filing_lookback_rank": [2, 1],
            "filed": ["2025-02-15", "2026-02-15"],
            "period_end": ["2024-12-31", "2025-12-31"],
            "source_type": ["sec_filing_text_request", "sec_filing_text_request"],
            "source_url": ["https://www.sec.gov/old.txt", "https://www.sec.gov/new.txt"],
            "source_path": [
                "data/sec_filing_text/0000001234/0000001234-25-000001.txt",
                "data/sec_filing_text/0000001234/0000001234-26-000001.txt",
            ],
            "source_hash": [hashlib.sha256(old_payload).hexdigest(), hashlib.sha256(new_payload).hexdigest()],
            "date_limited_source": [True, True],
            "download_status": ["downloaded", "downloaded"],
            "downloaded_at": ["2026-05-19T00:00:00+00:00", "2026-05-19T00:00:00+00:00"],
            "http_status": [200, 200],
            "download_error": [None, None],
            "text_extraction_status": ["not_run", "not_run"],
            "quality_flags": [[], []],
        }
    ).to_parquet(requests_path, index=False)

    summary = extract_text_feature_panel_from_requests(
        requests_path=requests_path,
        feature_panel_path=tmp_path / "panel.parquet",
        root=tmp_path,
    )
    panel = pd.read_parquet(tmp_path / "panel.parquet")
    updated_requests = pd.read_parquet(requests_path)

    assert summary["row_count"] == 2
    assert set(panel["extraction_model_id"]) == {DETERMINISTIC_EXTRACTION_MODEL_ID}
    assert "raw_ai_implied_irr" not in panel.columns
    assert panel.loc[panel["filing_lookback_rank"] == 1, "risk_factor_novelty"].notna().all()
    assert panel.loc[panel["filing_lookback_rank"] == 1, "tone_change"].notna().all()
    assert set(updated_requests["text_extraction_status"]) == {"extracted"}


def test_write_experiment_c_artifacts(tmp_path):
    security_master = tmp_path / "security_master.parquet"
    quarterly = tmp_path / "quarterly.parquet"
    companyfacts_dir = tmp_path / "companyfacts"
    companyfacts_dir.mkdir()
    pd.DataFrame({"ticker": ["A"], "cik": [1]}).to_parquet(security_master, index=False)
    pd.DataFrame(
        {
            "ticker": ["A"],
            "cik": ["0000000001"],
            "company_name": ["A Inc."],
            "filing_accession": ["0000000001-26-000001"],
            "filing_form": ["10-Q"],
            "filed": ["2026-05-01"],
            "period_end": ["2026-03-31"],
        }
    ).to_parquet(quarterly, index=False)

    manifest = write_experiment_c_artifacts(
        feature_panel_path=tmp_path / "panel.parquet",
        text_corpus_requests_path=tmp_path / "requests.parquet",
        llm_extraction_requests_path=tmp_path / "llm_requests.parquet",
        llm_extraction_responses_path=tmp_path / "llm_responses.parquet",
        manifest_path=tmp_path / "manifest.json",
        metadata_path=tmp_path / "metadata.json",
        security_master_path=security_master,
        quarterly_fundamentals_path=quarterly,
        companyfacts_dir=companyfacts_dir,
    )

    assert (tmp_path / "panel.parquet").exists()
    assert (tmp_path / "requests.parquet").exists()
    assert (tmp_path / "llm_requests.parquet").exists()
    assert (tmp_path / "llm_responses.parquet").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "metadata.json").exists()
    assert manifest["claim_ceiling"] == "corpus_request_manifest_only_no_alpha_evidence"
    assert manifest["text_corpus_request_summary"]["row_count"] == 1
    assert manifest["llm_extraction_request_summary"]["row_count"] == 1
    assert manifest["llm_extraction_request_summary"]["ready_for_llm_extraction_count"] == 0
    assert manifest["llm_extraction_response_summary"]["row_count"] == 0
    assert "llm_prompt_contract" in manifest
    assert manifest["text_feature_panel_summary"]["row_count"] == 0
