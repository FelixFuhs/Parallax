from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
ABSOLUTE_PATH_RE = re.compile(r"([A-Za-z]:\\|/Users/|/private/|/var/folders/)")
DEFAULT_COMPLETION_REPORT = ROOT / "docs" / "ai_label_decomposition_completion_report.md"


def _iter_json_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for item in value.values():
            strings.extend(_iter_json_strings(item))
        return strings
    if isinstance(value, list):
        strings = []
        for item in value:
            strings.extend(_iter_json_strings(item))
        return strings
    return []


def assert_no_absolute_paths(paths: Sequence[Path]) -> list[str]:
    failures: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for value in _iter_json_strings(payload):
            if ABSOLUTE_PATH_RE.search(value):
                failures.append(f"{path}: absolute path value {value!r}")
    return failures


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _resolve_repo_path(path_value: Any) -> Path | None:
    if _is_missing(path_value):
        return None
    candidate = Path(str(path_value))
    return candidate if candidate.is_absolute() else ROOT / candidate


def verify_completion_report(report_path: Path = DEFAULT_COMPLETION_REPORT) -> list[str]:
    if not report_path.exists():
        return [f"missing AI label decomposition completion report: {report_path}"]
    text = report_path.read_text(encoding="utf-8")
    failures: list[str] = []
    required_phrases = {
        "# AI Label Decomposition Completion Report",
        "Full objective status: incomplete",
        "Claim ceiling: diagnostic/private research",
        "## Research Layers Changed",
        "## Artifacts Written Or Updated",
        "## Result Summary",
        "## Commands Run",
        "## Critic And Verifier Review",
        "## Remaining Blockers",
        "## Acceptance Audit",
        "118 passed",
        "Artifact verification passed.",
        "The full research objective remains incomplete",
    }
    for phrase in sorted(required_phrases):
        if phrase not in text:
            failures.append(f"completion report missing phrase: {phrase}")
    forbidden_claims = {
        "Full objective status: complete",
        "Claim ceiling: institutional",
        "Claim ceiling: production",
    }
    for phrase in sorted(forbidden_claims):
        if phrase in text:
            failures.append(f"completion report contains unsupported completion/claim phrase: {phrase}")
    return failures


def verify_label_panel(panel_path: Path, summary_path: Path) -> list[str]:
    failures: list[str] = []
    if not panel_path.exists():
        return [f"missing label panel: {panel_path}"]
    if not summary_path.exists():
        return [f"missing label panel summary: {summary_path}"]
    panel = pd.read_parquet(panel_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("row_count", -1)) != len(panel):
        failures.append("label panel row_count does not match summary")
    if int(summary.get("ticker_count", -1)) != int(panel["ticker"].nunique()):
        failures.append("label panel ticker_count does not match summary")
    clean_count = int((~panel["exclude_from_clean_label"]).sum())
    if int(summary.get("clean_label_count", -1)) != clean_count:
        failures.append("label panel clean_label_count does not match summary")
    for key in ("failure_rates_by_model_tier", "failure_rates_by_market_cap_bucket"):
        if key not in summary:
            failures.append(f"label panel summary missing {key}")
        elif not isinstance(summary[key], list):
            failures.append(f"label panel summary {key} must be a list")
    if "sector_coverage" in summary and "sector" in panel.columns:
        sector_coverage = float(panel["sector"].notna().mean()) if len(panel) else 0.0
        if abs(float(summary["sector_coverage"]) - sector_coverage) > 1e-12:
            failures.append("label panel sector_coverage does not match panel")
    raw_price_coverage = float(panel["raw_close_price"].notna().mean()) if "raw_close_price" in panel.columns and len(panel) else 0.0
    if "raw_price_coverage" not in summary:
        failures.append("label panel summary missing raw_price_coverage")
    elif abs(float(summary["raw_price_coverage"]) - raw_price_coverage) > 1e-12:
        failures.append("label panel raw_price_coverage does not match panel")
    if len(panel) and raw_price_coverage <= 0.0:
        failures.append("label panel has zero raw close price coverage")
    if "mechanical_price_source" in panel.columns:
        mechanical_mask = panel["mechanical_dcf_implied_irr"].notna() if "mechanical_dcf_implied_irr" in panel.columns else pd.Series(False, index=panel.index)
        report_price_count = int(((panel["mechanical_price_source"] == "report_current_price") & mechanical_mask).sum())
        if report_price_count > 0:
            failures.append("mechanical DCF rows must not use AI report current_price as a control input")
    required_columns = {
        "raw_ai_implied_irr",
        "raw_ai_annualized_value_gap",
        "mechanical_dcf_implied_irr",
        "ai_minus_mechanical_irr",
        "factor_compressible_ai_score",
        "ai_factor_residual",
        "ai_irr_iqr",
        "ai_irr_rank_std",
        "model_disagreement",
        "tier_disagreement",
        "prompt_disagreement",
        "uncertainty_adjusted_label_weight",
        "quality_flags",
        "mechanical_price_source",
        "raw_close_price",
        "adjusted_close_price",
        "sector_source",
        "sub_industry",
    }
    missing_columns = sorted(required_columns - set(panel.columns))
    if missing_columns:
        failures.append(f"label panel missing required columns: {missing_columns}")
    return failures


def verify_sector_map(sector_map_path: Path, summary_path: Path) -> list[str]:
    failures: list[str] = []
    if not sector_map_path.exists():
        return [f"missing sector map: {sector_map_path}"]
    if not summary_path.exists():
        return [f"missing sector map summary: {summary_path}"]

    frame = pd.read_csv(sector_map_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required_columns = {"ticker", "company_name", "sector", "sub_industry", "sector_source", "source_url"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        failures.append(f"sector map missing required columns: {missing_columns}")
    if int(summary.get("row_count", -1)) != len(frame):
        failures.append("sector map row_count does not match summary")
    if int(summary.get("ticker_count", -1)) != int(frame["ticker"].nunique()):
        failures.append("sector map ticker_count does not match summary")
    if "sector" in frame.columns:
        sector_coverage = float(frame["sector"].notna().mean()) if len(frame) else 0.0
        if sector_coverage < 0.95:
            failures.append("sector map must have at least 95% sector coverage")
    if frame["ticker"].duplicated().any():
        failures.append("sector map contains duplicate tickers")
    return failures


def verify_sp500_changes(changes_path: Path, summary_path: Path) -> list[str]:
    failures: list[str] = []
    if not changes_path.exists():
        return [f"missing S&P 500 changes artifact: {changes_path}"]
    if not summary_path.exists():
        return [f"missing S&P 500 changes summary: {summary_path}"]

    frame = pd.read_csv(changes_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required_columns = {
        "effective_date",
        "added_ticker",
        "added_security",
        "removed_ticker",
        "removed_security",
        "reason",
        "approximate_membership_history",
        "point_in_time_membership",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        failures.append(f"S&P 500 changes artifact missing required columns: {missing_columns}")
    if int(summary.get("row_count", -1)) != len(frame):
        failures.append("S&P 500 changes row_count does not match summary")
    if summary.get("point_in_time_membership") is not False:
        failures.append("S&P 500 changes summary must not claim point-in-time membership")
    if summary.get("approximate_membership_history") is not True:
        failures.append("S&P 500 changes summary must mark approximate_membership_history true")
    if "effective_date" in frame.columns and pd.to_datetime(frame["effective_date"], errors="coerce").isna().any():
        failures.append("S&P 500 changes artifact contains invalid effective_date values")
    return failures


def verify_v2_status(status_path: Path) -> list[str]:
    if not status_path.exists():
        return [f"missing v2 status artifact: {status_path}"]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if status.get("status") == "complete" and status.get("blockers"):
        failures.append("v2 status cannot be complete while blockers are present")
    diagnostic_month_count = int(status.get("diagnostic_month_count") or 0)
    if status.get("status") == "complete" and diagnostic_month_count < 12:
        failures.append("v2 status cannot be complete with fewer than 12 diagnostic return months")
    forward_coverage = status.get("forward_return_coverage", {})
    zero_coverage_horizons = [
        horizon
        for horizon, coverage in forward_coverage.items()
        if isinstance(coverage, dict) and int(coverage.get("non_null") or 0) == 0
    ]
    if status.get("status") == "complete" and zero_coverage_horizons:
        failures.append("v2 status cannot be complete while forward-return horizons have zero coverage")
    return failures


def verify_freeze_artifacts(models_dir: Path = ROOT / "models", docs_dir: Path = ROOT / "docs") -> list[str]:
    failures: list[str] = []
    primary_model_path = models_dir / "frozen_xgb_regressor.json"
    primary_metadata_path = models_dir / "frozen_model_metadata.json"
    baseline_metadata_path = models_dir / "frozen_elasticnet_metadata.json"
    baseline_doc_path = docs_dir / "freeze_elasticnet_baseline.md"
    for label, path in (
        ("primary XGBoost model", primary_model_path),
        ("primary model metadata", primary_metadata_path),
        ("Elastic Net baseline metadata", baseline_metadata_path),
        ("Elastic Net baseline freeze doc", baseline_doc_path),
    ):
        if not path.exists():
            failures.append(f"missing {label}: {path}")
    if failures:
        return failures

    primary_metadata = json.loads(primary_metadata_path.read_text(encoding="utf-8"))
    baseline_metadata = json.loads(baseline_metadata_path.read_text(encoding="utf-8"))
    if primary_metadata.get("artifact_role") != "primary_model_freeze":
        failures.append("primary frozen metadata must have artifact_role=primary_model_freeze")
    if str(primary_metadata.get("model_type", "")).lower() not in {"xgbregressor", "xgboost_regressor"}:
        failures.append("primary frozen metadata must identify XGBoost regressor model_type")
    if primary_metadata.get("source_artifacts", {}).get("frozen_model") != "models/frozen_xgb_regressor.json":
        failures.append("primary frozen metadata must use repo-relative frozen_model path")
    if baseline_metadata.get("artifact_role") != "baseline_model_freeze":
        failures.append("Elastic Net metadata must have artifact_role=baseline_model_freeze")
    if baseline_metadata.get("primary_model") is not False:
        failures.append("Elastic Net metadata must mark primary_model false")
    if baseline_metadata.get("primary_model_reference") != "models/frozen_xgb_regressor.json":
        failures.append("Elastic Net metadata must point back to the XGBoost primary model")
    if baseline_metadata.get("model_family") != "ElasticNetCV":
        failures.append("Elastic Net metadata must identify ElasticNetCV model_family")
    return failures


def verify_critic_report(report_path: Path) -> list[str]:
    failures: list[str] = []
    if not report_path.exists():
        return [f"missing critic report artifact: {report_path}"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("experiment_id") != "ai_label_decomposition_critic_review":
        failures.append("critic report has wrong experiment_id")
    if report.get("goal_completion_claimed") is not False:
        failures.append("critic report must not claim goal completion while blockers remain")
    expected_passes = {"implementation_critic", "verification_critic", "wording_critic"}
    review_passes = report.get("review_passes", [])
    if not isinstance(review_passes, list) or not review_passes:
        failures.append("critic report must include review_passes")
        review_passes = []
    observed_passes = {
        review.get("review_type")
        for review in review_passes
        if isinstance(review, dict)
    }
    missing_passes = sorted(expected_passes - observed_passes)
    if missing_passes:
        failures.append(f"critic report missing review passes: {missing_passes}")
    for review in review_passes:
        if not isinstance(review, dict):
            failures.append("critic report review_passes entries must be objects")
            continue
        for key in ("review_type", "reviewer", "completed_at", "scope", "findings"):
            if key not in review:
                failures.append(f"critic report review missing {key}")
        findings = review.get("findings", [])
        if not isinstance(findings, list):
            failures.append("critic report findings must be lists")
            continue
        for finding in findings:
            if not isinstance(finding, dict):
                failures.append("critic report findings must be objects")
                continue
            for key in ("severity", "title", "evidence", "disposition", "completion_blocker"):
                if key not in finding:
                    failures.append(f"critic report finding missing {key}")
    high_medium_open = [
        finding
        for review in review_passes
        if isinstance(review, dict)
        for finding in review.get("findings", [])
        if isinstance(finding, dict)
        and finding.get("severity") in {"high", "medium"}
        and finding.get("disposition") not in {"fixed", "documented_blocker", "accepted_low_risk"}
    ]
    if high_medium_open:
        failures.append("critic report has high/medium findings without an accepted disposition")
    blockers = report.get("remaining_completion_blockers", [])
    if not isinstance(blockers, list) or not blockers:
        failures.append("critic report must list remaining_completion_blockers")
    return failures


def verify_rank_ic_artifacts(results_dir: Path) -> list[str]:
    failures: list[str] = []
    artifact_specs = {
        "rank_ic.parquet": {"date", "signal", "horizon", "decomposition", "n", "rank_ic"},
        "rank_ic_summary.parquet": {
            "signal",
            "horizon",
            "decomposition",
            "months",
            "mean_ic",
            "median_ic",
            "ic_std",
            "newey_west_tstat",
            "positive_ic_hit_rate",
        },
        "rank_ic_by_year.parquet": {
            "year",
            "signal",
            "horizon",
            "decomposition",
            "months",
            "mean_ic",
            "median_ic",
            "positive_ic_hit_rate",
        },
        "rank_ic_by_sector.parquet": {
            "sector",
            "signal",
            "horizon",
            "months",
            "mean_ic",
            "median_ic",
            "positive_ic_hit_rate",
            "mean_n",
        },
        "rank_ic_coverage.parquet": {
            "date",
            "signal",
            "horizon",
            "universe_n",
            "score_non_null",
            "return_non_null",
            "paired_n",
            "paired_coverage",
            "sector_count",
        },
        "signal_comparison.parquet": {
            "signal",
            "signal_label",
            "horizon",
            "global_mean_ic",
            "sector_neutral_mean_ic",
            "across_sector_mean_ic",
            "global_months",
            "sector_neutral_months",
            "across_sector_months",
            "comparison_status",
        },
    }
    for filename, required_columns in artifact_specs.items():
        path = results_dir / filename
        if not path.exists():
            failures.append(f"missing rank IC artifact: {path}")
            continue
        frame = pd.read_parquet(path)
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            failures.append(f"{filename} missing required columns: {missing_columns}")
    status_path = results_dir / "v2_experiment_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        artifacts = status.get("artifacts", {})
        for key in ("rank_ic_by_year", "rank_ic_by_sector", "rank_ic_coverage", "signal_comparison"):
            if key not in artifacts:
                failures.append(f"v2 status missing artifact reference: {key}")
    return failures


def verify_portfolio_audit_artifacts(results_dir: Path) -> list[str]:
    failures: list[str] = []
    artifact_specs = {
        "holdings.parquet": {
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
            "entry_price",
            "exit_price",
            "raw_return",
            "transaction_cost",
            "net_return",
        },
        "monthly_returns.parquet": {
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
        },
        "turnover.parquet": {
            "date",
            "signal_name",
            "portfolio_mode",
            "weighting_method",
            "bucket",
            "cost_bps_one_way",
            "turnover",
            "transaction_cost_drag",
        },
        "exposures.parquet": {
            "date",
            "signal_name",
            "portfolio_mode",
            "weighting_method",
            "bucket",
            "sector_weights",
            "average_market_cap",
            "name_count",
        },
    }
    frames: dict[str, pd.DataFrame] = {}
    for filename, required_columns in artifact_specs.items():
        path = results_dir / filename
        if not path.exists():
            failures.append(f"missing portfolio audit artifact: {path}")
            continue
        frame = pd.read_parquet(path)
        frames[filename] = frame
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            failures.append(f"{filename} missing required columns: {missing_columns}")

    status_path = results_dir / "v2_experiment_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "complete" and status.get("portfolio_return_column"):
            monthly = frames.get("monthly_returns.parquet", pd.DataFrame())
            modes = set(monthly["portfolio_mode"].dropna()) if "portfolio_mode" in monthly else set()
            if "unconstrained" not in modes:
                failures.append("portfolio audit artifacts missing unconstrained mode")
            if status.get("sector_coverage", 0.0) and "sector_neutral" not in modes:
                failures.append("portfolio audit artifacts missing sector_neutral mode despite sector coverage")
            if sorted(modes) != sorted(status.get("portfolio_modes", [])):
                failures.append("v2 status portfolio_modes do not match monthly_returns modes")
    return failures


def verify_forward_returns(returns_path: Path, summary_path: Path) -> list[str]:
    failures: list[str] = []
    if not returns_path.exists():
        return [f"missing forward returns artifact: {returns_path}"]
    if not summary_path.exists():
        return [f"missing forward returns summary: {summary_path}"]

    frame = pd.read_parquet(returns_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("row_count", -1)) != len(frame):
        failures.append("forward returns row_count does not match summary")
    if int(summary.get("ticker_count", -1)) != int(frame["ticker"].nunique()):
        failures.append("forward returns ticker_count does not match summary")

    coverage = summary.get("coverage", {})
    total_non_null = 0
    for column, column_summary in coverage.items():
        if column not in frame.columns:
            failures.append(f"forward returns summary references missing column: {column}")
            continue
        non_null = int(frame[column].notna().sum())
        total_non_null += non_null
        if int(column_summary.get("non_null", -1)) != non_null:
            failures.append(f"forward returns non_null mismatch for {column}")
    blocker_codes = {blocker.get("code") for blocker in summary.get("blockers", [])}
    if total_non_null == 0 and "no_usable_forward_returns" not in blocker_codes:
        failures.append("forward returns with zero coverage must carry no_usable_forward_returns blocker")
    return failures


def verify_universe_snapshot(security_master_path: Path, snapshot_path: Path) -> list[str]:
    failures: list[str] = []
    if not security_master_path.exists():
        return [f"missing security master: {security_master_path}"]
    if not snapshot_path.exists():
        return [f"missing universe snapshot: {snapshot_path}"]

    frame = pd.read_parquet(security_master_path)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if int(snapshot.get("ticker_count", -1)) != len(frame):
        failures.append("universe snapshot ticker_count does not match security master")
    if snapshot.get("point_in_time_membership") is not False:
        failures.append("universe snapshot must explicitly mark point_in_time_membership false")
    if snapshot.get("membership_history_source") and snapshot.get("membership_history_quality") != "selected_public_changes_not_full_point_in_time_membership":
        failures.append("universe snapshot membership history must be marked approximate")
    warning = str(snapshot.get("survivor_bias_warning", ""))
    if "not CRSP/Compustat-quality" not in warning:
        failures.append("universe snapshot missing survivor-bias warning")
    required_columns = {"ticker", "cik", "company_name", "point_in_time_membership", "membership_history_source"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        failures.append(f"security master missing required columns: {missing_columns}")
    return failures


def verify_approximate_membership(panel_path: Path, summary_path: Path) -> list[str]:
    failures: list[str] = []
    if not panel_path.exists():
        return [f"missing approximate membership panel: {panel_path}"]
    if not summary_path.exists():
        return [f"missing approximate membership summary: {summary_path}"]
    panel = pd.read_parquet(panel_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("point_in_time_membership") is not False:
        failures.append("approximate membership summary must not claim point-in-time membership")
    if int(summary.get("row_count", -1)) != len(panel):
        failures.append("approximate membership row_count does not match panel")
    required_columns = {
        "date",
        "ticker",
        "approximate_member",
        "in_current_security_master",
        "company_tickers_match",
        "has_cik",
        "cik_source",
        "membership_basis",
        "point_in_time_membership",
        "membership_history_quality",
    }
    missing_columns = sorted(required_columns - set(panel.columns))
    if missing_columns:
        failures.append(f"approximate membership panel missing required columns: {missing_columns}")
    if not panel.empty:
        if panel["point_in_time_membership"].any():
            failures.append("approximate membership panel contains point_in_time_membership=true rows")
        missing_count = int((~panel["in_current_security_master"]).sum())
        if missing_count <= 0:
            failures.append("approximate membership panel should quantify at least one missing historical constituent")
        if int(summary.get("missing_from_current_security_master_ticker_count", -1)) != int(
            panel.loc[~panel["in_current_security_master"], "ticker"].nunique()
        ):
            failures.append("approximate membership missing ticker count does not match panel")
        missing_matches = int(
            panel.loc[(~panel["in_current_security_master"]) & (panel["company_tickers_match"]), "ticker"].nunique()
        )
        if int(summary.get("missing_with_sec_company_ticker_match_count", -1)) != missing_matches:
            failures.append("approximate membership SEC ticker match count does not match panel")
    blocker_codes = {blocker.get("code") for blocker in summary.get("blockers", [])}
    for required in ("removed_names_missing_security_master_rows", "selected_changes_not_full_point_in_time_membership"):
        if required not in blocker_codes:
            failures.append(f"approximate membership summary missing blocker: {required}")
    claim_limit = str(summary.get("claim_limit", ""))
    if "not CRSP/Compustat-quality" not in claim_limit:
        failures.append("approximate membership summary must state it is not CRSP/Compustat-quality")
    return failures


def verify_quarterly_fundamentals(panel_path: Path, summary_path: Path) -> list[str]:
    failures: list[str] = []
    if not panel_path.exists():
        return [f"missing quarterly fundamentals: {panel_path}"]
    if not summary_path.exists():
        return [f"missing quarterly fundamentals summary: {summary_path}"]

    frame = pd.read_parquet(panel_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("row_count", -1)) != len(frame):
        failures.append("quarterly fundamentals row_count does not match summary")
    if int(summary.get("ticker_count", -1)) != int(frame["ticker"].nunique()):
        failures.append("quarterly fundamentals ticker_count does not match summary")
    required_columns = {
        "ticker",
        "cik",
        "fiscal_period",
        "period_end",
        "filed",
        "revenue",
        "operating_cash_flow",
        "capex",
        "free_cash_flow",
        "revenue_ttm",
        "revenue_qoq_change",
        "revenue_yoy_change",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        failures.append(f"quarterly fundamentals missing required columns: {missing_columns}")
    notes = " ".join(str(note) for note in summary.get("method_notes", []))
    if "do not infer Q4" not in notes:
        failures.append("quarterly fundamentals summary must document Q4 inference policy")
    return failures


def verify_experiment_c_text_features(panel_path: Path, manifest_path: Path) -> list[str]:
    failures: list[str] = []
    if not panel_path.exists():
        return [f"missing Experiment C text feature panel: {panel_path}"]
    if not manifest_path.exists():
        return [f"missing Experiment C manifest: {manifest_path}"]

    panel = pd.read_parquet(panel_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    request_path = ROOT / artifacts["text_corpus_requests"] if artifacts.get("text_corpus_requests") else None
    llm_request_path = (
        ROOT / artifacts["llm_extraction_requests"] if artifacts.get("llm_extraction_requests") else None
    )
    llm_response_path = (
        ROOT / artifacts["llm_extraction_responses"] if artifacts.get("llm_extraction_responses") else None
    )
    requests = pd.DataFrame()
    llm_requests = pd.DataFrame()
    llm_responses = pd.DataFrame()
    if request_path is None:
        failures.append("Experiment C manifest missing text_corpus_requests artifact reference")
    elif not request_path.exists():
        failures.append(f"missing Experiment C text corpus request artifact: {request_path}")
    else:
        requests = pd.read_parquet(request_path)
    if llm_request_path is None:
        failures.append("Experiment C manifest missing llm_extraction_requests artifact reference")
    elif not llm_request_path.exists():
        failures.append(f"missing Experiment C LLM extraction request artifact: {llm_request_path}")
    else:
        llm_requests = pd.read_parquet(llm_request_path)
    if llm_response_path is None:
        failures.append("Experiment C manifest missing llm_extraction_responses artifact reference")
    elif not llm_response_path.exists():
        failures.append(f"missing Experiment C LLM extraction response artifact: {llm_response_path}")
    else:
        llm_responses = pd.read_parquet(llm_response_path)
    if manifest.get("experiment_id") != "experiment_c_llm_text_features":
        failures.append("Experiment C manifest has wrong experiment_id")
    if manifest.get("separate_from_dcf_labels") is not True:
        failures.append("Experiment C manifest must be separate_from_dcf_labels")
    prompt_contract = manifest.get("llm_prompt_contract", {})
    if not isinstance(prompt_contract, dict) or prompt_contract.get("llm_extraction_not_run") is not True:
        failures.append("Experiment C manifest must include an unrun LLM prompt contract")
    if "output_schema" not in prompt_contract:
        failures.append("Experiment C manifest missing LLM output schema")
    blocker_codes = {blocker.get("code") for blocker in manifest.get("blockers", [])}
    if manifest.get("status") == "blocked" and not (
        {"missing_date_limited_text_corpus", "text_extraction_not_run", "llm_text_extraction_not_run"} & blocker_codes
    ):
        failures.append("blocked Experiment C manifest must name a corpus or extraction blocker")
    forbidden_columns = {
        "raw_ai_implied_irr",
        "mechanical_dcf_implied_irr",
        "ai_minus_mechanical_irr",
        "ai_factor_residual",
    }
    present_forbidden = sorted(forbidden_columns & set(panel.columns))
    if present_forbidden:
        failures.append(f"Experiment C panel must not include DCF label columns: {present_forbidden}")
    required_columns = {
        "ticker",
        "filing_accession",
        "source_hash",
        "extraction_model_id",
        "prompt_id",
        "prompt_version",
        "risk_factor_novelty",
        "quality_flags",
    }
    missing_columns = sorted(required_columns - set(panel.columns))
    if missing_columns:
        failures.append(f"Experiment C panel missing required columns: {missing_columns}")
    feature_summary = manifest.get("text_feature_panel_summary", {})
    if feature_summary and int(feature_summary.get("row_count", -1)) != len(panel):
        failures.append("Experiment C text feature panel row_count does not match manifest")
    if not panel.empty:
        if panel["source_hash"].isna().any():
            failures.append("extracted Experiment C feature rows must include source_hash")
        if panel["extraction_model_id"].isna().any():
            failures.append("extracted Experiment C feature rows must include extraction_model_id")
        if "date_limited_source" in panel.columns and not panel["date_limited_source"].all():
            failures.append("extracted Experiment C feature rows must use date-limited sources")
        if "source_path" in panel.columns:
            missing_panel_files = [
                str(path)
                for path in panel["source_path"].map(_resolve_repo_path)
                if path is None or not path.exists()
            ]
            if missing_panel_files:
                failures.append("extracted Experiment C feature rows must point to existing local text files")
    if request_path is not None and request_path.exists():
        required_request_columns = {
            "ticker",
            "cik",
            "filing_accession",
            "filing_form",
            "filing_lookback_rank",
            "filed",
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
        }
        missing_request_columns = sorted(required_request_columns - set(requests.columns))
        if missing_request_columns:
            failures.append(f"Experiment C text corpus requests missing required columns: {missing_request_columns}")
        summary = manifest.get("text_corpus_request_summary", {})
        if int(summary.get("row_count", -1)) != len(requests):
            failures.append("Experiment C text corpus request row_count does not match manifest")
        if not requests.empty and not requests["date_limited_source"].all():
            failures.append("Experiment C text corpus requests must be date-limited sources")
        if "download_status" in requests.columns:
            downloaded = requests.loc[requests["download_status"] == "downloaded"]
            if not downloaded.empty:
                if downloaded["source_hash"].isna().any():
                    failures.append("downloaded Experiment C text corpus requests must include source_hash")
                missing_files = [
                    str(path)
                    for path in downloaded["source_path"].map(_resolve_repo_path)
                    if path is None or not path.exists()
                ]
                if missing_files:
                    failures.append(
                        "downloaded Experiment C text corpus requests must point to existing local text files"
                    )
    if llm_request_path is not None and llm_request_path.exists():
        required_llm_columns = {
            "ticker",
            "filing_accession",
            "filing_form",
            "source_path",
            "source_hash",
            "date_limited_source",
            "download_status",
            "llm_request_status",
            "llm_extraction_status",
            "prompt_id",
            "prompt_version",
            "system_prompt",
            "user_prompt",
            "output_schema_json",
            "quality_flags",
        }
        missing_llm_columns = sorted(required_llm_columns - set(llm_requests.columns))
        if missing_llm_columns:
            failures.append(f"Experiment C LLM extraction requests missing required columns: {missing_llm_columns}")
        llm_summary = manifest.get("llm_extraction_request_summary", {})
        if int(llm_summary.get("row_count", -1)) != len(llm_requests):
            failures.append("Experiment C LLM extraction request row_count does not match manifest")
        if not llm_requests.empty:
            if not llm_requests["date_limited_source"].all():
                failures.append("Experiment C LLM extraction requests must be date-limited")
            if not (llm_requests["llm_extraction_status"] == "not_run").all():
                failures.append("Experiment C LLM extraction requests must not claim extraction ran")
            prompt_text = " ".join(
                str(value)
                for value in [
                    *llm_requests["system_prompt"].fillna("").tolist(),
                    *llm_requests["user_prompt"].fillna("").tolist(),
                ]
            )
            if "Do not use DCF labels" not in prompt_text:
                failures.append("Experiment C LLM prompts must forbid DCF label usage")
            if llm_requests["output_schema_json"].str.contains("raw_ai_implied_irr", na=False).any():
                failures.append("Experiment C LLM output schema must not include DCF label fields")
    if llm_response_path is not None and llm_response_path.exists():
        required_response_columns = {
            "ticker",
            "filing_accession",
            "filing_form",
            "source_hash",
            "prompt_id",
            "prompt_version",
            "llm_model_id",
            "response_json",
            "validation_status",
            "validation_errors",
            "quality_flags",
        }
        missing_response_columns = sorted(required_response_columns - set(llm_responses.columns))
        if missing_response_columns:
            failures.append(f"Experiment C LLM extraction responses missing required columns: {missing_response_columns}")
        response_summary = manifest.get("llm_extraction_response_summary", {})
        if int(response_summary.get("row_count", -1)) != len(llm_responses):
            failures.append("Experiment C LLM extraction response row_count does not match manifest")
        if not llm_responses.empty:
            if not set(llm_responses["validation_status"]).issubset({"valid", "invalid"}):
                failures.append("Experiment C LLM response validation_status must be valid or invalid")
            valid_responses = llm_responses.loc[llm_responses["validation_status"] == "valid"]
            if not valid_responses.empty and valid_responses["source_hash"].isna().any():
                failures.append("valid Experiment C LLM responses must include source_hash")
    return failures


def verify_experiment_b_factor_portability(
    predictions_path: Path,
    coefficients_path: Path,
    summary_path: Path,
) -> list[str]:
    failures: list[str] = []
    if not predictions_path.exists():
        return [f"missing Experiment B predictions artifact: {predictions_path}"]
    if not coefficients_path.exists():
        return [f"missing Experiment B coefficients artifact: {coefficients_path}"]
    if not summary_path.exists():
        return [f"missing Experiment B summary: {summary_path}"]

    predictions = pd.read_parquet(predictions_path)
    coefficients = pd.read_csv(coefficients_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("experiment_id") != "experiment_b_ai_implied_factor_portability":
        failures.append("Experiment B summary has wrong experiment_id")
    if summary.get("status") == "complete":
        failures.append("Experiment B summary must not claim complete historical portability")
    warning_codes = {warning.get("code") for warning in summary.get("warnings", [])}
    if "historical_backcast_not_run" not in warning_codes:
        failures.append("Experiment B summary must warn that historical backcast was not run")
    required_prediction_columns = {
        "ticker",
        "target",
        "observed_label",
        "cv_predicted_label",
        "fit_predicted_label",
        "fit_residual",
        "feature_null_count",
    }
    missing_prediction_columns = sorted(required_prediction_columns - set(predictions.columns))
    if missing_prediction_columns:
        failures.append(f"Experiment B predictions missing required columns: {missing_prediction_columns}")
    required_coefficient_columns = {"target", "feature", "coefficient"}
    missing_coefficient_columns = sorted(required_coefficient_columns - set(coefficients.columns))
    if missing_coefficient_columns:
        failures.append(f"Experiment B coefficients missing required columns: {missing_coefficient_columns}")
    if int(summary.get("row_count", -1)) != len(predictions):
        failures.append("Experiment B row_count does not match predictions")
    return failures


def verify_experiment_b_historical_backcast(
    scores_path: Path,
    monthly_returns_path: Path,
    holdings_path: Path,
    rank_ic_path: Path,
    rank_ic_summary_path: Path,
    rank_ic_by_year_path: Path,
    rank_ic_by_sector_path: Path,
    rank_ic_coverage_path: Path,
    summary_path: Path,
) -> list[str]:
    failures: list[str] = []
    required_paths = {
        "scores": scores_path,
        "monthly returns": monthly_returns_path,
        "holdings": holdings_path,
        "rank IC": rank_ic_path,
        "rank IC summary": rank_ic_summary_path,
        "rank IC by year": rank_ic_by_year_path,
        "rank IC by sector": rank_ic_by_sector_path,
        "rank IC coverage": rank_ic_coverage_path,
        "summary": summary_path,
    }
    for label, path in required_paths.items():
        if not path.exists():
            failures.append(f"missing Experiment B historical backcast {label}: {path}")
    if failures:
        return failures

    scores = pd.read_parquet(scores_path)
    monthly_returns = pd.read_parquet(monthly_returns_path)
    holdings = pd.read_parquet(holdings_path)
    rank_ic = pd.read_parquet(rank_ic_path)
    rank_ic_summary = pd.read_parquet(rank_ic_summary_path)
    rank_ic_by_year = pd.read_parquet(rank_ic_by_year_path)
    rank_ic_by_sector = pd.read_parquet(rank_ic_by_sector_path)
    rank_ic_coverage = pd.read_parquet(rank_ic_coverage_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("experiment_id") != "experiment_b_ai_implied_factor_portability_backcast":
        failures.append("Experiment B historical backcast summary has wrong experiment_id")
    if summary.get("status") == "complete":
        failures.append("Experiment B historical backcast must not claim full completion")
    if int(summary.get("score_row_count", -1)) != len(scores):
        failures.append("Experiment B historical backcast score_row_count does not match scores")
    if int(summary.get("monthly_return_row_count", -1)) != len(monthly_returns):
        failures.append("Experiment B historical backcast monthly_return_row_count does not match monthly returns")
    if int(summary.get("holding_row_count", -1)) != len(holdings):
        failures.append("Experiment B historical backcast holding_row_count does not match holdings")
    if int(summary.get("rank_ic_row_count", -1)) != len(rank_ic):
        failures.append("Experiment B historical backcast rank_ic_row_count does not match rank IC")
    required_benchmark_signals = {"benchmark_composite_vqmia_score", "benchmark_fcf_to_ev"}
    if not scores.empty and {"signal_name", "date", "score"}.issubset(scores.columns):
        score_variation = scores.groupby(["signal_name", "date"])["score"].agg(["count", "nunique"])
        tied = score_variation[(score_variation["count"] > 1) & (score_variation["nunique"] < 2)]
        if not tied.empty:
            failures.append("Experiment B historical backcast contains tied/constant score panels")
        observed_signals = set(scores["signal_name"].dropna())
        missing_benchmark_signals = sorted(required_benchmark_signals - observed_signals)
        if missing_benchmark_signals:
            failures.append(f"Experiment B historical backcast missing benchmark score signals: {missing_benchmark_signals}")
        summary_benchmark_signals = set(summary.get("benchmark_signals", []))
        missing_summary_benchmarks = sorted(required_benchmark_signals - summary_benchmark_signals)
        if missing_summary_benchmarks:
            failures.append(f"Experiment B historical backcast summary missing benchmark signals: {missing_summary_benchmarks}")

    required_score_columns = {
        "date",
        "ticker",
        "target",
        "signal_name",
        "score",
        "sector",
        "market_cap",
        "feature_null_count",
    }
    missing_score_columns = sorted(required_score_columns - set(scores.columns))
    if missing_score_columns:
        failures.append(f"Experiment B historical backcast scores missing required columns: {missing_score_columns}")
    required_monthly_columns = {
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
    }
    missing_monthly_columns = sorted(required_monthly_columns - set(monthly_returns.columns))
    if missing_monthly_columns:
        failures.append(f"Experiment B historical backcast monthly returns missing required columns: {missing_monthly_columns}")
    required_holding_columns = {
        "date",
        "ticker",
        "signal_name",
        "portfolio_mode",
        "weighting_method",
        "bucket",
        "weight",
        "raw_return",
        "transaction_cost",
        "net_return",
    }
    missing_holding_columns = sorted(required_holding_columns - set(holdings.columns))
    if missing_holding_columns:
        failures.append(f"Experiment B historical backcast holdings missing required columns: {missing_holding_columns}")
    if not monthly_returns.empty:
        observed_modes = set(monthly_returns["portfolio_mode"].dropna())
        for required_mode in ("unconstrained", "sector_neutral"):
            if required_mode not in observed_modes:
                failures.append(f"Experiment B historical backcast missing portfolio mode: {required_mode}")
        if sorted(observed_modes) != sorted(summary.get("portfolio_modes", [])):
            failures.append("Experiment B historical backcast summary portfolio_modes do not match monthly returns")
        observed_monthly_signals = set(monthly_returns["signal_name"].dropna()) if "signal_name" in monthly_returns else set()
        missing_monthly_benchmarks = sorted(required_benchmark_signals - observed_monthly_signals)
        if missing_monthly_benchmarks:
            failures.append(f"Experiment B historical backcast monthly returns missing benchmark signals: {missing_monthly_benchmarks}")
    if not holdings.empty:
        observed_holding_signals = set(holdings["signal_name"].dropna()) if "signal_name" in holdings else set()
        missing_holding_benchmarks = sorted(required_benchmark_signals - observed_holding_signals)
        if missing_holding_benchmarks:
            failures.append(f"Experiment B historical backcast holdings missing benchmark signals: {missing_holding_benchmarks}")
    rank_ic_specs = {
        "rank IC": (
            rank_ic,
            {"date", "signal", "horizon", "decomposition", "n", "rank_ic", "sector_status"},
        ),
        "rank IC summary": (
            rank_ic_summary,
            {
                "signal",
                "horizon",
                "decomposition",
                "months",
                "mean_ic",
                "median_ic",
                "ic_std",
                "newey_west_tstat",
                "positive_ic_hit_rate",
            },
        ),
        "rank IC by year": (
            rank_ic_by_year,
            {"year", "signal", "horizon", "decomposition", "months", "mean_ic", "median_ic", "positive_ic_hit_rate"},
        ),
        "rank IC by sector": (
            rank_ic_by_sector,
            {"sector", "signal", "horizon", "months", "mean_ic", "median_ic", "positive_ic_hit_rate", "mean_n"},
        ),
        "rank IC coverage": (
            rank_ic_coverage,
            {
                "date",
                "signal",
                "horizon",
                "universe_n",
                "score_non_null",
                "return_non_null",
                "paired_n",
                "paired_coverage",
                "sector_count",
            },
        ),
    }
    for label, (frame, required_columns) in rank_ic_specs.items():
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            failures.append(f"Experiment B historical backcast {label} missing required columns: {missing_columns}")
    if not rank_ic_summary.empty and "signal" in rank_ic_summary:
        observed_rank_ic_signals = set(rank_ic_summary["signal"].dropna())
        missing_rank_ic_benchmarks = sorted(required_benchmark_signals - observed_rank_ic_signals)
        if missing_rank_ic_benchmarks:
            failures.append(f"Experiment B historical backcast rank IC summary missing benchmark signals: {missing_rank_ic_benchmarks}")
    required_horizons = {"return_1m", "return_3m", "return_6m", "return_12m"}
    if not required_horizons.issubset(set(summary.get("rank_ic_return_horizons", []))):
        failures.append("Experiment B historical backcast summary missing required rank IC horizons")
    approximate_membership_gap = summary.get("approximate_membership_gap")
    if not isinstance(approximate_membership_gap, dict):
        failures.append("Experiment B historical backcast summary missing approximate_membership_gap")
    else:
        if approximate_membership_gap.get("point_in_time_membership") is not False:
            failures.append("Experiment B historical backcast membership gap must mark point_in_time_membership false")
        if approximate_membership_gap.get("status") in {
            "missing_approximate_membership_artifact",
            "unreadable_approximate_membership_artifact",
        }:
            failures.append("Experiment B historical backcast membership gap artifact was not loaded")
        missing_count = approximate_membership_gap.get("missing_from_current_security_master_ticker_count")
        if not isinstance(missing_count, int) or missing_count <= 0:
            failures.append("Experiment B historical backcast membership gap must quantify missing historical tickers")
        overlap = approximate_membership_gap.get("backcast_rebalance_overlap", {})
        if not isinstance(overlap, dict) or int(overlap.get("overlap_month_count", 0)) <= 0:
            failures.append("Experiment B historical backcast membership gap must overlap backcast rebalance dates")
    artifacts = summary.get("artifacts", {})
    for key in (
        "rank_ic",
        "rank_ic_summary",
        "rank_ic_by_year",
        "rank_ic_by_sector",
        "rank_ic_coverage",
        "approximate_membership",
        "approximate_membership_summary",
    ):
        if key not in artifacts:
            failures.append(f"Experiment B historical backcast summary missing artifact reference: {key}")
    warning_codes = {warning.get("code") for warning in summary.get("warnings", [])}
    for required_warning in ("current_label_projection", "survivor_universe", "current_sector_map_used"):
        if required_warning not in warning_codes:
            failures.append(f"Experiment B historical backcast missing warning: {required_warning}")
    warning_missing_count = (
        approximate_membership_gap.get("missing_from_current_security_master_ticker_count")
        if isinstance(approximate_membership_gap, dict)
        else 0
    )
    if (
        isinstance(warning_missing_count, int)
        and warning_missing_count > 0
        and "removed_names_missing_from_backcast_universe" not in warning_codes
    ):
        failures.append("Experiment B historical backcast missing removed-name universe warning")
    if summary.get("status") == "blocked" and not summary.get("blockers"):
        failures.append("blocked Experiment B historical backcast must include blockers")
    return failures


def run_verification(results_dir: Path = ROOT / "results") -> list[str]:
    failures: list[str] = []
    failures.extend(
        assert_no_absolute_paths(
            [
                results_dir / "label_panel_summary.json",
                results_dir / "label_panel_experiment_metadata.json",
                results_dir / "sector_map_summary.json",
                results_dir / "sector_map_metadata.json",
                results_dir / "sp500_changes_summary.json",
                results_dir / "sp500_changes_metadata.json",
                results_dir / "forward_returns_summary.json",
                results_dir / "forward_returns_metadata.json",
                results_dir / "universe_snapshot.json",
                results_dir / "security_master_metadata.json",
                results_dir / "approx_sp500_membership_summary.json",
                results_dir / "approx_sp500_membership_metadata.json",
                results_dir / "quarterly_fundamentals_summary.json",
                results_dir / "quarterly_fundamentals_metadata.json",
                results_dir / "experiment_c_text_features_manifest.json",
                results_dir / "experiment_c_text_features_metadata.json",
                results_dir / "experiment_b_factor_portability_summary.json",
                results_dir / "experiment_b_factor_portability_metadata.json",
                results_dir / "experiment_b_historical_backcast_summary.json",
                results_dir / "experiment_b_historical_backcast_metadata.json",
                results_dir / "ai_label_decomposition_critic_report.json",
                results_dir / "v2_experiment_status.json",
                results_dir / "v2_experiment_metadata.json",
                results_dir / "backtest_summary.json",
                ROOT / "models" / "frozen_model_metadata.json",
                ROOT / "models" / "frozen_elasticnet_metadata.json",
                ROOT / "models" / "elasticnet_stability.json",
                ROOT / "models" / "distill_v2_metadata.json",
            ]
        )
    )
    failures.extend(verify_completion_report(DEFAULT_COMPLETION_REPORT))
    failures.extend(verify_freeze_artifacts(ROOT / "models", ROOT / "docs"))
    failures.extend(verify_critic_report(results_dir / "ai_label_decomposition_critic_report.json"))
    failures.extend(
        verify_sector_map(ROOT / "data" / "sector_map_wikipedia.csv", results_dir / "sector_map_summary.json")
    )
    failures.extend(
        verify_sp500_changes(ROOT / "data" / "sp500_changes_wikipedia.csv", results_dir / "sp500_changes_summary.json")
    )
    failures.extend(verify_label_panel(results_dir / "label_panel.parquet", results_dir / "label_panel_summary.json"))
    failures.extend(
        verify_forward_returns(results_dir / "forward_returns.parquet", results_dir / "forward_returns_summary.json")
    )
    failures.extend(verify_rank_ic_artifacts(results_dir))
    failures.extend(verify_portfolio_audit_artifacts(results_dir))
    failures.extend(verify_universe_snapshot(ROOT / "data" / "security_master.parquet", results_dir / "universe_snapshot.json"))
    failures.extend(
        verify_approximate_membership(
            ROOT / "data" / "approx_sp500_membership.parquet",
            results_dir / "approx_sp500_membership_summary.json",
        )
    )
    failures.extend(
        verify_quarterly_fundamentals(
            ROOT / "data" / "quarterly_fundamentals.parquet",
            results_dir / "quarterly_fundamentals_summary.json",
        )
    )
    failures.extend(
        verify_experiment_c_text_features(
            results_dir / "experiment_c_text_features.parquet",
            results_dir / "experiment_c_text_features_manifest.json",
        )
    )
    failures.extend(
        verify_experiment_b_factor_portability(
            results_dir / "experiment_b_factor_portability.parquet",
            results_dir / "experiment_b_factor_portability_coefficients.csv",
            results_dir / "experiment_b_factor_portability_summary.json",
        )
    )
    failures.extend(
        verify_experiment_b_historical_backcast(
            results_dir / "experiment_b_historical_backcast_scores.parquet",
            results_dir / "experiment_b_historical_backcast_monthly_returns.parquet",
            results_dir / "experiment_b_historical_backcast_holdings.parquet",
            results_dir / "experiment_b_historical_backcast_rank_ic.parquet",
            results_dir / "experiment_b_historical_backcast_rank_ic_summary.parquet",
            results_dir / "experiment_b_historical_backcast_rank_ic_by_year.parquet",
            results_dir / "experiment_b_historical_backcast_rank_ic_by_sector.parquet",
            results_dir / "experiment_b_historical_backcast_rank_ic_coverage.parquet",
            results_dir / "experiment_b_historical_backcast_summary.json",
        )
    )
    failures.extend(verify_v2_status(results_dir / "v2_experiment_status.json"))
    return failures


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify generated Parallax research artifacts.")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    failures = run_verification(Path(args.results_dir))
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("Artifact verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
