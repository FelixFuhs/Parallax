import json

import pandas as pd

from verify_artifacts import (
    assert_no_absolute_paths,
    verify_approximate_membership,
    verify_completion_report,
    verify_critic_report,
    verify_experiment_b_factor_portability,
    verify_experiment_b_historical_backcast,
    verify_experiment_c_text_features,
    verify_forward_returns,
    verify_freeze_artifacts,
    verify_label_panel,
    verify_portfolio_audit_artifacts,
    verify_quarterly_fundamentals,
    verify_rank_ic_artifacts,
    verify_sector_map,
    verify_sp500_changes,
    verify_universe_snapshot,
    verify_v2_status,
)


def test_assert_no_absolute_paths_flags_windows_and_unix_paths(tmp_path):
    path = tmp_path / "payload.json"
    path.write_text(json.dumps({"bad": "C:\\Users\\Name\\repo\\file.json", "also": "/Users/name/repo/file"}), encoding="utf-8")

    failures = assert_no_absolute_paths([path])

    assert len(failures) == 2


def test_verify_completion_report_requires_claim_ceiling_and_blockers(tmp_path):
    report_path = tmp_path / "completion.md"
    report_path.write_text(
        "\n".join(
            [
                "# AI Label Decomposition Completion Report",
                "Full objective status: incomplete",
                "Claim ceiling: diagnostic/private research",
                "## Research Layers Changed",
                "## Artifacts Written Or Updated",
                "## Result Summary",
                "## Commands Run",
                    "118 passed",
                "Artifact verification passed.",
                "## Critic And Verifier Review",
                "## Remaining Blockers",
                "## Acceptance Audit",
                "The full research objective remains incomplete",
            ]
        ),
        encoding="utf-8",
    )

    assert verify_completion_report(report_path) == []


def test_verify_completion_report_rejects_completion_claim(tmp_path):
    report_path = tmp_path / "completion.md"
    report_path.write_text(
        "\n".join(
            [
                "# AI Label Decomposition Completion Report",
                "Full objective status: complete",
                "Claim ceiling: diagnostic/private research",
            ]
        ),
        encoding="utf-8",
    )

    failures = verify_completion_report(report_path)

    assert any("Full objective status: incomplete" in failure for failure in failures)
    assert any("unsupported completion" in failure for failure in failures)


def test_verify_label_panel_matches_summary_counts(tmp_path):
    panel_path = tmp_path / "label_panel.parquet"
    summary_path = tmp_path / "summary.json"
    panel = pd.DataFrame(
        {
            "ticker": ["A"],
            "exclude_from_clean_label": [False],
            "raw_ai_implied_irr": [0.1],
            "raw_ai_annualized_value_gap": [0.1],
            "mechanical_dcf_implied_irr": [0.05],
            "ai_minus_mechanical_irr": [0.05],
            "factor_compressible_ai_score": [0.08],
            "ai_factor_residual": [0.02],
            "ai_irr_iqr": [None],
            "ai_irr_rank_std": [None],
            "model_disagreement": [None],
            "tier_disagreement": [None],
            "prompt_disagreement": [None],
            "uncertainty_adjusted_label_weight": [1.0],
            "quality_flags": [[]],
            "mechanical_price_source": ["raw_close_price"],
            "raw_close_price": [10.0],
            "adjusted_close_price": [9.5],
            "sector": ["Technology"],
            "sector_source": ["test_map"],
            "sub_industry": ["Software"],
        }
    )
    panel.to_parquet(panel_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "row_count": 1,
                "ticker_count": 1,
                "clean_label_count": 1,
                "sector_coverage": 1.0,
                "raw_price_coverage": 1.0,
                "failure_rates_by_model_tier": [],
                "failure_rates_by_market_cap_bucket": [],
            }
        ),
        encoding="utf-8",
    )

    assert verify_label_panel(panel_path, summary_path) == []


def test_verify_label_panel_rejects_report_price_mechanical_control(tmp_path):
    panel_path = tmp_path / "label_panel.parquet"
    summary_path = tmp_path / "summary.json"
    panel = pd.DataFrame(
        {
            "ticker": ["A"],
            "exclude_from_clean_label": [False],
            "raw_ai_implied_irr": [0.1],
            "raw_ai_annualized_value_gap": [0.1],
            "mechanical_dcf_implied_irr": [0.05],
            "ai_minus_mechanical_irr": [0.05],
            "factor_compressible_ai_score": [0.08],
            "ai_factor_residual": [0.02],
            "ai_irr_iqr": [None],
            "ai_irr_rank_std": [None],
            "model_disagreement": [None],
            "tier_disagreement": [None],
            "prompt_disagreement": [None],
            "uncertainty_adjusted_label_weight": [1.0],
            "quality_flags": [[]],
            "mechanical_price_source": ["report_current_price"],
            "raw_close_price": [None],
            "adjusted_close_price": [9.5],
            "sector": ["Technology"],
            "sector_source": ["test_map"],
            "sub_industry": ["Software"],
        }
    )
    panel.to_parquet(panel_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "row_count": 1,
                "ticker_count": 1,
                "clean_label_count": 1,
                "sector_coverage": 1.0,
                "raw_price_coverage": 0.0,
                "failure_rates_by_model_tier": [],
                "failure_rates_by_market_cap_bucket": [],
            }
        ),
        encoding="utf-8",
    )

    failures = verify_label_panel(panel_path, summary_path)

    assert any("must not use AI report current_price" in failure for failure in failures)


def test_verify_sector_map_matches_summary_counts(tmp_path):
    sector_map_path = tmp_path / "sector_map.csv"
    summary_path = tmp_path / "sector_map_summary.json"
    pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "company_name": ["A Inc.", "B Inc."],
            "sector": ["Technology", "Health Care"],
            "sub_industry": ["Software", "Biotech"],
            "sector_source": ["wikipedia_sp500_constituents", "wikipedia_sp500_constituents"],
            "source_url": ["https://example.test", "https://example.test"],
        }
    ).to_csv(sector_map_path, index=False)
    summary_path.write_text(json.dumps({"row_count": 2, "ticker_count": 2}), encoding="utf-8")

    assert verify_sector_map(sector_map_path, summary_path) == []


def test_verify_sp500_changes_matches_summary_counts(tmp_path):
    changes_path = tmp_path / "sp500_changes.csv"
    summary_path = tmp_path / "sp500_changes_summary.json"
    pd.DataFrame(
        {
            "effective_date": ["2026-05-07"],
            "added_ticker": ["VEEV"],
            "added_security": ["Veeva Systems"],
            "removed_ticker": ["CTRA"],
            "removed_security": ["Coterra Energy"],
            "reason": ["Synthetic test change"],
            "approximate_membership_history": [True],
            "point_in_time_membership": [False],
        }
    ).to_csv(changes_path, index=False)
    summary_path.write_text(
        json.dumps({"row_count": 1, "point_in_time_membership": False, "approximate_membership_history": True}),
        encoding="utf-8",
    )

    assert verify_sp500_changes(changes_path, summary_path) == []


def test_verify_forward_returns_requires_zero_coverage_blocker(tmp_path):
    returns_path = tmp_path / "forward_returns.parquet"
    summary_path = tmp_path / "forward_returns_summary.json"
    pd.DataFrame({"ticker": ["A"], "return_1m": [None]}).to_parquet(returns_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "row_count": 1,
                "ticker_count": 1,
                "coverage": {"return_1m": {"non_null": 0, "coverage": 0.0}},
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )

    failures = verify_forward_returns(returns_path, summary_path)

    assert "forward returns with zero coverage must carry no_usable_forward_returns blocker" in failures


def test_verify_rank_ic_artifacts_require_diagnostic_tables(tmp_path):
    pd.DataFrame(
        {"date": ["2026-01-31"], "signal": ["s"], "horizon": ["return_1m"], "decomposition": ["global"], "n": [2], "rank_ic": [1.0]}
    ).to_parquet(tmp_path / "rank_ic.parquet", index=False)
    pd.DataFrame(
        {
            "signal": ["s"],
            "horizon": ["return_1m"],
            "decomposition": ["global"],
            "months": [1],
            "mean_ic": [1.0],
            "median_ic": [1.0],
            "ic_std": [0.0],
            "newey_west_tstat": [None],
            "positive_ic_hit_rate": [1.0],
        }
    ).to_parquet(tmp_path / "rank_ic_summary.parquet", index=False)
    pd.DataFrame(
        {
            "year": [2026],
            "signal": ["s"],
            "horizon": ["return_1m"],
            "decomposition": ["global"],
            "months": [1],
            "mean_ic": [1.0],
            "median_ic": [1.0],
            "positive_ic_hit_rate": [1.0],
        }
    ).to_parquet(tmp_path / "rank_ic_by_year.parquet", index=False)
    pd.DataFrame(
        {
            "sector": ["Tech"],
            "signal": ["s"],
            "horizon": ["return_1m"],
            "months": [1],
            "mean_ic": [1.0],
            "median_ic": [1.0],
            "positive_ic_hit_rate": [1.0],
            "mean_n": [2.0],
        }
    ).to_parquet(tmp_path / "rank_ic_by_sector.parquet", index=False)
    pd.DataFrame(
        {
            "date": ["2026-01-31"],
            "signal": ["s"],
            "horizon": ["return_1m"],
            "universe_n": [2],
            "score_non_null": [2],
            "return_non_null": [2],
            "paired_n": [2],
            "paired_coverage": [1.0],
            "sector_count": [1],
        }
    ).to_parquet(tmp_path / "rank_ic_coverage.parquet", index=False)
    pd.DataFrame(
        {
            "signal": ["s"],
            "signal_label": ["S"],
            "horizon": ["return_1m"],
            "global_mean_ic": [1.0],
            "sector_neutral_mean_ic": [1.0],
            "across_sector_mean_ic": [1.0],
            "global_months": [1],
            "sector_neutral_months": [1],
            "across_sector_months": [1],
            "comparison_status": ["diagnostic_insufficient_history"],
        }
    ).to_parquet(tmp_path / "signal_comparison.parquet", index=False)
    (tmp_path / "v2_experiment_status.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "rank_ic_by_year": "results/rank_ic_by_year.parquet",
                    "rank_ic_by_sector": "results/rank_ic_by_sector.parquet",
                    "rank_ic_coverage": "results/rank_ic_coverage.parquet",
                    "signal_comparison": "results/signal_comparison.parquet",
                }
            }
        ),
        encoding="utf-8",
    )

    assert verify_rank_ic_artifacts(tmp_path) == []


def test_verify_freeze_artifacts_enforces_primary_and_baseline_roles(tmp_path):
    models_dir = tmp_path / "models"
    docs_dir = tmp_path / "docs"
    models_dir.mkdir()
    docs_dir.mkdir()
    (models_dir / "frozen_xgb_regressor.json").write_text("{}", encoding="utf-8")
    (docs_dir / "freeze_elasticnet_baseline.md").write_text("# Elastic Net Baseline\n", encoding="utf-8")
    (models_dir / "frozen_model_metadata.json").write_text(
        json.dumps(
            {
                "artifact_role": "primary_model_freeze",
                "model_type": "xgboost_regressor",
                "source_artifacts": {"frozen_model": "models/frozen_xgb_regressor.json"},
            }
        ),
        encoding="utf-8",
    )
    (models_dir / "frozen_elasticnet_metadata.json").write_text(
        json.dumps(
            {
                "artifact_role": "baseline_model_freeze",
                "primary_model": False,
                "primary_model_reference": "models/frozen_xgb_regressor.json",
                "model_family": "ElasticNetCV",
            }
        ),
        encoding="utf-8",
    )

    assert verify_freeze_artifacts(models_dir, docs_dir) == []


def test_verify_critic_report_requires_three_review_passes_and_blockers(tmp_path):
    report_path = tmp_path / "critic_report.json"
    report_path.write_text(
        json.dumps(
            {
                "experiment_id": "ai_label_decomposition_critic_review",
                "goal_completion_claimed": False,
                "review_passes": [
                    {
                        "review_type": "implementation_critic",
                        "reviewer": "test",
                        "completed_at": "2026-05-19T00:00:00Z",
                        "scope": "implementation intent",
                        "findings": [
                            {
                                "severity": "high",
                                "title": "No repeated label vintages",
                                "evidence": ["results/label_panel_summary.json"],
                                "disposition": "documented_blocker",
                                "completion_blocker": True,
                            }
                        ],
                    },
                    {
                        "review_type": "verification_critic",
                        "reviewer": "test",
                        "completed_at": "2026-05-19T00:00:00Z",
                        "scope": "verification risks",
                        "findings": [],
                    },
                    {
                        "review_type": "wording_critic",
                        "reviewer": "test",
                        "completed_at": "2026-05-19T00:00:00Z",
                        "scope": "claim wording",
                        "findings": [],
                    },
                ],
                "remaining_completion_blockers": [
                    {"code": "missing_historical_label_vintages", "blocker_type": "data"}
                ],
            }
        ),
        encoding="utf-8",
    )

    assert verify_critic_report(report_path) == []


def test_verify_v2_status_rejects_complete_one_month_smoke_test(tmp_path):
    status_path = tmp_path / "v2_experiment_status.json"
    status_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "blockers": [],
                "diagnostic_month_count": 1,
                "forward_return_coverage": {
                    "return_1m": {"non_null": 10},
                    "return_3m": {"non_null": 0},
                },
            }
        ),
        encoding="utf-8",
    )

    failures = verify_v2_status(status_path)

    assert "fewer than 12 diagnostic return months" in " ".join(failures)
    assert "zero coverage" in " ".join(failures)


def test_verify_portfolio_audit_artifacts_require_portfolio_modes(tmp_path):
    pd.DataFrame(
        {
            "date": ["2026-03-21", "2026-03-21"],
            "ticker": ["A", "B"],
            "sector": ["Tech", "Energy"],
            "market_cap": [100.0, 90.0],
            "score": [1.0, 2.0],
            "signal_name": ["s", "s"],
            "portfolio_mode": ["unconstrained", "sector_neutral"],
            "weighting_method": ["equal_name", "equal_sector_then_equal_name"],
            "bucket": ["Q1", "Q1"],
            "weight": [1.0, 1.0],
            "entry_price": [10.0, 10.0],
            "exit_price": [11.0, 11.0],
            "raw_return": [0.1, 0.1],
            "transaction_cost": [0.0, 0.0],
            "net_return": [0.1, 0.1],
        }
    ).to_parquet(tmp_path / "holdings.parquet", index=False)
    pd.DataFrame(
        {
            "date": ["2026-03-21", "2026-03-21"],
            "signal_name": ["s", "s"],
            "portfolio_mode": ["unconstrained", "sector_neutral"],
            "weighting_method": ["equal_name", "equal_sector_then_equal_name"],
            "bucket": ["Q1", "Q1"],
            "cost_bps_one_way": [0, 0],
            "gross_return": [0.1, 0.1],
            "transaction_cost_drag": [0.0, 0.0],
            "net_return": [0.1, 0.1],
            "name_count": [1, 1],
        }
    ).to_parquet(tmp_path / "monthly_returns.parquet", index=False)
    pd.DataFrame(
        {
            "date": ["2026-03-21", "2026-03-21"],
            "signal_name": ["s", "s"],
            "portfolio_mode": ["unconstrained", "sector_neutral"],
            "weighting_method": ["equal_name", "equal_sector_then_equal_name"],
            "bucket": ["Q1", "Q1"],
            "cost_bps_one_way": [0, 0],
            "turnover": [1.0, 1.0],
            "transaction_cost_drag": [0.0, 0.0],
        }
    ).to_parquet(tmp_path / "turnover.parquet", index=False)
    pd.DataFrame(
        {
            "date": ["2026-03-21", "2026-03-21"],
            "signal_name": ["s", "s"],
            "portfolio_mode": ["unconstrained", "sector_neutral"],
            "weighting_method": ["equal_name", "equal_sector_then_equal_name"],
            "bucket": ["Q1", "Q1"],
            "sector_weights": ["{}", "{}"],
            "average_market_cap": [100.0, 90.0],
            "name_count": [1, 1],
        }
    ).to_parquet(tmp_path / "exposures.parquet", index=False)
    (tmp_path / "v2_experiment_status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "portfolio_return_column": "return_1m",
                "sector_coverage": 1.0,
                "portfolio_modes": ["sector_neutral", "unconstrained"],
            }
        ),
        encoding="utf-8",
    )

    assert verify_portfolio_audit_artifacts(tmp_path) == []


def test_verify_universe_snapshot_matches_security_master(tmp_path):
    security_master_path = tmp_path / "security_master.parquet"
    snapshot_path = tmp_path / "universe_snapshot.json"
    pd.DataFrame(
        {
            "ticker": ["A"],
            "cik": [1],
            "company_name": ["A Inc."],
            "point_in_time_membership": [False],
            "membership_history_source": [None],
        }
    ).to_parquet(security_master_path, index=False)
    snapshot_path.write_text(
        json.dumps(
            {
                "ticker_count": 1,
                "point_in_time_membership": False,
                "survivor_bias_warning": "Universe is not CRSP/Compustat-quality point-in-time membership.",
            }
        ),
        encoding="utf-8",
    )

    assert verify_universe_snapshot(security_master_path, snapshot_path) == []


def test_verify_approximate_membership_quantifies_gap(tmp_path):
    panel_path = tmp_path / "approx_membership.parquet"
    summary_path = tmp_path / "approx_membership_summary.json"
    pd.DataFrame(
        {
            "date": ["2025-01-31", "2025-01-31"],
            "ticker": ["A", "OLD"],
            "approximate_member": [True, True],
            "in_current_security_master": [True, False],
            "company_tickers_match": [True, True],
            "has_cik": [True, False],
            "cik_source": ["current_security_master", None],
            "membership_basis": ["current_security_master", "selected_changes_removed_ticker_backfill"],
            "point_in_time_membership": [False, False],
            "membership_history_quality": [
                "approximate_public_selected_changes_not_full_constituent_history",
                "approximate_public_selected_changes_not_full_constituent_history",
            ],
        }
    ).to_parquet(panel_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "row_count": 2,
                "point_in_time_membership": False,
                "missing_from_current_security_master_ticker_count": 1,
                "missing_with_sec_company_ticker_match_count": 1,
                "claim_limit": "This is not CRSP/Compustat-quality membership.",
                "blockers": [
                    {"code": "removed_names_missing_security_master_rows"},
                    {"code": "selected_changes_not_full_point_in_time_membership"},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert verify_approximate_membership(panel_path, summary_path) == []


def test_verify_quarterly_fundamentals_matches_summary(tmp_path):
    panel_path = tmp_path / "quarterly.parquet"
    summary_path = tmp_path / "quarterly_summary.json"
    pd.DataFrame(
        {
            "ticker": ["A"],
            "cik": ["0000000001"],
            "fiscal_period": ["Q1"],
            "period_end": ["2026-03-31"],
            "filed": ["2026-05-01"],
            "revenue": [1.0],
            "operating_cash_flow": [2.0],
            "capex": [0.5],
            "free_cash_flow": [1.5],
            "revenue_ttm": [None],
            "revenue_qoq_change": [None],
            "revenue_yoy_change": [None],
        }
    ).to_parquet(panel_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "row_count": 1,
                "ticker_count": 1,
                "method_notes": ["TTM rollups do not infer Q4 from 10-K annual facts."],
            }
        ),
        encoding="utf-8",
    )

    assert verify_quarterly_fundamentals(panel_path, summary_path) == []


def test_verify_experiment_c_text_features_enforces_separation(tmp_path):
    panel_path = tmp_path / "experiment_c.parquet"
    requests_path = tmp_path / "experiment_c_requests.parquet"
    llm_requests_path = tmp_path / "experiment_c_llm_requests.parquet"
    llm_responses_path = tmp_path / "experiment_c_llm_responses.parquet"
    manifest_path = tmp_path / "experiment_c_manifest.json"
    pd.DataFrame(
        columns=[
            "ticker",
            "filing_accession",
            "source_hash",
            "extraction_model_id",
            "prompt_id",
            "prompt_version",
            "risk_factor_novelty",
            "quality_flags",
        ]
    ).to_parquet(panel_path, index=False)
    pd.DataFrame(
        {
            "ticker": ["A"],
            "cik": ["0000000001"],
            "filing_accession": ["0000000001-26-000001"],
            "filing_form": ["10-Q"],
            "filing_lookback_rank": [1],
            "filed": ["2026-05-01"],
            "source_url": ["https://www.sec.gov/Archives/edgar/data/1/000000000126000001/0000000001-26-000001.txt"],
            "source_path": ["data/sec_filing_text/0000000001/0000000001-26-000001.txt"],
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
    pd.DataFrame(
        {
            "ticker": ["A"],
            "filing_accession": ["0000000001-26-000001"],
            "filing_form": ["10-Q"],
            "source_path": ["data/sec_filing_text/0000000001/0000000001-26-000001.txt"],
            "source_hash": [None],
            "date_limited_source": [True],
            "download_status": ["not_downloaded"],
            "llm_request_status": ["blocked_missing_downloaded_text"],
            "llm_extraction_status": ["not_run"],
            "prompt_id": ["experiment_c_sec_filing_text_features"],
            "prompt_version": ["v1_date_limited_sec_filings"],
            "system_prompt": ["Do not use DCF labels."],
            "user_prompt": ["Extract filing features. <FILING_TEXT>"],
            "output_schema_json": [json.dumps({"type": "object", "properties": {"risk_factor_novelty": {}}})],
            "quality_flags": [[]],
        }
    ).to_parquet(llm_requests_path, index=False)
    pd.DataFrame(
        columns=[
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
        ]
    ).to_parquet(llm_responses_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "experiment_id": "experiment_c_llm_text_features",
                "separate_from_dcf_labels": True,
                "status": "blocked",
                "blockers": [{"code": "missing_date_limited_text_corpus"}],
                "llm_prompt_contract": {"llm_extraction_not_run": True, "output_schema": {"type": "object"}},
                "artifacts": {
                    "text_corpus_requests": str(requests_path),
                    "llm_extraction_requests": str(llm_requests_path),
                    "llm_extraction_responses": str(llm_responses_path),
                },
                "text_corpus_request_summary": {"row_count": 1},
                "llm_extraction_request_summary": {"row_count": 1, "ready_for_llm_extraction_count": 0},
                "llm_extraction_response_summary": {"row_count": 0, "valid_response_count": 0},
            }
        ),
        encoding="utf-8",
    )

    assert verify_experiment_c_text_features(panel_path, manifest_path) == []


def test_verify_experiment_c_text_features_rejects_downloaded_rows_without_local_source(tmp_path):
    panel_path = tmp_path / "experiment_c.parquet"
    requests_path = tmp_path / "experiment_c_requests.parquet"
    llm_requests_path = tmp_path / "experiment_c_llm_requests.parquet"
    llm_responses_path = tmp_path / "experiment_c_llm_responses.parquet"
    manifest_path = tmp_path / "experiment_c_manifest.json"
    pd.DataFrame(
        columns=[
            "ticker",
            "filing_accession",
            "source_hash",
            "extraction_model_id",
            "prompt_id",
            "prompt_version",
            "risk_factor_novelty",
            "quality_flags",
        ]
    ).to_parquet(panel_path, index=False)
    pd.DataFrame(
        {
            "ticker": ["A"],
            "cik": ["0000000001"],
            "filing_accession": ["0000000001-26-000001"],
            "filing_form": ["10-Q"],
            "filing_lookback_rank": [1],
            "filed": ["2026-05-01"],
            "source_url": ["https://www.sec.gov/Archives/edgar/data/1/000000000126000001/0000000001-26-000001.txt"],
            "source_path": [str(tmp_path / "missing.txt")],
            "source_hash": [None],
            "date_limited_source": [True],
            "download_status": ["downloaded"],
            "downloaded_at": ["2026-05-19T00:00:00+00:00"],
            "http_status": [200],
            "download_error": [None],
            "text_extraction_status": ["not_run"],
            "quality_flags": [[]],
        }
    ).to_parquet(requests_path, index=False)
    pd.DataFrame(
        {
            "ticker": ["A"],
            "filing_accession": ["0000000001-26-000001"],
            "filing_form": ["10-Q"],
            "source_path": [str(tmp_path / "missing.txt")],
            "source_hash": [None],
            "date_limited_source": [True],
            "download_status": ["downloaded"],
            "llm_request_status": ["blocked_missing_source_hash"],
            "llm_extraction_status": ["not_run"],
            "prompt_id": ["experiment_c_sec_filing_text_features"],
            "prompt_version": ["v1_date_limited_sec_filings"],
            "system_prompt": ["Do not use DCF labels."],
            "user_prompt": ["Extract filing features. <FILING_TEXT>"],
            "output_schema_json": [json.dumps({"type": "object", "properties": {"risk_factor_novelty": {}}})],
            "quality_flags": [[]],
        }
    ).to_parquet(llm_requests_path, index=False)
    pd.DataFrame(
        columns=[
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
        ]
    ).to_parquet(llm_responses_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "experiment_id": "experiment_c_llm_text_features",
                "separate_from_dcf_labels": True,
                "status": "blocked",
                "blockers": [{"code": "missing_date_limited_text_corpus"}],
                "llm_prompt_contract": {"llm_extraction_not_run": True, "output_schema": {"type": "object"}},
                "artifacts": {
                    "text_corpus_requests": str(requests_path),
                    "llm_extraction_requests": str(llm_requests_path),
                    "llm_extraction_responses": str(llm_responses_path),
                },
                "text_corpus_request_summary": {"row_count": 1},
                "llm_extraction_request_summary": {"row_count": 1, "ready_for_llm_extraction_count": 0},
                "llm_extraction_response_summary": {"row_count": 0, "valid_response_count": 0},
            }
        ),
        encoding="utf-8",
    )

    failures = verify_experiment_c_text_features(panel_path, manifest_path)

    assert any("source_hash" in failure for failure in failures)
    assert any("existing local text files" in failure for failure in failures)


def test_verify_experiment_b_factor_portability_enforces_claim_ceiling(tmp_path):
    predictions_path = tmp_path / "experiment_b.parquet"
    coefficients_path = tmp_path / "experiment_b_coefficients.csv"
    summary_path = tmp_path / "experiment_b_summary.json"
    pd.DataFrame(
        {
            "ticker": ["A"],
            "target": ["raw_ai_implied_irr"],
            "observed_label": [0.1],
            "cv_predicted_label": [0.09],
            "fit_predicted_label": [0.1],
            "fit_residual": [0.0],
            "feature_null_count": [0],
        }
    ).to_parquet(predictions_path, index=False)
    pd.DataFrame({"target": ["raw_ai_implied_irr"], "feature": ["fcf_to_ev"], "coefficient": [0.1]}).to_csv(
        coefficients_path,
        index=False,
    )
    summary_path.write_text(
        json.dumps(
            {
                "experiment_id": "experiment_b_ai_implied_factor_portability",
                "status": "fit_current_cross_section_only",
                "row_count": 1,
                "warnings": [{"code": "historical_backcast_not_run"}],
            }
        ),
        encoding="utf-8",
    )

    assert verify_experiment_b_factor_portability(predictions_path, coefficients_path, summary_path) == []


def test_verify_experiment_b_historical_backcast_matches_summary_counts(tmp_path):
    scores_path = tmp_path / "experiment_b_historical_backcast_scores.parquet"
    monthly_path = tmp_path / "experiment_b_historical_backcast_monthly_returns.parquet"
    holdings_path = tmp_path / "experiment_b_historical_backcast_holdings.parquet"
    rank_ic_path = tmp_path / "experiment_b_historical_backcast_rank_ic.parquet"
    rank_ic_summary_path = tmp_path / "experiment_b_historical_backcast_rank_ic_summary.parquet"
    rank_ic_by_year_path = tmp_path / "experiment_b_historical_backcast_rank_ic_by_year.parquet"
    rank_ic_by_sector_path = tmp_path / "experiment_b_historical_backcast_rank_ic_by_sector.parquet"
    rank_ic_coverage_path = tmp_path / "experiment_b_historical_backcast_rank_ic_coverage.parquet"
    summary_path = tmp_path / "experiment_b_historical_backcast_summary.json"
    pd.DataFrame(
        {
            "date": ["2025-01-31", "2025-01-31", "2025-01-31"],
            "ticker": ["A", "B", "C"],
            "target": ["raw_ai_implied_irr", "benchmark_composite_vqmia", "benchmark_fcf_to_ev"],
            "signal_name": [
                "experiment_b_raw_ai_implied_irr",
                "benchmark_composite_vqmia_score",
                "benchmark_fcf_to_ev",
            ],
            "score": [0.1, 0.2, 0.3],
            "sector": ["Technology", "Technology", "Technology"],
            "market_cap": [100.0, 110.0, 120.0],
            "feature_null_count": [0, 0, 0],
        }
    ).to_parquet(scores_path, index=False)
    audit_signals = ["experiment_b_raw_ai_implied_irr", "benchmark_composite_vqmia_score", "benchmark_fcf_to_ev"]
    audit_rows = [
        {
            "date": "2025-01-31",
            "signal_name": signal,
            "portfolio_mode": mode,
            "weighting_method": "equal_name" if mode == "unconstrained" else "equal_sector_then_equal_name",
            "bucket": "Q1",
            "cost_bps_one_way": 0,
            "gross_return": 0.1,
            "transaction_cost_drag": 0.0,
            "net_return": 0.1,
            "name_count": 1,
        }
        for signal in audit_signals
        for mode in ("unconstrained", "sector_neutral")
    ]
    pd.DataFrame(
        audit_rows
    ).to_parquet(monthly_path, index=False)
    pd.DataFrame(
        [
            {
                **row,
                "ticker": "A",
                "weight": 1.0,
                "raw_return": 0.1,
                "transaction_cost": 0.0,
            }
            for row in audit_rows
        ]
    ).to_parquet(holdings_path, index=False)
    pd.DataFrame(
        {
            "date": ["2025-01-31"] * len(audit_signals),
            "signal": audit_signals,
            "horizon": ["return_1m"] * len(audit_signals),
            "decomposition": ["global"] * len(audit_signals),
            "n": [2] * len(audit_signals),
            "rank_ic": [1.0] * len(audit_signals),
            "sector_status": ["available"] * len(audit_signals),
        }
    ).to_parquet(rank_ic_path, index=False)
    pd.DataFrame(
        {
            "signal": audit_signals,
            "horizon": ["return_1m"] * len(audit_signals),
            "decomposition": ["global"] * len(audit_signals),
            "months": [1] * len(audit_signals),
            "mean_ic": [1.0] * len(audit_signals),
            "median_ic": [1.0] * len(audit_signals),
            "ic_std": [0.0] * len(audit_signals),
            "newey_west_tstat": [None] * len(audit_signals),
            "positive_ic_hit_rate": [1.0] * len(audit_signals),
        }
    ).to_parquet(rank_ic_summary_path, index=False)
    pd.DataFrame(
        {
            "year": [2025] * len(audit_signals),
            "signal": audit_signals,
            "horizon": ["return_1m"] * len(audit_signals),
            "decomposition": ["global"] * len(audit_signals),
            "months": [1] * len(audit_signals),
            "mean_ic": [1.0] * len(audit_signals),
            "median_ic": [1.0] * len(audit_signals),
            "positive_ic_hit_rate": [1.0] * len(audit_signals),
        }
    ).to_parquet(rank_ic_by_year_path, index=False)
    pd.DataFrame(
        {
            "sector": ["Technology"] * len(audit_signals),
            "signal": audit_signals,
            "horizon": ["return_1m"] * len(audit_signals),
            "months": [1] * len(audit_signals),
            "mean_ic": [1.0] * len(audit_signals),
            "median_ic": [1.0] * len(audit_signals),
            "positive_ic_hit_rate": [1.0] * len(audit_signals),
            "mean_n": [2.0] * len(audit_signals),
        }
    ).to_parquet(rank_ic_by_sector_path, index=False)
    pd.DataFrame(
        {
            "date": ["2025-01-31"] * len(audit_signals),
            "signal": audit_signals,
            "horizon": ["return_1m"] * len(audit_signals),
            "universe_n": [2] * len(audit_signals),
            "score_non_null": [2] * len(audit_signals),
            "return_non_null": [2] * len(audit_signals),
            "paired_n": [2] * len(audit_signals),
            "paired_coverage": [1.0] * len(audit_signals),
            "sector_count": [1] * len(audit_signals),
        }
    ).to_parquet(rank_ic_coverage_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "experiment_id": "experiment_b_ai_implied_factor_portability_backcast",
                "status": "historical_backcast_screen",
                "score_row_count": 3,
                "monthly_return_row_count": len(audit_rows),
                "holding_row_count": len(audit_rows),
                "portfolio_modes": ["sector_neutral", "unconstrained"],
                "rank_ic_row_count": len(audit_signals),
                "rank_ic_return_horizons": ["return_1m", "return_3m", "return_6m", "return_12m"],
                "benchmark_signals": ["benchmark_composite_vqmia_score", "benchmark_fcf_to_ev"],
                "approximate_membership_gap": {
                    "status": "approximate_gap_analysis_not_point_in_time_membership",
                    "point_in_time_membership": False,
                    "missing_from_current_security_master_ticker_count": 1,
                    "missing_with_sec_company_ticker_match_count": 1,
                    "missing_without_sec_company_ticker_match_count": 0,
                    "backcast_rebalance_overlap": {
                        "rebalance_date_count": 1,
                        "overlap_month_count": 1,
                        "average_missing_from_current_security_master_rate": 0.5,
                        "max_missing_from_current_security_master_rate": 0.5,
                    },
                },
                "artifacts": {
                    "rank_ic": "results/experiment_b_historical_backcast_rank_ic.parquet",
                    "rank_ic_summary": "results/experiment_b_historical_backcast_rank_ic_summary.parquet",
                    "rank_ic_by_year": "results/experiment_b_historical_backcast_rank_ic_by_year.parquet",
                    "rank_ic_by_sector": "results/experiment_b_historical_backcast_rank_ic_by_sector.parquet",
                    "rank_ic_coverage": "results/experiment_b_historical_backcast_rank_ic_coverage.parquet",
                    "approximate_membership": "data/approx_sp500_membership.parquet",
                    "approximate_membership_summary": "results/approx_sp500_membership_summary.json",
                },
                "warnings": [
                    {"code": "current_label_projection"},
                    {"code": "survivor_universe"},
                    {"code": "current_sector_map_used"},
                    {"code": "removed_names_missing_from_backcast_universe"},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert (
        verify_experiment_b_historical_backcast(
            scores_path,
            monthly_path,
            holdings_path,
            rank_ic_path,
            rank_ic_summary_path,
            rank_ic_by_year_path,
            rank_ic_by_sector_path,
            rank_ic_coverage_path,
            summary_path,
        )
        == []
    )
