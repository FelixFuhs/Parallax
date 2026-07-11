import json

import pandas as pd

from universe_membership import (
    build_approximate_membership_panel,
    membership_on_date,
    write_approximate_membership_artifacts,
)


def test_membership_on_date_walks_selected_changes_backward():
    changes = pd.DataFrame(
        {
            "effective_date": pd.to_datetime(["2025-06-30", "2026-06-30"]),
            "added_ticker": ["C", "D"],
            "removed_ticker": ["X", "B"],
        }
    )

    members_2024 = membership_on_date(
        current_tickers={"A", "C", "D"},
        changes=changes,
        membership_date=pd.Timestamp("2024-12-31"),
        as_of_date=pd.Timestamp("2026-12-31"),
    )
    members_2025 = membership_on_date(
        current_tickers={"A", "C", "D"},
        changes=changes,
        membership_date=pd.Timestamp("2025-12-31"),
        as_of_date=pd.Timestamp("2026-12-31"),
    )

    assert members_2024 == {"A", "B", "X"}
    assert members_2025 == {"A", "B", "C"}


def test_build_approximate_membership_panel_quantifies_missing_removed_names(tmp_path):
    security_master = tmp_path / "security.parquet"
    changes = tmp_path / "changes.csv"
    snapshot = tmp_path / "snapshot.json"
    company_tickers = tmp_path / "company_tickers.json"
    pd.DataFrame(
        {
            "ticker": ["A", "C", "D"],
            "cik": [1, 3, 4],
            "sector": ["Tech", "Energy", "Health"],
            "sector_source": ["test", "test", "test"],
        }
    ).to_parquet(security_master, index=False)
    pd.DataFrame(
        {
            "effective_date": ["2025-06-30", "2026-06-30"],
            "added_ticker": ["C", "D"],
            "added_security": ["C Inc.", "D Inc."],
            "removed_ticker": ["X", "B"],
            "removed_security": ["X Inc.", "B Inc."],
            "reason": ["test", "test"],
        }
    ).to_csv(changes, index=False)
    snapshot.write_text(json.dumps({"ticker_source_retrieved_date": "2026-12-31"}), encoding="utf-8")
    company_tickers.write_text(
        json.dumps({"0": {"ticker": "B", "cik_str": 22, "title": "B Inc."}}),
        encoding="utf-8",
    )

    panel, summary = build_approximate_membership_panel(
        security_master_path=security_master,
        changes_path=changes,
        universe_snapshot_path=snapshot,
        company_tickers_cache_path=company_tickers,
        start_year=2024,
        end_year=2025,
    )

    early = panel.loc[panel["date"] == "2024-12-31"]
    assert set(early["ticker"]) == {"A", "B", "X"}
    assert set(early.loc[~early["in_current_security_master"], "ticker"]) == {"B", "X"}
    assert early.loc[early["ticker"] == "B", "company_tickers_match"].iloc[0]
    assert early.loc[early["ticker"] == "B", "cik_source"].iloc[0] == "sec_company_tickers_cache"
    assert summary["point_in_time_membership"] is False
    assert summary["missing_from_current_security_master_ticker_count"] == 2
    assert summary["missing_with_sec_company_ticker_match_count"] == 1
    assert "removed_names_missing_security_master_rows" in {blocker["code"] for blocker in summary["blockers"]}


def test_write_approximate_membership_artifacts(tmp_path):
    security_master = tmp_path / "security.parquet"
    changes = tmp_path / "changes.csv"
    snapshot = tmp_path / "snapshot.json"
    pd.DataFrame({"ticker": ["A"], "cik": [1]}).to_parquet(security_master, index=False)
    pd.DataFrame(
        {
            "effective_date": ["2025-06-30"],
            "added_ticker": ["A"],
            "added_security": ["A Inc."],
            "removed_ticker": ["OLD"],
            "removed_security": ["Old Inc."],
            "reason": ["test"],
        }
    ).to_csv(changes, index=False)
    snapshot.write_text(json.dumps({"ticker_source_retrieved_date": "2025-12-31"}), encoding="utf-8")

    summary = write_approximate_membership_artifacts(
        security_master_path=security_master,
        changes_path=changes,
        universe_snapshot_path=snapshot,
        company_tickers_cache_path=None,
        output_path=tmp_path / "membership.parquet",
        summary_path=tmp_path / "summary.json",
        metadata_path=tmp_path / "metadata.json",
        start_year=2025,
        end_year=2025,
    )

    assert (tmp_path / "membership.parquet").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "metadata.json").exists()
    assert summary["artifacts"]["approximate_membership"].endswith("membership.parquet")
