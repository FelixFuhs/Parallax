import json

from security_master import build_security_master, parse_ticker_file, write_security_master


def test_parse_ticker_file_keeps_comment_sector_groups(tmp_path):
    tickers = tmp_path / "tickers.txt"
    tickers.write_text("# Retrieved 2026-03-21\n# Technology\nAAPL\nMSFT\n# Utilities\nNEE\n", encoding="utf-8")

    names, comments, groups = parse_ticker_file(tickers)

    assert names == ["AAPL", "MSFT", "NEE"]
    assert "Retrieved 2026-03-21" in comments
    assert groups == {"AAPL": "Technology", "MSFT": "Technology", "NEE": "Utilities"}


def test_security_master_records_cik_mapping_and_survivor_bias(tmp_path):
    tickers = tmp_path / "tickers.txt"
    tickers.write_text("# S&P list retrieved 2026-03-21\nAAPL\nMSFT\n", encoding="utf-8")
    company_cache = tmp_path / "company_tickers.json"
    company_cache.write_text(
        json.dumps(
            {
                "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
                "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp."},
            }
        ),
        encoding="utf-8",
    )
    edgar_features = tmp_path / "features.json"
    edgar_features.write_text(
        json.dumps(
            {
                "AAPL": {
                    "company_name": "Apple Inc.",
                    "filing_accession": "0000320193-26-000001",
                    "filing_form": "10-K",
                    "period_end": "2025-09-30",
                }
            }
        ),
        encoding="utf-8",
    )

    frame, snapshot = build_security_master(
        tickers_file=tickers,
        company_tickers_cache=company_cache,
        edgar_features_path=edgar_features,
        sector_map_path=None,
        membership_changes_path=None,
    )

    assert frame.loc[frame["ticker"] == "AAPL", "cik"].iloc[0] == 320193
    assert frame.loc[frame["ticker"] == "AAPL", "filing_accession"].iloc[0] == "0000320193-26-000001"
    assert snapshot["point_in_time_membership"] is False
    assert "not CRSP/Compustat-quality" in snapshot["survivor_bias_warning"]
    assert snapshot["ticker_source_retrieved_date"] == "2026-03-21"


def test_write_security_master_outputs_artifacts(tmp_path):
    tickers = tmp_path / "tickers.txt"
    tickers.write_text("AAPL\n", encoding="utf-8")
    company_cache = tmp_path / "company_tickers.json"
    company_cache.write_text(
        json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}),
        encoding="utf-8",
    )

    snapshot = write_security_master(
        tickers_file=tickers,
        company_tickers_cache=company_cache,
        edgar_features_path=None,
        sector_map_path=None,
        membership_changes_path=None,
        output_path=tmp_path / "security_master.parquet",
        snapshot_path=tmp_path / "universe_snapshot.json",
        metadata_path=tmp_path / "metadata.json",
    )

    assert (tmp_path / "security_master.parquet").exists()
    assert (tmp_path / "universe_snapshot.json").exists()
    assert (tmp_path / "metadata.json").exists()
    assert snapshot["artifacts"]["security_master"].endswith("security_master.parquet")


def test_security_master_prefers_sector_map_over_ticker_group(tmp_path):
    tickers = tmp_path / "tickers.txt"
    tickers.write_text("# Utilities\nAAPL\n", encoding="utf-8")
    company_cache = tmp_path / "company_tickers.json"
    company_cache.write_text(
        json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}),
        encoding="utf-8",
    )
    sector_map = tmp_path / "sector_map.csv"
    sector_map.write_text(
        "ticker,sector,sub_industry,sector_source,source_url\n"
        "AAPL,Information Technology,Technology Hardware,wikipedia_sp500_constituents,https://example.test\n",
        encoding="utf-8",
    )

    frame, snapshot = build_security_master(
        tickers_file=tickers,
        company_tickers_cache=company_cache,
        edgar_features_path=None,
        sector_map_path=sector_map,
        membership_changes_path=None,
    )

    row = frame.iloc[0]
    assert row["sector"] == "Information Technology"
    assert row["sub_industry"] == "Technology Hardware"
    assert row["sector_source"] == "wikipedia_sp500_constituents"
    assert snapshot["sector_coverage"] == 1.0


def test_security_master_records_approximate_membership_changes(tmp_path):
    tickers = tmp_path / "tickers.txt"
    tickers.write_text("AAPL\n", encoding="utf-8")
    company_cache = tmp_path / "company_tickers.json"
    company_cache.write_text(
        json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}),
        encoding="utf-8",
    )
    changes = tmp_path / "sp500_changes.csv"
    changes.write_text(
        "effective_date,added_ticker,added_security,removed_ticker,removed_security,reason\n"
        "2026-01-02,AAPL,Apple Inc.,OLD,Old Inc.,Synthetic test change\n",
        encoding="utf-8",
    )

    frame, snapshot = build_security_master(
        tickers_file=tickers,
        company_tickers_cache=company_cache,
        edgar_features_path=None,
        sector_map_path=None,
        membership_changes_path=changes,
    )

    assert frame["membership_history_source"].iloc[0].endswith("sp500_changes.csv")
    assert snapshot["membership_change_event_count"] == 1
    assert snapshot["membership_history_quality"] == "selected_public_changes_not_full_point_in_time_membership"
