import pandas as pd
import pytest

from signal_diagnostics import (
    build_signal_comparison_table,
    decompose_signal,
    rank_ic_by_sector,
    rank_ic_coverage,
    rank_ic_diagnostics,
    summarize_rank_ic,
    summarize_rank_ic_by_sector,
    summarize_rank_ic_by_year,
)


def test_decompose_signal_builds_within_and_across_sector_scores():
    frame = pd.DataFrame(
        {
            "sector": ["Tech", "Tech", "Energy", "Energy"],
            "score": [1.0, 3.0, 10.0, 14.0],
        }
    )

    decomposed = decompose_signal(frame, "score")

    assert decomposed["within_sector_score"].tolist() == pytest.approx([-1.0, 1.0, -2.0, 2.0])
    assert decomposed["sector_score"].tolist() == pytest.approx([2.0, 2.0, 12.0, 12.0])


def test_rank_ic_diagnostics_reports_global_sector_neutral_and_across_sector():
    frame = pd.DataFrame(
        {
            "date": ["2026-01-31"] * 4,
            "sector": ["Tech", "Tech", "Energy", "Energy"],
            "signal": [1.0, 2.0, 3.0, 4.0],
            "return_1m": [0.01, 0.02, 0.03, 0.04],
        }
    )

    ic = rank_ic_diagnostics(
        frame,
        date_column="date",
        signal_columns=["signal"],
        return_columns=["return_1m"],
    )
    summary = summarize_rank_ic(ic)

    assert set(ic["decomposition"]) == {"global", "within_sector", "across_sector"}
    assert summary.loc[summary["decomposition"] == "global", "mean_ic"].iloc[0] == pytest.approx(1.0)


def test_rank_ic_summaries_include_year_sector_and_coverage():
    frame = pd.DataFrame(
        {
            "date": ["2026-01-31"] * 4 + ["2027-01-31"] * 4,
            "sector": ["Tech", "Tech", "Energy", "Energy"] * 2,
            "signal": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 4.0, 3.0],
            "return_1m": [0.01, 0.02, 0.03, 0.04, 0.01, 0.02, 0.04, 0.03],
        }
    )

    ic = rank_ic_diagnostics(
        frame,
        date_column="date",
        signal_columns=["signal"],
        return_columns=["return_1m"],
    )
    yearly = summarize_rank_ic_by_year(ic)
    sector_detail = rank_ic_by_sector(
        frame,
        date_column="date",
        signal_columns=["signal"],
        return_columns=["return_1m"],
    )
    sector_summary = summarize_rank_ic_by_sector(sector_detail)
    coverage = rank_ic_coverage(
        frame,
        date_column="date",
        signal_columns=["signal"],
        return_columns=["return_1m"],
    )

    assert set(yearly["year"]) == {2026, 2027}
    assert set(sector_summary["sector"]) == {"Energy", "Tech"}
    assert coverage["paired_n"].tolist() == [4, 4]
    assert coverage["paired_coverage"].tolist() == pytest.approx([1.0, 1.0])


def test_signal_comparison_table_pivots_required_decompositions():
    summary = pd.DataFrame(
        {
            "signal": ["raw_ai_implied_irr", "raw_ai_implied_irr", "raw_ai_implied_irr"],
            "horizon": ["return_1m", "return_1m", "return_1m"],
            "decomposition": ["global", "within_sector", "across_sector"],
            "months": [24, 24, 24],
            "mean_ic": [0.03, 0.02, 0.01],
            "newey_west_tstat": [2.0, 1.5, 0.5],
            "positive_ic_hit_rate": [0.6, 0.55, 0.5],
        }
    )

    comparison = build_signal_comparison_table(
        summary,
        signal_labels={"raw_ai_implied_irr": "AI IRR"},
    )
    row = comparison.iloc[0]

    assert row["signal_label"] == "AI IRR"
    assert row["global_mean_ic"] == pytest.approx(0.03)
    assert row["sector_neutral_mean_ic"] == pytest.approx(0.02)
    assert row["across_sector_mean_ic"] == pytest.approx(0.01)
    assert row["comparison_status"] == "available"


def test_rank_ic_marks_sector_decomposition_blocked_without_sector_coverage():
    frame = pd.DataFrame(
        {
            "date": ["2026-01-31"] * 3,
            "signal": [1.0, 2.0, 3.0],
            "return_1m": [0.01, 0.02, 0.03],
        }
    )

    ic = rank_ic_diagnostics(
        frame,
        date_column="date",
        signal_columns=["signal"],
        return_columns=["return_1m"],
    )

    blocked = ic[ic["decomposition"] == "across_sector"].iloc[0]
    assert blocked["sector_status"] == "blocked_missing_sector"
    assert blocked["n"] == 0
