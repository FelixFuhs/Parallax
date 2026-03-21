import pandas as pd
import pytest

from stability import coefficient_sign_summary, summary_stats, top_quartile_tickers


def test_summary_stats_reports_basic_moments():
    stats = summary_stats([0.1, 0.2, 0.3, 0.4, 0.5])

    assert stats.mean == pytest.approx(0.3)
    assert stats.std == pytest.approx(0.1414213562)
    assert stats.lower == pytest.approx(0.12)
    assert stats.upper == pytest.approx(0.48)


def test_top_quartile_tickers_uses_ceiling_cutoff():
    predictions = pd.Series(
        [9, 8, 7, 6, 5, 4, 3, 2, 1],
        index=["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        dtype=float,
    )

    assert top_quartile_tickers(predictions) == {"A", "B", "C"}


def test_coefficient_sign_summary_flags_expected_sign_and_flips():
    coefficient_frame = pd.DataFrame(
        {
            "fcf_to_ev": [0.2, 0.1, 0.0],
            "gross_profitability_assets": [0.3, -0.2, 0.0],
            "asset_growth_1y": [-0.1, -0.2, 0.0],
            "cash_earnings_gap": [0.05, 0.02, 0.01],
            "momentum_12_1": [0.0, -0.1, 0.2],
        }
    )

    summary = coefficient_sign_summary(coefficient_frame)

    assert summary.loc["fcf_to_ev", "expected_sign_share"] == pytest.approx(2 / 3)
    assert bool(summary.loc["gross_profitability_assets", "sign_flip"]) is True
    assert summary.loc["asset_growth_1y", "expected_sign_share"] == pytest.approx(2 / 3)
    assert summary.loc["cash_earnings_gap", "expected_sign_share"] == pytest.approx(1.0)
