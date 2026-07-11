import pandas as pd

from benchmarks import build_composite_vqmia


def test_composite_vqmia_uses_equal_blocks_without_future_returns():
    frame = pd.DataFrame(
        {
            "fcf_to_ev": [0.10, 0.20, 0.30],
            "gross_profitability_assets": [0.30, 0.20, 0.10],
            "momentum_12_1": [0.05, 0.10, 0.15],
            "asset_growth_1y": [0.30, 0.20, 0.10],
            "accruals": [0.05, 0.00, -0.05],
            "debt_to_equity": [2.0, 1.0, 0.5],
            "future_return": [-1.0, 1.0, -1.0],
        },
        index=["A", "B", "C"],
    )

    scores = build_composite_vqmia(frame)

    assert "composite_vqmia_score" in scores.columns
    assert scores["composite_vqmia_block_count"].min() == 6
    assert scores.loc["C", "investment_block_score"] > scores.loc["A", "investment_block_score"]

