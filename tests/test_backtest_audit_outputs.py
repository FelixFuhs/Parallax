import pandas as pd
import pytest

from backtest import run_audited_signal_backtest, write_backtest_audit_artifacts
from price_model import PricePanels


def test_audited_backtest_uses_t_plus_one_adjusted_returns_and_costs(tmp_path):
    rebalance_dates = [pd.Timestamp("2026-01-31"), pd.Timestamp("2026-02-28")]
    matrices = {
        rebalance_dates[0]: pd.DataFrame(
            {
                "score": [3.0, 2.0, 1.0],
                "sector": ["Tech", "Tech", "Energy"],
                "market_cap": [100.0, 80.0, 60.0],
                "feature_null_count": [0, 0, 0],
            },
            index=["A", "B", "C"],
        )
    }
    adjusted = pd.DataFrame(
        {
            "A": [10.0, 11.0],
            "B": [10.0, 10.5],
            "C": [10.0, 9.0],
        },
        index=pd.to_datetime(["2026-02-02", "2026-03-02"]),
    )
    raw = adjusted * 2.0
    panels = PricePanels(raw_close=raw, adjusted_close=adjusted)

    artifacts = run_audited_signal_backtest(
        signal_name="test_signal",
        score_column="score",
        matrices=matrices,
        rebalance_dates=rebalance_dates,
        price_frame=panels,
        cost_bps_levels=(0, 10),
    )

    q1_10bps = artifacts["monthly_returns"][
        (artifacts["monthly_returns"]["bucket"] == "Q1")
        & (artifacts["monthly_returns"]["cost_bps_one_way"] == 10)
        & (artifacts["monthly_returns"]["portfolio_mode"] == "unconstrained")
    ].iloc[0]
    assert q1_10bps["gross_return"] == pytest.approx(0.10)
    assert q1_10bps["net_return"] == pytest.approx(0.10 - 0.001)
    assert artifacts["holdings"].loc[artifacts["holdings"]["ticker"] == "A", "raw_close_entry_price"].iloc[0] == 20.0
    assert "portfolio_mode" in artifacts["monthly_returns"].columns
    assert "weighting_method" in artifacts["holdings"].columns

    written = write_backtest_audit_artifacts(artifacts, tmp_path)
    assert {"holdings", "monthly_returns", "turnover", "exposures"} <= set(written)
    assert (tmp_path / "holdings.parquet").exists()


def test_audited_backtest_can_build_sector_neutral_buckets():
    rebalance_dates = [pd.Timestamp("2026-01-31"), pd.Timestamp("2026-02-28")]
    tickers = ["A", "B", "C", "D", "E", "F"]
    matrices = {
        rebalance_dates[0]: pd.DataFrame(
            {
                "score": [3.0, 2.0, 1.0, 6.0, 5.0, 4.0],
                "sector": ["Tech", "Tech", "Tech", "Energy", "Energy", "Energy"],
                "market_cap": [100.0, 90.0, 80.0, 70.0, 60.0, 50.0],
                "feature_null_count": [0, 0, 0, 0, 0, 0],
            },
            index=tickers,
        )
    }
    adjusted = pd.DataFrame(
        {ticker: [10.0, exit_price] for ticker, exit_price in zip(tickers, [11.0, 10.5, 10.1, 12.0, 11.0, 10.2], strict=True)},
        index=pd.to_datetime(["2026-02-02", "2026-03-02"]),
    )
    panels = PricePanels(raw_close=adjusted.copy(), adjusted_close=adjusted.copy())

    artifacts = run_audited_signal_backtest(
        signal_name="test_signal",
        score_column="score",
        matrices=matrices,
        rebalance_dates=rebalance_dates,
        price_frame=panels,
        cost_bps_levels=(0,),
    )
    sector_neutral_q1 = artifacts["holdings"][
        (artifacts["holdings"]["portfolio_mode"] == "sector_neutral")
        & (artifacts["holdings"]["bucket"] == "Q1")
    ]

    assert set(sector_neutral_q1["ticker"]) == {"A", "D"}
    assert sector_neutral_q1.set_index("ticker")["weight"].to_dict() == pytest.approx({"A": 0.5, "D": 0.5})
    assert {"sector_neutral", "unconstrained"} <= set(artifacts["monthly_returns"]["portfolio_mode"])
