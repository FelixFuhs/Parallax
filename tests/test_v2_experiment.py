import json

import pandas as pd
import pytest

from v2_experiment import add_benchmark_signals, build_portfolio_audit_artifacts, run_v2_experiment


def test_add_benchmark_signals_adds_composite_and_control_scores():
    label_panel = pd.DataFrame({"ticker": ["A", "B", "C"], "report_date": ["2026-03-21"] * 3})
    features = pd.DataFrame(
        {
            "fcf_to_ev": [0.1, 0.2, 0.3],
            "gross_profitability_assets": [0.1, 0.2, 0.3],
            "momentum_12_1": [0.1, 0.2, 0.3],
            "asset_growth_1y": [0.3, 0.2, 0.1],
            "accruals": [0.1, 0.0, -0.1],
            "debt_to_equity": [2.0, 1.0, 0.5],
        },
        index=["A", "B", "C"],
    )

    output = add_benchmark_signals(label_panel, features)

    assert "composite_vqmia_score" in output.columns
    assert "value_block_score" in output.columns
    assert output["composite_vqmia_score"].notna().all()


def test_v2_experiment_writes_blocked_status_without_forward_returns(tmp_path):
    label_panel = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "report_date": ["2026-03-21", "2026-03-21"],
            "raw_ai_implied_irr": [0.1, 0.2],
            "mechanical_dcf_implied_irr": [0.05, 0.06],
        }
    )
    label_path = tmp_path / "label_panel.parquet"
    label_panel.to_parquet(label_path, index=False)

    status = run_v2_experiment(label_panel_path=label_path, edgar_features_path=None, output_dir=tmp_path)

    assert status["status"] == "blocked"
    assert any(blocker["code"] == "missing_forward_returns" for blocker in status["blockers"])
    assert (tmp_path / "rank_ic.parquet").exists()
    assert (tmp_path / "rank_ic_by_year.parquet").exists()
    assert (tmp_path / "rank_ic_by_sector.parquet").exists()
    assert (tmp_path / "rank_ic_coverage.parquet").exists()
    assert (tmp_path / "signal_comparison.parquet").exists()
    assert (tmp_path / "holdings.parquet").exists()
    saved = json.loads((tmp_path / "v2_experiment_status.json").read_text(encoding="utf-8"))
    assert saved["status"] == "blocked"


def test_v2_experiment_writes_real_artifacts_with_forward_returns(tmp_path):
    label_panel = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E", "F"],
            "report_date": ["2026-03-21"] * 6,
            "raw_ai_implied_irr": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "mechanical_dcf_implied_irr": [0.1] * 6,
            "ai_minus_mechanical_irr": [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            "factor_compressible_ai_score": [0.55, 0.45, 0.35, 0.25, 0.15, 0.05],
            "ai_factor_residual": [0.05, 0.04, 0.03, 0.02, 0.01, 0.0],
            "market_cap": [100.0, 90.0, 80.0, 70.0, 60.0, 50.0],
            "sector": ["Tech", "Tech", "Tech", "Energy", "Energy", "Energy"],
        }
    )
    returns = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E", "F"],
            "date": ["2026-03-21"] * 6,
            "return_1m": [0.10, 0.08, 0.06, 0.04, 0.02, -0.01],
            "entry_price_1m": [10.0] * 6,
            "exit_price_1m": [11.0, 10.8, 10.6, 10.4, 10.2, 9.9],
        }
    )
    label_path = tmp_path / "label_panel.parquet"
    return_path = tmp_path / "forward_returns.parquet"
    edgar_path = tmp_path / "edgar.json"
    label_panel.to_parquet(label_path, index=False)
    returns.to_parquet(return_path, index=False)
    edgar_path.write_text(
        json.dumps(
            {
                ticker: {
                    "fcf_to_ev": index / 10,
                    "gross_profitability_assets": index / 10,
                    "momentum_12_1": index / 10,
                    "asset_growth_1y": (4 - index) / 10,
                    "accruals": -index / 100,
                    "debt_to_equity": 4 - index,
                    "market_cap": 100.0 * index,
                }
                for index, ticker in enumerate(["A", "B", "C", "D", "E", "F"], start=1)
            }
        ),
        encoding="utf-8",
    )

    status = run_v2_experiment(
        label_panel_path=label_path,
        forward_returns_path=return_path,
        edgar_features_path=edgar_path,
        output_dir=tmp_path,
    )

    assert not any(blocker["code"] == "missing_forward_returns" for blocker in status["blockers"])
    monthly = pd.read_parquet(tmp_path / "monthly_returns.parquet")
    holdings = pd.read_parquet(tmp_path / "holdings.parquet")
    rank_ic = pd.read_parquet(tmp_path / "rank_ic.parquet")
    rank_ic_by_year = pd.read_parquet(tmp_path / "rank_ic_by_year.parquet")
    rank_ic_by_sector = pd.read_parquet(tmp_path / "rank_ic_by_sector.parquet")
    rank_ic_coverage = pd.read_parquet(tmp_path / "rank_ic_coverage.parquet")
    signal_comparison = pd.read_parquet(tmp_path / "signal_comparison.parquet")
    assert not monthly.empty
    assert not holdings.empty
    assert not rank_ic.empty
    assert not rank_ic_by_year.empty
    assert not rank_ic_by_sector.empty
    assert not rank_ic_coverage.empty
    assert not signal_comparison.empty
    assert holdings["net_return"].notna().all()
    assert "composite_vqmia_score" in set(status["signals"])
    assert "composite_vqmia_score" in set(rank_ic["signal"])
    assert "rank_ic_by_sector" in status["artifacts"]
    assert "signal_comparison" in status["artifacts"]
    assert {"AI IRR", "AI residual", "Mechanical IRR", "Composite VQMIA", "FCF/EV"} <= set(
        signal_comparison["signal_label"]
    )
    assert {"global_mean_ic", "sector_neutral_mean_ic", "across_sector_mean_ic"} <= set(signal_comparison.columns)
    assert {"unconstrained", "sector_neutral"} <= set(monthly["portfolio_mode"])
    assert {"unconstrained", "sector_neutral"} <= set(holdings["portfolio_mode"])
    assert "sector_neutral" in status["portfolio_modes"]


def test_v2_experiment_blocks_when_forward_returns_are_all_null(tmp_path):
    label_panel = pd.DataFrame(
        {
            "ticker": ["A"],
            "report_date": ["2026-03-21"],
            "raw_ai_implied_irr": [0.3],
        }
    )
    returns = pd.DataFrame(
        {
            "ticker": ["A"],
            "date": ["2026-03-21"],
            "return_source": ["adjusted_close"],
            "return_1m": [None],
        }
    )
    label_path = tmp_path / "label_panel.parquet"
    return_path = tmp_path / "forward_returns.parquet"
    label_panel.to_parquet(label_path, index=False)
    returns.to_parquet(return_path, index=False)

    status = run_v2_experiment(
        label_panel_path=label_path,
        forward_returns_path=return_path,
        edgar_features_path=None,
        output_dir=tmp_path,
    )

    assert any(blocker["code"] == "insufficient_forward_return_coverage" for blocker in status["blockers"])
    assert any(blocker["code"] == "portfolio_backtest_not_run" for blocker in status["blockers"])
    assert status["portfolio_return_column"] is None
    holdings = pd.read_parquet(tmp_path / "holdings.parquet")
    assert list(holdings.columns[:3]) == ["date", "ticker", "sector"]
    assert "portfolio_mode" in holdings.columns


def test_sector_neutral_portfolio_weights_sectors_equally():
    diagnostic_input = pd.DataFrame(
        {
            "date": ["2026-03-21"] * 6,
            "ticker": ["A", "B", "C", "D", "E", "F"],
            "sector": ["Tech", "Tech", "Tech", "Energy", "Energy", "Energy"],
            "market_cap": [100, 90, 80, 70, 60, 50],
            "signal": [3.0, 2.0, 1.0, 6.0, 5.0, 4.0],
            "return_1m": [0.10, 0.05, 0.01, 0.20, 0.10, 0.02],
            "entry_price_1m": [10.0] * 6,
            "exit_price_1m": [11.0, 10.5, 10.1, 12.0, 11.0, 10.2],
        }
    )

    artifacts = build_portfolio_audit_artifacts(
        diagnostic_input,
        signal_columns=["signal"],
        return_column="return_1m",
        cost_bps_levels=(0,),
    )
    sector_neutral_q1 = artifacts["holdings"][
        (artifacts["holdings"]["portfolio_mode"] == "sector_neutral")
        & (artifacts["holdings"]["bucket"] == "Q1")
    ]

    assert set(sector_neutral_q1["ticker"]) == {"A", "D"}
    assert sector_neutral_q1.set_index("ticker")["weight"].to_dict() == pytest.approx({"A": 0.5, "D": 0.5})
    monthly_q1 = artifacts["monthly_returns"][
        (artifacts["monthly_returns"]["portfolio_mode"] == "sector_neutral")
        & (artifacts["monthly_returns"]["bucket"] == "Q1")
    ].iloc[0]
    assert monthly_q1["gross_return"] == pytest.approx(0.15)
