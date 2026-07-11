import json

import pandas as pd

from experiment_b_factor_portability import build_training_frame
from experiment_b_historical_backcast import (
    run_experiment_b_historical_backcast,
    select_historical_compatible_features,
)
from price_model import PricePanels


def _label_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E", "F"],
            "company_name": ["A Inc.", "B Inc.", "C Inc.", "D Inc.", "E Inc.", "F Inc."],
            "sector": ["Tech", "Tech", "Tech", "Energy", "Energy", "Energy"],
            "market_cap": [100.0, 110.0, 90.0, 80.0, 70.0, 60.0],
            "raw_ai_implied_irr": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
            "ai_minus_mechanical_irr": [0.01, 0.03, 0.01, 0.02, 0.04, 0.05],
            "ai_factor_residual": [0.00, -0.01, 0.02, -0.02, 0.03, 0.04],
            "label_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "exclude_from_clean_label": [False, False, False, False, False, False],
        }
    )


def _edgar_payload() -> dict[str, dict[str, float]]:
    return {
        ticker: {
            "fcf_to_ev": index / 10,
            "gross_profitability_assets": index / 20,
            "momentum_12_1": index / 30,
            "asset_growth_1y": (6 - index) / 20,
            "cash_earnings_gap": index / 40,
            "market_cap": 100.0 * index,
        }
        for index, ticker in enumerate(["A", "B", "C", "D", "E", "F"], start=1)
    }


def _matrix() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fcf_to_ev": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
            "gross_profitability_assets": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "momentum_12_1": [0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
            "asset_growth_1y": [0.30, 0.25, 0.20, 0.15, 0.10, 0.05],
            "cash_earnings_gap": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "market_cap": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            "feature_null_count": [0, 0, 0, 0, 0, 0],
        },
        index=["A", "B", "C", "D", "E", "F"],
    )


def test_select_historical_compatible_features_intersects_current_and_matrix_features():
    training_frame = build_training_frame(_label_panel(), _edgar_payload())
    features = select_historical_compatible_features(training_frame, {pd.Timestamp("2025-01-31"): _matrix()})

    assert "fcf_to_ev" in features
    assert "gross_profitability_assets" in features
    assert "log_market_cap" in features
    assert "book_to_market" not in features


def test_run_experiment_b_historical_backcast_writes_scores_and_audit_artifacts(tmp_path):
    label_path = tmp_path / "labels.parquet"
    edgar_path = tmp_path / "edgar.json"
    sector_path = tmp_path / "sectors.csv"
    changes_path = tmp_path / "sp500_changes.csv"
    approximate_membership_path = tmp_path / "approx_sp500_membership.parquet"
    approximate_membership_summary_path = tmp_path / "approx_sp500_membership_summary.json"
    scores_path = tmp_path / "scores.parquet"
    monthly_path = tmp_path / "monthly.parquet"
    holdings_path = tmp_path / "holdings.parquet"
    turnover_path = tmp_path / "turnover.parquet"
    exposures_path = tmp_path / "exposures.parquet"
    rank_ic_path = tmp_path / "rank_ic.parquet"
    rank_ic_summary_path = tmp_path / "rank_ic_summary.parquet"
    rank_ic_by_year_path = tmp_path / "rank_ic_by_year.parquet"
    rank_ic_by_sector_path = tmp_path / "rank_ic_by_sector.parquet"
    rank_ic_coverage_path = tmp_path / "rank_ic_coverage.parquet"
    summary_path = tmp_path / "summary.json"
    metadata_path = tmp_path / "metadata.json"
    _label_panel().to_parquet(label_path, index=False)
    edgar_path.write_text(json.dumps(_edgar_payload()), encoding="utf-8")
    pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E", "F"],
            "sector": ["Tech", "Tech", "Tech", "Energy", "Energy", "Energy"],
        }
    ).to_csv(sector_path, index=False)
    pd.DataFrame(columns=["effective_date", "added_ticker"]).to_csv(changes_path, index=False)
    pd.DataFrame(
        {
            "date": ["2025-01-31", "2025-01-31", "2025-02-28", "2025-02-28"],
            "ticker": ["A", "Z", "B", "Z"],
            "approximate_member": [True, True, True, True],
            "in_current_security_master": [True, False, True, False],
            "company_tickers_match": [True, True, True, False],
            "membership_basis": [
                "current_security_master",
                "selected_changes_removed_ticker_backfill",
                "current_security_master",
                "selected_changes_removed_ticker_backfill",
            ],
            "point_in_time_membership": [False, False, False, False],
            "membership_history_quality": [
                "approximate_public_selected_changes_not_full_constituent_history",
            ]
            * 4,
        }
    ).to_parquet(approximate_membership_path, index=False)
    approximate_membership_summary_path.write_text(
        json.dumps(
            {
                "status": "approximate_gap_analysis_not_point_in_time_membership",
                "row_count": 4,
                "ticker_count": 3,
                "date_count": 2,
                "current_security_master_ticker_count": 2,
                "missing_from_current_security_master_ticker_count": 1,
                "missing_with_sec_company_ticker_match_count": 1,
                "missing_without_sec_company_ticker_match_count": 0,
                "average_monthly_missing_from_security_master_rate": 0.5,
                "max_monthly_missing_from_security_master_rate": 0.5,
                "point_in_time_membership": False,
                "claim_limit": "Not CRSP/Compustat-quality membership.",
                "blockers": [
                    {"code": "removed_names_missing_security_master_rows"},
                    {"code": "selected_changes_not_full_point_in_time_membership"},
                ],
            }
        ),
        encoding="utf-8",
    )

    rebalance_dates = [pd.Timestamp("2025-01-31"), pd.Timestamp("2025-02-28")]
    matrices = {rebalance_dates[0]: _matrix(), rebalance_dates[1]: _matrix()}
    prices = pd.DataFrame(
        {
            "A": [10.0, 11.0],
            "B": [10.0, 10.5],
            "C": [10.0, 10.2],
            "D": [10.0, 9.8],
            "E": [10.0, 9.5],
            "F": [10.0, 9.0],
        },
        index=pd.to_datetime(["2025-02-03", "2025-03-03"]),
    )
    price_panels = PricePanels(raw_close=prices.copy(), adjusted_close=prices.copy())

    summary = run_experiment_b_historical_backcast(
        label_panel_path=label_path,
        edgar_features_path=edgar_path,
        sector_map_path=sector_path,
        sp500_changes_path=changes_path,
        approximate_membership_path=approximate_membership_path,
        approximate_membership_summary_path=approximate_membership_summary_path,
        scores_path=scores_path,
        monthly_returns_path=monthly_path,
        holdings_path=holdings_path,
        turnover_path=turnover_path,
        exposures_path=exposures_path,
        rank_ic_path=rank_ic_path,
        rank_ic_summary_path=rank_ic_summary_path,
        rank_ic_by_year_path=rank_ic_by_year_path,
        rank_ic_by_sector_path=rank_ic_by_sector_path,
        rank_ic_coverage_path=rank_ic_coverage_path,
        summary_path=summary_path,
        metadata_path=metadata_path,
        targets=("raw_ai_implied_irr",),
        cost_bps_levels=(0,),
        matrices=matrices,
        rebalance_dates=rebalance_dates,
        price_panels=price_panels,
    )

    assert summary["status"] == "historical_backcast_screen"
    assert scores_path.exists()
    assert monthly_path.exists()
    assert holdings_path.exists()
    assert rank_ic_path.exists()
    assert rank_ic_summary_path.exists()
    assert not pd.read_parquet(monthly_path).empty
    score_signals = set(pd.read_parquet(scores_path)["signal_name"])
    assert {"benchmark_composite_vqmia_score", "benchmark_fcf_to_ev"} <= score_signals
    monthly = pd.read_parquet(monthly_path)
    assert {"sector_neutral", "unconstrained"} <= set(monthly["portfolio_mode"])
    assert {"benchmark_composite_vqmia_score", "benchmark_fcf_to_ev"} <= set(monthly["signal_name"])
    assert set(summary["rank_ic_return_horizons"]) == {"return_1m", "return_3m", "return_6m", "return_12m"}
    assert {"sector_neutral", "unconstrained"} <= set(summary["portfolio_modes"])
    assert {"benchmark_composite_vqmia_score", "benchmark_fcf_to_ev"} <= set(summary["benchmark_signals"])
    assert summary["rank_ic_usable_month_count"] == 1
    assert summary["approximate_membership_gap"]["point_in_time_membership"] is False
    assert summary["approximate_membership_gap"]["missing_from_current_security_master_ticker_count"] == 1
    assert summary["approximate_membership_gap"]["backcast_rebalance_overlap"]["overlap_month_count"] == 2
    assert summary["approximate_membership_gap"]["backcast_rebalance_overlap"][
        "max_missing_from_current_security_master_count"
    ] == 1
    warning_codes = {warning["code"] for warning in summary["warnings"]}
    assert "current_label_projection" in warning_codes
    assert "removed_names_missing_from_backcast_universe" in warning_codes
