import json

import pandas as pd
import pytest

from label_panel import (
    add_factor_residuals,
    add_repeated_label_uncertainty,
    build_label_panel,
    dcf_sanity_flags,
    mechanical_dcf_implied_irr,
    summarize_label_panel,
)
from price_model import PricePanels


def _report(ticker: str, per_share: float = 15.0) -> dict[str, object]:
    return {
        "_meta": {
            "generated_at": "2026-03-21T00:00:00Z",
            "model": "test-model",
            "tier": "cheap",
            "quality_flags": [],
        },
        "ticker": ticker,
        "company_name": f"{ticker} Inc.",
        "forecast_years": 5,
        "forecast": {
            "revenue_growth": {"2027": 0.031, "2028": 0.029},
            "ebit_margin": {"2027": 0.123, "2028": 0.127},
        },
        "assumptions": {
            "wacc": 0.09,
            "terminal_growth": 0.025,
            "diluted_shares": 10,
        },
        "comps": [{"company_name": "Peer", "ticker": "P", "ev_ntm_ebitda": 10}],
        "_valuation": {
            "scenarios": {
                "bear": {
                    "current_price": 10.0,
                    "per_share_value": 12.0,
                    "upside_downside_pct": 0.2,
                    "enterprise_value": 100.0,
                    "terminal_value": 80.0,
                    "terminal_value_present_value": 50.0,
                    "implied_market_cap": 120.0,
                    "net_debt": 0.0,
                    "investments_adjustment": 0.0,
                    "projected_years": [{"fcff": 10.0}, {"fcff": 10.0}, {"fcff": 10.0}, {"fcff": 10.0}, {"fcff": 10.0}],
                },
                "base": {
                    "current_price": 10.0,
                    "per_share_value": per_share,
                    "upside_downside_pct": per_share / 10.0 - 1.0,
                    "enterprise_value": 120.0,
                    "terminal_value": 120.0,
                    "terminal_value_present_value": 60.0,
                    "implied_market_cap": per_share * 10.0,
                    "net_debt": 0.0,
                    "investments_adjustment": 0.0,
                    "projected_years": [{"fcff": 10.0}, {"fcff": 10.0}, {"fcff": 10.0}, {"fcff": 10.0}, {"fcff": 10.0}],
                },
                "bull": {
                    "current_price": 10.0,
                    "per_share_value": 18.0,
                    "upside_downside_pct": 0.8,
                    "enterprise_value": 150.0,
                    "terminal_value": 180.0,
                    "terminal_value_present_value": 75.0,
                    "implied_market_cap": 180.0,
                    "net_debt": 0.0,
                    "investments_adjustment": 0.0,
                    "projected_years": [{"fcff": 10.0}, {"fcff": 10.0}, {"fcff": 10.0}, {"fcff": 10.0}, {"fcff": 10.0}],
                },
            }
        },
    }


def _edgar_record() -> dict[str, float | str | None]:
    return {
        "company_name": "Test Inc.",
        "revenue": 1000.0,
        "operating_income": 120.0,
        "shares_outstanding": 10.0,
        "current_price": 10.0,
        "raw_close_price": 10.0,
        "da": 20.0,
        "capex": 30.0,
        "cash": 5.0,
        "total_debt": 40.0,
        "market_cap": 100.0,
        "fcf_to_ev": 0.1,
        "gross_profitability_assets": 0.3,
        "asset_growth_1y": 0.05,
        "cash_earnings_gap": 0.02,
        "momentum_12_1": 0.12,
    }


def test_build_label_panel_extracts_ai_and_mechanical_irr(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "TST_2026-03-21_cheap.json").write_text(json.dumps(_report("TST")), encoding="utf-8")

    panel = build_label_panel(
        reports_dir=reports_dir,
        edgar_payload={"TST": _edgar_record()},
        sector_map={"TST": {"sector": "Industrials", "sub_industry": "Machinery", "sector_source": "test_map"}},
    )
    row = panel.iloc[0]

    assert row["raw_ai_upside"] == pytest.approx(0.5)
    assert pd.notna(row["raw_ai_implied_irr"])
    assert row["raw_ai_annualized_value_gap"] == pytest.approx((15.0 / 10.0) ** (1 / 5) - 1.0)
    assert pd.notna(row["mechanical_dcf_implied_irr"])
    assert row["ai_minus_mechanical_irr"] == pytest.approx(
        row["raw_ai_implied_irr"] - row["mechanical_dcf_implied_irr"]
    )
    assert row["sector"] == "Industrials"
    assert row["sub_industry"] == "Machinery"
    assert row["sector_source"] == "test_map"
    assert bool(row["exclude_from_clean_label"]) is False


def test_dcf_sanity_flags_scenario_order_failure():
    payload = _report("TST")
    payload["_valuation"]["scenarios"]["bear"]["per_share_value"] = 20.0

    assert "scenario_order_fail" in dcf_sanity_flags(payload)


def test_mechanical_dcf_reports_missing_inputs_honestly():
    irr, flags, details = mechanical_dcf_implied_irr("BAD", {"revenue": None}, report_current_price=None)

    assert irr is None
    assert "missing_mechanical_revenue" in flags
    assert "revenue" in details["missing_critical_inputs"]


def test_mechanical_dcf_prefers_legacy_edgar_price_before_report_price_when_raw_price_missing():
    record = _edgar_record()
    record.pop("raw_close_price")

    irr, flags, details = mechanical_dcf_implied_irr("TST", record, report_current_price=12.0)

    assert irr is not None
    assert "raw_price_unavailable" in flags
    assert details["mechanical_price_source"] == "legacy_edgar_current_price"


def test_mechanical_dcf_refuses_report_price_as_control_input():
    record = _edgar_record()
    record.pop("raw_close_price")
    record.pop("current_price")

    irr, flags, details = mechanical_dcf_implied_irr("TST", record, report_current_price=12.0)

    assert irr is None
    assert {"raw_price_unavailable", "report_price_ignored_for_mechanical_dcf", "missing_mechanical_price"} <= set(flags)
    assert details["missing_critical_inputs"] == ["price"]


def test_build_label_panel_uses_cached_raw_price_for_mechanical_dcf(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "TST_2026-03-21_cheap.json").write_text(json.dumps(_report("TST")), encoding="utf-8")
    edgar_record = _edgar_record()
    edgar_record.pop("raw_close_price")
    edgar_record.pop("current_price")
    price_panels = PricePanels(
        raw_close=pd.DataFrame({"TST": [9.5]}, index=pd.to_datetime(["2026-03-20"])),
        adjusted_close=pd.DataFrame({"TST": [9.0]}, index=pd.to_datetime(["2026-03-20"])),
    )

    panel = build_label_panel(
        reports_dir=reports_dir,
        edgar_payload={"TST": edgar_record},
        price_panels=price_panels,
    )
    row = panel.iloc[0]

    assert row["raw_close_price"] == pytest.approx(9.5)
    assert row["adjusted_close_price"] == pytest.approx(9.0)
    assert row["market_cap"] == pytest.approx(95.0)
    assert row["mechanical_price_source"] == "raw_close_price"
    assert "mechanical_price_from_ai_report" not in row["quality_flags"]


def test_factor_residualization_adds_fitted_and_residual_scores():
    panel = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "raw_ai_implied_irr": [0.10, 0.20, 0.30, 0.40],
                "label_weight": [1.0, 1.0, 1.0, 1.0],
            }
    )
    features = pd.DataFrame(
        {
            "fcf_to_ev": [1.0, 2.0, 3.0, 4.0],
            "gross_profitability_assets": [0.1, 0.2, 0.3, 0.4],
        },
        index=["A", "B", "C", "D"],
    )

    output = add_factor_residuals(panel, features)

    assert output["factor_compressible_ai_score"].notna().all()
    assert output["ai_factor_residual"].abs().max() < 0.002


def test_repeated_label_uncertainty_adds_dispersion_and_disagreement_metrics():
    panel = pd.DataFrame(
        {
            "ticker": ["A", "A", "A", "B"],
            "raw_ai_implied_irr": [0.10, 0.30, 0.20, 0.40],
            "model_id": ["m1", "m2", "m1", "m1"],
            "tier": ["cheap", "full", "cheap", "cheap"],
            "prompt_id": ["p1", "p2", "p1", "p1"],
            "exclude_from_clean_label": [False, False, True, False],
            "label_observation_count": [1, 1, 1, 1],
            "parse_failure_rate": [0.0, 0.0, 0.0, 0.0],
            "label_weight": [1.0, 1.0, 0.0, 1.0],
        }
    )

    output = add_repeated_label_uncertainty(panel)
    a_rows = output[output["ticker"] == "A"]
    b_row = output[output["ticker"] == "B"].iloc[0]

    assert a_rows["label_observation_count"].iloc[0] == 3
    assert a_rows["ai_irr_iqr"].iloc[0] == pytest.approx(0.10)
    assert a_rows["model_disagreement"].iloc[0] == pytest.approx(0.15)
    assert a_rows["tier_disagreement"].iloc[0] == pytest.approx(0.15)
    assert a_rows["prompt_disagreement"].iloc[0] == pytest.approx(0.15)
    assert a_rows["rank_std"].notna().all()
    assert b_row["label_observation_count"] == 1
    assert pd.isna(b_row["ai_label_uncertainty"])
    assert b_row["uncertainty_adjusted_label_weight"] == pytest.approx(1.0)


def test_label_summary_reports_failure_rates_by_tier_and_market_cap_bucket():
    panel = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "tier": ["cheap", "cheap", "full"],
            "sector": ["Technology", None, "Health Care"],
            "sector_source": ["test", None, "test"],
            "market_cap": [100.0, 200.0, 300.0],
            "exclude_from_clean_label": [False, True, False],
            "label_weight": [1.0, 0.0, 0.5],
            "quality_flags": [[], ["missing_ai_irr"], ["default_terminal_growth"]],
            "raw_ai_implied_irr": [0.1, None, 0.2],
            "mechanical_dcf_implied_irr": [0.05, 0.04, None],
            "ai_minus_mechanical_irr": [0.05, None, None],
            "ai_factor_residual": [0.01, None, 0.02],
        }
    )

    summary = summarize_label_panel(panel)

    assert {row["tier"] for row in summary["failure_rates_by_model_tier"]} == {"cheap", "full"}
    cheap = next(row for row in summary["failure_rates_by_model_tier"] if row["tier"] == "cheap")
    assert cheap["exclude_rate"] == pytest.approx(0.5)
    assert summary["sector_coverage"] == pytest.approx(2 / 3)
    assert summary["sector_source_counts"] == {"test": 2, "missing": 1}
    assert {row["market_cap_bucket"] for row in summary["failure_rates_by_market_cap_bucket"]} == {
        "small",
        "mid",
        "large",
    }
