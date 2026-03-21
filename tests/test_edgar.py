import pytest

from edgar import (
    FIELD_SPECS,
    compute_derived_fields,
    extract_company_features,
    extract_gross_profit,
    iter_field_facts,
    select_fact,
)


def _duration_fact(
    value: float,
    *,
    start: str,
    end: str,
    filed: str = "2026-02-01",
    fy: int = 2025,
    form: str = "10-K",
    accn: str = "0000000000-26-000001",
) -> dict[str, object]:
    return {
        "val": value,
        "start": start,
        "end": end,
        "fy": fy,
        "fp": "FY",
        "form": form,
        "filed": filed,
        "accn": accn,
    }


def _instant_fact(
    value: float,
    *,
    end: str,
    filed: str = "2026-02-01",
    fy: int = 2025,
    form: str = "10-K",
    accn: str = "0000000000-26-000001",
) -> dict[str, object]:
    return {
        "val": value,
        "end": end,
        "fy": fy,
        "fp": "FY",
        "form": form,
        "filed": filed,
        "accn": accn,
    }


def test_extract_company_features_computes_gross_profit_from_cost_of_revenue():
    company_facts = {
        "entityName": "Test Co",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [_duration_fact(100.0, start="2025-01-01", end="2025-12-31")],
                    }
                },
                "GrossProfit": {
                    "units": {
                        "USD": [
                            _duration_fact(24.0, start="2025-10-01", end="2025-12-31"),
                        ],
                    }
                },
                "CostOfRevenue": {
                    "units": {
                        "USD": [_duration_fact(60.0, start="2025-01-01", end="2025-12-31")],
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [_duration_fact(10.0, start="2025-01-01", end="2025-12-31")],
                    }
                },
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [_duration_fact(15.0, start="2025-01-01", end="2025-12-31")],
                    }
                },
                "Assets": {
                    "units": {
                        "USD": [
                            _instant_fact(200.0, end="2025-12-31"),
                            _instant_fact(
                                160.0,
                                end="2024-12-31",
                                filed="2025-02-01",
                                fy=2024,
                                accn="0000000000-25-000001",
                            ),
                        ],
                    }
                },
                "StockholdersEquity": {
                    "units": {"USD": [_instant_fact(80.0, end="2025-12-31")]}
                },
                "CashAndCashEquivalentsAtCarryingValue": {
                    "units": {"USD": [_instant_fact(20.0, end="2025-12-31")]}
                },
                "LongTermDebt": {
                    "units": {"USD": [_instant_fact(30.0, end="2025-12-31")]}
                },
                "CommonStockSharesOutstanding": {
                    "units": {"shares": [_instant_fact(5.0, end="2025-12-31")]}
                },
            }
        },
    }

    record = extract_company_features(
        "TST",
        {"cik_str": 1, "title": "Test Co"},
        company_facts,
        {
            "current_price": 10.0,
            "price_return_1m": None,
            "price_return_3m": None,
            "price_return_6m": None,
            "price_return_12m": None,
        },
    )

    assert record["gross_profit"] == pytest.approx(40.0)
    assert record["gross_margin"] == pytest.approx(0.4)
    assert record["gross_profitability_assets"] == pytest.approx(0.2)
    assert record["book_to_market"] == pytest.approx(1.6)
    assert record["asset_growth_1y"] == pytest.approx(0.25)


def test_extract_gross_profit_prefers_cost_of_revenue_first():
    company_facts = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [_duration_fact(100.0, start="2025-01-01", end="2025-12-31")],
                    }
                },
                "CostOfRevenue": {
                    "units": {
                        "USD": [_duration_fact(60.0, start="2025-01-01", end="2025-12-31")],
                    }
                },
                "CostOfGoodsAndServicesSold": {
                    "units": {
                        "USD": [_duration_fact(55.0, start="2025-01-01", end="2025-12-31")],
                    }
                },
            }
        }
    }

    revenue_facts = list(iter_field_facts(company_facts, "revenue", FIELD_SPECS["revenue"]))
    revenue_fact = select_fact(revenue_facts, anchor_end=None, anchor_accn=None)

    gross_profit, supporting_fact = extract_gross_profit(
        company_facts,
        direct_fact=None,
        direct_value=None,
        revenue_fact=revenue_fact,
        revenue_value=100.0,
        anchor_end=revenue_fact.end,
        anchor_accn=revenue_fact.accn,
    )

    assert gross_profit == pytest.approx(40.0)
    assert supporting_fact is not None
    assert supporting_fact.tag == "CostOfRevenue"


def test_compute_derived_fields_leaves_asset_growth_null_without_prior_assets():
    record = {
        "operating_cash_flow": None,
        "capex": None,
        "current_price": 10.0,
        "price_return_1m": None,
        "price_return_12m": None,
        "shares_outstanding": 5.0,
        "operating_income": None,
        "total_equity": 80.0,
        "total_debt": None,
        "cash": None,
        "net_income": None,
        "revenue": 100.0,
        "total_assets": 200.0,
        "gross_profit": 50.0,
        "current_assets": None,
        "current_liabilities": None,
    }

    compute_derived_fields(
        record,
        current_total_assets_for_growth=200.0,
        prior_total_assets=None,
    )

    assert record["gross_profitability_assets"] == pytest.approx(0.25)
    assert record["book_to_market"] == pytest.approx(1.6)
    assert record["asset_growth_1y"] is None


def test_compute_derived_fields_computes_consensus_features():
    record = {
        "operating_cash_flow": 120.0,
        "capex": -20.0,
        "current_price": 10.0,
        "price_return_1m": 0.10,
        "price_return_12m": 0.32,
        "shares_outstanding": 5.0,
        "operating_income": 30.0,
        "total_equity": 80.0,
        "total_debt": 40.0,
        "cash": 5.0,
        "net_income": 80.0,
        "revenue": 100.0,
        "total_assets": 200.0,
        "gross_profit": 50.0,
        "current_assets": 60.0,
        "current_liabilities": 30.0,
    }

    compute_derived_fields(
        record,
        current_total_assets_for_growth=200.0,
        prior_total_assets=160.0,
    )

    assert record["free_cash_flow"] == pytest.approx(100.0)
    assert record["fcf_to_ev"] == pytest.approx(100.0 / 85.0)
    assert record["cash_earnings_gap"] == pytest.approx(0.2)
    assert record["momentum_12_1"] == pytest.approx(0.2)
    assert record["gross_profitability_assets"] == pytest.approx(0.25)
    assert record["asset_growth_1y"] == pytest.approx(0.25)


def test_compute_derived_fields_nulls_fcf_to_ev_for_non_positive_ev():
    record = {
        "operating_cash_flow": 120.0,
        "capex": 20.0,
        "current_price": 4.0,
        "price_return_1m": -1.0,
        "price_return_12m": 0.15,
        "shares_outstanding": 5.0,
        "operating_income": None,
        "total_equity": 80.0,
        "total_debt": 5.0,
        "cash": 30.0,
        "net_income": 80.0,
        "revenue": 100.0,
        "total_assets": 200.0,
        "gross_profit": 50.0,
        "current_assets": None,
        "current_liabilities": None,
    }

    compute_derived_fields(record)

    assert record["fcf_to_ev"] is None
    assert record["momentum_12_1"] is None
