from datetime import date

import pandas as pd
import pytest

from historical import build_point_in_time_feature_matrix, build_ticker_history, latest_snapshot_before


def _duration_fact(
    value: float,
    *,
    start: str,
    end: str,
    filed: str,
    fy: int,
    accn: str,
    form: str = "10-K",
) -> dict[str, object]:
    return {
        "val": value,
        "start": start,
        "end": end,
        "filed": filed,
        "fy": fy,
        "fp": "FY",
        "form": form,
        "accn": accn,
    }


def _instant_fact(
    value: float,
    *,
    end: str,
    filed: str,
    fy: int,
    accn: str,
    form: str = "10-K",
) -> dict[str, object]:
    return {
        "val": value,
        "end": end,
        "filed": filed,
        "fy": fy,
        "fp": "FY",
        "form": form,
        "accn": accn,
    }


def _company_facts() -> dict[str, object]:
    return {
        "entityName": "Test Co",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                100.0,
                                start="2024-01-01",
                                end="2024-12-31",
                                filed="2025-02-01",
                                fy=2024,
                                accn="0001",
                            ),
                            _duration_fact(
                                110.0,
                                start="2025-01-01",
                                end="2025-12-31",
                                filed="2026-02-01",
                                fy=2025,
                                accn="0002",
                            ),
                        ]
                    }
                },
                "GrossProfit": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                40.0,
                                start="2024-01-01",
                                end="2024-12-31",
                                filed="2025-02-01",
                                fy=2024,
                                accn="0001",
                            ),
                            _duration_fact(
                                50.0,
                                start="2025-01-01",
                                end="2025-12-31",
                                filed="2026-02-01",
                                fy=2025,
                                accn="0002",
                            ),
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                64.0,
                                start="2024-01-01",
                                end="2024-12-31",
                                filed="2025-02-01",
                                fy=2024,
                                accn="0001",
                            ),
                            _duration_fact(
                                80.0,
                                start="2025-01-01",
                                end="2025-12-31",
                                filed="2026-02-01",
                                fy=2025,
                                accn="0002",
                            ),
                        ]
                    }
                },
                "Assets": {
                    "units": {
                        "USD": [
                            _instant_fact(160.0, end="2024-12-31", filed="2025-02-01", fy=2024, accn="0001"),
                            _instant_fact(200.0, end="2025-12-31", filed="2026-02-01", fy=2025, accn="0002"),
                        ]
                    }
                },
                "CashAndCashEquivalentsAtCarryingValue": {
                    "units": {
                        "USD": [
                            _instant_fact(4.0, end="2024-12-31", filed="2025-02-01", fy=2024, accn="0001"),
                            _instant_fact(5.0, end="2025-12-31", filed="2026-02-01", fy=2025, accn="0002"),
                        ]
                    }
                },
                "NetCashProvidedByOperatingActivities": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                96.0,
                                start="2024-01-01",
                                end="2024-12-31",
                                filed="2025-02-01",
                                fy=2024,
                                accn="0001",
                            ),
                            _duration_fact(
                                120.0,
                                start="2025-01-01",
                                end="2025-12-31",
                                filed="2026-02-01",
                                fy=2025,
                                accn="0002",
                            ),
                        ]
                    }
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {
                        "USD": [
                            _duration_fact(
                                16.0,
                                start="2024-01-01",
                                end="2024-12-31",
                                filed="2025-02-01",
                                fy=2024,
                                accn="0001",
                            ),
                            _duration_fact(
                                20.0,
                                start="2025-01-01",
                                end="2025-12-31",
                                filed="2026-02-01",
                                fy=2025,
                                accn="0002",
                            ),
                        ]
                    }
                },
                "LongTermDebt": {
                    "units": {
                        "USD": [
                            _instant_fact(30.0, end="2024-12-31", filed="2025-02-01", fy=2024, accn="0001"),
                            _instant_fact(40.0, end="2025-12-31", filed="2026-02-01", fy=2025, accn="0002"),
                        ]
                    }
                },
                "CommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            _instant_fact(5.0, end="2024-12-31", filed="2025-02-01", fy=2024, accn="0001"),
                            _instant_fact(5.0, end="2025-12-31", filed="2026-02-01", fy=2025, accn="0002"),
                        ]
                    }
                },
            }
        },
    }


def test_build_ticker_history_orders_filings_by_filed_date():
    history = build_ticker_history("TST", {"cik_str": 1, "title": "Test Co"}, _company_facts())

    assert len(history.filings) == 2
    assert [snapshot.filed for snapshot in history.filings] == [date(2025, 2, 1), date(2026, 2, 1)]
    assert history.filings[-1].raw_fields["gross_profit"] == pytest.approx(50.0)
    assert history.filings[-1].raw_fields["total_debt"] == pytest.approx(40.0)


def test_point_in_time_feature_matrix_uses_latest_accepted_filing_and_prior_assets():
    history = build_ticker_history("TST", {"cik_str": 1, "title": "Test Co"}, _company_facts())
    snapshot_index, snapshot = latest_snapshot_before(history, pd.Timestamp("2026-03-31"))

    assert snapshot_index == 1
    assert snapshot is not None
    assert snapshot.accn == "0002"

    price_frame = pd.DataFrame(
        {
            "TST": [8.0, 9.0, 10.0, 11.0, 12.0],
        },
        index=pd.to_datetime(
            [
                "2025-02-28",
                "2025-03-31",
                "2026-02-27",
                "2026-03-31",
                "2026-04-30",
            ]
        ),
    )

    matrix = build_point_in_time_feature_matrix({"TST": history}, price_frame, pd.Timestamp("2026-03-31"))
    row = matrix.loc["TST"]

    assert row["filing_date"] == pd.Timestamp("2026-02-01")
    assert row["fcf_to_ev"] == pytest.approx(100.0 / 90.0)
    assert row["gross_profitability_assets"] == pytest.approx(0.25)
    assert row["asset_growth_1y"] == pytest.approx(0.25)
    assert row["cash_earnings_gap"] == pytest.approx(0.2)
    assert row["momentum_12_1"] == pytest.approx(((1.0 + (11.0 / 9.0 - 1.0)) / (1.0 + (11.0 / 10.0 - 1.0))) - 1.0)
    assert row["feature_null_count"] == 0
