import json

import pandas as pd
import pytest

from quarterly_fundamentals import build_quarterly_fundamentals, extract_quarterly_company_fundamentals


def _duration_fact(
    value: float,
    *,
    start: str,
    end: str,
    filed: str = "2026-05-01",
    fy: int = 2026,
    fp: str = "Q1",
    form: str = "10-Q",
    accn: str = "0001",
) -> dict[str, object]:
    return {
        "val": value,
        "start": start,
        "end": end,
        "filed": filed,
        "fy": fy,
        "fp": fp,
        "form": form,
        "accn": accn,
    }


def _instant_fact(
    value: float,
    *,
    end: str,
    filed: str = "2026-05-01",
    fy: int = 2026,
    fp: str = "Q1",
    form: str = "10-Q",
    accn: str = "0001",
) -> dict[str, object]:
    return {
        "val": value,
        "end": end,
        "filed": filed,
        "fy": fy,
        "fp": fp,
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
                            _duration_fact(100.0, start="2025-01-01", end="2025-03-31", fy=2025, fp="Q1", accn="q1"),
                            _duration_fact(220.0, start="2025-01-01", end="2025-06-30", fy=2025, fp="Q2", accn="q2"),
                            _duration_fact(120.0, start="2025-04-01", end="2025-06-30", fy=2025, fp="Q2", accn="q2"),
                            _duration_fact(130.0, start="2025-07-01", end="2025-09-30", fy=2025, fp="Q3", accn="q3"),
                            _duration_fact(140.0, start="2026-01-01", end="2026-03-31", fy=2026, fp="Q1", accn="q4"),
                        ]
                    }
                },
                "GrossProfit": {
                    "units": {
                        "USD": [
                            _duration_fact(40.0, start="2025-01-01", end="2025-03-31", fy=2025, fp="Q1", accn="q1"),
                            _duration_fact(50.0, start="2025-04-01", end="2025-06-30", fy=2025, fp="Q2", accn="q2"),
                            _duration_fact(55.0, start="2025-07-01", end="2025-09-30", fy=2025, fp="Q3", accn="q3"),
                            _duration_fact(60.0, start="2026-01-01", end="2026-03-31", fy=2026, fp="Q1", accn="q4"),
                        ]
                    }
                },
                "NetCashProvidedByOperatingActivities": {
                    "units": {
                        "USD": [
                            _duration_fact(30.0, start="2025-01-01", end="2025-03-31", fy=2025, fp="Q1", accn="q1"),
                            _duration_fact(35.0, start="2025-04-01", end="2025-06-30", fy=2025, fp="Q2", accn="q2"),
                            _duration_fact(40.0, start="2025-07-01", end="2025-09-30", fy=2025, fp="Q3", accn="q3"),
                            _duration_fact(45.0, start="2026-01-01", end="2026-03-31", fy=2026, fp="Q1", accn="q4"),
                        ]
                    }
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {
                        "USD": [
                            _duration_fact(10.0, start="2025-01-01", end="2025-03-31", fy=2025, fp="Q1", accn="q1"),
                            _duration_fact(11.0, start="2025-04-01", end="2025-06-30", fy=2025, fp="Q2", accn="q2"),
                            _duration_fact(12.0, start="2025-07-01", end="2025-09-30", fy=2025, fp="Q3", accn="q3"),
                            _duration_fact(13.0, start="2026-01-01", end="2026-03-31", fy=2026, fp="Q1", accn="q4"),
                        ]
                    }
                },
                "Assets": {
                    "units": {
                        "USD": [
                            _instant_fact(500.0, end="2025-03-31", fy=2025, fp="Q1", accn="q1"),
                            _instant_fact(510.0, end="2025-06-30", fy=2025, fp="Q2", accn="q2"),
                        ]
                    }
                },
                "CashAndCashEquivalentsAtCarryingValue": {
                    "units": {"USD": [_instant_fact(20.0, end="2025-03-31", fy=2025, fp="Q1", accn="q1")]}
                },
                "LongTermDebt": {
                    "units": {"USD": [_instant_fact(80.0, end="2025-03-31", fy=2025, fp="Q1", accn="q1")]}
                },
                "ShortTermBorrowings": {
                    "units": {"USD": [_instant_fact(5.0, end="2025-03-31", fy=2025, fp="Q1", accn="q1")]}
                },
                "CommonStockSharesOutstanding": {
                    "units": {"shares": [_instant_fact(10.0, end="2025-03-31", fy=2025, fp="Q1", accn="q1")]}
                },
            }
        },
    }


def test_extract_quarterly_company_fundamentals_uses_quarter_length_duration_not_ytd():
    frame = extract_quarterly_company_fundamentals("TST", 1, _company_facts())
    q2 = frame[(frame["fiscal_year"] == 2025) & (frame["fiscal_period"] == "Q2")].iloc[0]
    q1 = frame[(frame["fiscal_year"] == 2025) & (frame["fiscal_period"] == "Q1")].iloc[0]

    assert q2["revenue"] == pytest.approx(120.0)
    assert q1["total_debt"] == pytest.approx(85.0)
    assert q1["free_cash_flow"] == pytest.approx(20.0)


def test_build_quarterly_fundamentals_adds_ttm_and_change_fields(tmp_path):
    security_master = pd.DataFrame({"ticker": ["TST"], "cik": [1]})
    security_path = tmp_path / "security_master.parquet"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    security_master.to_parquet(security_path, index=False)
    (cache_dir / "0000000001.json").write_text(json.dumps(_company_facts()), encoding="utf-8")

    frame = build_quarterly_fundamentals(security_master_path=security_path, companyfacts_dir=cache_dir)

    assert "revenue_ttm" in frame.columns
    assert "revenue_qoq_change" in frame.columns
    assert frame["free_cash_flow"].notna().sum() == 4
    assert frame["revenue_ttm"].dropna().iloc[-1] == pytest.approx(490.0)
