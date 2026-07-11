import pandas as pd
import pytest

import historical
from edgar import compute_derived_fields
from price_model import extract_price_panels, load_price_panels


def test_rebalance_price_features_use_raw_for_valuation_and_adjusted_for_momentum():
    raw = pd.Series(
        [50.0, 100.0],
        index=pd.to_datetime(["2025-02-28", "2025-03-31"]),
    )
    adjusted = pd.Series(
        [80.0, 88.0],
        index=pd.to_datetime(["2025-02-28", "2025-03-31"]),
    )

    features = historical.build_price_features_for_rebalance(
        raw,
        pd.Timestamp("2025-03-31"),
        adjusted_series=adjusted,
    )

    assert features["current_price"] == pytest.approx(100.0)
    assert features["raw_close_price"] == pytest.approx(100.0)
    assert features["adjusted_close_price"] == pytest.approx(88.0)
    assert features["price_return_1m"] == pytest.approx(0.10)


def test_derived_valuation_uses_raw_close_not_adjusted_close():
    record = {
        "operating_cash_flow": 120.0,
        "capex": 20.0,
        "current_price": 88.0,
        "raw_close_price": 100.0,
        "adjusted_close_price": 88.0,
        "price_return_1m": 0.10,
        "price_return_12m": 0.32,
        "shares_outstanding": 10.0,
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

    compute_derived_fields(record)

    assert record["market_cap"] == pytest.approx(1000.0)
    assert record["fcf_to_ev"] == pytest.approx(100.0 / 1035.0)


def test_price_panel_download_does_not_overwrite_cache_with_empty_download(tmp_path, monkeypatch):
    cache_path = tmp_path / "price_panels.parquet"
    cached = pd.concat(
        {
            "raw_close": pd.DataFrame({"A": [10.0]}, index=pd.to_datetime(["2025-01-02"])),
            "adjusted_close": pd.DataFrame({"A": [9.0]}, index=pd.to_datetime(["2025-01-02"])),
        },
        axis=1,
    )
    cached.to_parquet(cache_path)

    monkeypatch.setattr("price_model.yf.download", lambda *args, **kwargs: pd.DataFrame())

    panels = load_price_panels(["A"], start="2024-01-01", end="2026-01-31", cache_path=cache_path)

    stored = pd.read_parquet(cache_path)
    assert stored.shape == cached.shape
    assert panels.raw_close["A"].dropna().iloc[0] == pytest.approx(10.0)


def test_extract_price_panels_does_not_treat_adjusted_close_as_raw_close():
    history = pd.DataFrame({"Adj Close": [9.0, 10.0]}, index=pd.to_datetime(["2025-01-02", "2025-01-03"]))

    panels = extract_price_panels(history, ["A"])

    assert panels.raw_close["A"].dropna().empty
    assert panels.adjusted_close["A"].dropna().iloc[-1] == pytest.approx(10.0)
