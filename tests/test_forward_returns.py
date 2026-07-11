import pandas as pd
import pytest

from forward_returns import (
    build_forward_returns,
    forward_return_for_horizon,
    summarize_forward_returns,
    write_forward_returns,
)
from price_model import PricePanels


def test_forward_return_uses_first_adjusted_close_after_label_and_horizon():
    series = pd.Series(
        [10.0, 12.0, 15.0],
        index=pd.to_datetime(["2026-03-22", "2026-04-22", "2026-04-23"]),
    )

    result, entry_date, exit_date, entry_price, exit_price = forward_return_for_horizon(
        series,
        pd.Timestamp("2026-03-21"),
        months=1,
    )

    assert result == pytest.approx(0.2)
    assert entry_date == pd.Timestamp("2026-03-22")
    assert exit_date == pd.Timestamp("2026-04-22")
    assert entry_price == pytest.approx(10.0)
    assert exit_price == pytest.approx(12.0)


def test_build_forward_returns_from_price_panel():
    label_panel = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "report_date": ["2026-03-21", "2026-03-21"],
        }
    )
    adjusted = pd.DataFrame(
        {
            "A": [10.0, 11.0],
            "B": [20.0, 18.0],
        },
        index=pd.to_datetime(["2026-03-23", "2026-04-22"]),
    )
    panels = PricePanels(raw_close=adjusted * 2.0, adjusted_close=adjusted)

    output = build_forward_returns(label_panel, panels, horizons=(1,))

    assert output.loc[output["ticker"] == "A", "return_1m"].iloc[0] == pytest.approx(0.1)
    assert output.loc[output["ticker"] == "B", "return_1m"].iloc[0] == pytest.approx(-0.1)


def test_forward_return_summary_flags_zero_coverage():
    frame = pd.DataFrame({"ticker": ["A"], "return_1m": [None]})

    summary = summarize_forward_returns(frame, horizons=(1,))

    assert summary["blockers"][0]["code"] == "no_usable_forward_returns"


def test_write_forward_returns_offline_skips_missing_price_cache(tmp_path):
    label_path = tmp_path / "label_panel.parquet"
    output_path = tmp_path / "forward_returns.parquet"
    summary_path = tmp_path / "summary.json"
    metadata_path = tmp_path / "metadata.json"
    pd.DataFrame({"ticker": ["A"], "report_date": ["2026-03-21"]}).to_parquet(label_path, index=False)

    summary = write_forward_returns(
        label_panel_path=label_path,
        output_path=output_path,
        summary_path=summary_path,
        metadata_path=metadata_path,
        price_cache_path=tmp_path / "missing_prices.parquet",
        offline=True,
        horizons=(1,),
    )

    assert output_path.exists()
    assert summary["price_window"]["offline"] is True
    assert {blocker["code"] for blocker in summary["blockers"]} == {
        "no_usable_forward_returns",
        "price_cache_missing_or_incomplete",
    }
