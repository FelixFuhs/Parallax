from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dcf import DCFError, run_dcf
from edgar import normalize_ticker
from experiment_registry import build_experiment_metadata, repo_relative, write_experiment_metadata
from parser import Assumptions, ForecastDrivers, HistoricalFinancials, ValuationInput
from price_model import PricePanels, load_cached_price_panels, price_on_or_before
from sector_map import DEFAULT_OUTPUT as DEFAULT_SECTOR_MAP
from sector_map import load_sector_map

ROOT = Path(__file__).resolve().parent
DEFAULT_REPORTS_DIR = ROOT / "reports"
DEFAULT_EDGAR_PATH = ROOT / "data" / "edgar_features_full.json"
DEFAULT_OUTPUT_PATH = ROOT / "results" / "label_panel.parquet"
DEFAULT_SUMMARY_PATH = ROOT / "results" / "label_panel_summary.json"
DEFAULT_METADATA_PATH = ROOT / "results" / "label_panel_experiment_metadata.json"
DEFAULT_PRICE_PANEL_CACHE = ROOT / "data" / "forward_price_panels.parquet"

EXCLUDE_FLAGS = {
    "missing_price",
    "missing_ai_irr",
    "missing_wacc",
    "missing_scenario",
    "scenario_order_fail",
    "internal_inconsistency",
    "nonpositive_ai_value",
    "wacc_terminal_spread_too_small",
    "share_count_outlier",
    "ev_negative",
}
DOWNWEIGHT_FLAGS = {
    "default_terminal_growth",
    "suspiciously_round_forecast",
    "terminal_value_dominates",
    "forecast_discontinuity",
    "stale_price",
}
WARNING_FLAGS = {
    "missing_comps",
    "missing_da_mechanical",
    "missing_capex_mechanical",
    "missing_cash_mechanical",
    "missing_debt_mechanical",
    "missing_mechanical_inputs",
    "missing_mechanical_revenue",
    "missing_mechanical_operating_income",
    "missing_mechanical_shares_outstanding",
    "missing_mechanical_price",
    "mechanical_price_asof_unverified",
    "mechanical_price_from_ai_report",
    "report_price_ignored_for_mechanical_dcf",
    "mechanical_dcf_error",
    "margin_reversal",
    "raw_price_unavailable",
}
QUALITY_FLAG_POLICY = {
    **{flag: "exclude" for flag in EXCLUDE_FLAGS},
    **{flag: "downweight" for flag in DOWNWEIGHT_FLAGS},
    **{flag: "warning" for flag in WARNING_FLAGS},
}
FACTOR_FEATURES = (
    "fcf_to_ev",
    "book_to_market",
    "fcf_yield",
    "gross_profitability_assets",
    "roic",
    "roe",
    "operating_margin",
    "momentum_12_1",
    "price_return_6m",
    "asset_growth_1y",
    "cash_earnings_gap",
    "accruals",
    "debt_to_equity",
    "current_ratio",
    "market_cap",
)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _annualized_return(current_price: Any, terminal_price: Any, years: int | float | None) -> float | None:
    current = _coerce_float(current_price)
    terminal = _coerce_float(terminal_price)
    horizon = _coerce_float(years)
    if current is None or terminal is None or horizon is None:
        return None
    if current <= 0.0 or terminal <= 0.0 or horizon <= 0.0:
        return None
    return float((terminal / current) ** (1.0 / horizon) - 1.0)


def _solve_discount_rate(cash_flows: Sequence[float], target_value: float) -> float | None:
    if target_value <= 0.0 or not cash_flows:
        return None
    if not all(math.isfinite(value) for value in cash_flows):
        return None

    def npv(rate: float) -> float:
        return sum(flow / ((1.0 + rate) ** index) for index, flow in enumerate(cash_flows, start=1))

    grid = [-0.95, -0.75, -0.5, -0.25, -0.1, 0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    previous_rate = grid[0]
    previous_value = npv(previous_rate) - target_value
    if abs(previous_value) < 1e-9:
        return previous_rate

    bracket: tuple[float, float] | None = None
    for rate in grid[1:]:
        value = npv(rate) - target_value
        if abs(value) < 1e-9:
            return rate
        if previous_value * value < 0.0:
            bracket = (previous_rate, rate)
            break
        previous_rate = rate
        previous_value = value
    if bracket is None:
        return None

    low, high = bracket
    low_value = npv(low) - target_value
    for _ in range(100):
        mid = (low + high) / 2.0
        mid_value = npv(mid) - target_value
        if abs(mid_value) < 1e-10:
            return float(mid)
        if low_value * mid_value <= 0.0:
            high = mid
        else:
            low = mid
            low_value = mid_value
    return float((low + high) / 2.0)


def _cash_flow_implied_irr(
    scenario: Mapping[str, Any] | None,
    assumptions: Mapping[str, Any],
) -> tuple[float | None, list[str], dict[str, Any]]:
    if scenario is None:
        return None, ["missing_scenario"], {}
    projected_years = scenario.get("projected_years")
    if not isinstance(projected_years, list) or not projected_years:
        return None, ["missing_ai_irr"], {}

    fcff: list[float] = []
    for year in projected_years:
        if not isinstance(year, Mapping):
            return None, ["missing_ai_irr"], {}
        value = _coerce_float(year.get("fcff"))
        if value is None:
            return None, ["missing_ai_irr"], {}
        fcff.append(value)

    terminal_value = _coerce_float(scenario.get("terminal_value"))
    current_price = _coerce_float(scenario.get("current_price"))
    diluted_shares = _coerce_float(assumptions.get("diluted_shares"))
    net_debt = _coerce_float(scenario.get("net_debt"))
    investments_adjustment = _coerce_float(scenario.get("investments_adjustment"))
    minority_interest = _coerce_float(assumptions.get("minority_interest")) or 0.0
    preferred_equity = _coerce_float(assumptions.get("preferred_equity")) or 0.0
    if (
        terminal_value is None
        or current_price is None
        or current_price <= 0.0
        or diluted_shares is None
        or diluted_shares <= 0.0
        or net_debt is None
    ):
        return None, ["missing_ai_irr"], {}
    if investments_adjustment is None:
        investments_adjustment = 0.0

    current_equity_value = current_price * diluted_shares
    current_enterprise_value = (
        current_equity_value
        + net_debt
        + minority_interest
        + preferred_equity
        - investments_adjustment
    )
    cash_flows = [*fcff[:-1], fcff[-1] + terminal_value]
    irr = _solve_discount_rate(cash_flows, current_enterprise_value)
    if irr is None:
        return None, ["missing_ai_irr"], {
            "ai_irr_target_enterprise_value": current_enterprise_value,
        }
    return irr, [], {
        "ai_irr_target_enterprise_value": current_enterprise_value,
        "ai_irr_method": "cash_flow_discount_rate_to_current_enterprise_value",
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object.")
    return payload


def load_edgar_payload(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    return {normalize_ticker(str(ticker)): dict(record) for ticker, record in payload.items() if isinstance(record, Mapping)}


def _report_identity(path: Path) -> tuple[str | None, str | None]:
    parts = path.stem.split("_")
    ticker = normalize_ticker(parts[0]) if parts else None
    report_date = parts[1] if len(parts) >= 2 else None
    return ticker, report_date


def _scenario(payload: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    value = payload.get("_valuation", {}).get("scenarios", {}).get(name)
    return value if isinstance(value, Mapping) else None


def _forecast_years(payload: Mapping[str, Any], base_case: Mapping[str, Any] | None) -> int:
    raw = _coerce_float(payload.get("forecast_years"))
    if raw is not None and raw > 0:
        return int(raw)
    if base_case is not None:
        projected_years = base_case.get("projected_years")
        if isinstance(projected_years, list) and projected_years:
            return len(projected_years)
    return 5


def _scenario_annualized_value_gap(payload: Mapping[str, Any], scenario_name: str) -> float | None:
    case = _scenario(payload, scenario_name)
    if case is None:
        return None
    return _annualized_return(
        case.get("current_price"),
        case.get("per_share_value"),
        _forecast_years(payload, case),
    )


def _quality_flags_from_meta(payload: Mapping[str, Any]) -> list[str]:
    raw_flags = payload.get("_meta", {}).get("quality_flags", [])
    if isinstance(raw_flags, list):
        return [str(flag) for flag in raw_flags]
    return []


def _scenario_order_flags(payload: Mapping[str, Any]) -> list[str]:
    flags: list[str] = []
    cases = {name: _scenario(payload, name) for name in ("bear", "base", "bull")}
    if any(case is None for case in cases.values()):
        return ["missing_scenario"]

    values = [_coerce_float(cases[name].get("per_share_value")) for name in ("bear", "base", "bull")]
    assumptions = payload.get("assumptions") if isinstance(payload.get("assumptions"), Mapping) else {}
    irrs = [_cash_flow_implied_irr(cases[name], assumptions)[0] for name in ("bear", "base", "bull")]
    if all(value is not None for value in values) and not (values[0] <= values[1] <= values[2]):
        flags.append("scenario_order_fail")
    if all(value is not None for value in irrs) and not (irrs[0] <= irrs[1] <= irrs[2]):
        flags.append("scenario_order_fail")
    return flags


def _forecast_roundness_flag(payload: Mapping[str, Any]) -> bool:
    forecast = payload.get("forecast")
    if not isinstance(forecast, Mapping):
        return False
    values: list[float] = []
    for field_name in ("revenue_growth", "ebit_margin", "da_pct_sales", "capex_pct_sales", "nwc_pct_sales"):
        series = forecast.get(field_name)
        if not isinstance(series, Mapping):
            continue
        for value in series.values():
            number = _coerce_float(value)
            if number is not None:
                values.append(number)
    if len(values) < 6:
        return False
    round_count = sum(abs(value * 100.0 - round(value * 100.0)) < 1e-9 for value in values)
    return (round_count / len(values)) >= 0.85


def _forecast_discontinuity_flag(payload: Mapping[str, Any]) -> bool:
    forecast = payload.get("forecast")
    if not isinstance(forecast, Mapping):
        return False
    for field_name, threshold in (("revenue_growth", 0.25), ("ebit_margin", 0.15)):
        series = forecast.get(field_name)
        if not isinstance(series, Mapping) or len(series) < 2:
            continue
        ordered = [_coerce_float(series[key]) for key in sorted(series)]
        clean = [value for value in ordered if value is not None]
        if any(abs(next_value - value) > threshold for value, next_value in zip(clean, clean[1:])):
            return True
    return False


def dcf_sanity_flags(payload: Mapping[str, Any]) -> list[str]:
    flags = _quality_flags_from_meta(payload)
    base_case = _scenario(payload, "base")
    assumptions = payload.get("assumptions") if isinstance(payload.get("assumptions"), Mapping) else {}

    if base_case is None:
        flags.append("missing_scenario")
    else:
        current_price = _coerce_float(base_case.get("current_price"))
        per_share = _coerce_float(base_case.get("per_share_value"))
        upside = _coerce_float(base_case.get("upside_downside_pct"))
        if current_price is None or current_price <= 0.0:
            flags.append("missing_price")
        if per_share is None or per_share <= 0.0:
            flags.append("nonpositive_ai_value")
        if current_price not in (None, 0.0) and per_share is not None and upside is not None:
            expected_upside = per_share / current_price - 1.0
            if abs(expected_upside - upside) > 0.01:
                flags.append("internal_inconsistency")

        enterprise_value = _coerce_float(base_case.get("enterprise_value"))
        if enterprise_value is not None and enterprise_value <= 0.0:
            flags.append("ev_negative")
        terminal_pv = _coerce_float(base_case.get("terminal_value_present_value"))
        if terminal_pv is not None and enterprise_value not in (None, 0.0):
            if abs(terminal_pv / enterprise_value) > 0.85:
                flags.append("terminal_value_dominates")

        implied_market_cap = _coerce_float(base_case.get("implied_market_cap"))
        diluted_shares = _coerce_float(assumptions.get("diluted_shares")) if isinstance(assumptions, Mapping) else None
        if diluted_shares is None or diluted_shares <= 0.0:
            flags.append("share_count_outlier")
        elif current_price is not None and implied_market_cap is not None:
            implied_shares = implied_market_cap / per_share if per_share not in (None, 0.0) else None
            if implied_shares is not None and abs(implied_shares / diluted_shares - 1.0) > 0.05:
                flags.append("share_count_outlier")

    wacc = _coerce_float(assumptions.get("wacc")) if isinstance(assumptions, Mapping) else None
    terminal_growth = _coerce_float(assumptions.get("terminal_growth")) if isinstance(assumptions, Mapping) else None
    if wacc is None or wacc <= 0.0:
        flags.append("missing_wacc")
    if wacc is not None and terminal_growth is not None and (wacc - terminal_growth) <= 0.01:
        flags.append("wacc_terminal_spread_too_small")

    comps = payload.get("comps")
    if not isinstance(comps, list) or not comps:
        flags.append("missing_comps")
    if _forecast_roundness_flag(payload):
        flags.append("suspiciously_round_forecast")
    if _forecast_discontinuity_flag(payload):
        flags.append("forecast_discontinuity")
    flags.extend(_scenario_order_flags(payload))
    return sorted(set(flags))


def label_weight(flags: Sequence[str]) -> float:
    if any(flag in EXCLUDE_FLAGS for flag in flags):
        return 0.0
    downweight_count = sum(flag in DOWNWEIGHT_FLAGS for flag in flags)
    return float(1.0 / (1.0 + downweight_count))


def _latest_year(record: Mapping[str, Any]) -> int:
    fiscal_year = _coerce_float(record.get("fiscal_year"))
    return int(fiscal_year) if fiscal_year is not None and fiscal_year > 1900 else 2025


def _record_price(
    record: Mapping[str, Any],
    report_price: float | None,
) -> tuple[float | None, list[str], str | None]:
    raw_close = _coerce_float(record.get("raw_close_price"))
    if raw_close is not None and raw_close > 0.0:
        return raw_close, [], "raw_close_price"
    current_price = _coerce_float(record.get("current_price"))
    if current_price is not None and current_price > 0.0:
        return current_price, ["raw_price_unavailable", "mechanical_price_asof_unverified"], "legacy_edgar_current_price"
    if report_price is not None and report_price > 0.0:
        return None, ["raw_price_unavailable", "report_price_ignored_for_mechanical_dcf"], None
    return None, ["missing_price"], None


def _price_snapshot(
    price_panels: PricePanels | None,
    ticker: str,
    report_date: str | None,
) -> dict[str, float]:
    if price_panels is None or not ticker or not report_date:
        return {}
    target = pd.Timestamp(report_date)
    if pd.isna(target):
        return {}
    if target.tzinfo is not None:
        target = target.tz_convert(None)
    target = target.normalize()

    snapshot: dict[str, float] = {}
    if ticker in price_panels.raw_close.columns:
        raw = price_on_or_before(price_panels.raw_close[ticker].dropna(), target)
        if raw is not None and raw > 0.0:
            snapshot["raw_close_price"] = float(raw)
    if ticker in price_panels.adjusted_close.columns:
        adjusted = price_on_or_before(price_panels.adjusted_close[ticker].dropna(), target)
        if adjusted is not None and adjusted > 0.0:
            snapshot["adjusted_close_price"] = float(adjusted)
    return snapshot


def mechanical_dcf_implied_irr(
    ticker: str,
    record: Mapping[str, Any] | None,
    *,
    report_current_price: float | None = None,
    forecast_years: int = 5,
    wacc: float = 0.09,
    terminal_growth: float = 0.025,
    revenue_growth: float = 0.03,
) -> tuple[float | None, list[str], dict[str, Any]]:
    if record is None:
        return None, ["missing_mechanical_inputs"], {}

    flags: list[str] = []
    revenue = _coerce_float(record.get("revenue"))
    operating_income = _coerce_float(record.get("operating_income"))
    if operating_income is None:
        margin = _coerce_float(record.get("operating_margin"))
        operating_income = revenue * margin if revenue is not None and margin is not None else None
    shares = _coerce_float(record.get("shares_outstanding"))
    price, price_flags, price_source = _record_price(record, report_current_price)
    flags.extend(price_flags)

    missing_critical = []
    if revenue is None or revenue <= 0.0:
        missing_critical.append("revenue")
    if operating_income is None:
        missing_critical.append("operating_income")
    if shares is None or shares <= 0.0:
        missing_critical.append("shares_outstanding")
    if price is None or price <= 0.0:
        missing_critical.append("price")
    if missing_critical:
        flags.extend(f"missing_mechanical_{name}" for name in missing_critical)
        return None, sorted(set(flags)), {"missing_critical_inputs": missing_critical}

    da = _coerce_float(record.get("da"))
    if da is None:
        da = 0.0
        flags.append("missing_da_mechanical")
    capex = _coerce_float(record.get("capex"))
    if capex is None:
        capex = 0.0
        flags.append("missing_capex_mechanical")
    cash = _coerce_float(record.get("cash"))
    if cash is None:
        cash = 0.0
        flags.append("missing_cash_mechanical")
    debt = _coerce_float(record.get("total_debt"))
    if debt is None:
        debt = 0.0
        flags.append("missing_debt_mechanical")

    last_year = _latest_year(record)
    forecast = ForecastDrivers(
        revenue_growth={last_year + offset: revenue_growth for offset in range(1, forecast_years + 1)},
        ebit_margin={last_year + offset: operating_income / revenue for offset in range(1, forecast_years + 1)},
        da_pct_sales={last_year + offset: da / revenue for offset in range(1, forecast_years + 1)},
        capex_pct_sales={last_year + offset: abs(capex) / revenue for offset in range(1, forecast_years + 1)},
        nwc_pct_sales={last_year + offset: 0.0 for offset in range(1, forecast_years + 1)},
    )
    valuation_input = ValuationInput(
        company_name=str(record.get("company_name")) if record.get("company_name") else None,
        ticker=ticker,
        currency="USD",
        forecast_years=forecast_years,
        historical=HistoricalFinancials(
            revenue={last_year: revenue},
            ebit={last_year: operating_income},
            da={last_year: da},
            capex={last_year: abs(capex)},
            nwc={last_year: 0.0},
        ),
        forecast=forecast,
        assumptions=Assumptions(
            tax_rate=0.21,
            wacc=wacc,
            terminal_growth=terminal_growth,
            terminal_method="gordon_growth",
            cash=cash,
            debt=debt,
            diluted_shares=shares,
            current_price=price,
        ),
    )
    try:
        result = run_dcf(valuation_input)
    except DCFError as exc:
        flags.append("mechanical_dcf_error")
        return None, sorted(set(flags)), {"error": str(exc)}

    base = result.scenarios["base"]
    mechanical_cash_flows = [
        *[year.fcff for year in base.projected_years[:-1]],
        base.projected_years[-1].fcff + base.terminal_value,
    ]
    current_enterprise_value = price * shares + base.net_debt - base.investments_adjustment
    irr = _solve_discount_rate(mechanical_cash_flows, current_enterprise_value)
    if irr is None:
        flags.append("mechanical_dcf_error")
    details = {
        "mechanical_per_share_value": base.per_share_value,
        "mechanical_upside": base.upside_downside_pct,
        "mechanical_annualized_value_gap": _annualized_return(price, base.per_share_value, forecast_years),
        "mechanical_wacc": wacc,
        "mechanical_terminal_growth": terminal_growth,
        "mechanical_revenue_growth": revenue_growth,
        "mechanical_method": "cash_flow_discount_rate_to_current_enterprise_value",
        "mechanical_price_source": price_source,
        "mechanical_irr_target_enterprise_value": current_enterprise_value,
    }
    return irr, sorted(set(flags)), details


def _report_row(
    path: Path,
    payload: Mapping[str, Any],
    edgar_record: Mapping[str, Any] | None,
    sector_record: Mapping[str, Any] | None = None,
    price_panels: PricePanels | None = None,
) -> dict[str, Any]:
    path_ticker, path_report_date = _report_identity(path)
    ticker = normalize_ticker(str(payload.get("ticker") or path_ticker or ""))
    meta = payload.get("_meta") if isinstance(payload.get("_meta"), Mapping) else {}
    base_case = _scenario(payload, "base")
    report_date = path_report_date
    generated_at = meta.get("generated_at")
    if report_date is None and isinstance(generated_at, str) and len(generated_at) >= 10:
        report_date = generated_at[:10]
    record = dict(edgar_record or {})
    record.update(_price_snapshot(price_panels, ticker, report_date))

    forecast_years = _forecast_years(payload, base_case)
    raw_ai_upside = _coerce_float(base_case.get("upside_downside_pct")) if base_case is not None else None
    assumptions = payload.get("assumptions") if isinstance(payload.get("assumptions"), Mapping) else {}
    raw_ai_irr, ai_irr_flags, ai_irr_details = _cash_flow_implied_irr(base_case, assumptions)
    raw_ai_annualized_value_gap = _scenario_annualized_value_gap(payload, "base")
    report_price = _coerce_float(base_case.get("current_price")) if base_case is not None else None
    mechanical_irr, mechanical_flags, mechanical_details = mechanical_dcf_implied_irr(
        ticker,
        record if record else None,
        report_current_price=report_price,
        forecast_years=forecast_years,
    )
    flags = sorted(set(dcf_sanity_flags(payload) + ai_irr_flags + mechanical_flags))
    ai_minus_mechanical = (
        raw_ai_irr - mechanical_irr if raw_ai_irr is not None and mechanical_irr is not None else None
    )
    raw_close_price = _coerce_float(record.get("raw_close_price"))
    adjusted_close_price = _coerce_float(record.get("adjusted_close_price"))
    shares_outstanding = _coerce_float(record.get("shares_outstanding"))
    market_cap = raw_close_price * shares_outstanding if raw_close_price is not None and shares_outstanding else None
    if market_cap is None:
        market_cap = _coerce_float(record.get("market_cap")) if record else None
    edgar_sector = record.get("sector") if record else None
    mapped_sector = sector_record.get("sector") if sector_record is not None else None
    sector = edgar_sector or mapped_sector
    sub_industry = (
        record.get("sub_industry")
        if record and record.get("sub_industry")
        else sector_record.get("sub_industry")
        if sector_record is not None
        else None
    )
    if edgar_sector:
        sector_source = "edgar_features"
    elif mapped_sector:
        sector_source = sector_record.get("sector_source") if sector_record is not None else None
        sector_source = sector_source or "sector_map"
    else:
        sector_source = None

    row: dict[str, Any] = {
        "ticker": ticker,
        "report_id": path.stem,
        "report_path": repo_relative(path),
        "report_date": report_date,
        "generated_at": generated_at,
        "model_id": meta.get("model"),
        "tier": meta.get("tier"),
        "reasoning_effort": meta.get("reasoning_effort"),
        "sector": sector,
        "sub_industry": sub_industry,
        "sector_source": sector_source,
        "company_name": payload.get("company_name") or (record.get("company_name") if record else None),
        "raw_ai_upside": raw_ai_upside,
        "raw_ai_implied_irr": raw_ai_irr,
        "raw_ai_annualized_value_gap": raw_ai_annualized_value_gap,
        "mechanical_dcf_implied_irr": mechanical_irr,
        "ai_minus_mechanical_irr": ai_minus_mechanical,
        "factor_compressible_ai_score": None,
        "ai_factor_residual": None,
        "ai_label_uncertainty": None,
        "label_observation_count": 1,
        "parse_failure_rate": 0.0,
        "quality_flags": flags,
        "quality_flag_policy": {flag: QUALITY_FLAG_POLICY.get(flag, "warning") for flag in flags},
        "exclude_from_clean_label": bool(any(flag in EXCLUDE_FLAGS for flag in flags)),
        "label_weight": label_weight(flags),
        "raw_close_price": raw_close_price,
        "adjusted_close_price": adjusted_close_price,
        "market_cap": market_cap,
        "source_report_path": repo_relative(path),
        "edgar_feature_source": "data/edgar_features_full.json" if edgar_record is not None else None,
    }
    row.update(mechanical_details)
    row.update(ai_irr_details)
    return row


def _feature_frame_from_edgar(edgar_payload: Mapping[str, Mapping[str, Any]] | None) -> pd.DataFrame:
    if not edgar_payload:
        return pd.DataFrame()
    rows = []
    for ticker, record in edgar_payload.items():
        row = {"ticker": normalize_ticker(str(ticker))}
        row.update(record)
        rows.append(row)
    return pd.DataFrame(rows).drop_duplicates("ticker").set_index("ticker")


def _winsorized_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce").astype(float)
        clean = series.dropna()
        if clean.empty:
            continue
        lower = float(clean.quantile(0.01))
        upper = float(clean.quantile(0.99))
        clipped = series.clip(lower=lower, upper=upper)
        median = float(clipped.median()) if clipped.notna().any() else 0.0
        filled = clipped.fillna(median)
        std = float(filled.std(ddof=0))
        output[column] = 0.0 if std == 0.0 or not math.isfinite(std) else (filled - float(filled.mean())) / std
    return output


def add_factor_residuals(
    panel: pd.DataFrame,
    feature_frame: pd.DataFrame,
    *,
    target_column: str = "raw_ai_implied_irr",
    sector_column: str = "sector",
    fitted_column: str = "factor_compressible_ai_score",
    residual_column: str = "ai_factor_residual",
    metadata_prefix: str = "factor_residualization",
) -> pd.DataFrame:
    output = panel.copy()
    if output.empty or feature_frame.empty or target_column not in output.columns:
        return output

    merged = output.merge(
        feature_frame,
        how="left",
        left_on="ticker",
        right_index=True,
        suffixes=("", "_feature"),
    )
    fit_mask = (
        pd.to_numeric(merged[target_column], errors="coerce").notna()
        & (merged["label_weight"].astype(float) > 0.0)
    )
    available_features = [feature for feature in FACTOR_FEATURES if feature in merged.columns]
    if not available_features:
        return output

    fit_source = merged.loc[fit_mask, ["ticker", target_column, *available_features]].copy()
    usable = (
        fit_source.groupby("ticker", as_index=True)
        .agg({target_column: "mean", **{feature: "first" for feature in available_features}})
        .copy()
    )
    if len(usable) < 3:
        return output

    x_numeric = _winsorized_zscore(usable[available_features])
    if x_numeric.empty:
        return output

    x_parts = [pd.Series(1.0, index=x_numeric.index, name="intercept"), x_numeric]
    sector_source = None
    if sector_column in merged.columns:
        sector_source = merged.loc[fit_mask, ["ticker", sector_column]].drop_duplicates("ticker").set_index("ticker")[sector_column].reindex(x_numeric.index)
    elif f"{sector_column}_feature" in merged.columns:
        sector_source = (
            merged.loc[fit_mask, ["ticker", f"{sector_column}_feature"]]
            .drop_duplicates("ticker")
            .set_index("ticker")[f"{sector_column}_feature"]
            .reindex(x_numeric.index)
        )
    if sector_source is not None and sector_source.nunique(dropna=True) > 1 and len(x_numeric) >= 8:
        dummies = pd.get_dummies(sector_source.fillna("Unknown"), prefix="sector", drop_first=True, dtype=float)
        if len(dummies.columns) < max(1, len(x_numeric) // 3):
            x_parts.append(dummies)

    x = pd.concat(x_parts, axis=1).astype(float)
    y = pd.to_numeric(usable[target_column], errors="coerce").astype(float)
    coefficients, *_ = np.linalg.lstsq(x.to_numpy(dtype=float), y.to_numpy(dtype=float), rcond=None)
    fitted = pd.Series(x.to_numpy(dtype=float) @ coefficients, index=x.index, dtype=float)

    fitted_by_ticker = output["ticker"].map(fitted)
    target_values = pd.to_numeric(output[target_column], errors="coerce").astype(float)
    output[fitted_column] = fitted_by_ticker
    output[residual_column] = target_values - fitted_by_ticker
    output[f"{metadata_prefix}_target"] = target_column
    output[f"{metadata_prefix}_n"] = int(len(y))
    output[f"{metadata_prefix}_unique_tickers"] = int(len(y))
    return output


def add_repeated_label_uncertainty(panel: pd.DataFrame) -> pd.DataFrame:
    output = panel.copy()
    if output.empty:
        return output

    raw_ai_irr = pd.to_numeric(output["raw_ai_implied_irr"], errors="coerce").astype(float)
    output["raw_ai_irr_cross_section_rank"] = raw_ai_irr.rank(pct=True, method="average")
    grouped = output.assign(_raw_ai_irr=raw_ai_irr).groupby("ticker", dropna=False)["_raw_ai_irr"]
    stats = grouped.agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75),
        min="min",
        max="max",
    ).rename(
        columns={
            "count": "label_observation_count",
            "mean": "mean_ai_irr",
            "median": "median_ai_irr",
            "std": "std_ai_irr",
            "q25": "q25_ai_irr",
            "q75": "q75_ai_irr",
            "min": "min_ai_irr",
            "max": "max_ai_irr",
        }
    )
    stats["ai_irr_iqr"] = stats["q75_ai_irr"] - stats["q25_ai_irr"]
    stats["ai_irr_range"] = stats["max_ai_irr"] - stats["min_ai_irr"]
    rank_std = output.groupby("ticker", dropna=False)["raw_ai_irr_cross_section_rank"].std().rename("ai_irr_rank_std")
    stats = stats.join(rank_std)
    output = output.merge(stats, how="left", left_on="ticker", right_index=True, suffixes=("", "_agg"))
    output["label_observation_count"] = output["label_observation_count_agg"].fillna(
        output["label_observation_count"]
    )
    output = output.drop(columns=["label_observation_count_agg"])
    output["ai_label_uncertainty"] = output["std_ai_irr"]
    output["rank_std"] = output["ai_irr_rank_std"]
    output["uncertainty_inverse_weight"] = np.where(
        output["ai_label_uncertainty"].astype(float) > 0.0,
        1.0 / output["ai_label_uncertainty"].astype(float),
        np.nan,
    )
    uncertainty_penalty = 1.0 / (1.0 + output["ai_label_uncertainty"].fillna(0.0).astype(float))
    output["uncertainty_adjusted_label_weight"] = output["label_weight"].astype(float) * uncertainty_penalty
    for column_name, group_column in (
        ("model_disagreement", "model_id"),
        ("tier_disagreement", "tier"),
        ("prompt_disagreement", _prompt_group_column(output)),
    ):
        output[column_name] = output["ticker"].map(_label_disagreement(output, group_column)) if group_column else np.nan
    failure = output.groupby("ticker", dropna=False)["exclude_from_clean_label"].mean().rename("parse_failure_rate_agg")
    output = output.merge(failure, how="left", left_on="ticker", right_index=True)
    output["parse_failure_rate"] = output["parse_failure_rate_agg"].fillna(output["parse_failure_rate"])
    output = output.drop(columns=["parse_failure_rate_agg"])
    return output


def _prompt_group_column(frame: pd.DataFrame) -> str | None:
    for column in ("prompt_id", "prompt_template", "prompt_variant"):
        if column in frame.columns:
            return column
    return None


def _label_disagreement(frame: pd.DataFrame, group_column: str | None) -> pd.Series:
    if group_column is None or group_column not in frame.columns:
        return pd.Series(dtype=float)
    values = frame[["ticker", group_column, "raw_ai_implied_irr"]].copy()
    values["raw_ai_implied_irr"] = pd.to_numeric(values["raw_ai_implied_irr"], errors="coerce").astype(float)
    values = values.dropna(subset=[group_column, "raw_ai_implied_irr"])
    if values.empty:
        return pd.Series(dtype=float)
    by_group = values.groupby(["ticker", group_column], dropna=False)["raw_ai_implied_irr"].mean()
    return by_group.groupby(level=0).agg(lambda series: float(series.max() - series.min()) if len(series) > 1 else np.nan)


def build_label_panel(
    *,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    edgar_payload: Mapping[str, Mapping[str, Any]] | None = None,
    sector_map: Mapping[str, Mapping[str, Any]] | None = None,
    price_panels: PricePanels | None = None,
    include_factor_residuals: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    normalized_edgar = (
        {normalize_ticker(str(ticker)): record for ticker, record in edgar_payload.items()}
        if edgar_payload is not None
        else {}
    )
    normalized_sector_map = (
        {normalize_ticker(str(ticker)): record for ticker, record in sector_map.items()}
        if sector_map is not None
        else {}
    )
    for path in sorted(reports_dir.glob("*.json")):
        try:
            payload = _load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        ticker = normalize_ticker(str(payload.get("ticker") or _report_identity(path)[0] or ""))
        edgar_record = normalized_edgar.get(ticker)
        rows.append(_report_row(path, payload, edgar_record, normalized_sector_map.get(ticker), price_panels))

    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    panel = add_repeated_label_uncertainty(panel)
    panel = panel.sort_values(["ticker", "report_date", "report_id"], kind="mergesort").reset_index(drop=True)
    if include_factor_residuals:
        feature_frame = _feature_frame_from_edgar(normalized_edgar)
        panel = add_factor_residuals(panel, feature_frame)
        panel = add_factor_residuals(
            panel,
            feature_frame,
            target_column="ai_minus_mechanical_irr",
            fitted_column="mechanical_adjusted_factor_score",
            residual_column="mechanical_adjusted_factor_residual",
            metadata_prefix="mechanical_adjusted_residualization",
        )
    return panel


def summarize_label_panel(panel: pd.DataFrame) -> dict[str, Any]:
    if panel.empty:
        return {"row_count": 0}

    flag_counts: dict[str, int] = {}
    for flags in panel["quality_flags"]:
        for flag in flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    sector_summary = []
    if "sector" in panel.columns:
        for sector, group in panel.groupby("sector", dropna=False):
            sector_summary.append(
                {
                    "sector": None if pd.isna(sector) else sector,
                    "rows": int(len(group)),
                    "clean_label_rate": float((~group["exclude_from_clean_label"]).mean()),
                    "mechanical_coverage": float(group["mechanical_dcf_implied_irr"].notna().mean()),
                }
            )

    tier_summary = []
    if "tier" in panel.columns:
        for tier, group in panel.groupby("tier", dropna=False):
            tier_summary.append(_failure_summary_row("tier", tier, group))

    market_cap_summary = []
    if "market_cap" in panel.columns and pd.to_numeric(panel["market_cap"], errors="coerce").notna().any():
        market_cap = pd.to_numeric(panel["market_cap"], errors="coerce")
        bucket_labels = pd.Series("unknown", index=panel.index, dtype="object")
        positive = market_cap[market_cap > 0.0].dropna()
        if positive.nunique() >= 3:
            try:
                bucket_labels.loc[positive.index] = pd.qcut(
                    positive,
                    q=min(3, positive.nunique()),
                    labels=["small", "mid", "large"][: min(3, positive.nunique())],
                    duplicates="drop",
                ).astype("object")
            except ValueError:
                bucket_labels.loc[positive.index] = "known"
        elif not positive.empty:
            bucket_labels.loc[positive.index] = "known"
        for bucket, group in panel.groupby(bucket_labels, dropna=False):
            market_cap_summary.append(_failure_summary_row("market_cap_bucket", bucket, group))

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "row_count": int(len(panel)),
        "ticker_count": int(panel["ticker"].nunique()),
        "clean_label_count": int((~panel["exclude_from_clean_label"]).sum()),
        "raw_ai_irr_coverage": float(panel["raw_ai_implied_irr"].notna().mean()),
        "mechanical_irr_coverage": float(panel["mechanical_dcf_implied_irr"].notna().mean()),
        "ai_minus_mechanical_coverage": float(panel["ai_minus_mechanical_irr"].notna().mean()),
        "factor_residual_coverage": float(panel["ai_factor_residual"].notna().mean()),
        "raw_price_coverage": float(panel["raw_close_price"].notna().mean()) if "raw_close_price" in panel.columns else 0.0,
        "adjusted_price_coverage": (
            float(panel["adjusted_close_price"].notna().mean()) if "adjusted_close_price" in panel.columns else 0.0
        ),
        "mechanical_price_source_counts": (
            {str(key): int(value) for key, value in panel["mechanical_price_source"].fillna("missing").value_counts().items()}
            if "mechanical_price_source" in panel.columns
            else {}
        ),
        "sector_coverage": float(panel["sector"].notna().mean()) if "sector" in panel.columns else 0.0,
        "sector_source_counts": (
            {str(key): int(value) for key, value in panel["sector_source"].fillna("missing").value_counts().items()}
            if "sector_source" in panel.columns
            else {}
        ),
        "quality_flag_counts": dict(sorted(flag_counts.items())),
        "quality_flag_policy": QUALITY_FLAG_POLICY,
        "failure_rates_by_sector": sector_summary,
        "failure_rates_by_model_tier": tier_summary,
        "failure_rates_by_market_cap_bucket": market_cap_summary,
    }


def _failure_summary_row(group_name: str, group_value: Any, group: pd.DataFrame) -> dict[str, Any]:
    return {
        group_name: None if pd.isna(group_value) else str(group_value),
        "rows": int(len(group)),
        "clean_label_rate": float((~group["exclude_from_clean_label"]).mean()),
        "exclude_rate": float(group["exclude_from_clean_label"].mean()),
        "downweighted_rate": float((group["label_weight"].astype(float).between(0.0, 1.0, inclusive="neither")).mean()),
        "raw_ai_irr_coverage": float(group["raw_ai_implied_irr"].notna().mean()),
        "mechanical_coverage": float(group["mechanical_dcf_implied_irr"].notna().mean()),
        "ai_minus_mechanical_coverage": float(group["ai_minus_mechanical_irr"].notna().mean()),
    }


def _write_panel(panel: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = panel.copy()
    for column in ("quality_flags", "quality_flag_policy"):
        if column in serializable.columns:
            serializable[column] = serializable[column].map(json.dumps)
    if output_path.suffix == ".csv":
        serializable.to_csv(output_path, index=False)
    else:
        serializable.to_parquet(output_path, index=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Parallax v2 AI label decomposition panel.")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--edgar-file", default=str(DEFAULT_EDGAR_PATH))
    parser.add_argument("--sector-map", default=str(DEFAULT_SECTOR_MAP))
    parser.add_argument("--price-panel-cache", default=str(DEFAULT_PRICE_PANEL_CACHE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--csv-output")
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--metadata-output", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--no-factor-residuals", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    edgar_path = Path(args.edgar_file)
    edgar_payload = load_edgar_payload(edgar_path) if edgar_path.exists() else {}
    sector_map_path = Path(args.sector_map) if args.sector_map else None
    sector_map = load_sector_map(sector_map_path)
    price_panel_cache = Path(args.price_panel_cache) if args.price_panel_cache else None
    price_panels = (
        load_cached_price_panels(edgar_payload.keys(), price_panel_cache)
        if price_panel_cache is not None and price_panel_cache.exists()
        else None
    )
    panel = build_label_panel(
        reports_dir=Path(args.reports_dir),
        edgar_payload=edgar_payload,
        sector_map=sector_map,
        price_panels=price_panels,
        include_factor_residuals=not args.no_factor_residuals,
    )
    output_path = Path(args.output)
    _write_panel(panel, output_path)
    if args.csv_output:
        _write_panel(panel, Path(args.csv_output))

    summary = summarize_label_panel(panel)
    summary["artifacts"] = {
        "label_panel": repo_relative(output_path),
        "summary": repo_relative(args.summary_output),
        "metadata": repo_relative(args.metadata_output),
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    metadata = build_experiment_metadata(
        experiment_id="ai_label_decomposition_v2",
        feature_config={
            "factor_features": list(FACTOR_FEATURES),
            "sector_source": repo_relative(sector_map_path) if sector_map_path is not None and sector_map_path.exists() else None,
            "quality_flag_policy": QUALITY_FLAG_POLICY,
            "raw_price_source": (
                repo_relative(price_panel_cache) if price_panel_cache is not None and price_panel_cache.exists() else None
            ),
            "repeated_label_uncertainty": [
                "std_ai_irr",
                "ai_irr_iqr",
                "ai_irr_rank_std",
                "model_disagreement",
                "tier_disagreement",
                "prompt_disagreement",
                "uncertainty_adjusted_label_weight",
            ],
        },
        model_config={"factor_residualization": "cross_sectional_ols_with_sector_dummies_when_available"},
        universe_config={
            "reports_dir": repo_relative(args.reports_dir),
            "survivor_bias_caveat": True,
            "sector_map_current_snapshot_only": bool(sector_map),
        },
        backtest_config={"rank_ic_required_before_portfolio_backtests": True},
        data_snapshot_paths={
            "edgar_features": edgar_path,
            **({"price_panel_cache": price_panel_cache} if price_panel_cache is not None and price_panel_cache.exists() else {}),
            **({"sector_map": sector_map_path} if sector_map_path is not None and sector_map_path.exists() else {}),
        },
        label_snapshot_paths={"label_panel": output_path},
        artifacts={"summary": args.summary_output, "label_panel": output_path},
    )
    write_experiment_metadata(args.metadata_output, metadata)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
