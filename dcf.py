from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from parser import (
    Assumptions,
    ForecastDrivers,
    ParseError,
    ScenarioOverrides,
    TerminalMethod,
    ValuationInput,
    merge_assumptions,
    merge_forecast,
    parse_input,
)


class DCFError(ValueError):
    """Raised when the normalized inputs still do not support a valid DCF."""


@dataclass(frozen=True)
class ProjectedYear:
    year: int
    revenue: float
    revenue_growth: float
    ebit_margin: float
    ebit: float
    da: float
    ebitda: float
    capex: float
    nwc: float
    delta_nwc: float
    cash_taxes: float
    nol_used: float
    nol_balance_end: float
    fcff: float
    discount_factor: float
    present_value_fcff: float


@dataclass(frozen=True)
class ScenarioValuation:
    scenario: str
    description: str | None
    terminal_method: TerminalMethod
    enterprise_value: float
    equity_value: float
    net_debt: float
    investments_adjustment: float
    per_share_value: float
    current_price: float | None
    upside_downside_pct: float | None
    implied_market_cap: float
    terminal_value: float
    terminal_value_present_value: float
    gordon_terminal_value: float | None
    exit_multiple_terminal_value: float | None
    projected_years: tuple[ProjectedYear, ...]


@dataclass(frozen=True)
class DCFResult:
    company_name: str | None
    ticker: str | None
    currency: str | None
    scenarios: dict[str, ScenarioValuation]


@dataclass(frozen=True)
class CliInput:
    raw_payload: Mapping[str, Any]
    output_root: Path


def run_dcf(valuation_input: ValuationInput) -> DCFResult:
    scenarios = {
        "base": value_scenario(valuation_input, "base"),
        "bull": value_scenario(valuation_input, "bull"),
        "bear": value_scenario(valuation_input, "bear"),
    }
    return DCFResult(
        company_name=valuation_input.company_name,
        ticker=valuation_input.ticker,
        currency=valuation_input.currency,
        scenarios=scenarios,
    )


def value_scenario(valuation_input: ValuationInput, scenario_name: str) -> ScenarioValuation:
    override = valuation_input.scenarios.get(scenario_name, ScenarioOverrides())
    forecast, assumptions, description = _resolve_scenario(
        valuation_input,
        scenario_name,
        override,
    )

    if assumptions.wacc <= 0.0:
        raise DCFError(f"{scenario_name} scenario requires a positive WACC.")
    if assumptions.diluted_shares <= 0.0:
        raise DCFError(f"{scenario_name} scenario requires a positive diluted share count.")

    last_year = valuation_input.last_historical_year
    revenue = valuation_input.historical.revenue[last_year]
    nwc = _starting_nwc(valuation_input, last_year)
    nol_balance = max(assumptions.nol_balance, 0.0)

    projected_years: list[ProjectedYear] = []
    pv_fcff_total = 0.0

    for index, year in enumerate(forecast.years, start=1):
        growth = forecast.revenue_growth[year]
        revenue *= 1.0 + growth
        ebit_margin = forecast.ebit_margin[year]
        da_pct_sales = forecast.da_pct_sales[year]
        capex_pct_sales = forecast.capex_pct_sales[year]
        nwc_pct_sales = forecast.nwc_pct_sales[year]

        ebit = revenue * ebit_margin
        da = revenue * da_pct_sales
        ebitda = ebit + da
        capex = revenue * capex_pct_sales
        projected_nwc = revenue * nwc_pct_sales
        delta_nwc = projected_nwc - nwc

        cash_taxes, nol_used, nol_balance = _cash_taxes(
            ebit=ebit,
            tax_rate=assumptions.tax_rate,
            nol_balance=nol_balance,
            nol_utilization_pct=assumptions.nol_utilization_pct,
        )
        fcff = ebit - cash_taxes + da - capex - delta_nwc
        discount_factor = 1.0 / ((1.0 + assumptions.wacc) ** index)
        present_value_fcff = fcff * discount_factor

        projected_years.append(
            ProjectedYear(
                year=year,
                revenue=revenue,
                revenue_growth=growth,
                ebit_margin=ebit_margin,
                ebit=ebit,
                da=da,
                ebitda=ebitda,
                capex=capex,
                nwc=projected_nwc,
                delta_nwc=delta_nwc,
                cash_taxes=cash_taxes,
                nol_used=nol_used,
                nol_balance_end=nol_balance,
                fcff=fcff,
                discount_factor=discount_factor,
                present_value_fcff=present_value_fcff,
            )
        )
        pv_fcff_total += present_value_fcff
        nwc = projected_nwc

    terminal_values = _terminal_value(projected_years[-1], assumptions)
    terminal_discount_factor = projected_years[-1].discount_factor
    terminal_value_present_value = terminal_values["selected"] * terminal_discount_factor
    enterprise_value = pv_fcff_total + terminal_value_present_value

    net_debt = _net_debt(assumptions)
    investments_adjustment = _investments_adjustment(assumptions, net_debt)
    equity_value = (
        enterprise_value
        - net_debt
        - assumptions.minority_interest
        - assumptions.preferred_equity
        + investments_adjustment
    )
    per_share_value = equity_value / assumptions.diluted_shares
    current_price = assumptions.current_price
    upside_downside_pct = None
    if current_price not in (None, 0):
        upside_downside_pct = (per_share_value / current_price) - 1.0

    return ScenarioValuation(
        scenario=scenario_name,
        description=description,
        terminal_method=assumptions.terminal_method,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        net_debt=net_debt,
        investments_adjustment=investments_adjustment,
        per_share_value=per_share_value,
        current_price=current_price,
        upside_downside_pct=upside_downside_pct,
        implied_market_cap=per_share_value * assumptions.diluted_shares,
        terminal_value=terminal_values["selected"],
        terminal_value_present_value=terminal_value_present_value,
        gordon_terminal_value=terminal_values["gordon"],
        exit_multiple_terminal_value=terminal_values["exit"],
        projected_years=tuple(projected_years),
    )


def _resolve_scenario(
    valuation_input: ValuationInput,
    scenario_name: str,
    override: ScenarioOverrides,
) -> tuple[ForecastDrivers, Assumptions, str | None]:
    if scenario_name == "base":
        return valuation_input.forecast, valuation_input.assumptions, "Base case"

    forecast = merge_forecast(valuation_input.forecast, override.forecast)
    assumptions = merge_assumptions(valuation_input.assumptions, override.assumptions)
    description = override.description or f"{scenario_name.title()} case"
    return forecast, assumptions, description


def _starting_nwc(valuation_input: ValuationInput, last_year: int) -> float:
    if last_year in valuation_input.historical.nwc:
        return valuation_input.historical.nwc[last_year]

    historical = valuation_input.historical
    common_years = sorted(set(historical.nwc) & set(historical.revenue))
    if not common_years:
        return 0.0
    latest_nwc_year = common_years[-1]
    revenue = historical.revenue[latest_nwc_year]
    if revenue == 0:
        return 0.0

    ratio = historical.nwc[latest_nwc_year] / revenue
    return valuation_input.historical.revenue[last_year] * ratio


def _cash_taxes(
    *,
    ebit: float,
    tax_rate: float,
    nol_balance: float,
    nol_utilization_pct: float,
) -> tuple[float, float, float]:
    tax_rate = max(tax_rate, 0.0)
    nol_balance = max(nol_balance, 0.0)
    utilization = min(max(nol_utilization_pct, 0.0), 1.0)

    if ebit <= 0.0:
        return 0.0, 0.0, nol_balance + abs(ebit)

    nol_used = min(nol_balance, ebit * utilization)
    taxable_ebit = max(ebit - nol_used, 0.0)
    cash_taxes = taxable_ebit * tax_rate
    return cash_taxes, nol_used, nol_balance - nol_used


def _terminal_value(last_year: ProjectedYear, assumptions: Assumptions) -> dict[str, float | None]:
    gordon_value: float | None = None
    exit_value: float | None = None

    if assumptions.terminal_method in {"gordon_growth", "average"}:
        spread = assumptions.wacc - assumptions.terminal_growth
        if spread <= 0.0:
            raise DCFError("Terminal growth must be below WACC for Gordon growth valuation.")
        gordon_value = last_year.fcff * (1.0 + assumptions.terminal_growth) / spread

    if assumptions.terminal_method in {"exit_multiple", "average"}:
        multiple = assumptions.terminal_exit_ebitda_multiple
        if multiple is None:
            raise DCFError(
                "Terminal exit EBITDA multiple is required for exit multiple or average terminal valuation."
            )
        exit_value = last_year.ebitda * multiple

    if assumptions.terminal_method == "gordon_growth":
        selected = gordon_value
    elif assumptions.terminal_method == "exit_multiple":
        selected = exit_value
    else:
        if gordon_value is None or exit_value is None:
            raise DCFError("Average terminal valuation requires both Gordon growth and exit multiple inputs.")
        selected = (gordon_value + exit_value) / 2.0

    return {
        "selected": selected,
        "gordon": gordon_value,
        "exit": exit_value,
    }


def _net_debt(assumptions: Assumptions) -> float:
    if assumptions.net_debt_override is not None:
        return assumptions.net_debt_override
    return assumptions.debt - assumptions.cash


def _investments_adjustment(assumptions: Assumptions, net_debt: float) -> float:
    investments = assumptions.investments
    if investments == 0.0:
        return 0.0

    if assumptions.net_debt_override is None:
        return investments

    net_debt_excluding_investments = assumptions.debt - assumptions.cash
    net_debt_including_investments = net_debt_excluding_investments - investments
    tolerance = max(1.0, abs(investments) * 0.01)

    if abs(net_debt - net_debt_including_investments) <= tolerance and abs(
        net_debt - net_debt_excluding_investments
    ) > tolerance:
        return 0.0

    return investments


def _load_cli_input() -> CliInput:
    cli = argparse.ArgumentParser(description="Run a three-scenario FCFF DCF from normalized research JSON.")
    cli.add_argument(
        "input_path",
        nargs="?",
        help="Path to a JSON file. If omitted, JSON is read from stdin.",
    )
    cli.add_argument(
        "--output-dir",
        default="valuations",
        help="Root directory where company valuation artifacts will be saved.",
    )
    args = cli.parse_args()

    if args.input_path:
        try:
            raw_text = Path(args.input_path).read_text(encoding="utf-8")
        except OSError as exc:
            raise ParseError(f"Unable to read JSON file '{args.input_path}'.") from exc
        raw_payload = _load_raw_payload(raw_text)
        return CliInput(raw_payload=raw_payload, output_root=Path(args.output_dir))

    payload = sys.stdin.read()
    if not payload.strip():
        raise ParseError("Provide JSON via stdin or pass a file path argument.")
    return CliInput(raw_payload=_load_raw_payload(payload), output_root=Path(args.output_dir))


def _load_raw_payload(text: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"Invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise ParseError("Top-level JSON value must be an object.")
    return payload


def _save_run_artifacts(
    *,
    raw_payload: Mapping[str, Any],
    valuation_input: ValuationInput,
    results: DCFResult,
    output_root: Path,
) -> Path:
    company_dir = output_root / _company_folder_name(valuation_input)
    company_dir.mkdir(parents=True, exist_ok=True)

    _write_json(company_dir / "research_input.json", raw_payload)
    _write_json(company_dir / "normalized_input.json", asdict(valuation_input))
    _write_json(company_dir / "dcf_output.json", asdict(results))
    return company_dir


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _company_folder_name(valuation_input: ValuationInput) -> str:
    parts = [valuation_input.ticker, valuation_input.company_name]
    slug_parts = [_slugify(part) for part in parts if part]
    if not slug_parts:
        return "unknown-company"
    return "_".join(dict.fromkeys(slug_parts))


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unknown"


def main() -> int:
    try:
        cli_input = _load_cli_input()
        valuation_input = parse_input(cli_input.raw_payload)
        results = run_dcf(valuation_input)
        artifacts_dir = _save_run_artifacts(
            raw_payload=cli_input.raw_payload,
            valuation_input=valuation_input,
            results=results,
            output_root=cli_input.output_root,
        )
    except (ParseError, DCFError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(asdict(results), indent=2))
    print(f"Saved valuation artifacts to {artifacts_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
