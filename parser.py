from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping, TypeAlias, cast


TerminalMethod: TypeAlias = Literal["gordon_growth", "exit_multiple", "average"]

_FORECAST_FIELDS = (
    "revenue_growth",
    "ebit_margin",
    "da_pct_sales",
    "capex_pct_sales",
    "nwc_pct_sales",
)
_TERMINAL_METHODS = {"gordon_growth", "exit_multiple", "average"}
_MISSING_STRINGS = {"", "na", "n/a", "none", "null", "-"}
_CURRENCY_MARKERS = "$\u20ac\u00a3\u00a5"
_MARKDOWN_LINK_RE = re.compile(r"^\[(?P<label>[^\]]+)\]\((?P<url>https?://[^)]+)\)$")


class ParseError(ValueError):
    """Raised when the research JSON cannot be normalized into a DCF input."""


@dataclass(frozen=True)
class HistoricalFinancials:
    revenue: dict[int, float]
    ebit: dict[int, float]
    da: dict[int, float] = field(default_factory=dict)
    capex: dict[int, float] = field(default_factory=dict)
    nwc: dict[int, float] = field(default_factory=dict)

    def latest_year(self, series_name: str) -> int | None:
        series = getattr(self, series_name)
        return max(series) if series else None

    def latest_common_year(self, *series_names: str) -> int | None:
        names = series_names or ("revenue", "ebit")
        common_years: set[int] | None = None
        for name in names:
            years = set(getattr(self, name).keys())
            if not years:
                return None
            common_years = years if common_years is None else common_years & years
        if not common_years:
            return None
        return max(common_years)


@dataclass(frozen=True)
class ForecastDrivers:
    revenue_growth: dict[int, float]
    ebit_margin: dict[int, float]
    da_pct_sales: dict[int, float]
    capex_pct_sales: dict[int, float]
    nwc_pct_sales: dict[int, float]

    @property
    def years(self) -> list[int]:
        return sorted(self.revenue_growth)


@dataclass(frozen=True)
class PartialForecastDrivers:
    revenue_growth: dict[int, float] = field(default_factory=dict)
    ebit_margin: dict[int, float] = field(default_factory=dict)
    da_pct_sales: dict[int, float] = field(default_factory=dict)
    capex_pct_sales: dict[int, float] = field(default_factory=dict)
    nwc_pct_sales: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Assumptions:
    tax_rate: float = 0.0
    wacc: float = 0.0
    terminal_growth: float = 0.0
    terminal_exit_ebitda_multiple: float | None = None
    terminal_method: TerminalMethod = "average"
    cash: float = 0.0
    debt: float = 0.0
    net_debt_override: float | None = None
    investments: float = 0.0
    minority_interest: float = 0.0
    preferred_equity: float = 0.0
    diluted_shares: float = 0.0
    current_price: float | None = None
    target_ebit_margin: float | None = None
    nol_balance: float = 0.0
    nol_utilization_pct: float = 0.8


@dataclass(frozen=True)
class PartialAssumptions:
    tax_rate: float | None = None
    wacc: float | None = None
    terminal_growth: float | None = None
    terminal_exit_ebitda_multiple: float | None = None
    terminal_method: TerminalMethod | None = None
    cash: float | None = None
    debt: float | None = None
    net_debt_override: float | None = None
    investments: float | None = None
    minority_interest: float | None = None
    preferred_equity: float | None = None
    diluted_shares: float | None = None
    current_price: float | None = None
    target_ebit_margin: float | None = None
    nol_balance: float | None = None
    nol_utilization_pct: float | None = None


@dataclass(frozen=True)
class ComparableCompany:
    company_name: str | None
    ticker: str | None
    ev_ntm_revenue: float | None = None
    ev_ntm_ebitda: float | None = None
    pe_ntm: float | None = None
    source: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class SourceReference:
    label: str | None
    url: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class PresentationContent:
    subtitle: str | None = None
    as_of_date: str | None = None
    company_overview: tuple[str, ...] = ()
    current_context: tuple[str, ...] = ()
    investment_highlights: tuple[str, ...] = ()
    valuation_summary: tuple[str, ...] = ()
    catalysts: tuple[str, ...] = ()
    key_risks: tuple[str, ...] = ()
    sources: tuple[SourceReference, ...] = ()


@dataclass(frozen=True)
class ScenarioOverrides:
    description: str | None = None
    forecast: PartialForecastDrivers = field(default_factory=PartialForecastDrivers)
    assumptions: PartialAssumptions = field(default_factory=PartialAssumptions)


@dataclass(frozen=True)
class ValuationInput:
    company_name: str | None
    ticker: str | None
    currency: str | None
    forecast_years: int
    historical: HistoricalFinancials
    forecast: ForecastDrivers
    assumptions: Assumptions
    comps: tuple[ComparableCompany, ...] = ()
    scenarios: dict[str, ScenarioOverrides] = field(default_factory=dict)
    presentation: PresentationContent = field(default_factory=PresentationContent)
    notes: tuple[str, ...] = ()

    @property
    def last_historical_year(self) -> int:
        year = self.historical.latest_common_year("revenue", "ebit")
        if year is None:
            raise ParseError(
                "Historical revenue and EBIT must share at least one fiscal year."
            )
        return year


def parse_input(payload: str | bytes | os.PathLike[str] | Mapping[str, Any]) -> ValuationInput:
    return parse_valuation_json(_load_payload(payload))


def parse_valuation_json(raw: Mapping[str, Any]) -> ValuationInput:
    historical_data = _mapping(raw.get("historical"), "historical")
    historical = HistoricalFinancials(
        revenue=_parse_year_map(historical_data.get("revenue"), "historical.revenue"),
        ebit=_parse_year_map(historical_data.get("ebit"), "historical.ebit"),
        da=_parse_year_map(historical_data.get("da"), "historical.da"),
        capex=_parse_year_map(historical_data.get("capex"), "historical.capex"),
        nwc=_parse_year_map(historical_data.get("nwc"), "historical.nwc"),
    )

    if not historical.revenue:
        raise ParseError(
            "Missing critical field 'historical.revenue'. At least one historical revenue value is required."
        )
    if not historical.ebit:
        raise ParseError(
            "Missing critical field 'historical.ebit'. At least one historical EBIT value is required."
        )
    if historical.latest_common_year("revenue", "ebit") is None:
        raise ParseError(
            "Historical revenue and EBIT must overlap in at least one fiscal year."
        )

    assumptions = _parse_assumptions(
        raw.get("assumptions"),
        "assumptions",
        partial=False,
    )
    if assumptions.wacc <= 0.0:
        raise ParseError(
            "Missing critical field 'assumptions.wacc'. A positive WACC is required."
        )
    if assumptions.diluted_shares <= 0.0:
        raise ParseError(
            "Missing critical field 'assumptions.diluted_shares'. A positive share count is required."
        )

    declared_forecast_years = _coerce_int(raw.get("forecast_years"), "forecast_years") or 5
    forecast_data = _mapping(raw.get("forecast"), "forecast")
    forecast_years = _resolve_forecast_years(
        historical=historical,
        forecast_data=forecast_data,
        declared_forecast_years=declared_forecast_years,
    )
    forecast = _build_forecast(
        forecast_data=forecast_data,
        historical=historical,
        assumptions=assumptions,
        years=forecast_years,
    )

    scenarios_data = _mapping(raw.get("scenarios"), "scenarios")
    scenarios = {
        name: _parse_scenario(scenarios_data.get(name), f"scenarios.{name}")
        for name in ("bull", "bear")
    }

    return ValuationInput(
        company_name=_coerce_text(raw.get("company_name")),
        ticker=_coerce_text(raw.get("ticker")),
        currency=_coerce_text(raw.get("currency")),
        forecast_years=len(forecast_years),
        historical=historical,
        forecast=forecast,
        assumptions=assumptions,
        comps=_parse_comps(raw.get("comps")),
        scenarios=scenarios,
        presentation=_parse_presentation(raw.get("presentation")),
        notes=_parse_string_list(raw.get("notes"), "notes"),
    )


def merge_forecast(base: ForecastDrivers, override: PartialForecastDrivers) -> ForecastDrivers:
    years = sorted(
        set(base.years)
        | set(override.revenue_growth)
        | set(override.ebit_margin)
        | set(override.da_pct_sales)
        | set(override.capex_pct_sales)
        | set(override.nwc_pct_sales)
    )
    return ForecastDrivers(
        revenue_growth=_merge_year_series(base.revenue_growth, override.revenue_growth, years),
        ebit_margin=_merge_year_series(base.ebit_margin, override.ebit_margin, years),
        da_pct_sales=_merge_year_series(base.da_pct_sales, override.da_pct_sales, years),
        capex_pct_sales=_merge_year_series(base.capex_pct_sales, override.capex_pct_sales, years),
        nwc_pct_sales=_merge_year_series(base.nwc_pct_sales, override.nwc_pct_sales, years),
    )


def merge_assumptions(base: Assumptions, override: PartialAssumptions) -> Assumptions:
    updates: dict[str, Any] = {}
    for field_name in override.__dataclass_fields__:
        value = getattr(override, field_name)
        if value is not None:
            updates[field_name] = value
    return replace(base, **updates)


def _load_payload(payload: str | bytes | os.PathLike[str] | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload

    if isinstance(payload, bytes):
        try:
            return _load_payload(payload.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ParseError("Input bytes are not valid UTF-8 JSON.") from exc

    if isinstance(payload, os.PathLike):
        return _load_json_file(Path(payload))

    text = payload.strip()
    if text.startswith("{"):
        return _load_json_text(text)

    candidate = Path(payload)
    if candidate.exists():
        return _load_json_file(candidate)

    raise ParseError("Input must be a JSON object, JSON text, or a valid file path.")


def _load_json_file(path: Path) -> Mapping[str, Any]:
    try:
        return _load_json_text(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ParseError(f"Unable to read JSON file '{path}'.") from exc


def _load_json_text(text: str) -> Mapping[str, Any]:
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"Invalid JSON: {exc.msg}") from exc
    if not isinstance(loaded, Mapping):
        raise ParseError("Top-level JSON value must be an object.")
    return loaded


def _parse_assumptions(
    raw_value: Any,
    path: str,
    *,
    partial: bool,
) -> Assumptions | PartialAssumptions:
    data = _mapping(raw_value, path)
    method = _parse_terminal_method(data.get("terminal_method"), f"{path}.terminal_method", partial=partial)

    common = {
        "tax_rate": _coerce_number(data.get("tax_rate"), f"{path}.tax_rate", ratio=True),
        "wacc": _coerce_number(data.get("wacc"), f"{path}.wacc", ratio=True),
        "terminal_growth": _coerce_number(
            data.get("terminal_growth"),
            f"{path}.terminal_growth",
            ratio=True,
        ),
        "terminal_exit_ebitda_multiple": _coerce_number(
            data.get("terminal_exit_ebitda_multiple"),
            f"{path}.terminal_exit_ebitda_multiple",
        ),
        "terminal_method": method,
        "cash": _coerce_number(data.get("cash"), f"{path}.cash"),
        "debt": _coerce_number(data.get("debt"), f"{path}.debt"),
        "net_debt_override": _coerce_number(data.get("net_debt_override"), f"{path}.net_debt_override"),
        "investments": _coerce_number(data.get("investments"), f"{path}.investments"),
        "minority_interest": _coerce_number(data.get("minority_interest"), f"{path}.minority_interest"),
        "preferred_equity": _coerce_number(data.get("preferred_equity"), f"{path}.preferred_equity"),
        "diluted_shares": _coerce_number(data.get("diluted_shares"), f"{path}.diluted_shares"),
        "current_price": _coerce_number(data.get("current_price"), f"{path}.current_price"),
        "target_ebit_margin": _coerce_number(
            data.get("target_ebit_margin"),
            f"{path}.target_ebit_margin",
            ratio=True,
        ),
        "nol_balance": _coerce_number(data.get("nol_balance"), f"{path}.nol_balance"),
        "nol_utilization_pct": _coerce_number(
            data.get("nol_utilization_pct"),
            f"{path}.nol_utilization_pct",
            ratio=True,
        ),
    }

    if partial:
        return PartialAssumptions(**common)

    return Assumptions(
        tax_rate=common["tax_rate"] or 0.0,
        wacc=common["wacc"] or 0.0,
        terminal_growth=common["terminal_growth"] or 0.0,
        terminal_exit_ebitda_multiple=common["terminal_exit_ebitda_multiple"],
        terminal_method=cast(TerminalMethod, common["terminal_method"] or "average"),
        cash=common["cash"] or 0.0,
        debt=common["debt"] or 0.0,
        net_debt_override=common["net_debt_override"],
        investments=common["investments"] or 0.0,
        minority_interest=common["minority_interest"] or 0.0,
        preferred_equity=common["preferred_equity"] or 0.0,
        diluted_shares=common["diluted_shares"] or 0.0,
        current_price=common["current_price"],
        target_ebit_margin=common["target_ebit_margin"],
        nol_balance=common["nol_balance"] or 0.0,
        nol_utilization_pct=common["nol_utilization_pct"] or 0.8,
    )


def _build_forecast(
    forecast_data: Mapping[str, Any],
    historical: HistoricalFinancials,
    assumptions: Assumptions,
    years: list[int],
) -> ForecastDrivers:
    revenue_growth_raw = _parse_year_map(
        forecast_data.get("revenue_growth"),
        "forecast.revenue_growth",
        ratio=True,
    )
    ebit_margin_raw = _parse_year_map(
        forecast_data.get("ebit_margin"),
        "forecast.ebit_margin",
        ratio=True,
    )
    da_pct_sales_raw = _parse_year_map(
        forecast_data.get("da_pct_sales"),
        "forecast.da_pct_sales",
        ratio=True,
    )
    capex_pct_sales_raw = _parse_year_map(
        forecast_data.get("capex_pct_sales"),
        "forecast.capex_pct_sales",
        ratio=True,
    )
    nwc_pct_sales_raw = _parse_year_map(
        forecast_data.get("nwc_pct_sales"),
        "forecast.nwc_pct_sales",
        ratio=True,
    )

    latest_margin = _latest_ratio(historical.ebit, historical.revenue)
    if not ebit_margin_raw and assumptions.target_ebit_margin is not None:
        margin_default = assumptions.target_ebit_margin
    else:
        margin_default = latest_margin

    return ForecastDrivers(
        revenue_growth=_fill_series(revenue_growth_raw, years, default=0.0),
        ebit_margin=_fill_series(ebit_margin_raw, years, default=margin_default or 0.0),
        da_pct_sales=_fill_series(
            da_pct_sales_raw,
            years,
            default=_latest_ratio(historical.da, historical.revenue) or 0.0,
        ),
        capex_pct_sales=_fill_series(
            capex_pct_sales_raw,
            years,
            default=_latest_ratio(historical.capex, historical.revenue) or 0.0,
        ),
        nwc_pct_sales=_fill_series(
            nwc_pct_sales_raw,
            years,
            default=_latest_ratio(historical.nwc, historical.revenue) or 0.0,
        ),
    )


def _parse_scenario(raw_value: Any, path: str) -> ScenarioOverrides:
    data = _mapping(raw_value, path)
    forecast_data = _mapping(data.get("forecast"), f"{path}.forecast")
    return ScenarioOverrides(
        description=_coerce_text(data.get("description")),
        forecast=PartialForecastDrivers(
            revenue_growth=_parse_year_map(
                forecast_data.get("revenue_growth"),
                f"{path}.forecast.revenue_growth",
                ratio=True,
            ),
            ebit_margin=_parse_year_map(
                forecast_data.get("ebit_margin"),
                f"{path}.forecast.ebit_margin",
                ratio=True,
            ),
            da_pct_sales=_parse_year_map(
                forecast_data.get("da_pct_sales"),
                f"{path}.forecast.da_pct_sales",
                ratio=True,
            ),
            capex_pct_sales=_parse_year_map(
                forecast_data.get("capex_pct_sales"),
                f"{path}.forecast.capex_pct_sales",
                ratio=True,
            ),
            nwc_pct_sales=_parse_year_map(
                forecast_data.get("nwc_pct_sales"),
                f"{path}.forecast.nwc_pct_sales",
                ratio=True,
            ),
        ),
        assumptions=cast(
            PartialAssumptions,
            _parse_assumptions(data.get("assumptions"), f"{path}.assumptions", partial=True),
        ),
    )


def _parse_comps(raw_value: Any) -> tuple[ComparableCompany, ...]:
    items = _list(raw_value, "comps")
    comps: list[ComparableCompany] = []
    for index, item in enumerate(items):
        data = _mapping(item, f"comps[{index}]")
        comps.append(
            ComparableCompany(
                company_name=_coerce_text(data.get("company_name")),
                ticker=_coerce_text(data.get("ticker")),
                ev_ntm_revenue=_coerce_number(
                    data.get("ev_ntm_revenue"),
                    f"comps[{index}].ev_ntm_revenue",
                ),
                ev_ntm_ebitda=_coerce_number(
                    data.get("ev_ntm_ebitda"),
                    f"comps[{index}].ev_ntm_ebitda",
                ),
                pe_ntm=_coerce_number(data.get("pe_ntm"), f"comps[{index}].pe_ntm"),
                source=_coerce_url_text(data.get("source")),
                notes=_coerce_text(data.get("notes")),
            )
        )
    return tuple(comps)


def _parse_presentation(raw_value: Any) -> PresentationContent:
    data = _mapping(raw_value, "presentation")
    sources_data = _list(data.get("sources"), "presentation.sources")
    sources: list[SourceReference] = []
    for index, item in enumerate(sources_data):
        source = _mapping(item, f"presentation.sources[{index}]")
        sources.append(
            SourceReference(
                label=_coerce_text(source.get("label")),
                url=_coerce_url_text(source.get("url")),
                notes=_coerce_text(source.get("notes")),
            )
        )
    return PresentationContent(
        subtitle=_coerce_text(data.get("subtitle")),
        as_of_date=_coerce_text(data.get("as_of_date")),
        company_overview=_parse_string_list(
            data.get("company_overview"),
            "presentation.company_overview",
        ),
        current_context=_parse_string_list(
            data.get("current_context"),
            "presentation.current_context",
        ),
        investment_highlights=_parse_string_list(
            data.get("investment_highlights"),
            "presentation.investment_highlights",
        ),
        valuation_summary=_parse_string_list(
            data.get("valuation_summary"),
            "presentation.valuation_summary",
        ),
        catalysts=_parse_string_list(data.get("catalysts"), "presentation.catalysts"),
        key_risks=_parse_string_list(data.get("key_risks"), "presentation.key_risks"),
        sources=tuple(sources),
    )


def _parse_string_list(raw_value: Any, path: str) -> tuple[str, ...]:
    values = _list(raw_value, path)
    items: list[str] = []
    for index, value in enumerate(values):
        text = _coerce_text(value)
        if text is None:
            continue
        items.append(text)
    return tuple(items)


def _resolve_forecast_years(
    historical: HistoricalFinancials,
    forecast_data: Mapping[str, Any],
    declared_forecast_years: int,
) -> list[int]:
    years: set[int] = set()
    for field_name in _FORECAST_FIELDS:
        raw_series = forecast_data.get(field_name)
        if not isinstance(raw_series, Mapping):
            continue
        years.update(_coerce_year(year_key, f"forecast.{field_name}") for year_key in raw_series)

    if years:
        return sorted(years)

    last_year = historical.latest_common_year("revenue", "ebit")
    if last_year is None:
        raise ParseError("Unable to infer forecast years without overlapping historical revenue and EBIT.")
    count = max(declared_forecast_years, 1)
    return [last_year + offset for offset in range(1, count + 1)]


def _fill_series(series: Mapping[int, float], years: list[int], *, default: float) -> dict[int, float]:
    result: dict[int, float] = {}
    last_value = default
    for year in years:
        value = series.get(year, last_value)
        result[year] = float(value)
        last_value = result[year]
    return result


def _merge_year_series(
    base: Mapping[int, float],
    override: Mapping[int, float],
    years: list[int],
) -> dict[int, float]:
    if not years:
        return {}

    base_years = sorted(base)
    if not base_years:
        raise ParseError("Cannot merge forecast overrides without a base forecast series.")

    result: dict[int, float] = {}
    last_base_value = base[base_years[0]]
    for year in years:
        if year in base:
            last_base_value = base[year]
        result[year] = float(override.get(year, last_base_value))
    return result


def _latest_ratio(numerator_map: Mapping[int, float], denominator_map: Mapping[int, float]) -> float | None:
    common_years = sorted(set(numerator_map) & set(denominator_map))
    if not common_years:
        return None
    year = common_years[-1]
    denominator = denominator_map[year]
    if denominator == 0:
        return None
    return numerator_map[year] / denominator


def _parse_year_map(raw_value: Any, path: str, *, ratio: bool = False) -> dict[int, float]:
    data = _mapping(raw_value, path)
    parsed: dict[int, float] = {}
    for year_key, raw_number in data.items():
        number = _coerce_number(raw_number, f"{path}.{year_key}", ratio=ratio)
        if number is None:
            continue
        parsed[_coerce_year(year_key, path)] = number
    return dict(sorted(parsed.items()))


def _coerce_number(value: Any, path: str, *, ratio: bool = False) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ParseError(f"{path} must be numeric, not boolean.")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if text.lower() in _MISSING_STRINGS:
            return None

        negative = text.startswith("(") and text.endswith(")")
        if negative:
            text = text[1:-1].strip()

        if text.endswith("%"):
            ratio = True
            text = text[:-1].strip()

        if text.lower().endswith("x"):
            text = text[:-1].strip()

        text = text.replace(",", "").replace("_", "")
        text = text.translate({ord(marker): None for marker in _CURRENCY_MARKERS})

        try:
            number = float(text)
        except ValueError as exc:
            raise ParseError(f"{path} could not be parsed as a number: {value!r}") from exc

        if negative:
            number = -number
    else:
        raise ParseError(f"{path} must be numeric or null.")

    if ratio and abs(number) > 1.0 and abs(number) <= 100.0:
        number /= 100.0
    return number


def _coerce_int(value: Any, path: str) -> int | None:
    number = _coerce_number(value, path)
    if number is None:
        return None
    return int(round(number))


def _coerce_year(value: Any, path: str) -> int:
    if isinstance(value, bool):
        raise ParseError(f"{path} year key must be an integer year.")
    if isinstance(value, int):
        year = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ParseError(f"{path} contains an empty year key.")
        if not text.lstrip("-").isdigit():
            raise ParseError(f"{path} year key must be numeric, got {value!r}.")
        year = int(text)
    else:
        raise ParseError(f"{path} year key must be a string or integer.")

    if year < 1900 or year > 2500:
        raise ParseError(f"{path} year key {year} is out of range.")
    return year


def _parse_terminal_method(value: Any, path: str, *, partial: bool) -> TerminalMethod | None:
    if value is None:
        return None if partial else "average"

    method = str(value).strip().lower()
    if method in _MISSING_STRINGS:
        return None if partial else "average"
    if method not in _TERMINAL_METHODS:
        choices = ", ".join(sorted(_TERMINAL_METHODS))
        raise ParseError(f"{path} must be one of: {choices}.")
    return cast(TerminalMethod, method)


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return str(value)


def _coerce_url_text(value: Any) -> str | None:
    text = _coerce_text(value)
    if text is None:
        return None

    match = _MARKDOWN_LINK_RE.match(text)
    if match:
        return match.group("url")
    return text


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ParseError(f"{path} must be an object.")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ParseError(f"{path} must be an array.")
    return value
