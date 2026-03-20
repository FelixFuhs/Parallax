from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import requests
import yfinance as yf


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_MAX_REQUESTS_PER_SECOND = 10.0
SEC_TIMEOUT_SECONDS = 30
PLACEHOLDER_SEC_NAME = "Parallax Research"
PLACEHOLDER_SEC_EMAIL = "sec-contact@example.com"
DEFAULT_SEC_NAME = os.getenv("SEC_USER_AGENT_NAME", PLACEHOLDER_SEC_NAME)
DEFAULT_SEC_EMAIL = os.getenv("SEC_USER_AGENT_EMAIL", PLACEHOLDER_SEC_EMAIL)
ANNUAL_FORMS = {
    "10-K",
    "10-K/A",
    "10-KT",
    "10-KT/A",
    "20-F",
    "20-F/A",
    "40-F",
    "40-F/A",
}
PRICE_WINDOWS = {
    "price_return_1m": pd.DateOffset(months=1),
    "price_return_3m": pd.DateOffset(months=3),
    "price_return_6m": pd.DateOffset(months=6),
    "price_return_12m": pd.DateOffset(months=12),
}
RAW_FIELD_NAMES = (
    "revenue",
    "gross_profit",
    "net_income",
    "total_assets",
    "total_equity",
    "operating_income",
    "total_debt",
    "cash",
    "operating_cash_flow",
    "capex",
    "da",
    "shares_outstanding",
    "current_assets",
    "current_liabilities",
)
DERIVED_FIELD_NAMES = (
    "current_price",
    "price_return_1m",
    "price_return_3m",
    "price_return_6m",
    "price_return_12m",
    "market_cap",
    "free_cash_flow",
    "fcf_yield",
    "roic",
    "roe",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "debt_to_equity",
    "current_ratio",
    "asset_turnover",
    "accruals",
    "capex_intensity",
)
LOGGER = logging.getLogger("edgar")


@dataclass(frozen=True)
class TagCandidate:
    taxonomy: str
    tag: str
    is_duration: bool
    units: tuple[str, ...] = ("USD",)


@dataclass(frozen=True)
class FieldSpec:
    candidates: tuple[TagCandidate, ...]


@dataclass(frozen=True)
class FactRecord:
    field_name: str
    taxonomy: str
    tag: str
    candidate_rank: int
    value: float
    start: date | None
    end: date | None
    filed: date | None
    fy: int | None
    fp: str | None
    form: str | None
    accn: str | None


FIELD_SPECS: dict[str, FieldSpec] = {
    "revenue": FieldSpec(
        candidates=(
            TagCandidate("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", True),
            TagCandidate("us-gaap", "Revenues", True),
            TagCandidate("us-gaap", "SalesRevenueNet", True),
        )
    ),
    "gross_profit": FieldSpec(
        candidates=(TagCandidate("us-gaap", "GrossProfit", True),)
    ),
    "net_income": FieldSpec(
        candidates=(
            TagCandidate("us-gaap", "NetIncomeLoss", True),
            TagCandidate("us-gaap", "ProfitLoss", True),
        )
    ),
    "total_assets": FieldSpec(
        candidates=(TagCandidate("us-gaap", "Assets", False),)
    ),
    "total_equity": FieldSpec(
        candidates=(
            TagCandidate("us-gaap", "StockholdersEquity", False),
            TagCandidate(
                "us-gaap",
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                False,
            ),
        )
    ),
    "operating_income": FieldSpec(
        candidates=(TagCandidate("us-gaap", "OperatingIncomeLoss", True),)
    ),
    "cash": FieldSpec(
        candidates=(TagCandidate("us-gaap", "CashAndCashEquivalentsAtCarryingValue", False),)
    ),
    "operating_cash_flow": FieldSpec(
        candidates=(
            TagCandidate("us-gaap", "NetCashProvidedByOperatingActivities", True),
            TagCandidate("us-gaap", "NetCashProvidedByUsedInOperatingActivities", True),
            TagCandidate(
                "us-gaap",
                "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
                True,
            ),
        )
    ),
    "capex": FieldSpec(
        candidates=(
            TagCandidate("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment", True),
            TagCandidate("us-gaap", "PropertyPlantAndEquipmentAdditions", True),
            TagCandidate("us-gaap", "PaymentsToAcquireProductiveAssets", True),
        )
    ),
    "da": FieldSpec(
        candidates=(
            TagCandidate("us-gaap", "DepreciationDepletionAndAmortization", True),
            TagCandidate("us-gaap", "DepreciationAmortizationAndAccretionNet", True),
            TagCandidate("us-gaap", "DepreciationAndAmortization", True),
        )
    ),
    "shares_outstanding": FieldSpec(
        candidates=(
            TagCandidate("us-gaap", "CommonStockSharesOutstanding", False, ("shares",)),
            TagCandidate("dei", "EntityCommonStockSharesOutstanding", False, ("shares",)),
            TagCandidate("us-gaap", "WeightedAverageNumberOfShareOutstandingsBasic", True, ("shares",)),
            TagCandidate("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic", True, ("shares",)),
            TagCandidate(
                "us-gaap",
                "WeightedAverageNumberOfShareOutstandingsBasicAndDiluted",
                True,
                ("shares",),
            ),
            TagCandidate(
                "us-gaap",
                "WeightedAverageNumberOfSharesOutstandingBasicAndDiluted",
                True,
                ("shares",),
            ),
        )
    ),
    "current_assets": FieldSpec(
        candidates=(TagCandidate("us-gaap", "AssetsCurrent", False),)
    ),
    "current_liabilities": FieldSpec(
        candidates=(TagCandidate("us-gaap", "LiabilitiesCurrent", False),)
    ),
}
TOTAL_DEBT_SPEC = FieldSpec(
    candidates=(
        TagCandidate("us-gaap", "LongTermDebtAndCapitalLeaseObligations", False),
        TagCandidate("us-gaap", "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities", False),
        TagCandidate("us-gaap", "LongTermDebtAndFinanceLeaseObligations", False),
        TagCandidate("us-gaap", "LongTermDebtAndFinanceLeaseObligationsIncludingCurrentMaturities", False),
    )
)
LONG_TERM_DEBT_SPEC = FieldSpec(
    candidates=(
        TagCandidate("us-gaap", "LongTermDebt", False),
        TagCandidate("us-gaap", "LongTermDebtNoncurrent", False),
    )
)
SHORT_TERM_DEBT_SPEC = FieldSpec(
    candidates=(
        TagCandidate("us-gaap", "ShortTermBorrowings", False),
        TagCandidate("us-gaap", "LongTermDebtCurrent", False),
        TagCandidate("us-gaap", "ShortTermBankLoansAndNotesPayable", False),
        TagCandidate("us-gaap", "ShortTermDebt", False),
        TagCandidate("us-gaap", "CommercialPaper", False),
        TagCandidate("us-gaap", "LongTermDebtAndCapitalLeaseObligationsCurrent", False),
        TagCandidate("us-gaap", "LongTermDebtAndFinanceLeaseObligationsCurrent", False),
    )
)
DEPRECIATION_SPEC = FieldSpec(
    candidates=(TagCandidate("us-gaap", "Depreciation", True),)
)
AMORTIZATION_SPEC = FieldSpec(
    candidates=(
        TagCandidate("us-gaap", "AmortizationOfIntangibleAssets", True),
        TagCandidate("us-gaap", "FiniteLivedIntangibleAssetsAmortizationExpense", True),
    )
)


class EdgarError(RuntimeError):
    """Raised when feature extraction cannot proceed."""


class RateLimiter:
    def __init__(self, max_requests_per_second: float) -> None:
        self.min_interval = 1.0 / max_requests_per_second
        self.last_request_at = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self.last_request_at
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_at = time.monotonic()


class SecClient:
    def __init__(self, *, user_agent_name: str, user_agent_email: str) -> None:
        user_agent = f"{user_agent_name} ({user_agent_email})"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "application/json",
                "From": user_agent_email,
            }
        )
        self.rate_limiter = RateLimiter(SEC_MAX_REQUESTS_PER_SECOND)

    def get_json(self, url: str) -> Any:
        last_error: Exception | None = None
        for attempt in range(3):
            self.rate_limiter.wait()
            try:
                response = self.session.get(url, timeout=SEC_TIMEOUT_SECONDS)
                if response.status_code in {429, 500, 502, 503, 504}:
                    response.raise_for_status()
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(1.0 + attempt)
        raise EdgarError(f"Unable to fetch SEC JSON from {url}: {last_error}") from last_error

    def load_company_tickers(self) -> dict[str, dict[str, Any]]:
        payload = self.get_json(SEC_TICKERS_URL)
        if not isinstance(payload, dict):
            raise EdgarError("Unexpected SEC ticker payload shape.")

        by_ticker: dict[str, dict[str, Any]] = {}
        for value in payload.values():
            if not isinstance(value, dict):
                continue
            ticker = normalize_ticker(str(value.get("ticker", "")))
            if not ticker:
                continue
            by_ticker[ticker] = value
        return by_ticker

    def load_company_facts(self, cik: str) -> dict[str, Any]:
        payload = self.get_json(SEC_COMPANYFACTS_URL.format(cik=cik))
        if not isinstance(payload, dict):
            raise EdgarError(f"Unexpected companyfacts payload for CIK {cik}.")
        return payload


def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


def cik_to_str(value: int | str) -> str:
    return f"{int(value):010d}"


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def is_annual_fact(fact: dict[str, Any], *, is_duration: bool) -> bool:
    if fact.get("form") not in ANNUAL_FORMS:
        return False

    end = parse_date(fact.get("end"))
    if end is None:
        return False

    if not is_duration:
        return True

    start = parse_date(fact.get("start"))
    if start is None:
        return False

    duration_days = (end - start).days
    return 300 <= duration_days <= 380


def preferred_units(units: dict[str, list[dict[str, Any]]], priorities: tuple[str, ...]) -> list[str]:
    ordered: list[str] = []
    for priority in priorities:
        if priority in units:
            ordered.append(priority)
    for unit_name in units:
        if unit_name not in ordered:
            ordered.append(unit_name)
    return ordered


def iter_field_facts(company_facts: dict[str, Any], field_name: str, spec: FieldSpec) -> Iterable[FactRecord]:
    facts_root = company_facts.get("facts", {})
    if not isinstance(facts_root, dict):
        return

    for candidate_rank, candidate in enumerate(spec.candidates):
        taxonomy_facts = facts_root.get(candidate.taxonomy, {})
        if not isinstance(taxonomy_facts, dict):
            continue
        tag_payload = taxonomy_facts.get(candidate.tag)
        if not isinstance(tag_payload, dict):
            continue
        units = tag_payload.get("units", {})
        if not isinstance(units, dict):
            continue

        for unit_name in preferred_units(units, candidate.units):
            values = units.get(unit_name)
            if not isinstance(values, list):
                continue

            for raw_fact in values:
                if not isinstance(raw_fact, dict) or not is_annual_fact(
                    raw_fact,
                    is_duration=candidate.is_duration,
                ):
                    continue

                value = coerce_float(raw_fact.get("val"))
                if value is None:
                    continue

                yield FactRecord(
                    field_name=field_name,
                    taxonomy=candidate.taxonomy,
                    tag=candidate.tag,
                    candidate_rank=candidate_rank,
                    value=value,
                    start=parse_date(raw_fact.get("start")),
                    end=parse_date(raw_fact.get("end")),
                    filed=parse_date(raw_fact.get("filed")),
                    fy=int(raw_fact["fy"]) if isinstance(raw_fact.get("fy"), int) else coerce_int(raw_fact.get("fy")),
                    fp=str(raw_fact.get("fp")) if raw_fact.get("fp") else None,
                    form=str(raw_fact.get("form")) if raw_fact.get("form") else None,
                    accn=str(raw_fact.get("accn")) if raw_fact.get("accn") else None,
                )
            if unit_name in candidate.units:
                break


def coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def latest_fact(facts: Sequence[FactRecord]) -> FactRecord | None:
    if not facts:
        return None
    return max(
        facts,
        key=lambda fact: (
            fact.end or date.min,
            fact.filed or date.min,
            -fact.candidate_rank,
        ),
    )


def select_fact(
    facts: Sequence[FactRecord],
    *,
    anchor_end: date | None,
    anchor_accn: str | None,
) -> FactRecord | None:
    if not facts:
        return None

    if anchor_end is None and anchor_accn is None:
        return latest_fact(facts)

    def sort_key(fact: FactRecord) -> tuple[int, int, int, date, date, int]:
        if anchor_end is not None and fact.end is not None:
            distance = -abs((fact.end - anchor_end).days)
        else:
            distance = -10**9
        return (
            1 if anchor_accn and fact.accn == anchor_accn else 0,
            1 if anchor_end and fact.end == anchor_end else 0,
            distance,
            fact.end or date.min,
            fact.filed or date.min,
            -fact.candidate_rank,
        )

    return max(facts, key=sort_key)


def extract_anchor(company_facts: dict[str, Any]) -> FactRecord | None:
    for field_name in ("revenue", "net_income", "total_assets"):
        facts = list(iter_field_facts(company_facts, field_name, FIELD_SPECS[field_name]))
        fact = latest_fact(facts)
        if fact is not None:
            return fact
    return None


def abs_if_present(value: float | None) -> float | None:
    if value is None:
        return None
    return abs(value)


def extract_total_debt(
    company_facts: dict[str, Any],
    *,
    anchor_end: date | None,
    anchor_accn: str | None,
) -> tuple[float | None, FactRecord | None]:
    total_debt_facts = list(iter_field_facts(company_facts, "total_debt", TOTAL_DEBT_SPEC))
    total_debt_fact = select_fact(total_debt_facts, anchor_end=anchor_end, anchor_accn=anchor_accn)
    if total_debt_fact is not None:
        return abs(total_debt_fact.value), total_debt_fact

    long_term_facts = list(iter_field_facts(company_facts, "long_term_debt", LONG_TERM_DEBT_SPEC))
    short_term_facts = list(iter_field_facts(company_facts, "short_term_debt", SHORT_TERM_DEBT_SPEC))
    long_term_fact = select_fact(long_term_facts, anchor_end=anchor_end, anchor_accn=anchor_accn)
    short_term_fact = select_fact(short_term_facts, anchor_end=anchor_end, anchor_accn=anchor_accn)

    if long_term_fact is None and short_term_fact is None:
        return None, None

    total_value = 0.0
    if long_term_fact is not None:
        total_value += abs(long_term_fact.value)
    if short_term_fact is not None:
        total_value += abs(short_term_fact.value)

    return total_value, long_term_fact or short_term_fact


def extract_da(
    company_facts: dict[str, Any],
    *,
    anchor_end: date | None,
    anchor_accn: str | None,
) -> tuple[float | None, FactRecord | None]:
    direct_facts = list(iter_field_facts(company_facts, "da", FIELD_SPECS["da"]))
    direct_fact = select_fact(direct_facts, anchor_end=anchor_end, anchor_accn=anchor_accn)
    if direct_fact is not None:
        return abs(direct_fact.value), direct_fact

    depreciation_facts = list(iter_field_facts(company_facts, "depreciation", DEPRECIATION_SPEC))
    amortization_facts = list(iter_field_facts(company_facts, "amortization", AMORTIZATION_SPEC))
    depreciation_fact = select_fact(
        depreciation_facts,
        anchor_end=anchor_end,
        anchor_accn=anchor_accn,
    )
    amortization_fact = select_fact(
        amortization_facts,
        anchor_end=anchor_end,
        anchor_accn=anchor_accn,
    )

    if depreciation_fact is None and amortization_fact is None:
        return None, None

    total_value = 0.0
    if depreciation_fact is not None:
        total_value += abs(depreciation_fact.value)
    if amortization_fact is not None:
        total_value += abs(amortization_fact.value)
    return total_value, depreciation_fact or amortization_fact


def extract_company_features(
    ticker: str,
    company_record: dict[str, Any],
    company_facts: dict[str, Any],
    price_features: dict[str, float | None],
) -> dict[str, Any]:
    record = empty_feature_record(ticker)
    record["company_name"] = company_facts.get("entityName") or company_record.get("title")
    record["cik"] = cik_to_str(company_record["cik_str"])

    anchor = extract_anchor(company_facts)
    anchor_end = anchor.end if anchor else None
    anchor_accn = anchor.accn if anchor else None

    selected_facts: dict[str, FactRecord | None] = {}
    values: dict[str, float | None] = {}
    missing_fields: list[str] = []

    for field_name, spec in FIELD_SPECS.items():
        if field_name == "da":
            value, fact = extract_da(
                company_facts,
                anchor_end=anchor_end,
                anchor_accn=anchor_accn,
            )
        else:
            facts = list(iter_field_facts(company_facts, field_name, spec))
            fact = select_fact(facts, anchor_end=anchor_end, anchor_accn=anchor_accn)
            value = fact.value if fact is not None else None
            if field_name == "capex":
                value = abs_if_present(value)
        selected_facts[field_name] = fact
        values[field_name] = value
        if value is None:
            missing_fields.append(field_name)

    total_debt, debt_fact = extract_total_debt(
        company_facts,
        anchor_end=anchor_end,
        anchor_accn=anchor_accn,
    )
    values["total_debt"] = total_debt
    selected_facts["total_debt"] = debt_fact
    if total_debt is None:
        missing_fields.append("total_debt")

    best_meta_fact = anchor or latest_fact(
        [fact for fact in selected_facts.values() if fact is not None]
    )
    if best_meta_fact is not None:
        record["fiscal_year"] = best_meta_fact.fy
        record["filing_form"] = best_meta_fact.form
        record["filing_accession"] = best_meta_fact.accn
        record["period_end"] = best_meta_fact.end.isoformat() if best_meta_fact.end else None

    for field_name in RAW_FIELD_NAMES:
        record[field_name] = values.get(field_name)

    record.update(price_features)
    compute_derived_fields(record)

    if missing_fields:
        LOGGER.warning("%s missing EDGAR fields: %s", ticker, ", ".join(sorted(missing_fields)))

    return record


def empty_feature_record(ticker: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "ticker": ticker,
        "company_name": None,
        "cik": None,
        "fiscal_year": None,
        "filing_form": None,
        "filing_accession": None,
        "period_end": None,
        "error": None,
    }
    for field_name in RAW_FIELD_NAMES:
        record[field_name] = None
    for field_name in DERIVED_FIELD_NAMES:
        record[field_name] = None
    return record


def safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def compute_derived_fields(record: dict[str, Any]) -> None:
    operating_cash_flow = coerce_float(record.get("operating_cash_flow"))
    capex = abs_if_present(coerce_float(record.get("capex")))
    current_price = coerce_float(record.get("current_price"))
    shares_outstanding = coerce_float(record.get("shares_outstanding"))
    operating_income = coerce_float(record.get("operating_income"))
    total_equity = coerce_float(record.get("total_equity"))
    total_debt = coerce_float(record.get("total_debt"))
    cash = coerce_float(record.get("cash"))
    net_income = coerce_float(record.get("net_income"))
    revenue = coerce_float(record.get("revenue"))
    total_assets = coerce_float(record.get("total_assets"))
    gross_profit = coerce_float(record.get("gross_profit"))
    current_assets = coerce_float(record.get("current_assets"))
    current_liabilities = coerce_float(record.get("current_liabilities"))

    market_cap = None
    if current_price is not None and shares_outstanding not in (None, 0):
        market_cap = current_price * shares_outstanding

    free_cash_flow = None
    if operating_cash_flow is not None and capex is not None:
        free_cash_flow = operating_cash_flow - capex

    roic_denominator = None
    if total_equity is not None and total_debt is not None and cash is not None:
        roic_denominator = total_equity + total_debt - cash

    record["capex"] = capex
    record["market_cap"] = market_cap
    record["free_cash_flow"] = free_cash_flow
    record["fcf_yield"] = safe_divide(free_cash_flow, market_cap)
    record["roic"] = safe_divide(
        operating_income * (1.0 - 0.21) if operating_income is not None else None,
        roic_denominator,
    )
    record["roe"] = safe_divide(net_income, total_equity)
    record["gross_margin"] = safe_divide(gross_profit, revenue)
    record["operating_margin"] = safe_divide(operating_income, revenue)
    record["net_margin"] = safe_divide(net_income, revenue)
    record["debt_to_equity"] = safe_divide(total_debt, total_equity)
    record["current_ratio"] = safe_divide(current_assets, current_liabilities)
    record["asset_turnover"] = safe_divide(revenue, total_assets)
    record["accruals"] = safe_divide(
        net_income - operating_cash_flow
        if net_income is not None and operating_cash_flow is not None
        else None,
        total_assets,
    )
    record["capex_intensity"] = safe_divide(capex, revenue)


def price_on_or_before(series: pd.Series, target: pd.Timestamp) -> float | None:
    eligible = series[series.index <= target]
    if eligible.empty:
        return None
    return float(eligible.iloc[-1])


def extract_close_frame(history: pd.DataFrame, tickers: Sequence[str]) -> dict[str, pd.Series]:
    if history.empty:
        return {}

    close_by_ticker: dict[str, pd.Series] = {}
    if isinstance(history.columns, pd.MultiIndex):
        if "Close" not in history.columns.get_level_values(0):
            return {}
        close_frame = history["Close"]
        for ticker in tickers:
            if ticker not in close_frame.columns:
                continue
            series = close_frame[ticker].dropna()
            if not series.empty:
                close_by_ticker[ticker] = series
        return close_by_ticker

    if "Close" not in history.columns:
        return {}
    series = history["Close"].dropna()
    if series.empty:
        return {}
    close_by_ticker[tickers[0]] = series
    return close_by_ticker


def fetch_price_features(tickers: Sequence[str]) -> dict[str, dict[str, float | None]]:
    normalized_tickers = [normalize_ticker(ticker) for ticker in tickers]
    unique_tickers = list(dict.fromkeys(normalized_tickers))
    features = {
        ticker: {
            "current_price": None,
            "price_return_1m": None,
            "price_return_3m": None,
            "price_return_6m": None,
            "price_return_12m": None,
        }
        for ticker in unique_tickers
    }
    if not unique_tickers:
        return features

    start = (datetime.now(timezone.utc) - pd.Timedelta(days=400)).date().isoformat()
    try:
        history = yf.download(
            unique_tickers,
            start=start,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        LOGGER.warning("Unable to fetch yfinance history: %s", exc)
        return features

    close_by_ticker = extract_close_frame(history, unique_tickers)
    for ticker in unique_tickers:
        series = close_by_ticker.get(ticker)
        if series is None or series.empty:
            LOGGER.warning("%s missing yfinance price history", ticker)
            continue

        index = pd.to_datetime(series.index)
        if getattr(index, "tz", None) is not None:
            index = index.tz_localize(None)
        series.index = index
        latest_price = float(series.iloc[-1])
        latest_date = pd.Timestamp(series.index[-1])

        features[ticker]["current_price"] = latest_price
        for field_name, offset in PRICE_WINDOWS.items():
            baseline = price_on_or_before(series, latest_date - offset)
            features[ticker][field_name] = (
                (latest_price / baseline) - 1.0 if baseline not in (None, 0.0) else None
            )

    return features


def load_tickers_from_file(path: str | os.PathLike[str]) -> list[str]:
    tickers: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tickers.append(normalize_ticker(stripped))
    return tickers


def collect_features(
    tickers: Sequence[str],
    *,
    sec_client: SecClient,
) -> dict[str, dict[str, Any]]:
    normalized_tickers = list(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers if ticker))
    if not normalized_tickers:
        raise EdgarError("Provide at least one ticker or pass --file.")

    company_index = sec_client.load_company_tickers()
    price_features = fetch_price_features(normalized_tickers)
    output: dict[str, dict[str, Any]] = {}

    for ticker in normalized_tickers:
        company_record = company_index.get(ticker)
        if company_record is None:
            record = empty_feature_record(ticker)
            record["error"] = "Ticker not found in SEC company_tickers.json"
            output[ticker] = record
            LOGGER.warning("%s not found in SEC company_tickers.json", ticker)
            continue

        record = empty_feature_record(ticker)
        record.update(price_features.get(ticker, {}))
        record["company_name"] = company_record.get("title")
        record["cik"] = cik_to_str(company_record["cik_str"])

        try:
            company_facts = sec_client.load_company_facts(record["cik"])
            record = extract_company_features(
                ticker,
                company_record,
                company_facts,
                price_features.get(ticker, {}),
            )
        except EdgarError as exc:
            record["error"] = str(exc)
            LOGGER.warning("%s failed: %s", ticker, exc)

        output[ticker] = record

    return output


def write_output(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch SEC EDGAR companyfacts and price features for one or more tickers."
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Ticker symbols for a quick debug run, for example: python edgar.py AAPL MSFT",
    )
    parser.add_argument(
        "--file",
        help="Text file containing one ticker per line. Blank lines and lines starting with # are ignored.",
    )
    parser.add_argument(
        "--output",
        help="Optional output JSON path. If omitted, the JSON payload is printed to stdout.",
    )
    parser.add_argument(
        "--user-agent-name",
        default=DEFAULT_SEC_NAME,
        help="Name to include in the SEC User-Agent header.",
    )
    parser.add_argument(
        "--user-agent-email",
        default=DEFAULT_SEC_EMAIL,
        help="Email to include in the SEC User-Agent header.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    tickers: list[str] = []
    if args.file:
        try:
            tickers.extend(load_tickers_from_file(args.file))
        except OSError as exc:
            print(f"Error: unable to read ticker file '{args.file}': {exc}", file=sys.stderr)
            return 1
    tickers.extend(normalize_ticker(ticker) for ticker in args.tickers)

    if not tickers:
        print("Error: provide ticker arguments or pass --file.", file=sys.stderr)
        return 1

    if (
        args.user_agent_email == PLACEHOLDER_SEC_EMAIL
        and args.user_agent_name == PLACEHOLDER_SEC_NAME
    ):
        LOGGER.warning(
            "Using default SEC contact info. Override with --user-agent-name/--user-agent-email or set "
            "SEC_USER_AGENT_NAME and SEC_USER_AGENT_EMAIL for production runs."
        )

    try:
        payload = collect_features(
            tickers,
            sec_client=SecClient(
                user_agent_name=args.user_agent_name,
                user_agent_email=args.user_agent_email,
            ),
        )
    except EdgarError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        try:
            write_output(args.output, payload)
        except OSError as exc:
            print(f"Error: unable to write output file '{args.output}': {exc}", file=sys.stderr)
            return 1
        print(f"Saved features to {args.output}", file=sys.stderr)
        return 0

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
