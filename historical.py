import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from edgar import (
    DEFAULT_SEC_EMAIL,
    DEFAULT_SEC_NAME,
    FIELD_SPECS,
    GROSS_PROFIT_FALLBACK_SPECS,
    LONG_TERM_DEBT_SPEC,
    RAW_FIELD_NAMES,
    SHORT_TERM_DEBT_SPEC,
    TOTAL_DEBT_SPEC,
    FactRecord,
    SecClient,
    cik_to_str,
    compute_derived_fields,
    empty_feature_record,
    iter_field_facts,
    normalize_ticker,
    price_on_or_before,
)
from price_model import PricePanels, extract_price_panels, load_price_panels

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
EDGAR_CACHE_DIR = DATA_DIR / "edgar_cache"
COMPANY_TICKERS_CACHE_PATH = EDGAR_CACHE_DIR / "company_tickers.json"
PRICE_CACHE_PATH = DATA_DIR / "price_cache.parquet"
PRICE_PANEL_CACHE_PATH = DATA_DIR / "price_panels.parquet"
FEATURE_ORDER = (
    "fcf_to_ev",
    "gross_profitability_assets",
    "asset_growth_1y",
    "cash_earnings_gap",
    "momentum_12_1",
)
HISTORICAL_FEATURE_COLUMNS = (
    "current_price",
    "raw_close_price",
    "adjusted_close_price",
    "price_return_1m",
    "price_return_3m",
    "price_return_6m",
    "price_return_12m",
    "momentum_12_1",
    "market_cap",
    "free_cash_flow",
    "fcf_yield",
    "fcf_to_ev",
    "roic",
    "roe",
    "gross_margin",
    "gross_profitability_assets",
    "operating_margin",
    "net_margin",
    "book_to_market",
    "debt_to_equity",
    "current_ratio",
    "asset_turnover",
    "asset_growth_1y",
    "cash_earnings_gap",
    "accruals",
    "capex_intensity",
)
LOGGER = logging.getLogger("historical")


@dataclass(frozen=True)
class FilingMeta:
    key: str
    accn: str | None
    filed: date
    fy: int | None
    form: str | None
    period_end: date | None


@dataclass(frozen=True)
class FilingSnapshot:
    accn: str | None
    filed: date
    fy: int | None
    form: str | None
    period_end: date | None
    raw_fields: dict[str, float | None]


@dataclass(frozen=True)
class TickerHistory:
    ticker: str
    company_name: str | None
    cik: str
    filings: tuple[FilingSnapshot, ...]


def ensure_cache_dirs() -> None:
    EDGAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PRICE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_company_index(
    *,
    sec_client: SecClient | None = None,
    refresh: bool = False,
) -> dict[str, dict[str, Any]]:
    ensure_cache_dirs()
    if COMPANY_TICKERS_CACHE_PATH.exists() and not refresh:
        payload = _read_json(COMPANY_TICKERS_CACHE_PATH)
        if isinstance(payload, dict):
            return payload

    if sec_client is None:
        sec_client = SecClient(
            user_agent_name=DEFAULT_SEC_NAME,
            user_agent_email=DEFAULT_SEC_EMAIL,
        )

    payload = sec_client.load_company_tickers()
    _write_json(COMPANY_TICKERS_CACHE_PATH, payload)
    return payload


def load_company_facts_cached(
    cik: str,
    *,
    sec_client: SecClient,
    refresh: bool = False,
) -> dict[str, Any]:
    ensure_cache_dirs()
    cache_path = EDGAR_CACHE_DIR / f"{cik}.json"
    if cache_path.exists() and not refresh:
        payload = _read_json(cache_path)
        if isinstance(payload, dict):
            return payload

    payload = sec_client.load_company_facts(cik)
    _write_json(cache_path, payload)
    return payload


def extract_adjusted_close_frame(history: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    return extract_price_panels(history, tickers).adjusted_close


def load_price_panel_history(
    tickers: Sequence[str],
    *,
    start: str,
    end: str,
    refresh: bool = False,
) -> PricePanels:
    return load_price_panels(
        tickers,
        start=start,
        end=end,
        cache_path=PRICE_PANEL_CACHE_PATH,
        refresh=refresh,
    )


def load_price_history(
    tickers: Sequence[str],
    *,
    start: str,
    end: str,
    refresh: bool = False,
) -> pd.DataFrame:
    ensure_cache_dirs()
    normalized_tickers = list(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers if ticker))
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    if PRICE_CACHE_PATH.exists() and not refresh:
        cached = pd.read_parquet(PRICE_CACHE_PATH)
        cached.index = pd.to_datetime(cached.index)
        if getattr(cached.index, "tz", None) is not None:
            cached.index = cached.index.tz_localize(None)
        cached = cached.sort_index()
        if (
            set(normalized_tickers).issubset(cached.columns)
            and not cached.empty
            and cached.index.min() <= start_ts
            and cached.index.max() >= end_ts
        ):
            return cached.loc[:, normalized_tickers]

    LOGGER.info(
        "Downloading raw and adjusted close history for %d tickers from %s to %s",
        len(normalized_tickers),
        start,
        end,
    )
    history = yf.download(
        normalized_tickers,
        start=start,
        end=(end_ts + pd.Timedelta(days=5)).date().isoformat(),
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    close_frame = extract_price_panels(history, normalized_tickers).adjusted_close
    missing_tickers = [
        ticker for ticker in normalized_tickers if ticker not in close_frame.columns
    ]
    if missing_tickers:
        LOGGER.warning(
            "Price history download omitted %d ticker(s): %s",
            len(missing_tickers),
            ", ".join(missing_tickers),
        )
    close_frame = close_frame.reindex(columns=normalized_tickers)
    close_frame.to_parquet(PRICE_CACHE_PATH)
    return close_frame.loc[:, normalized_tickers]


def _filing_key(fact: FactRecord) -> str:
    if fact.accn:
        return fact.accn
    return "::".join(
        [
            fact.filed.isoformat() if fact.filed else "",
            fact.end.isoformat() if fact.end else "",
            fact.form or "",
            str(fact.fy) if fact.fy is not None else "",
        ]
    )


def _iter_relevant_facts(company_facts: dict[str, Any]) -> dict[str, list[FactRecord]]:
    relevant_specs = {
        "revenue": FIELD_SPECS["revenue"],
        "gross_profit": FIELD_SPECS["gross_profit"],
        "net_income": FIELD_SPECS["net_income"],
        "total_assets": FIELD_SPECS["total_assets"],
        "total_equity": FIELD_SPECS["total_equity"],
        "operating_income": FIELD_SPECS["operating_income"],
        "cash": FIELD_SPECS["cash"],
        "operating_cash_flow": FIELD_SPECS["operating_cash_flow"],
        "capex": FIELD_SPECS["capex"],
        "da": FIELD_SPECS["da"],
        "shares_outstanding": FIELD_SPECS["shares_outstanding"],
        "current_assets": FIELD_SPECS["current_assets"],
        "current_liabilities": FIELD_SPECS["current_liabilities"],
    }
    facts_by_name = {
        field_name: list(iter_field_facts(company_facts, field_name, spec))
        for field_name, spec in relevant_specs.items()
    }
    facts_by_name["total_debt_direct"] = list(
        iter_field_facts(company_facts, "total_debt", TOTAL_DEBT_SPEC)
    )
    facts_by_name["long_term_debt"] = list(
        iter_field_facts(company_facts, "long_term_debt", LONG_TERM_DEBT_SPEC)
    )
    facts_by_name["short_term_debt"] = list(
        iter_field_facts(company_facts, "short_term_debt", SHORT_TERM_DEBT_SPEC)
    )
    for index, spec in enumerate(GROSS_PROFIT_FALLBACK_SPECS):
        facts_by_name[f"gross_profit_cost_{index}"] = list(
            iter_field_facts(company_facts, f"gross_profit_cost_{index}", spec)
        )
    return facts_by_name


def _build_filing_metas(facts_by_name: dict[str, list[FactRecord]]) -> tuple[FilingMeta, ...]:
    filings: dict[str, FilingMeta] = {}
    for facts in facts_by_name.values():
        for fact in facts:
            if fact.filed is None:
                continue
            key = _filing_key(fact)
            filing = filings.get(key)
            candidate = FilingMeta(
                key=key,
                accn=fact.accn,
                filed=fact.filed,
                fy=fact.fy,
                form=fact.form,
                period_end=fact.end,
            )
            if filing is None:
                filings[key] = candidate
                continue
            period_end = max(
                [value for value in (filing.period_end, candidate.period_end) if value is not None],
                default=None,
            )
            filings[key] = FilingMeta(
                key=key,
                accn=filing.accn or candidate.accn,
                filed=max(filing.filed, candidate.filed),
                fy=candidate.fy if candidate.fy is not None else filing.fy,
                form=candidate.form or filing.form,
                period_end=period_end,
            )
    return tuple(
        sorted(
            filings.values(),
            key=lambda filing: (filing.filed, filing.period_end or date.min, filing.accn or filing.key),
        )
    )


def _facts_for_filing(facts: Iterable[FactRecord], filing: FilingMeta) -> list[FactRecord]:
    return [fact for fact in facts if _filing_key(fact) == filing.key]


def _select_fact_for_filing(
    facts: Iterable[FactRecord],
    filing: FilingMeta,
) -> FactRecord | None:
    candidates = _facts_for_filing(facts, filing)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda fact: (
            1 if filing.period_end and fact.end == filing.period_end else 0,
            fact.end or date.min,
            -(fact.candidate_rank),
        ),
    )


def _extract_total_debt_for_filing(
    filing: FilingMeta,
    *,
    direct_facts: list[FactRecord],
    long_term_facts: list[FactRecord],
    short_term_facts: list[FactRecord],
) -> float | None:
    direct_fact = _select_fact_for_filing(direct_facts, filing)
    if direct_fact is not None:
        return abs(direct_fact.value)

    long_term_fact = _select_fact_for_filing(long_term_facts, filing)
    short_term_fact = _select_fact_for_filing(short_term_facts, filing)
    if long_term_fact is None and short_term_fact is None:
        return None

    total_value = 0.0
    if long_term_fact is not None:
        total_value += abs(long_term_fact.value)
    if short_term_fact is not None:
        total_value += abs(short_term_fact.value)
    return total_value


def _extract_gross_profit_for_filing(
    filing: FilingMeta,
    *,
    direct_fact: FactRecord | None,
    revenue_value: float | None,
    fallback_cost_facts: Sequence[list[FactRecord]],
) -> float | None:
    if direct_fact is not None:
        return direct_fact.value
    if revenue_value is None:
        return None

    for cost_facts in fallback_cost_facts:
        cost_fact = _select_fact_for_filing(cost_facts, filing)
        if cost_fact is not None:
            return revenue_value - abs(cost_fact.value)
    return None


def build_ticker_history(
    ticker: str,
    company_record: dict[str, Any],
    company_facts: dict[str, Any],
) -> TickerHistory:
    facts_by_name = _iter_relevant_facts(company_facts)
    filings = _build_filing_metas(facts_by_name)
    snapshots: list[FilingSnapshot] = []

    for filing in filings:
        revenue_fact = _select_fact_for_filing(facts_by_name["revenue"], filing)
        gross_profit_fact = _select_fact_for_filing(facts_by_name["gross_profit"], filing)
        net_income_fact = _select_fact_for_filing(facts_by_name["net_income"], filing)
        total_assets_fact = _select_fact_for_filing(facts_by_name["total_assets"], filing)
        total_equity_fact = _select_fact_for_filing(facts_by_name["total_equity"], filing)
        operating_income_fact = _select_fact_for_filing(facts_by_name["operating_income"], filing)
        cash_fact = _select_fact_for_filing(facts_by_name["cash"], filing)
        operating_cash_flow_fact = _select_fact_for_filing(
            facts_by_name["operating_cash_flow"],
            filing,
        )
        capex_fact = _select_fact_for_filing(facts_by_name["capex"], filing)
        da_fact = _select_fact_for_filing(facts_by_name["da"], filing)
        shares_outstanding_fact = _select_fact_for_filing(
            facts_by_name["shares_outstanding"],
            filing,
        )
        current_assets_fact = _select_fact_for_filing(facts_by_name["current_assets"], filing)
        current_liabilities_fact = _select_fact_for_filing(
            facts_by_name["current_liabilities"],
            filing,
        )

        revenue_value = revenue_fact.value if revenue_fact is not None else None
        gross_profit_value = _extract_gross_profit_for_filing(
            filing,
            direct_fact=gross_profit_fact,
            revenue_value=revenue_value,
            fallback_cost_facts=[
                facts_by_name["gross_profit_cost_0"],
                facts_by_name["gross_profit_cost_1"],
                facts_by_name["gross_profit_cost_2"],
            ],
        )
        total_debt_value = _extract_total_debt_for_filing(
            filing,
            direct_facts=facts_by_name["total_debt_direct"],
            long_term_facts=facts_by_name["long_term_debt"],
            short_term_facts=facts_by_name["short_term_debt"],
        )

        raw_fields = {field_name: None for field_name in RAW_FIELD_NAMES}
        raw_fields["revenue"] = revenue_value
        raw_fields["gross_profit"] = gross_profit_value
        raw_fields["net_income"] = net_income_fact.value if net_income_fact is not None else None
        raw_fields["total_assets"] = total_assets_fact.value if total_assets_fact is not None else None
        raw_fields["total_equity"] = total_equity_fact.value if total_equity_fact is not None else None
        raw_fields["operating_income"] = (
            operating_income_fact.value if operating_income_fact is not None else None
        )
        raw_fields["cash"] = cash_fact.value if cash_fact is not None else None
        raw_fields["operating_cash_flow"] = (
            operating_cash_flow_fact.value if operating_cash_flow_fact is not None else None
        )
        raw_fields["capex"] = abs(capex_fact.value) if capex_fact is not None else None
        raw_fields["da"] = abs(da_fact.value) if da_fact is not None else None
        raw_fields["shares_outstanding"] = (
            shares_outstanding_fact.value if shares_outstanding_fact is not None else None
        )
        raw_fields["current_assets"] = (
            current_assets_fact.value if current_assets_fact is not None else None
        )
        raw_fields["current_liabilities"] = (
            current_liabilities_fact.value if current_liabilities_fact is not None else None
        )
        raw_fields["total_debt"] = total_debt_value

        snapshots.append(
            FilingSnapshot(
                accn=filing.accn,
                filed=filing.filed,
                fy=filing.fy,
                form=filing.form,
                period_end=filing.period_end,
                raw_fields=raw_fields,
            )
        )

    company_name = company_facts.get("entityName") or company_record.get("title")
    cik = cik_to_str(company_record["cik_str"])
    return TickerHistory(
        ticker=normalize_ticker(ticker),
        company_name=str(company_name) if company_name else None,
        cik=cik,
        filings=tuple(snapshots),
    )


def load_ticker_histories(
    tickers: Sequence[str],
    *,
    sec_client: SecClient | None = None,
    refresh_sec_cache: bool = False,
) -> dict[str, TickerHistory]:
    normalized_tickers = list(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers if ticker))
    if sec_client is None:
        sec_client = SecClient(
            user_agent_name=DEFAULT_SEC_NAME,
            user_agent_email=DEFAULT_SEC_EMAIL,
        )

    company_index = load_company_index(sec_client=sec_client, refresh=refresh_sec_cache)
    histories: dict[str, TickerHistory] = {}
    total = len(normalized_tickers)

    for index, ticker in enumerate(normalized_tickers, start=1):
        company_record = company_index.get(ticker)
        if company_record is None:
            LOGGER.warning("%s missing from SEC company index", ticker)
            histories[ticker] = TickerHistory(
                ticker=ticker,
                company_name=None,
                cik="",
                filings=(),
            )
            continue

        cik = cik_to_str(company_record["cik_str"])
        company_facts = load_company_facts_cached(
            cik,
            sec_client=sec_client,
            refresh=refresh_sec_cache,
        )
        history = build_ticker_history(ticker, company_record, company_facts)
        histories[ticker] = history
        LOGGER.info(
            "Loaded %s historical annual filings for %s (%d/%d)",
            len(history.filings),
            ticker,
            index,
            total,
        )

    return histories


def latest_snapshot_before(
    history: TickerHistory,
    rebalance_date: pd.Timestamp,
) -> tuple[int | None, FilingSnapshot | None]:
    cutoff = rebalance_date.date()
    for index in range(len(history.filings) - 1, -1, -1):
        snapshot = history.filings[index]
        if snapshot.filed < cutoff:
            return index, snapshot
    return None, None


def previous_snapshot_for_growth(
    history: TickerHistory,
    snapshot_index: int,
) -> FilingSnapshot | None:
    current_snapshot = history.filings[snapshot_index]
    current_period_end = current_snapshot.period_end
    current_fy = current_snapshot.fy

    for index in range(snapshot_index - 1, -1, -1):
        candidate = history.filings[index]
        if current_period_end is not None and candidate.period_end is not None:
            if candidate.period_end < current_period_end:
                return candidate
            continue
        if current_fy is not None and candidate.fy is not None:
            if candidate.fy < current_fy:
                return candidate
            continue
        return candidate
    return None


def price_on_or_after(series: pd.Series, target: pd.Timestamp) -> float | None:
    eligible = series[series.index > target]
    if eligible.empty:
        return None
    return float(eligible.iloc[0])


def build_price_features_for_rebalance(
    raw_series: pd.Series,
    rebalance_date: pd.Timestamp,
    adjusted_series: pd.Series | None = None,
) -> dict[str, float | None]:
    if adjusted_series is None:
        adjusted_series = raw_series
    current_price = price_on_or_before(raw_series, rebalance_date)
    adjusted_close_price = price_on_or_before(adjusted_series, rebalance_date)
    baseline_1m = price_on_or_before(adjusted_series, rebalance_date - pd.DateOffset(months=1))
    baseline_3m = price_on_or_before(adjusted_series, rebalance_date - pd.DateOffset(months=3))
    baseline_6m = price_on_or_before(adjusted_series, rebalance_date - pd.DateOffset(months=6))
    baseline_12m = price_on_or_before(adjusted_series, rebalance_date - pd.DateOffset(months=12))
    return {
        "current_price": current_price,
        "raw_close_price": current_price,
        "adjusted_close_price": adjusted_close_price,
        "price_return_1m": (
            (adjusted_close_price / baseline_1m) - 1.0
            if adjusted_close_price not in (None, 0.0) and baseline_1m not in (None, 0.0)
            else None
        ),
        "price_return_3m": (
            (adjusted_close_price / baseline_3m) - 1.0
            if adjusted_close_price not in (None, 0.0) and baseline_3m not in (None, 0.0)
            else None
        ),
        "price_return_6m": (
            (adjusted_close_price / baseline_6m) - 1.0
            if adjusted_close_price not in (None, 0.0) and baseline_6m not in (None, 0.0)
            else None
        ),
        "price_return_12m": (
            (adjusted_close_price / baseline_12m) - 1.0
            if adjusted_close_price not in (None, 0.0) and baseline_12m not in (None, 0.0)
            else None
        ),
    }


def build_feature_row(
    history: TickerHistory,
    snapshot: FilingSnapshot,
    previous_snapshot: FilingSnapshot | None,
    *,
    rebalance_date: pd.Timestamp,
    price_features: dict[str, float | None],
) -> dict[str, Any]:
    record = empty_feature_record(history.ticker)
    record["company_name"] = history.company_name
    record["cik"] = history.cik
    record["fiscal_year"] = snapshot.fy
    record["filing_form"] = snapshot.form
    record["filing_accession"] = snapshot.accn
    record["period_end"] = snapshot.period_end.isoformat() if snapshot.period_end else None
    for field_name, value in snapshot.raw_fields.items():
        record[field_name] = value
    record.update(price_features)
    compute_derived_fields(
        record,
        current_total_assets_for_growth=snapshot.raw_fields.get("total_assets"),
        prior_total_assets=(
            previous_snapshot.raw_fields.get("total_assets") if previous_snapshot is not None else None
        ),
    )

    row: dict[str, Any] = {
        "ticker": history.ticker,
        "company_name": history.company_name,
        "cik": history.cik,
        "rebalance_date": rebalance_date.normalize(),
        "filing_date": pd.Timestamp(snapshot.filed),
        "period_end": pd.Timestamp(snapshot.period_end) if snapshot.period_end else pd.NaT,
        "feature_null_count": 0,
    }
    for feature_name in FEATURE_ORDER:
        row[feature_name] = record.get(feature_name)
    for feature_name in HISTORICAL_FEATURE_COLUMNS:
        row[feature_name] = record.get(feature_name)
    row["feature_null_count"] = int(sum(pd.isna(row[feature_name]) for feature_name in FEATURE_ORDER))
    return row


def build_point_in_time_feature_matrix(
    histories: dict[str, TickerHistory],
    price_frame: pd.DataFrame | PricePanels,
    rebalance_date: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    normalized_rebalance = pd.Timestamp(rebalance_date).normalize()
    if isinstance(price_frame, PricePanels):
        raw_price_frame = price_frame.raw_close
        adjusted_price_frame = price_frame.adjusted_close
    else:
        raw_price_frame = price_frame
        adjusted_price_frame = price_frame

    for ticker, history in histories.items():
        if ticker not in raw_price_frame.columns:
            continue

        raw_series = raw_price_frame[ticker].dropna()
        adjusted_series = (
            adjusted_price_frame[ticker].dropna()
            if ticker in adjusted_price_frame.columns
            else raw_series
        )
        if raw_series.empty or adjusted_series.empty:
            continue

        price_features = build_price_features_for_rebalance(
            raw_series,
            normalized_rebalance,
            adjusted_series=adjusted_series,
        )
        if price_features["current_price"] is None:
            continue

        snapshot_index, snapshot = latest_snapshot_before(history, normalized_rebalance)
        if snapshot is None or snapshot_index is None:
            continue

        previous_snapshot = previous_snapshot_for_growth(history, snapshot_index)
        rows.append(
            build_feature_row(
                history,
                snapshot,
                previous_snapshot,
                rebalance_date=normalized_rebalance,
                price_features=price_features,
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "company_name",
                "cik",
                "rebalance_date",
                "filing_date",
                "period_end",
                "feature_null_count",
                *FEATURE_ORDER,
                *[feature for feature in HISTORICAL_FEATURE_COLUMNS if feature not in FEATURE_ORDER],
            ]
        ).set_index(pd.Index([], name="ticker"))

    frame = pd.DataFrame(rows).set_index("ticker").sort_index()
    return frame
