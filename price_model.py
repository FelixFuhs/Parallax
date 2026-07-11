from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf


def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


@dataclass(frozen=True)
class PricePanels:
    raw_close: pd.DataFrame
    adjusted_close: pd.DataFrame


def empty_price_panels(tickers: Sequence[str]) -> PricePanels:
    normalized_tickers = list(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers if ticker))
    empty = pd.DataFrame(index=pd.DatetimeIndex([], name="date"), columns=normalized_tickers, dtype=float)
    return PricePanels(raw_close=empty.copy(), adjusted_close=empty.copy())


def _normalize_price_frame(frame: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.index = pd.to_datetime(normalized.index)
    if getattr(normalized.index, "tz", None) is not None:
        normalized.index = normalized.index.tz_localize(None)
    normalized.index.name = "date"
    normalized = normalized.sort_index()
    normalized.columns = [normalize_ticker(str(column)) for column in normalized.columns]
    return normalized.reindex(columns=list(tickers)).astype(float)


def extract_price_panels(history: pd.DataFrame, tickers: Sequence[str]) -> PricePanels:
    normalized_tickers = list(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers if ticker))
    empty = empty_price_panels(normalized_tickers)
    if history.empty:
        return empty

    if isinstance(history.columns, pd.MultiIndex):
        top_level = history.columns.get_level_values(0)
        if "Close" in top_level:
            raw_close = _normalize_price_frame(history["Close"], normalized_tickers)
        else:
            raw_close = empty.raw_close.copy()

        if "Adj Close" in top_level:
            adjusted_close = _normalize_price_frame(history["Adj Close"], normalized_tickers)
        elif "Close" in top_level:
            adjusted_close = _normalize_price_frame(history["Close"], normalized_tickers)
        else:
            adjusted_close = empty.adjusted_close.copy()
        return PricePanels(raw_close=raw_close, adjusted_close=adjusted_close)

    raw_key = "Close" if "Close" in history.columns else None
    adjusted_key = "Adj Close" if "Adj Close" in history.columns else raw_key
    if raw_key is None and adjusted_key is None:
        return empty

    raw_close = history[[raw_key]].copy() if raw_key is not None else empty.raw_close.copy()
    adjusted_close = history[[adjusted_key]].copy() if adjusted_key is not None else empty.adjusted_close.copy()
    if normalized_tickers and raw_key is not None:
        raw_close.columns = [normalized_tickers[0]]
    if normalized_tickers and adjusted_key is not None:
        adjusted_close.columns = [normalized_tickers[0]]
    return PricePanels(
        raw_close=_normalize_price_frame(raw_close, normalized_tickers),
        adjusted_close=_normalize_price_frame(adjusted_close, normalized_tickers),
    )


def load_cached_price_panels(tickers: Sequence[str], cache_path: Path | None) -> PricePanels:
    normalized_tickers = list(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers if ticker))
    if cache_path is None or not cache_path.exists():
        return empty_price_panels(normalized_tickers)

    cached = pd.read_parquet(cache_path)
    if not (
        isinstance(cached.columns, pd.MultiIndex)
        and {"raw_close", "adjusted_close"}.issubset(set(cached.columns.get_level_values(0)))
    ):
        return empty_price_panels(normalized_tickers)

    return PricePanels(
        raw_close=_normalize_price_frame(cached["raw_close"], normalized_tickers),
        adjusted_close=_normalize_price_frame(cached["adjusted_close"], normalized_tickers),
    )


def load_price_panels(
    tickers: Sequence[str],
    *,
    start: str,
    end: str,
    cache_path: Path | None = None,
    refresh: bool = False,
    chunk_size: int = 50,
) -> PricePanels:
    normalized_tickers = list(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers if ticker))
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    cached_fallback = empty_price_panels(normalized_tickers)

    if cache_path is not None and cache_path.exists() and not refresh:
        cached_panels = load_cached_price_panels(normalized_tickers, cache_path)
        cached_fallback = cached_panels
        if (
            set(normalized_tickers).issubset(cached_panels.raw_close.columns)
            and not cached_panels.raw_close.empty
            and cached_panels.raw_close.index.min() <= start_ts + pd.Timedelta(days=7)
            and cached_panels.raw_close.index.max() >= end_ts - pd.Timedelta(days=7)
        ):
            return PricePanels(
                raw_close=cached_panels.raw_close.loc[:, normalized_tickers],
                adjusted_close=cached_panels.adjusted_close.loc[:, normalized_tickers],
            )

    panel_parts: list[PricePanels] = []
    download_end = (end_ts + pd.Timedelta(days=5)).date().isoformat()
    for offset in range(0, len(normalized_tickers), max(1, int(chunk_size))):
        chunk = normalized_tickers[offset : offset + max(1, int(chunk_size))]
        history = yf.download(
            chunk,
            start=start,
            end=download_end,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        panel_parts.append(extract_price_panels(history, chunk))

    panels = (
        PricePanels(
            raw_close=pd.concat([part.raw_close for part in panel_parts], axis=1).reindex(columns=normalized_tickers),
            adjusted_close=pd.concat([part.adjusted_close for part in panel_parts], axis=1).reindex(columns=normalized_tickers),
        )
        if panel_parts
        else empty_price_panels(normalized_tickers)
    )
    has_download_data = (not panels.raw_close.empty) and bool(panels.raw_close.notna().any().any())
    if not has_download_data and not cached_fallback.raw_close.empty and cached_fallback.raw_close.notna().any().any():
        return PricePanels(
            raw_close=cached_fallback.raw_close.reindex(columns=normalized_tickers),
            adjusted_close=cached_fallback.adjusted_close.reindex(columns=normalized_tickers),
        )
    if cache_path is not None:
        if has_download_data:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            combined = pd.concat(
                {"raw_close": panels.raw_close, "adjusted_close": panels.adjusted_close},
                axis=1,
            )
            combined.to_parquet(cache_path)
    return panels


def price_on_or_before(series: pd.Series, target: pd.Timestamp) -> float | None:
    eligible = series[series.index <= target]
    if eligible.empty:
        return None
    return float(eligible.iloc[-1])


def price_on_or_after(series: pd.Series, target: pd.Timestamp) -> float | None:
    eligible = series[series.index > target]
    if eligible.empty:
        return None
    return float(eligible.iloc[0])
