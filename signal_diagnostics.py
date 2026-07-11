from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def decompose_signal(
    frame: pd.DataFrame,
    score_column: str,
    *,
    sector_column: str = "sector",
) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    score = pd.to_numeric(frame[score_column], errors="coerce").astype(float)
    output["global_score"] = score

    if sector_column not in frame.columns:
        output["within_sector_score"] = np.nan
        output["sector_score"] = np.nan
        output["sector_status"] = "blocked_missing_sector"
        return output

    if frame[sector_column].nunique(dropna=True) < 2:
        output["within_sector_score"] = np.nan
        output["sector_score"] = np.nan
        output["sector_status"] = "blocked_insufficient_sector_coverage"
        return output

    sector_means = score.groupby(frame[sector_column], dropna=False).transform("mean")
    output["within_sector_score"] = score - sector_means
    output["sector_score"] = sector_means
    output["sector_status"] = "available"
    return output


def safe_spearman_rank_ic(scores: pd.Series, returns: pd.Series) -> float | None:
    aligned = pd.concat({"score": scores, "return": returns}, axis=1).dropna()
    if len(aligned) < 2:
        return None
    if aligned["score"].nunique() < 2 or aligned["return"].nunique() < 2:
        return None
    statistic = spearmanr(aligned["score"], aligned["return"]).statistic
    if statistic is None or np.isnan(statistic):
        return None
    return float(statistic)


def _rank_ic_row(
    *,
    signal_name: str,
    horizon: str,
    date_value: object,
    decomposition: str,
    scores: pd.Series,
    returns: pd.Series,
    sector: pd.Series | None = None,
) -> dict[str, object]:
    rank_ic = safe_spearman_rank_ic(scores, returns)
    clean = pd.concat({"score": scores, "return": returns}, axis=1).dropna()
    row: dict[str, object] = {
        "date": date_value,
        "signal": signal_name,
        "horizon": horizon,
        "decomposition": decomposition,
        "n": int(len(clean)),
        "rank_ic": rank_ic,
    }
    if sector is not None and not clean.empty:
        row["sector_count"] = int(sector.reindex(clean.index).nunique(dropna=True))
    return row


def rank_ic_diagnostics(
    frame: pd.DataFrame,
    *,
    date_column: str,
    signal_columns: Sequence[str],
    return_columns: Sequence[str],
    sector_column: str = "sector",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_value, date_frame in frame.groupby(date_column, dropna=False):
        sector = date_frame[sector_column] if sector_column in date_frame.columns else None
        for signal_name in signal_columns:
            if signal_name not in date_frame.columns:
                continue
            decomposed = decompose_signal(date_frame, signal_name, sector_column=sector_column)
            sector_status = (
                str(decomposed["sector_status"].iloc[0])
                if "sector_status" in decomposed.columns and not decomposed.empty
                else "available"
            )
            for return_column in return_columns:
                if return_column not in date_frame.columns:
                    continue
                returns = pd.to_numeric(date_frame[return_column], errors="coerce")
                rows.append(
                    _rank_ic_row(
                        signal_name=signal_name,
                        horizon=return_column,
                        date_value=date_value,
                        decomposition="global",
                        scores=decomposed["global_score"],
                        returns=returns,
                        sector=sector,
                    )
                )
                rows[-1]["sector_status"] = sector_status
                rows.append(
                    _rank_ic_row(
                        signal_name=signal_name,
                        horizon=return_column,
                        date_value=date_value,
                        decomposition="within_sector",
                        scores=decomposed["within_sector_score"],
                        returns=returns,
                        sector=sector,
                    )
                )
                rows[-1]["sector_status"] = sector_status
                if sector is not None and sector_status == "available":
                    sector_level = pd.DataFrame(
                        {
                            "score": decomposed["sector_score"],
                            "return": returns,
                            "sector": sector,
                        }
                    ).dropna()
                    sector_level = sector_level.groupby("sector", dropna=True).mean(numeric_only=True)
                    across_row = _rank_ic_row(
                        signal_name=signal_name,
                        horizon=return_column,
                        date_value=date_value,
                        decomposition="across_sector",
                        scores=sector_level["score"],
                        returns=sector_level["return"],
                    )
                else:
                    across_row = {
                        "date": date_value,
                        "signal": signal_name,
                        "horizon": return_column,
                        "decomposition": "across_sector",
                        "n": 0,
                        "rank_ic": None,
                    }
                rows.append(across_row)
                rows[-1]["sector_status"] = sector_status
    return pd.DataFrame(rows)


def rank_ic_by_sector(
    frame: pd.DataFrame,
    *,
    date_column: str,
    signal_columns: Sequence[str],
    return_columns: Sequence[str],
    sector_column: str = "sector",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    columns = ["date", "sector", "signal", "horizon", "n", "rank_ic"]
    if sector_column not in frame.columns:
        return pd.DataFrame(columns=columns)

    for (date_value, sector_value), group in frame.groupby([date_column, sector_column], dropna=False):
        for signal_name in signal_columns:
            if signal_name not in group.columns:
                continue
            scores = pd.to_numeric(group[signal_name], errors="coerce")
            for return_column in return_columns:
                if return_column not in group.columns:
                    continue
                returns = pd.to_numeric(group[return_column], errors="coerce")
                rows.append(
                    {
                        "date": date_value,
                        "sector": None if pd.isna(sector_value) else str(sector_value),
                        "signal": signal_name,
                        "horizon": return_column,
                        "n": int(pd.concat({"score": scores, "return": returns}, axis=1).dropna().shape[0]),
                        "rank_ic": safe_spearman_rank_ic(scores, returns),
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def rank_ic_coverage(
    frame: pd.DataFrame,
    *,
    date_column: str,
    signal_columns: Sequence[str],
    return_columns: Sequence[str],
    sector_column: str = "sector",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    columns = [
        "date",
        "signal",
        "horizon",
        "universe_n",
        "score_non_null",
        "return_non_null",
        "paired_n",
        "paired_coverage",
        "sector_count",
    ]
    for date_value, date_frame in frame.groupby(date_column, dropna=False):
        universe_n = int(len(date_frame))
        sector_count = int(date_frame[sector_column].nunique(dropna=True)) if sector_column in date_frame.columns else 0
        for signal_name in signal_columns:
            if signal_name not in date_frame.columns:
                continue
            scores = pd.to_numeric(date_frame[signal_name], errors="coerce")
            score_non_null = int(scores.notna().sum())
            for return_column in return_columns:
                if return_column not in date_frame.columns:
                    continue
                returns = pd.to_numeric(date_frame[return_column], errors="coerce")
                paired = pd.concat({"score": scores, "return": returns}, axis=1).dropna()
                rows.append(
                    {
                        "date": date_value,
                        "signal": signal_name,
                        "horizon": return_column,
                        "universe_n": universe_n,
                        "score_non_null": score_non_null,
                        "return_non_null": int(returns.notna().sum()),
                        "paired_n": int(len(paired)),
                        "paired_coverage": float(len(paired) / universe_n) if universe_n else 0.0,
                        "sector_count": sector_count,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def newey_west_tstat(values: Iterable[float], *, max_lag: int | None = None) -> float | None:
    array = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    n = array.size
    if n < 2:
        return None
    demeaned = array - array.mean()
    if max_lag is None:
        max_lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    gamma0 = float(np.dot(demeaned, demeaned) / n)
    variance = gamma0
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        variance += 2.0 * weight * gamma
    if variance <= 0.0:
        return None
    standard_error = np.sqrt(variance / n)
    if standard_error == 0.0:
        return None
    return float(array.mean() / standard_error)


def summarize_rank_ic(ic_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if ic_frame.empty:
        return pd.DataFrame(
            columns=[
                "signal",
                "horizon",
                "decomposition",
                "months",
                "mean_ic",
                "median_ic",
                "ic_std",
                "newey_west_tstat",
                "positive_ic_hit_rate",
            ]
        )

    for keys, group in ic_frame.groupby(["signal", "horizon", "decomposition"], dropna=False):
        clean = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
        signal, horizon, decomposition = keys
        rows.append(
            {
                "signal": signal,
                "horizon": horizon,
                "decomposition": decomposition,
                "months": int(len(clean)),
                "mean_ic": float(clean.mean()) if not clean.empty else None,
                "median_ic": float(clean.median()) if not clean.empty else None,
                "ic_std": float(clean.std(ddof=0)) if not clean.empty else None,
                "newey_west_tstat": newey_west_tstat(clean),
                "positive_ic_hit_rate": float((clean > 0.0).mean()) if not clean.empty else None,
            }
        )
    return pd.DataFrame(rows)


SIGNAL_COMPARISON_COLUMNS = [
    "signal",
    "signal_label",
    "horizon",
    "global_mean_ic",
    "sector_neutral_mean_ic",
    "across_sector_mean_ic",
    "global_newey_west_tstat",
    "sector_neutral_newey_west_tstat",
    "across_sector_newey_west_tstat",
    "global_months",
    "sector_neutral_months",
    "across_sector_months",
    "global_positive_ic_hit_rate",
    "sector_neutral_positive_ic_hit_rate",
    "across_sector_positive_ic_hit_rate",
    "comparison_status",
    "missing_decompositions",
]


def build_signal_comparison_table(
    rank_ic_summary: pd.DataFrame,
    *,
    signal_labels: Mapping[str, str],
    horizons: Sequence[str] = ("return_1m",),
    minimum_months: int = 12,
) -> pd.DataFrame:
    if rank_ic_summary.empty:
        rows = [
            {
                "signal": signal,
                "signal_label": label,
                "horizon": horizon,
                "comparison_status": "missing_signal",
                "missing_decompositions": "global,sector_neutral,across_sector",
            }
            for signal, label in signal_labels.items()
            for horizon in horizons
        ]
        return pd.DataFrame(rows).reindex(columns=SIGNAL_COMPARISON_COLUMNS)

    decomposition_map = {
        "global": "global",
        "within_sector": "sector_neutral",
        "across_sector": "across_sector",
    }
    rows: list[dict[str, object]] = []
    for signal, label in signal_labels.items():
        for horizon in horizons:
            subset = rank_ic_summary[
                (rank_ic_summary["signal"] == signal)
                & (rank_ic_summary["horizon"] == horizon)
            ]
            row: dict[str, object] = {
                "signal": signal,
                "signal_label": label,
                "horizon": horizon,
            }
            missing_decompositions: list[str] = []
            observed_months: list[int] = []
            for decomposition, prefix in decomposition_map.items():
                metric_row = subset.loc[subset["decomposition"] == decomposition]
                if metric_row.empty:
                    missing_decompositions.append(prefix)
                    row[f"{prefix}_mean_ic"] = None
                    row[f"{prefix}_newey_west_tstat"] = None
                    row[f"{prefix}_months"] = 0
                    row[f"{prefix}_positive_ic_hit_rate"] = None
                    continue
                first = metric_row.iloc[0]
                months = int(first["months"]) if pd.notna(first.get("months")) else 0
                observed_months.append(months)
                row[f"{prefix}_mean_ic"] = first.get("mean_ic")
                row[f"{prefix}_newey_west_tstat"] = first.get("newey_west_tstat")
                row[f"{prefix}_months"] = months
                row[f"{prefix}_positive_ic_hit_rate"] = first.get("positive_ic_hit_rate")

            if subset.empty:
                status = "missing_signal"
            elif missing_decompositions:
                status = "blocked_missing_decomposition"
            elif observed_months and min(observed_months) < minimum_months:
                status = "diagnostic_insufficient_history"
            else:
                status = "available"
            row["comparison_status"] = status
            row["missing_decompositions"] = ",".join(missing_decompositions)
            rows.append(row)
    return pd.DataFrame(rows).reindex(columns=SIGNAL_COMPARISON_COLUMNS)


def summarize_rank_ic_by_year(ic_frame: pd.DataFrame) -> pd.DataFrame:
    if ic_frame.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "signal",
                "horizon",
                "decomposition",
                "months",
                "mean_ic",
                "median_ic",
                "positive_ic_hit_rate",
            ]
        )

    frame = ic_frame.copy()
    frame["year"] = pd.to_datetime(frame["date"], errors="coerce").dt.year
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(["year", "signal", "horizon", "decomposition"], dropna=False):
        year, signal, horizon, decomposition = keys
        clean = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
        rows.append(
            {
                "year": None if pd.isna(year) else int(year),
                "signal": signal,
                "horizon": horizon,
                "decomposition": decomposition,
                "months": int(len(clean)),
                "mean_ic": float(clean.mean()) if not clean.empty else None,
                "median_ic": float(clean.median()) if not clean.empty else None,
                "positive_ic_hit_rate": float((clean > 0.0).mean()) if not clean.empty else None,
            }
        )
    return pd.DataFrame(rows)


def summarize_rank_ic_by_sector(sector_ic_frame: pd.DataFrame) -> pd.DataFrame:
    if sector_ic_frame.empty:
        return pd.DataFrame(
            columns=[
                "sector",
                "signal",
                "horizon",
                "months",
                "mean_ic",
                "median_ic",
                "positive_ic_hit_rate",
                "mean_n",
            ]
        )

    rows: list[dict[str, object]] = []
    for keys, group in sector_ic_frame.groupby(["sector", "signal", "horizon"], dropna=False):
        sector, signal, horizon = keys
        clean = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
        rows.append(
            {
                "sector": None if pd.isna(sector) else str(sector),
                "signal": signal,
                "horizon": horizon,
                "months": int(len(clean)),
                "mean_ic": float(clean.mean()) if not clean.empty else None,
                "median_ic": float(clean.median()) if not clean.empty else None,
                "positive_ic_hit_rate": float((clean > 0.0).mean()) if not clean.empty else None,
                "mean_n": float(pd.to_numeric(group["n"], errors="coerce").mean()) if "n" in group else None,
            }
        )
    return pd.DataFrame(rows)
