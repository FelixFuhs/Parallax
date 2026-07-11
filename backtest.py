import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import historical
from edgar import DEFAULT_SEC_EMAIL, DEFAULT_SEC_NAME, SecClient, normalize_ticker

ROOT = Path(__file__).resolve().parent
TICKERS_PATH = ROOT / "tickers.txt"
FEATURE_SPEC_PATH = ROOT / "models" / "feature_spec.json"
FROZEN_MODEL_PATH = ROOT / "models" / "frozen_xgb_regressor.json"
DEFAULT_MODEL_PATH = FROZEN_MODEL_PATH
PLOTS_DIR = ROOT / "plots"
RESULTS_DIR = ROOT / "results"
SUMMARY_PATH = RESULTS_DIR / "backtest_summary.json"
UNIVERSE_PLOT_PATH = PLOTS_DIR / "universe_size.png"
CUMULATIVE_PLOT_PATH = PLOTS_DIR / "backtest_cumulative.png"
ANNUAL_RETURNS_PLOT_PATH = PLOTS_DIR / "backtest_annual_returns.png"
FEATURE_ORDER = (
    "fcf_to_ev",
    "gross_profitability_assets",
    "asset_growth_1y",
    "cash_earnings_gap",
    "momentum_12_1",
)
LOGGER = logging.getLogger("backtest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Parallax historical backcaster.")
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Frozen XGBoost regressor to score with.",
    )
    parser.add_argument(
        "--tickers-file",
        default=str(TICKERS_PATH),
        help="Ticker universe file.",
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
    parser.add_argument("--refresh-sec-cache", action="store_true")
    parser.add_argument("--refresh-price-cache", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def load_feature_order(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    feature_order = payload.get("feature_order")
    if not isinstance(feature_order, list) or feature_order != list(FEATURE_ORDER):
        raise ValueError("Feature spec did not match the frozen feature order.")
    return feature_order


def load_tickers(path: Path) -> list[str]:
    tickers: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tickers.append(normalize_ticker(stripped))
    return tickers


def load_model(path: Path) -> XGBRegressor:
    model = XGBRegressor()
    model.load_model(path)
    return model


def build_rebalance_dates(
    price_frame: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
) -> list[pd.Timestamp]:
    if price_frame.empty:
        return []

    calendar = pd.DatetimeIndex(price_frame.index).sort_values()
    calendar = calendar[(calendar >= pd.Timestamp(f"{start_year}-01-01")) & (calendar <= pd.Timestamp(f"{end_year}-12-31"))]
    if calendar.empty:
        return []

    grouped = pd.Series(calendar, index=calendar).groupby(calendar.to_period("M"))
    return [pd.Timestamp(group.iloc[-1]).normalize() for _, group in grouped]


def determine_bucket_labels(count: int) -> tuple[int, list[str]]:
    if count < 3:
        return 0, []
    if count < 50:
        return 3, ["Q1", "Q3", "Q5"]
    return 5, ["Q1", "Q2", "Q3", "Q4", "Q5"]


def assign_buckets(scores: pd.Series) -> pd.Series:
    ordered = scores.sort_values(ascending=False, kind="mergesort")
    bucket_count, labels = determine_bucket_labels(len(ordered))
    if bucket_count == 0:
        return pd.Series(index=ordered.index, dtype="object")

    boundaries = np.floor(np.arange(len(ordered)) * bucket_count / len(ordered)).astype(int)
    bucket_labels = pd.Series([labels[index] for index in boundaries], index=ordered.index, dtype="object")
    return bucket_labels.reindex(scores.index)


def _unconstrained_bucket_frame(base_frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    output = base_frame.copy()
    output["bucket"] = assign_buckets(pd.to_numeric(output[score_column], errors="coerce"))
    output = output[output["bucket"].notna()].copy()
    if output.empty:
        output["weight"] = pd.Series(dtype=float)
        return output
    output["weight"] = output.groupby("bucket")["bucket"].transform(lambda values: 1.0 / len(values))
    return output


def _sector_neutral_bucket_frame(base_frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    if "sector" not in base_frame.columns or base_frame["sector"].nunique(dropna=True) < 2:
        return pd.DataFrame(columns=[*base_frame.columns, "bucket", "weight"])

    pieces: list[pd.DataFrame] = []
    for sector, sector_frame in base_frame.dropna(subset=["sector"]).groupby("sector", dropna=False):
        if len(sector_frame) < 3:
            continue
        bucketed = sector_frame.copy()
        bucketed["bucket"] = assign_buckets(pd.to_numeric(bucketed[score_column], errors="coerce"))
        bucketed = bucketed[bucketed["bucket"].notna()].copy()
        if bucketed.empty:
            continue
        bucketed["_sector_key"] = str(sector)
        pieces.append(bucketed)
    if not pieces:
        return pd.DataFrame(columns=[*base_frame.columns, "bucket", "weight"])

    output = pd.concat(pieces, axis=0, ignore_index=False)
    output["weight"] = np.nan
    for bucket, bucket_frame in output.groupby("bucket", dropna=False):
        sectors = sorted(bucket_frame["_sector_key"].dropna().unique().tolist())
        if not sectors:
            continue
        sector_weight = 1.0 / len(sectors)
        for sector in sectors:
            sector_index = bucket_frame.index[bucket_frame["_sector_key"] == sector]
            output.loc[sector_index, "weight"] = sector_weight / len(sector_index)
    return output.drop(columns=["_sector_key"])


def median_impute_cross_section(frame: pd.DataFrame) -> pd.DataFrame:
    filled = frame.copy()
    for column in filled.columns:
        median_value = filled[column].median()
        if pd.isna(median_value):
            median_value = 0.0
        filled[column] = filled[column].fillna(median_value)
    return filled


def compute_additive_scores(frame: pd.DataFrame, *, universe_name: str) -> pd.Series:
    work = frame.loc[:, FEATURE_ORDER].copy()
    if universe_name == "broad":
        work = median_impute_cross_section(work)

    score_parts: list[pd.Series] = []
    for feature_name in FEATURE_ORDER:
        series = work[feature_name]
        std = float(series.std(ddof=0)) if len(series) else 0.0
        if not np.isfinite(std) or std == 0.0:
            z_score = pd.Series(0.0, index=series.index)
        else:
            z_score = (series - series.mean()) / std
        if feature_name == "asset_growth_1y":
            z_score = -z_score
        score_parts.append(z_score)
    return pd.concat(score_parts, axis=1).sum(axis=1)


def compute_group_returns(
    frame: pd.DataFrame,
    score: pd.Series,
) -> tuple[dict[str, float | None], str]:
    if frame.empty:
        return {label: None for label in ("Q1", "Q2", "Q3", "Q4", "Q5")}, "none"

    buckets = assign_buckets(score)
    if buckets.empty:
        return {label: None for label in ("Q1", "Q2", "Q3", "Q4", "Q5")}, "none"

    labeled = frame.copy()
    labeled["bucket"] = buckets
    grouped = labeled.groupby("bucket")["forward_return"].mean()
    returns = {label: float(grouped[label]) if label in grouped.index else None for label in ("Q1", "Q2", "Q3", "Q4", "Q5")}
    bucket_mode = "terciles" if labeled["bucket"].nunique() == 3 else "quintiles"
    return returns, bucket_mode


def aligned_outperformance(a: pd.Series, b: pd.Series) -> dict[str, Any]:
    overlap = pd.concat({"a": a, "b": b}, axis=1).dropna()
    if overlap.empty:
        return {
            "overlap_months": 0,
            "monthly_hit_rate": None,
            "annualized_return_gap": None,
            "cumulative_return_gap": None,
            "beats": False,
        }

    a_metrics = compute_performance_metrics(overlap["a"])
    b_metrics = compute_performance_metrics(overlap["b"])
    annualized_gap = None
    cumulative_gap = None
    if a_metrics["annualized_return"] is not None and b_metrics["annualized_return"] is not None:
        annualized_gap = a_metrics["annualized_return"] - b_metrics["annualized_return"]
    if a_metrics["cumulative_return"] is not None and b_metrics["cumulative_return"] is not None:
        cumulative_gap = a_metrics["cumulative_return"] - b_metrics["cumulative_return"]
    hit_rate = float((overlap["a"] > overlap["b"]).mean())
    beats = bool((annualized_gap is not None and annualized_gap > 0.0) and hit_rate > 0.5)
    return {
        "overlap_months": int(len(overlap)),
        "monthly_hit_rate": hit_rate,
        "annualized_return_gap": annualized_gap,
        "cumulative_return_gap": cumulative_gap,
        "beats": beats,
    }


def annual_returns(series: pd.Series) -> dict[str, float]:
    clean = series.dropna()
    if clean.empty:
        return {}
    return {
        str(int(year)): float((1.0 + values).prod() - 1.0)
        for year, values in clean.groupby(clean.index.year)
    }


def wealth_index(series: pd.Series) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return pd.Series(dtype=float)
    return (1.0 + clean).cumprod()


def compute_performance_metrics(series: pd.Series) -> dict[str, float | None]:
    clean = series.dropna()
    if clean.empty:
        return {
            "months": 0,
            "cumulative_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe": None,
            "max_drawdown": None,
        }

    wealth = wealth_index(clean)
    cumulative_return = float(wealth.iloc[-1] - 1.0)
    annualized_return = None
    if wealth.iloc[-1] > 0:
        annualized_return = float((wealth.iloc[-1] ** (12.0 / len(clean))) - 1.0)

    monthly_std = float(clean.std(ddof=0))
    annualized_volatility = float(monthly_std * np.sqrt(12.0))
    sharpe = None
    if monthly_std > 0.0:
        sharpe = float((clean.mean() / monthly_std) * np.sqrt(12.0))

    drawdown = wealth / wealth.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    return {
        "months": int(len(clean)),
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def first_price_after(series: pd.Series, date_value: pd.Timestamp) -> float | None:
    return historical.price_on_or_after(series, pd.Timestamp(date_value).normalize())


def compute_forward_return(
    price_frame: pd.DataFrame | historical.PricePanels,
    ticker: str,
    rebalance_date: pd.Timestamp,
    next_rebalance_date: pd.Timestamp,
) -> float | None:
    if isinstance(price_frame, historical.PricePanels):
        price_frame = price_frame.adjusted_close
    if ticker not in price_frame.columns:
        return None
    series = price_frame[ticker].dropna()
    if series.empty:
        return None

    entry_price = first_price_after(series, rebalance_date)
    exit_price = first_price_after(series, next_rebalance_date)
    if entry_price in (None, 0.0) or exit_price is None:
        return None
    return float((exit_price / entry_price) - 1.0)


def one_way_cost_rate(cost_bps: float) -> float:
    return float(cost_bps) / 10000.0


def portfolio_turnover(previous_weights: pd.Series, current_weights: pd.Series) -> float:
    if previous_weights.empty:
        return float(current_weights.abs().sum())
    aligned = pd.concat({"previous": previous_weights, "current": current_weights}, axis=1).fillna(0.0)
    if aligned.empty:
        return 0.0
    return float(0.5 * (aligned["current"] - aligned["previous"]).abs().sum())


def _price_frame_for_returns(price_frame: pd.DataFrame | historical.PricePanels) -> pd.DataFrame:
    return price_frame.adjusted_close if isinstance(price_frame, historical.PricePanels) else price_frame


def _price_frame_for_valuation(price_frame: pd.DataFrame | historical.PricePanels) -> pd.DataFrame:
    return price_frame.raw_close if isinstance(price_frame, historical.PricePanels) else price_frame


def run_audited_signal_backtest(
    *,
    signal_name: str,
    score_column: str,
    matrices: dict[pd.Timestamp, pd.DataFrame],
    rebalance_dates: list[pd.Timestamp],
    price_frame: pd.DataFrame | historical.PricePanels,
    cost_bps_levels: tuple[int, ...] = (0, 10, 25, 50),
) -> dict[str, pd.DataFrame]:
    adjusted_prices = _price_frame_for_returns(price_frame)
    raw_prices = _price_frame_for_valuation(price_frame)
    holding_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    turnover_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    previous_weights: dict[tuple[str, str, str], pd.Series] = {}

    for index, rebalance_date in enumerate(rebalance_dates[:-1], start=1):
        next_rebalance_date = rebalance_dates[index]
        if rebalance_date not in matrices or score_column not in matrices[rebalance_date].columns:
            continue
        base_frame = matrices[rebalance_date].copy()
        base_frame = base_frame[pd.to_numeric(base_frame[score_column], errors="coerce").notna()].copy()
        if base_frame.empty:
            continue

        base_frame["entry_price"] = [
            first_price_after(adjusted_prices[ticker].dropna(), rebalance_date)
            if ticker in adjusted_prices.columns
            else None
            for ticker in base_frame.index
        ]
        base_frame["exit_price"] = [
            first_price_after(adjusted_prices[ticker].dropna(), next_rebalance_date)
            if ticker in adjusted_prices.columns
            else None
            for ticker in base_frame.index
        ]
        base_frame["raw_return"] = [
            (exit_price / entry_price) - 1.0
            if entry_price not in (None, 0.0) and exit_price is not None
            else np.nan
            for entry_price, exit_price in zip(base_frame["entry_price"], base_frame["exit_price"], strict=True)
        ]
        base_frame = base_frame[base_frame["raw_return"].notna()].copy()
        if base_frame.empty:
            continue

        portfolio_frames = [
            ("unconstrained", "equal_name", _unconstrained_bucket_frame(base_frame, score_column)),
            ("sector_neutral", "equal_sector_then_equal_name", _sector_neutral_bucket_frame(base_frame, score_column)),
        ]
        for portfolio_mode, weighting_method, mode_frame in portfolio_frames:
            if mode_frame.empty:
                continue
            for bucket, bucket_frame in mode_frame.groupby("bucket"):
                current_weights = pd.to_numeric(bucket_frame["weight"], errors="coerce").astype(float)
                current_weights = current_weights[current_weights.notna() & (current_weights > 0.0)]
                bucket_frame = bucket_frame.loc[current_weights.index].copy()
                if bucket_frame.empty:
                    continue
                total_weight = float(current_weights.sum())
                if total_weight <= 0.0:
                    continue
                current_weights = current_weights / total_weight
                previous = previous_weights.get((signal_name, portfolio_mode, str(bucket)), pd.Series(dtype=float))
                turnover = portfolio_turnover(previous, current_weights)
                previous_weights[(signal_name, portfolio_mode, str(bucket))] = current_weights
                gross_return = float((bucket_frame["raw_return"] * current_weights).sum())
                sector_weights = (
                    bucket_frame.assign(weight=current_weights)
                    .groupby("sector", dropna=False)["weight"]
                    .sum()
                    .to_dict()
                    if "sector" in bucket_frame.columns
                    else {}
                )
                average_market_cap = (
                    float((pd.to_numeric(bucket_frame["market_cap"], errors="coerce") * current_weights).sum())
                    if "market_cap" in bucket_frame.columns and bucket_frame["market_cap"].notna().any()
                    else None
                )

                for cost_bps in cost_bps_levels:
                    cost_rate = one_way_cost_rate(cost_bps)
                    transaction_cost_drag = turnover * cost_rate
                    net_return = gross_return - transaction_cost_drag
                    monthly_rows.append(
                        {
                            "date": rebalance_date,
                            "next_rebalance_date": next_rebalance_date,
                            "signal_name": signal_name,
                            "portfolio_mode": portfolio_mode,
                            "weighting_method": weighting_method,
                            "bucket": bucket,
                            "cost_bps_one_way": int(cost_bps),
                            "gross_return": gross_return,
                            "transaction_cost_drag": transaction_cost_drag,
                            "net_return": net_return,
                            "name_count": int(len(bucket_frame)),
                        }
                    )
                    turnover_rows.append(
                        {
                            "date": rebalance_date,
                            "signal_name": signal_name,
                            "portfolio_mode": portfolio_mode,
                            "weighting_method": weighting_method,
                            "bucket": bucket,
                            "cost_bps_one_way": int(cost_bps),
                            "turnover": turnover,
                            "transaction_cost_drag": transaction_cost_drag,
                        }
                    )

                exposure_rows.append(
                    {
                        "date": rebalance_date,
                        "signal_name": signal_name,
                        "portfolio_mode": portfolio_mode,
                        "weighting_method": weighting_method,
                        "bucket": bucket,
                        "sector_weights": json.dumps({str(key): float(value) for key, value in sector_weights.items()}),
                        "average_market_cap": average_market_cap,
                        "name_count": int(len(bucket_frame)),
                    }
                )

                for ticker, row in bucket_frame.iterrows():
                    raw_entry_price = (
                        first_price_after(raw_prices[ticker].dropna(), rebalance_date)
                        if ticker in raw_prices.columns
                        else None
                    )
                    weight = float(current_weights.loc[ticker])
                    base_holding = {
                        "date": rebalance_date,
                        "next_rebalance_date": next_rebalance_date,
                        "ticker": ticker,
                        "sector": row.get("sector"),
                        "market_cap": row.get("market_cap"),
                        "score": row[score_column],
                        "signal_name": signal_name,
                        "portfolio_mode": portfolio_mode,
                        "weighting_method": weighting_method,
                        "bucket": bucket,
                        "weight": weight,
                        "feature_null_count": row.get("feature_null_count"),
                        "entry_price": row["entry_price"],
                        "exit_price": row["exit_price"],
                        "raw_close_entry_price": raw_entry_price,
                        "raw_return": row["raw_return"],
                    }
                    for feature_name in FEATURE_ORDER:
                        if feature_name in row.index:
                            base_holding[feature_name] = row[feature_name]
                    for label_name in (
                        "raw_ai_implied_irr",
                        "mechanical_dcf_implied_irr",
                        "ai_minus_mechanical_irr",
                        "factor_compressible_ai_score",
                        "ai_factor_residual",
                    ):
                        if label_name in row.index:
                            base_holding[label_name] = row[label_name]
                    for cost_bps in cost_bps_levels:
                        holding_cost = turnover * one_way_cost_rate(cost_bps) * weight
                        holding_rows.append(
                            {
                                **base_holding,
                                "cost_bps_one_way": int(cost_bps),
                                "transaction_cost": holding_cost,
                                "net_return": row["raw_return"] - holding_cost,
                            }
                        )

    return {
        "holdings": pd.DataFrame(holding_rows),
        "monthly_returns": pd.DataFrame(monthly_rows),
        "turnover": pd.DataFrame(turnover_rows),
        "exposures": pd.DataFrame(exposure_rows),
    }


def write_backtest_audit_artifacts(
    artifacts: dict[str, pd.DataFrame],
    output_dir: Path = RESULTS_DIR,
    *,
    prefix: str = "",
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, frame in artifacts.items():
        path = output_dir / f"{prefix}{name}.parquet"
        frame.to_parquet(path, index=False)
        try:
            written[name] = str(path.relative_to(ROOT))
        except ValueError:
            written[name] = str(path)
    return written


def run_backtest_pass(
    universe_name: str,
    matrices: dict[pd.Timestamp, pd.DataFrame],
    rebalance_dates: list[pd.Timestamp],
    price_frame: pd.DataFrame,
    model: XGBRegressor,
) -> dict[str, Any]:
    monthly_rows: list[dict[str, Any]] = []
    series_store: dict[str, list[tuple[pd.Timestamp, float]]] = defaultdict(list)

    for index, rebalance_date in enumerate(rebalance_dates[:-1], start=1):
        next_rebalance_date = rebalance_dates[index]
        base_frame = matrices[rebalance_date]
        if universe_name == "clean":
            universe_frame = base_frame[base_frame["feature_null_count"] == 0].copy()
        else:
            universe_frame = base_frame[base_frame["feature_null_count"] <= 2].copy()

        if universe_frame.empty:
            LOGGER.info("%s %s: no eligible names", universe_name, rebalance_date.date())
            continue

        universe_frame["forward_return"] = [
            compute_forward_return(price_frame, ticker, rebalance_date, next_rebalance_date)
            for ticker in universe_frame.index
        ]
        universe_frame = universe_frame[universe_frame["forward_return"].notna()].copy()
        if universe_frame.empty:
            LOGGER.info("%s %s: no forward returns", universe_name, rebalance_date.date())
            continue

        feature_matrix = universe_frame.loc[:, FEATURE_ORDER]
        predicted_scores = pd.Series(
            model.predict(feature_matrix),
            index=universe_frame.index,
            dtype=float,
        )
        model_group_returns, bucket_mode = compute_group_returns(universe_frame, predicted_scores)

        additive_scores = compute_additive_scores(universe_frame, universe_name=universe_name)
        additive_group_returns, _ = compute_group_returns(universe_frame, additive_scores)

        pure_fcf_frame = universe_frame[universe_frame["fcf_to_ev"].notna()].copy()
        pure_fcf_group_returns, pure_fcf_bucket_mode = compute_group_returns(
            pure_fcf_frame,
            pure_fcf_frame["fcf_to_ev"] if not pure_fcf_frame.empty else pd.Series(dtype=float),
        )

        ew_return = float(universe_frame["forward_return"].mean())
        model_spread = (
            model_group_returns["Q1"] - model_group_returns["Q5"]
            if model_group_returns["Q1"] is not None and model_group_returns["Q5"] is not None
            else None
        )
        additive_spread = (
            additive_group_returns["Q1"] - additive_group_returns["Q5"]
            if additive_group_returns["Q1"] is not None and additive_group_returns["Q5"] is not None
            else None
        )
        pure_fcf_spread = (
            pure_fcf_group_returns["Q1"] - pure_fcf_group_returns["Q5"]
            if pure_fcf_group_returns["Q1"] is not None and pure_fcf_group_returns["Q5"] is not None
            else None
        )

        record = {
            "rebalance_date": rebalance_date.date().isoformat(),
            "next_rebalance_date": next_rebalance_date.date().isoformat(),
            "universe_name": universe_name,
            "universe_size": int(len(universe_frame)),
            "pure_fcf_universe_size": int(len(pure_fcf_frame)),
            "bucket_mode": bucket_mode,
            "pure_fcf_bucket_mode": pure_fcf_bucket_mode,
            "ew_universe": ew_return,
            "model_Q1": model_group_returns["Q1"],
            "model_Q2": model_group_returns["Q2"],
            "model_Q3": model_group_returns["Q3"],
            "model_Q4": model_group_returns["Q4"],
            "model_Q5": model_group_returns["Q5"],
            "model_Q1_Q5_spread": model_spread,
            "pure_fcf_Q1": pure_fcf_group_returns["Q1"],
            "pure_fcf_Q5": pure_fcf_group_returns["Q5"],
            "pure_fcf_Q1_Q5_spread": pure_fcf_spread,
            "additive_Q1": additive_group_returns["Q1"],
            "additive_Q5": additive_group_returns["Q5"],
            "additive_Q1_Q5_spread": additive_spread,
        }
        monthly_rows.append(record)

        for key, value in (
            ("EW Universe", ew_return),
            ("Q1", model_group_returns["Q1"]),
            ("Q2", model_group_returns["Q2"]),
            ("Q3", model_group_returns["Q3"]),
            ("Q4", model_group_returns["Q4"]),
            ("Q5", model_group_returns["Q5"]),
            ("Q1-Q5 Spread", model_spread),
            ("Pure FCF/EV Q1", pure_fcf_group_returns["Q1"]),
            ("Pure FCF/EV Q5", pure_fcf_group_returns["Q5"]),
            ("Pure FCF/EV Q1-Q5 Spread", pure_fcf_spread),
            ("Additive Q1", additive_group_returns["Q1"]),
            ("Additive Q5", additive_group_returns["Q5"]),
            ("Additive Q1-Q5 Spread", additive_spread),
        ):
            if value is not None:
                series_store[key].append((rebalance_date, float(value)))

        LOGGER.info(
            "%s %s: %d names, %s, Q1=%s, Q5=%s, spread=%s",
            universe_name,
            rebalance_date.date(),
            len(universe_frame),
            bucket_mode,
            f"{model_group_returns['Q1']:.4f}" if model_group_returns["Q1"] is not None else "n/a",
            f"{model_group_returns['Q5']:.4f}" if model_group_returns["Q5"] is not None else "n/a",
            f"{model_spread:.4f}" if model_spread is not None else "n/a",
        )

    series_by_name = {
        name: pd.Series(
            data=[value for _, value in values],
            index=pd.DatetimeIndex([timestamp for timestamp, _ in values]),
            dtype=float,
        ).sort_index()
        for name, values in series_store.items()
    }

    metrics = {name: compute_performance_metrics(series) for name, series in series_by_name.items()}
    annual = {name: annual_returns(series) for name, series in series_by_name.items()}

    q1 = series_by_name.get("Q1", pd.Series(dtype=float))
    q5 = series_by_name.get("Q5", pd.Series(dtype=float))
    ew = series_by_name.get("EW Universe", pd.Series(dtype=float))
    pure_fcf_q1 = series_by_name.get("Pure FCF/EV Q1", pd.Series(dtype=float))
    additive_q1 = series_by_name.get("Additive Q1", pd.Series(dtype=float))
    positive_year_share = None
    spread_annual = annual.get("Q1-Q5 Spread", {})
    if spread_annual:
        positive_year_share = float(sum(value > 0.0 for value in spread_annual.values()) / len(spread_annual))

    summary_answers = {
        "does_q1_beat_q5_consistently": aligned_outperformance(q1, q5),
        "does_q1_beat_ew_universe": aligned_outperformance(q1, ew),
        "does_q1_beat_pure_fcf_ev_q1": aligned_outperformance(q1, pure_fcf_q1),
        "does_q1_beat_additive_linear_composite_q1": aligned_outperformance(q1, additive_q1),
        "is_q1_q5_spread_positive_in_most_years": {
            "positive_year_share": positive_year_share,
            "positive_years": int(sum(value > 0.0 for value in spread_annual.values())),
            "total_years": int(len(spread_annual)),
            "positive_in_most_years": bool(positive_year_share is not None and positive_year_share > 0.5),
        },
    }

    return {
        "monthly_returns": monthly_rows,
        "metrics": metrics,
        "annual_returns": annual,
        "summary_answers": summary_answers,
        "series": {name: {timestamp.date().isoformat(): float(value) for timestamp, value in series.items()} for name, series in series_by_name.items()},
    }


def plot_universe_size(count_frame: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    for column, color in (("broad_count", "#1f77b4"), ("clean_count", "#d62728")):
        ax.plot(count_frame.index, count_frame[column], label=column.replace("_", " "), linewidth=2.0, color=color)
    ax.set_title("Eligible Universe Size Over Time")
    ax.set_ylabel("Tickers")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(UNIVERSE_PLOT_PATH, dpi=160)
    plt.close(fig)


def plot_cumulative(runs: dict[str, dict[str, Any]]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    for axis, universe_name in zip(axes, ("broad", "clean")):
        run = runs[universe_name]
        series_payload = run.get("series", {})
        plotted = False
        for name, color in (
            ("Q1", "#1f77b4"),
            ("Q5", "#d62728"),
            ("Q1-Q5 Spread", "#2ca02c"),
            ("EW Universe", "#7f7f7f"),
            ("Pure FCF/EV Q1", "#ff7f0e"),
        ):
            raw_series = series_payload.get(name, {})
            if not raw_series:
                continue
            series = pd.Series(raw_series, dtype=float)
            series.index = pd.to_datetime(series.index)
            wealth = wealth_index(series.sort_index())
            if wealth.empty:
                continue
            axis.plot(wealth.index, wealth.values, label=name, linewidth=2.0, color=color)
            plotted = True
        axis.set_title(f"{universe_name.title()} Universe")
        axis.set_ylabel("Growth of $1")
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
        if plotted:
            axis.legend()
    fig.tight_layout()
    fig.savefig(CUMULATIVE_PLOT_PATH, dpi=160)
    plt.close(fig)


def plot_annual_returns(runs: dict[str, dict[str, Any]]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    for axis, universe_name in zip(axes, ("broad", "clean")):
        annual_payload = runs[universe_name].get("annual_returns", {})
        years = sorted(
            {
                int(year)
                for name in ("Q1", "Q5", "Q1-Q5 Spread")
                for year in annual_payload.get(name, {}).keys()
            }
        )
        if not years:
            continue
        positions = np.arange(len(years))
        width = 0.26
        for offset, name, color in (
            (-width, "Q1", "#1f77b4"),
            (0.0, "Q5", "#d62728"),
            (width, "Q1-Q5 Spread", "#2ca02c"),
        ):
            values = [annual_payload.get(name, {}).get(str(year), np.nan) for year in years]
            axis.bar(positions + offset, values, width=width, label=name, color=color)
        axis.set_title(f"{universe_name.title()} Universe Annual Returns")
        axis.set_ylabel("Return")
        axis.axhline(0.0, color="#444444", linewidth=1.0)
        axis.set_xticks(positions, [str(year) for year in years], rotation=45)
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
    fig.tight_layout()
    fig.savefig(ANNUAL_RETURNS_PLOT_PATH, dpi=160)
    plt.close(fig)


def print_summary(runs: dict[str, dict[str, Any]]) -> None:
    for universe_name in ("broad", "clean"):
        run = runs[universe_name]
        metrics = run.get("metrics", {})
        summary = run.get("summary_answers", {})
        q1_metrics = metrics.get("Q1", {})
        q5_metrics = metrics.get("Q5", {})
        spread_metrics = metrics.get("Q1-Q5 Spread", {})
        print(f"\n{universe_name.upper()} UNIVERSE")
        print(
            "Q1 vs Q5: "
            f"beat={summary['does_q1_beat_q5_consistently']['beats']}, "
            f"hit_rate={summary['does_q1_beat_q5_consistently']['monthly_hit_rate']}, "
            f"Q1_ann={q1_metrics.get('annualized_return')}, "
            f"Q5_ann={q5_metrics.get('annualized_return')}, "
            f"spread_ann={spread_metrics.get('annualized_return')}"
        )
        print(
            "Q1 vs EW Universe: "
            f"beat={summary['does_q1_beat_ew_universe']['beats']}, "
            f"hit_rate={summary['does_q1_beat_ew_universe']['monthly_hit_rate']}"
        )
        print(
            "Q1 vs Pure FCF/EV Q1: "
            f"beat={summary['does_q1_beat_pure_fcf_ev_q1']['beats']}, "
            f"hit_rate={summary['does_q1_beat_pure_fcf_ev_q1']['monthly_hit_rate']}"
        )
        print(
            "Q1 vs Additive Linear Composite Q1: "
            f"beat={summary['does_q1_beat_additive_linear_composite_q1']['beats']}, "
            f"hit_rate={summary['does_q1_beat_additive_linear_composite_q1']['monthly_hit_rate']}"
        )
        print(
            "Q1-Q5 Spread Positive In Most Years: "
            f"{summary['is_q1_q5_spread_positive_in_most_years']['positive_in_most_years']} "
            f"({summary['is_q1_q5_spread_positive_in_most_years']['positive_years']}/"
            f"{summary['is_q1_q5_spread_positive_in_most_years']['total_years']})"
        )


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    feature_order = load_feature_order(FEATURE_SPEC_PATH)
    if feature_order != list(FEATURE_ORDER):
        raise ValueError("Frozen feature order changed.")

    tickers = load_tickers(Path(args.tickers_file))
    LOGGER.info("Loaded %d tickers from %s", len(tickers), args.tickers_file)

    sec_client = SecClient(
        user_agent_name=args.user_agent_name,
        user_agent_email=args.user_agent_email,
    )
    histories = historical.load_ticker_histories(
        tickers,
        sec_client=sec_client,
        refresh_sec_cache=args.refresh_sec_cache,
    )

    price_start_year = max(args.start_year - 2, 1990)
    price_panels = historical.load_price_panel_history(
        tickers,
        start=f"{price_start_year}-01-01",
        end=f"{args.end_year + 1}-01-31",
        refresh=args.refresh_price_cache,
    )
    rebalance_dates = build_rebalance_dates(
        price_panels.adjusted_close,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    if len(rebalance_dates) < 2:
        raise ValueError("Need at least two rebalance dates to run the backtest.")

    matrices: dict[pd.Timestamp, pd.DataFrame] = {}
    counts: list[dict[str, Any]] = []
    for index, rebalance_date in enumerate(rebalance_dates, start=1):
        LOGGER.info("Building point-in-time matrix for %s (%d/%d)", rebalance_date.date(), index, len(rebalance_dates))
        matrix = historical.build_point_in_time_feature_matrix(histories, price_panels, rebalance_date)
        matrices[rebalance_date] = matrix
        counts.append(
            {
                "date": rebalance_date,
                "broad_count": int(len(matrix[matrix["feature_null_count"] <= 2])),
                "clean_count": int(len(matrix[matrix["feature_null_count"] == 0])),
            }
        )

    count_frame = pd.DataFrame(counts).set_index("date").sort_index()
    plot_universe_size(count_frame)

    model_path = Path(args.model_path)
    if FROZEN_MODEL_PATH.exists():
        LOGGER.info("Frozen model snapshot available at %s", FROZEN_MODEL_PATH)
    model = load_model(model_path)

    runs = {
        "broad": run_backtest_pass("broad", matrices, rebalance_dates, price_panels, model),
        "clean": run_backtest_pass("clean", matrices, rebalance_dates, price_panels, model),
    }

    plot_cumulative(runs)
    plot_annual_returns(runs)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "config": {
            "start_year": args.start_year,
            "end_year": args.end_year,
            "model_path": str(model_path.relative_to(ROOT)) if model_path.is_absolute() and model_path.is_relative_to(ROOT) else str(model_path),
            "tickers_file": str(Path(args.tickers_file).relative_to(ROOT)) if Path(args.tickers_file).is_absolute() and Path(args.tickers_file).is_relative_to(ROOT) else str(Path(args.tickers_file)),
            "feature_order": feature_order,
            "t_plus_one_trading": True,
        },
        "assumptions": {
            "filings_must_be_accepted_before_rebalance_date": True,
            "raw_close_price_used_for_market_cap_and_ev": True,
            "adjusted_close_price_used_for_returns_and_momentum": True,
            "broad_universe_allows_up_to_two_missing_features": True,
            "additive_benchmark_broad_mode_missing_features": "cross-sectional median imputation before z-scoring",
            "pure_fcf_ev_sort_uses_only_names_with_non_null_fcf_to_ev": True,
        },
        "artifacts": {
            "summary": str(SUMMARY_PATH.relative_to(ROOT)),
            "universe_plot": str(UNIVERSE_PLOT_PATH.relative_to(ROOT)),
            "cumulative_plot": str(CUMULATIVE_PLOT_PATH.relative_to(ROOT)),
            "annual_returns_plot": str(ANNUAL_RETURNS_PLOT_PATH.relative_to(ROOT)),
        },
        "universe_counts": {
            row["date"].date().isoformat(): {
                "broad_count": row["broad_count"],
                "clean_count": row["clean_count"],
            }
            for row in counts
        },
        "runs": runs,
    }
    SUMMARY_PATH.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    print_summary(runs)
    print(f"\nSaved summary to {SUMMARY_PATH}")
    print(f"Saved cumulative plot to {CUMULATIVE_PLOT_PATH}")
    print(f"Saved annual returns plot to {ANNUAL_RETURNS_PLOT_PATH}")
    print(f"Saved universe size plot to {UNIVERSE_PLOT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
