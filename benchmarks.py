from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureBlock:
    name: str
    features: Mapping[str, int]


FEATURE_BLOCKS: tuple[FeatureBlock, ...] = (
    FeatureBlock(
        "value",
        {
            "fcf_to_ev": 1,
            "ebit_to_ev": 1,
            "ebitda_to_ev": 1,
            "book_to_market": 1,
            "earnings_yield": 1,
            "fcf_yield": 1,
        },
    ),
    FeatureBlock(
        "quality",
        {
            "gross_profitability_assets": 1,
            "roic": 1,
            "roe": 1,
            "operating_margin": 1,
            "cash_conversion": 1,
            "cash_earnings_gap": 1,
        },
    ),
    FeatureBlock(
        "momentum",
        {
            "momentum_12_1": 1,
            "price_return_6m": 1,
            "price_return_1m": -1,
        },
    ),
    FeatureBlock(
        "investment",
        {
            "asset_growth_1y": -1,
            "capex_growth": -1,
            "working_capital_growth": -1,
            "issuance_yield": -1,
            "buyback_yield": 1,
        },
    ),
    FeatureBlock(
        "accruals",
        {
            "cash_earnings_gap": 1,
            "accruals": -1,
            "working_capital_accruals": -1,
        },
    ),
    FeatureBlock(
        "balance_sheet",
        {
            "net_debt_to_ebitda": -1,
            "interest_coverage": 1,
            "current_ratio": 1,
            "debt_to_equity": -1,
        },
    ),
)


def winsorize_series(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    clean = numeric.dropna()
    if clean.empty:
        return numeric
    lower_value = float(clean.quantile(lower))
    upper_value = float(clean.quantile(upper))
    return numeric.clip(lower=lower_value, upper=upper_value)


def zscore_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    std = float(numeric.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(0.0, index=numeric.index, dtype=float).where(numeric.notna())
    return (numeric - float(numeric.mean())) / std


def _available_block_features(frame: pd.DataFrame, block: FeatureBlock) -> list[str]:
    return [feature for feature in block.features if feature in frame.columns]


def build_composite_vqmia(
    frame: pd.DataFrame,
    *,
    sector_column: str = "sector",
    blocks: Sequence[FeatureBlock] = FEATURE_BLOCKS,
) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    block_columns: list[str] = []

    for block in blocks:
        standardized_parts: list[pd.Series] = []
        within_sector_parts: list[pd.Series] = []
        for feature_name in _available_block_features(frame, block):
            direction = block.features[feature_name]
            transformed = zscore_series(winsorize_series(frame[feature_name])) * direction
            standardized_parts.append(transformed.rename(feature_name))

            if sector_column in frame.columns:
                sector_values = frame[sector_column]
                within = transformed.groupby(sector_values, dropna=False).transform(
                    lambda values: zscore_series(pd.Series(values, index=values.index))
                )
                within_sector_parts.append(within.rename(feature_name))

        block_column = f"{block.name}_block_score"
        block_columns.append(block_column)
        if standardized_parts:
            output[block_column] = pd.concat(standardized_parts, axis=1).mean(axis=1, skipna=True)
            output[f"{block.name}_feature_count"] = pd.concat(standardized_parts, axis=1).notna().sum(axis=1)
        else:
            output[block_column] = np.nan
            output[f"{block.name}_feature_count"] = 0

        if within_sector_parts:
            output[f"{block.name}_within_sector_score"] = pd.concat(within_sector_parts, axis=1).mean(
                axis=1,
                skipna=True,
            )

    output["composite_vqmia_score"] = output[block_columns].mean(axis=1, skipna=True)
    output["composite_vqmia_block_count"] = output[block_columns].notna().sum(axis=1)
    if sector_column in frame.columns:
        within_columns = [f"{block.name}_within_sector_score" for block in blocks if f"{block.name}_within_sector_score" in output]
        if within_columns:
            output["composite_vqmia_within_sector_score"] = output[within_columns].mean(axis=1, skipna=True)
    return output

