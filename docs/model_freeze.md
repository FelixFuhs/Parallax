# Parallax Model Freeze

Updated: 2026-03-22

## Frozen Statement

This spec is frozen. Any changes after seeing backtest results must be documented as post-hoc modifications and flagged as such.

## Research Framing

- Primary frozen model: XGBoost regressor.
- Frozen training slice: `n=264` matched set, not `n=66`.
- Backtest framing: supervised interpolation today, economic falsification backward.
- This is a prototype-grade, survivor-biased historical falsification test, not proof of alpha.

## Model Status

- Primary model: `models/frozen_xgb_regressor.json`
- Primary metadata: `models/frozen_model_metadata.json`
- Elastic net is retained as a transparency baseline only, not the primary model.
- Baseline-only Elastic Net artifacts remain: `models/frozen_elasticnet_coefficients.json`, `models/frozen_elasticnet_metadata.json`, `models/distill_elasticnet_v2.pkl`, and `docs/freeze_elasticnet_baseline.md`
- XGBoost ranker is dropped from the frozen spec.

## Universe

- Conceptual universe: S&P 500 ex-Financials ex-REITs.
- Current frozen research slice: `tickers.txt` in this repo.
- Matched training sample used for the frozen model: 264 tickers.
- Entry requirements: latest successful cheap valuation report, no `stale_price` quality flag, base-case upside present, EDGAR row present with no extraction error, fiscal year >= 2024, and not in the hard-coded broken ticker list.
- Missing feature values are allowed at the row level.

## Exclusion Rules

- Exclude EDGAR rows with stale fiscal years (`fiscal_year < 2024`).
- Exclude rows with explicit EDGAR extraction errors.
- Exclude hard-coded broken data cases such as `MCD` (broken shares-outstanding / derived-feature path).
- Exclude valuation reports with missing base-case upside or `stale_price` flags.

## Target Variable

- Target: fractional percentile rank of AI DCF base-case upside (`_valuation.scenarios.base.upside_downside_pct`) across the matched set.
- Scale: `0.0` = lowest upside in the matched set, `1.0` = highest upside.
- Ranking is computed directly from raw upside values; the target is not winsorized.

## Winsorization

- None.
- No percentiles are clipped before ranking, after ranking, or before model fitting.

## Feature Handling

- Features are used in the fixed order: `fcf_to_ev`, `gross_profitability_assets`, `asset_growth_1y`, `cash_earnings_gap`, `momentum_12_1`.
- Monotonic constraints are fixed at `(1, 1, -1, 1, 0)` in that same feature order.
- The XGBoost regressor is fit on the raw feature matrix with missing values left as missing for native tree handling.
- No feature standardization or target normalization is applied in the frozen primary model.

## Frozen Primary Model

- Estimator: `XGBRegressor(objective="reg:squarederror")`
- Hyperparameters:
  - `max_depth=2`
  - `learning_rate=0.05`
  - `n_estimators=200`
  - `subsample=0.8`
  - `colsample_bytree=0.8`
  - `reg_alpha=1.0`
  - `reg_lambda=1.0`
  - `min_child_weight=5`
  - `random_state=42`
  - `n_jobs=1`
  - `monotone_constraints=(1, 1, -1, 1, 0)`

## Stability Results

- Spearman CV mean: `0.6196`
- Spearman CV std: `0.0901`
- Spearman CV p05: `0.4577`
- Spearman CV p95: `0.7448`
- Bootstrap CI: `[0.4959, 0.7193]`
- Top-Q stability: `90.8%`

These are the frozen stability numbers for the primary model decision.

## Feature Set

| Feature | Expected Sign | Economic Rationale |
| --- | --- | --- |
| fcf_to_ev | positive | Higher free cash flow relative to enterprise value indicates cheaper cash generation. |
| gross_profitability_assets | positive | Higher gross profits on the asset base are associated with better business quality. |
| asset_growth_1y | negative | Aggressive balance-sheet expansion is often linked to weaker subsequent returns. |
| cash_earnings_gap | positive | Cash earnings above accounting earnings can signal stronger earnings quality. |
| momentum_12_1 | positive | Intermediate-term momentum can capture trend persistence without the most recent month. |

## Exact Feature Definitions

EDGAR values are taken from the latest selected annual filing context; when a primary tag is absent, the documented fallback chain is used.

### `fcf_to_ev`

- Formula: `(operating_cash_flow - abs(capex)) / (market_cap + total_debt - cash)`
- Tags used:
  - operating_cash_flow: us-gaap/NetCashProvidedByOperatingActivities
  - operating_cash_flow fallback: us-gaap/NetCashProvidedByUsedInOperatingActivities
  - operating_cash_flow fallback: us-gaap/NetCashProvidedByUsedInOperatingActivitiesContinuingOperations
  - capex: us-gaap/PaymentsToAcquirePropertyPlantAndEquipment
  - capex fallback: us-gaap/PropertyPlantAndEquipmentAdditions
  - capex fallback: us-gaap/PaymentsToAcquireProductiveAssets
  - cash: us-gaap/CashAndCashEquivalentsAtCarryingValue
  - shares_outstanding: us-gaap/CommonStockSharesOutstanding
  - shares_outstanding fallback: dei/EntityCommonStockSharesOutstanding
  - shares_outstanding fallback: us-gaap/WeightedAverageNumberOfShareOutstandingsBasic
  - shares_outstanding fallback: us-gaap/WeightedAverageNumberOfSharesOutstandingBasic
  - shares_outstanding fallback: us-gaap/WeightedAverageNumberOfShareOutstandingsBasicAndDiluted
  - shares_outstanding fallback: us-gaap/WeightedAverageNumberOfSharesOutstandingBasicAndDiluted
  - total_debt direct: us-gaap/LongTermDebtAndCapitalLeaseObligations
  - total_debt direct fallback: us-gaap/LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities
  - total_debt direct fallback: us-gaap/LongTermDebtAndFinanceLeaseObligations
  - total_debt direct fallback: us-gaap/LongTermDebtAndFinanceLeaseObligationsIncludingCurrentMaturities
  - total_debt component fallback: us-gaap/LongTermDebt
  - total_debt component fallback: us-gaap/LongTermDebtNoncurrent
  - total_debt component fallback: us-gaap/ShortTermBorrowings
  - total_debt component fallback: us-gaap/LongTermDebtCurrent
  - total_debt component fallback: us-gaap/ShortTermBankLoansAndNotesPayable
  - total_debt component fallback: us-gaap/ShortTermDebt
  - total_debt component fallback: us-gaap/CommercialPaper
  - total_debt component fallback: us-gaap/LongTermDebtAndCapitalLeaseObligationsCurrent
  - total_debt component fallback: us-gaap/LongTermDebtAndFinanceLeaseObligationsCurrent
- Fallback logic:
  - Capex is forced positive with abs().
  - Legacy frozen artifacts used the available `current_price` field. New v2 price paths separate raw close for market cap and enterprise value from adjusted close for returns and momentum.
  - Total debt uses the direct total-debt tags first; if none are available, long-term and short-term debt components are summed.
  - If enterprise value is zero or negative, the feature is set to null.

### `gross_profitability_assets`

- Formula: `gross_profit / total_assets`
- Tags used:
  - gross_profit: us-gaap/GrossProfit
  - gross_profit fallback cost tag: us-gaap/CostOfRevenue
  - gross_profit fallback cost tag: us-gaap/CostOfGoodsAndServicesSold
  - gross_profit fallback cost tag: us-gaap/CostOfGoodsSold
  - revenue: us-gaap/RevenueFromContractWithCustomerExcludingAssessedTax
  - revenue fallback: us-gaap/RevenueFromContractWithCustomerIncludingAssessedTax
  - revenue fallback: us-gaap/Revenues
  - revenue fallback: us-gaap/SalesRevenueNet
  - total_assets: us-gaap/Assets
- Fallback logic:
  - If GrossProfit is missing but revenue is present, gross profit is reconstructed as revenue minus the first aligned cost tag available.
  - If neither direct nor fallback gross profit can be resolved, the feature is null.

### `asset_growth_1y`

- Formula: `(current_total_assets - prior_total_assets) / prior_total_assets`
- Tags used:
  - current_total_assets: us-gaap/Assets
  - prior_total_assets: us-gaap/Assets
- Fallback logic:
  - Uses the two most recent distinct annual asset facts selected from EDGAR.
  - If the prior-year asset value is unavailable, the feature is null.

### `cash_earnings_gap`

- Formula: `(operating_cash_flow - net_income) / total_assets`
- Tags used:
  - operating_cash_flow: us-gaap/NetCashProvidedByOperatingActivities
  - operating_cash_flow fallback: us-gaap/NetCashProvidedByUsedInOperatingActivities
  - operating_cash_flow fallback: us-gaap/NetCashProvidedByUsedInOperatingActivitiesContinuingOperations
  - net_income: us-gaap/NetIncomeLoss
  - net_income fallback: us-gaap/ProfitLoss
  - total_assets: us-gaap/Assets
- Fallback logic:
  - If any of operating_cash_flow, net_income, or total_assets are missing, the feature is null.

### `momentum_12_1`

- Formula: `(1 + price_return_12m) / (1 + price_return_1m) - 1`
- Tags used:
  - price source: adjusted close history for return inputs; raw close is used separately for valuation fields in the v2 price model.
  - price_return_1m: latest price versus price on or before latest_date - 1 month
  - price_return_12m: latest price versus price on or before latest_date - 12 months
- Fallback logic:
  - If either component return is missing, the feature is null.
  - If 1-month gross return equals zero (a -100% return), the feature is null to avoid division by zero.
