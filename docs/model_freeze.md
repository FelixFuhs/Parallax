# Parallax Model Freeze

Generated: 2026-03-21T16:35:52.326925+00:00

## Frozen Statement

This spec is frozen. Any changes after seeing backtest results must be documented as post-hoc modifications and flagged as such.

## Universe

- Conceptual universe: S&P 500 ex-Financials ex-REITs.
- Current frozen research slice: `tickers_100.txt` in this repo.
- Matched training sample used for the frozen model: 66 tickers.
- Entry requirements: latest successful cheap valuation report, no `stale_price` quality flag, base-case upside present, EDGAR row present with no extraction error, fiscal year >= 2024, and not in the hard-coded broken ticker list.
- Missing feature values are allowed at the row level and are median-imputed inside each training fold and in the final full-sample fit.

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
- No percentiles are clipped before ranking, after ranking, or before Elastic Net fitting.

## Scaling And Normalization

- Features are used in the fixed order: `fcf_to_ev`, `gross_profitability_assets`, `asset_growth_1y`, `cash_earnings_gap`, `momentum_12_1`.
- Missing feature values are imputed with the training-fold median (`SimpleImputer(strategy='median')`).
- Imputed features are standardized with `StandardScaler()` fit on the training fold only.
- The target is left on percentile-rank scale; no target normalization is applied.
- Final frozen coefficients come from refitting the same pipeline on the full matched sample.

## ElasticNetCV Search Space

- Estimator: `ElasticNetCV(random_state=42, max_iter=100000)`.
- `l1_ratio` remains at the sklearn default of `0.5` (not searched).
- Alpha search uses sklearn's default auto-generated log-spaced alpha path, from `alpha_max` down to `alpha_max * eps`, with `eps=1e-3` and the default number of alpha values.
- Inner CV is sklearn's default 5-fold cross-validation.
- On the frozen full-sample fit, the selected alpha is `0.079630` and the selected `l1_ratio` is `0.50`.

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
  - Market cap is current yfinance auto-adjusted close times shares_outstanding.
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
  - price source: yfinance auto-adjusted close history over the trailing ~400 calendar days
  - price_return_1m: latest price versus price on or before latest_date - 1 month
  - price_return_12m: latest price versus price on or before latest_date - 12 months
- Fallback logic:
  - If either component return is missing, the feature is null.
  - If 1-month gross return equals zero (a -100% return), the feature is null to avoid division by zero.

## Frozen CV Metrics

| Metric | Value |
| --- | --- |
| Spearman (CV) | 0.6473 |
| R^2 (CV) | 0.2947 |
| MAE (CV) | 0.1893 |

## Stability Checks

| Diagnostic | Value |
| --- | --- |
| Repeated-CV fold Spearman mean | 0.6619 |
| Repeated-CV fold Spearman std | 0.1582 |
| Repeated-CV fold Spearman 5th pct | 0.3624 |
| Repeated-CV fold Spearman 95th pct | 0.8791 |
| Bootstrap OOB Spearman mean | 0.6450 |
| Bootstrap OOB Spearman 95% CI | [0.3487, 0.8325] |

### Coefficient Sign Stability

| Feature | Expected | Expected Sign % | Positive % | Negative % | Zero % | Flip? |
| --- | --- | --- | --- | --- | --- | --- |
| fcf_to_ev | positive | 100.0% | 100.0% | 0.0% | 0.0% | No |
| gross_profitability_assets | positive | 0.0% | 0.0% | 15.6% | 84.4% | No |
| asset_growth_1y | negative | 10.4% | 0.4% | 10.4% | 89.2% | Yes |
| cash_earnings_gap | positive | 72.4% | 72.4% | 0.0% | 27.6% | No |
| momentum_12_1 | positive | 0.0% | 0.0% | 99.6% | 0.4% | No |

### Top-Quartile Stability

Top-quartile means the top 25% of predicted names inside each held-out fold.

Most consistently top-ranked:

| Ticker | Company | Top Quartile % | Hold-out Appearances |
| --- | --- | --- | --- |
| EPAM | EPAM SYSTEMS, INC. | 100.0% | 50 |
| MTCH | MATCH GROUP, INC. | 100.0% | 50 |
| OMC | OMNICOM GROUP INC. | 100.0% | 50 |
| TMUS | T-MOBILE US, INC. | 100.0% | 50 |
| DVN | DEVON ENERGY CORP/DE | 100.0% | 50 |
| GIS | GENERAL MILLS, INC. | 100.0% | 50 |
| ABNB | Airbnb, Inc. | 100.0% | 50 |
| BKNG | Booking Holdings Inc. | 98.0% | 50 |
| EQT | EQT CORPORATION | 96.0% | 50 |
| COP | ConocoPhillips | 96.0% | 50 |

Most consistently bottom-ranked:

| Ticker | Company | Top Quartile % | Hold-out Appearances |
| --- | --- | --- | --- |
| AVGO | Broadcom Inc. | 0.0% | 50 |
| GOOGL | ALPHABET INC. | 0.0% | 50 |
| CAT | CATERPILLAR INC | 0.0% | 50 |
| SO | The Southern Company | 0.0% | 50 |
| EXC | EXELON CORPORATION | 0.0% | 50 |
| TSLA | Tesla, Inc. | 0.0% | 50 |
| PWR | Quanta Services, Inc. | 0.0% | 50 |
| APD | AIR PRODUCTS AND CHEMICALS, INC. | 0.0% | 50 |
| MOS | Mosaic Co | 0.0% | 50 |
| CVX | Chevron Corp | 0.0% | 50 |

## Backcasting Verdict

Is the model stable enough to trust for backcasting? No.

- Repeated-CV fold Spearman is materially positive on average (0.662).
- The 5th percentile fold Spearman stays positive (0.362).
- The bootstrap OOB Spearman 95% interval stays above zero (0.349 to 0.832).
- Only 1/5 coefficients keep the expected sign in at least 75% of folds.
- The top-10 stable names land in the held-out top quartile 99.0% of the time on average.

## Frozen Linear Parameters

- Frozen scoring payload: `models/frozen_elasticnet_coefficients.json`.
- This payload includes the intercept, standardized-feature coefficients, imputer medians, scaler means/scales, and the selected alpha.
