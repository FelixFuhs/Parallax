# Parallax Lessons Learned

This document captures the operational lessons from building and running Parallax through the first EDGAR validation pass, Nano smoke tests, and the first full 100-ticker Nano batch.

## OpenRouter Cheap Tier

- `response_format: {"type": "json_object"}` cannot be combined with `:online` on the cheap tier reliably. The working pattern in this repo is:
  - do not request `json_object` for `openai/gpt-5.4-nano:online`
  - enable the `response-healing` plugin
  - parse the first valid JSON object with `json.JSONDecoder().raw_decode()`
- The cheap-tier healing path in [openrouter.py](C:/Users/Anwender/OneDrive/Desktop/Parallax/openrouter.py) is materially better than naive brace slicing. It can recover fenced JSON and ignore trailing commentary.
- Cheap-tier failures are not just `429`s. Under load, OpenRouter also returned payloads with no `choices[0]`, or empty message content. Those need to be treated as operational failures even when token usage is zero.

## Rate Limits And Retries

- Raising the cheap tier to 50 req/min worked mechanically, but it was too aggressive for a 100-name bulk pass.
- On the first 100-ticker Nano run at 50 req/min, about 34% of tickers failed with `OpenRouter response did not include choices[0].`
- The total first-pass failure rate at that setting was worse than the raw 34% payload error rate because some additional names failed with empty responses or incomplete JSON.
- Recommended practical rate for cheap-tier bulk runs: 20 to 30 req/min.
- Always plan a retry pass. The retry recovered a meaningful number of names without any code changes.
- Do not assume the rate limiter is the only control surface. The provider can still degrade structurally before it starts returning clean `429`s.

## EDGAR Validation Method

- Spot-check 5 tickers against StockAnalysis annual pages:
  - revenue
  - net income
  - total debt
  - tolerance: within 10%
- Always check fiscal years. Three bad pulls stood out immediately:
  - `DUK` pulled FY2016
  - `NEE` pulled FY2012
  - `TJX` pulled FY2017 / period end 2018-02-03
- Always inspect derived-feature sanity:
  - `MCD` had broken `shares_outstanding`, which corrupted `market_cap` and produced nonsense `fcf_yield`
  - extreme ROIC values for `BKNG`, `FTNT`, `GE`, and `ORLY` were likely driven by negative or tiny book equity / invested capital, not necessarily extraction bugs
- Gross margin coverage was weak, but XGBoost-style tabular models can tolerate that if the rest of the feature set is sound and missingness is handled explicitly.

## Parser Bugs Found And Fixed

- Ratio parsing corruption for sub-1% values:
  - values like `0.5%` must become `0.005`
  - the fix was to distinguish percent-suffixed inputs from already-decimal ratios so sub-1% values are not divided twice
- Silent `or`-based defaults overwriting explicit zeros:
  - explicit `0.0` values for fields like tax rate, terminal growth, and NOL utilization must survive parsing
  - the fix was to use `value is not None` semantics instead of truthiness-based defaults
- First-brace / last-brace JSON healing was too brittle:
  - it failed when the model returned valid JSON followed by trailing explanation text
  - the fix was to walk the string and use `json.JSONDecoder().raw_decode()` to extract the first valid JSON object
- The parser test suite in [tests/test_parser.py](C:/Users/Anwender/OneDrive/Desktop/Parallax/tests/test_parser.py) and [tests/test_openrouter.py](C:/Users/Anwender/OneDrive/Desktop/Parallax/tests/test_openrouter.py) now covers these cases.

## Nano Vs Full Tier

- Nano is cheap enough for broad sweeps:
  - roughly `$0.02` per report in typical runs
- Full tier is much more expensive:
  - budget roughly `$0.87` per report
  - real observed cost can vary materially with prompt length, output length, and online context
- Nano has a systematic bearish bias in the saved base-case upsides, but the cross-sectional spread is still wide enough to use as a distillation target.
- Nano still fails to produce complete JSON for about 10% to 15% of names with unusual accounting or awkward output structure, and can fail much more often if the request rate is pushed too hard.

## Parallelism

- `--parallel 100` is acceptable from the script's point of view. The internal rate limiter is still the real throttle.
- The main scaling risk is provider behavior, not local thread count.
- On Windows, backgrounding the Python process opens a visible console window. That is normal for this setup.
- Same-day reruns can be skipped because report filenames are date-keyed. If you need a true rerun, move or rename existing `YYYY-MM-DD` report files first.

## BYOK

- When using your own OpenAI key through OpenRouter, billing lands on the OpenAI dashboard, not the OpenRouter dashboard.
- OpenRouter still handles routing and response shape, but cost attribution follows the underlying provider key.

## Additional Lessons

- A saved report is not automatically a usable training row:
  - `CMS` saved successfully but carried `stale_price`, so it had no usable upside target
- Bulk matching should define "clean EDGAR" explicitly:
  - no EDGAR `error`
  - fiscal year at least 2024
  - manually exclude known broken rows like `MCD`
- Retry-only reruns are important for both cost control and data hygiene. Limiting every ticker to at most two attempts kept the batch process predictable.
- The dominant production failures were operational and structural, not valuation-specific:
  - missing `choices[0]`
  - empty message content
  - missing `historical.revenue`
  - missing `assumptions.wacc`
  - terminal-method outputs that referenced exit multiples without providing the multiple
- Keep a separate backup folder when moving prior reports out of the way for reruns. That preserves the original artifacts without polluting the active training set scan.
