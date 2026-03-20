Use this prompt with another AI when you only have a ticker and want a complete valuation input package for this DCF pipeline.

---

You are a senior equity research associate preparing a high-quality valuation input package for a DCF model.

Research the public company identified by ticker `{TICKER}` and return exactly one JSON object matching the schema below.

Hard rules:
- Output JSON only. Do not add commentary before or after the JSON.
- Use annual data only.
- Use the company's reporting currency when possible.
- Historical years should be the last 3 completed fiscal years if available.
- Forecast years should be the next 5 fiscal years.
- All currency amounts must be absolute values in full currency units, not millions.
- All margins, growth rates, tax rates, WACC, terminal growth, and utilization rates must be decimals, not percentages.
- If a field is unavailable, set it to `null` and explain the gap in `notes`.
- Distinguish clearly between historical facts, base-case assumptions, and scenario overrides.
- Prefer primary sources: company filings, annual reports, investor presentations, and official earnings releases. Use high-quality market-data summaries only when needed for current price or comp multiples.

What to collect:
- Company identity: company name, ticker, currency.
- Historical financials: revenue, EBIT, D&A, capex, and NWC for the last 3 completed fiscal years.
- Base-case forecast drivers for the next 5 fiscal years:
  - revenue growth
  - EBIT margin
  - D&A as a percent of sales
  - capex as a percent of sales
  - NWC as a percent of sales
- Base-case assumptions:
  - tax rate
  - WACC
  - terminal growth
  - terminal exit EBITDA multiple
  - terminal method: `gordon_growth`, `exit_multiple`, or `average`
  - cash
  - debt
  - net debt override if more reliable than separate cash and debt
  - investments
  - minority interest
  - preferred equity
  - diluted shares
  - current price
  - target EBIT margin
  - NOL balance
  - NOL utilization percent
- Comparable companies: 5 to 8 relevant peers with EV / NTM Revenue, EV / NTM EBITDA, and P / E NTM when available.
- Presentation content for an auto-generated pitchbook:
  - `subtitle`: short descriptor for the cover slide
  - `as_of_date`: exact date the current market / news context reflects
  - `company_overview`: 2 to 4 factual bullets on the business and positioning
  - `current_context`: 2 to 4 bullets on the latest earnings, guidance, operating update, or trading setup
  - `investment_highlights`: 3 to 5 bullets linking business drivers to the base-case thesis
  - `valuation_summary`: 2 to 4 bullets on why the selected DCF and comp outputs are reasonable
  - `catalysts`: 2 to 4 bullets on events that could move the stock
  - `key_risks`: 2 to 4 bullets on the main downside risks
  - `sources`: 3 to 8 primary or high-quality supporting sources with label, URL when available, and short notes
- Scenario overrides:
  - `bull`: modestly better growth, margins, and valuation assumptions
  - `bear`: modestly weaker growth, margins, and valuation assumptions
  - Only override values that change versus the base case.

Quality bar:
- Numbers must tie logically and be internally consistent.
- The base case should be sober, not promotional.
- Scenario deltas should be realistic, not cartoonish.
- Terminal assumptions must be economically sensible for the company and geography.
- Do not invent segment detail or balance-sheet items that are not defensible.
- Presentation bullets should read like concise sell-side / IB slide copy, not marketing copy.
- For `current_context` and `sources`, favor the most recent company filings, earnings releases, and investor materials available as of the research date.
- Use `notes` to flag uncertainty, definition choices, non-standard NWC construction, or data limitations.

Output schema:

```json
{
  "company_name": "string",
  "ticker": "string",
  "currency": "USD",
  "forecast_years": 5,
  "historical": {
    "revenue": {
      "2023": 0,
      "2024": 0,
      "2025": 0
    },
    "ebit": {
      "2023": 0,
      "2024": 0,
      "2025": 0
    },
    "da": {
      "2023": 0,
      "2024": 0,
      "2025": 0
    },
    "capex": {
      "2023": 0,
      "2024": 0,
      "2025": 0
    },
    "nwc": {
      "2023": 0,
      "2024": 0,
      "2025": 0
    }
  },
  "forecast": {
    "revenue_growth": {
      "2026": 0.00,
      "2027": 0.00,
      "2028": 0.00,
      "2029": 0.00,
      "2030": 0.00
    },
    "ebit_margin": {
      "2026": 0.00,
      "2027": 0.00,
      "2028": 0.00,
      "2029": 0.00,
      "2030": 0.00
    },
    "da_pct_sales": {
      "2026": 0.00,
      "2027": 0.00,
      "2028": 0.00,
      "2029": 0.00,
      "2030": 0.00
    },
    "capex_pct_sales": {
      "2026": 0.00,
      "2027": 0.00,
      "2028": 0.00,
      "2029": 0.00,
      "2030": 0.00
    },
    "nwc_pct_sales": {
      "2026": 0.00,
      "2027": 0.00,
      "2028": 0.00,
      "2029": 0.00,
      "2030": 0.00
    }
  },
  "assumptions": {
    "tax_rate": 0.00,
    "wacc": 0.00,
    "terminal_growth": 0.00,
    "terminal_exit_ebitda_multiple": null,
    "terminal_method": "average",
    "cash": 0,
    "debt": 0,
    "net_debt_override": null,
    "investments": 0,
    "minority_interest": 0,
    "preferred_equity": 0,
    "diluted_shares": 0,
    "current_price": null,
    "target_ebit_margin": 0.00,
    "nol_balance": 0,
    "nol_utilization_pct": 0.80
  },
  "comps": [
    {
      "company_name": "Peer 1",
      "ticker": "PEER1",
      "ev_ntm_revenue": 0.0,
      "ev_ntm_ebitda": 0.0,
      "pe_ntm": 0.0,
      "source": "string or null",
      "notes": "string or null"
    }
  ],
  "scenarios": {
    "bull": {
      "description": "short explanation",
      "forecast": {
        "revenue_growth": {
          "2026": 0.00,
          "2027": 0.00,
          "2028": 0.00,
          "2029": 0.00,
          "2030": 0.00
        },
        "ebit_margin": {
          "2026": 0.00,
          "2027": 0.00,
          "2028": 0.00,
          "2029": 0.00,
          "2030": 0.00
        },
        "da_pct_sales": {
          "2026": 0.00,
          "2027": 0.00,
          "2028": 0.00,
          "2029": 0.00,
          "2030": 0.00
        },
        "capex_pct_sales": {
          "2026": 0.00,
          "2027": 0.00,
          "2028": 0.00,
          "2029": 0.00,
          "2030": 0.00
        },
        "nwc_pct_sales": {
          "2026": 0.00,
          "2027": 0.00,
          "2028": 0.00,
          "2029": 0.00,
          "2030": 0.00
        }
      },
      "assumptions": {
        "wacc": 0.00,
        "terminal_growth": 0.00,
        "terminal_exit_ebitda_multiple": null,
        "terminal_method": "average",
        "current_price": null,
        "target_ebit_margin": 0.00
      }
    },
    "bear": {
      "description": "short explanation",
      "forecast": {
        "revenue_growth": {
          "2026": 0.00,
          "2027": 0.00,
          "2028": 0.00,
          "2029": 0.00,
          "2030": 0.00
        },
        "ebit_margin": {
          "2026": 0.00,
          "2027": 0.00,
          "2028": 0.00,
          "2029": 0.00,
          "2030": 0.00
        },
        "da_pct_sales": {
          "2026": 0.00,
          "2027": 0.00,
          "2028": 0.00,
          "2029": 0.00,
          "2030": 0.00
        },
        "capex_pct_sales": {
          "2026": 0.00,
          "2027": 0.00,
          "2028": 0.00,
          "2029": 0.00,
          "2030": 0.00
        },
        "nwc_pct_sales": {
          "2026": 0.00,
          "2027": 0.00,
          "2028": 0.00,
          "2029": 0.00,
          "2030": 0.00
        }
      },
      "assumptions": {
        "wacc": 0.00,
        "terminal_growth": 0.00,
        "terminal_exit_ebitda_multiple": null,
        "terminal_method": "average",
        "current_price": null,
        "target_ebit_margin": 0.00
      }
    }
  },
  "presentation": {
    "subtitle": "string",
    "as_of_date": "YYYY-MM-DD",
    "company_overview": [
      "short bullet 1",
      "short bullet 2"
    ],
    "current_context": [
      "short bullet 1",
      "short bullet 2"
    ],
    "investment_highlights": [
      "short bullet 1",
      "short bullet 2"
    ],
    "valuation_summary": [
      "short bullet 1",
      "short bullet 2"
    ],
    "catalysts": [
      "short bullet 1",
      "short bullet 2"
    ],
    "key_risks": [
      "short bullet 1",
      "short bullet 2"
    ],
    "sources": [
      {
        "label": "Latest annual report",
        "url": "https://example.com",
        "notes": "what it supports"
      }
    ]
  },
  "notes": [
    "short note 1",
    "short note 2"
  ]
}
```
