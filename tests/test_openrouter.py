import pytest

from dcf import run_dcf
from openrouter import (
    OpenRouterError,
    TIER_CONFIGS,
    UsageStats,
    _build_saved_report,
    _extract_json_object_with_healing,
)
from parser import parse_input


def test_build_saved_report_merges_parser_quality_flags_into_meta():
    valuation_input = parse_input(
        {
            "company_name": "Test Co",
            "ticker": "TST",
            "currency": "USD",
            "historical": {
                "revenue": {"2025": 1000},
                "ebit": {"2025": 200},
                "da": {"2025": 50},
                "capex": {"2025": 40},
                "nwc": {"2025": 100},
            },
            "forecast": {
                "revenue_growth": {"2026": 0.05},
                "ebit_margin": {"2026": 0.2},
                "da_pct_sales": {"2026": 0.05},
                "capex_pct_sales": {"2026": 0.04},
                "nwc_pct_sales": {"2026": 0.1},
            },
            "assumptions": {
                "wacc": 0.1,
                "diluted_shares": 100,
                "terminal_method": "gordon_growth",
            },
        }
    )

    report = _build_saved_report(
        valuation_input=valuation_input,
        valuation_result=run_dcf(valuation_input),
        config=TIER_CONFIGS["cheap"],
        usage=UsageStats(prompt_tokens=1, completion_tokens=2, reasoning_tokens=3),
        estimated_cost_usd=0.01,
        quality_flags=["missing_comps"],
    )

    assert "quality_flags" not in report
    assert report["_meta"]["quality_flags"] == [
        "default_tax_rate",
        "default_terminal_growth",
        "missing_comps",
    ]


def test_extract_json_object_with_healing_parses_fenced_json_with_trailing_commentary():
    parsed, warnings = _extract_json_object_with_healing(
        '```json\n{"a": 1}\n```\nExplanation with nested example {"b": 2}'
    )

    assert parsed == {"a": 1}
    assert warnings == [
        "healed JSON by stripping markdown code fences before parsing.",
        "healed JSON by extracting the first JSON object from surrounding text.",
    ]


def test_extract_json_object_with_healing_returns_first_json_object_when_multiple_are_present():
    parsed, warnings = _extract_json_object_with_healing('preface {"a": 1} {"b": 2}')

    assert parsed == {"a": 1}
    assert warnings == ["healed JSON by extracting the first JSON object from surrounding text."]


def test_extract_json_object_with_healing_raises_for_missing_json():
    with pytest.raises(OpenRouterError):
        _extract_json_object_with_healing("no json here")
