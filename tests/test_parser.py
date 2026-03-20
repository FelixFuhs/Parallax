import pytest

from parser import (
    DEFAULT_TAX_RATE,
    DEFAULT_TERMINAL_GROWTH,
    _coerce_number,
    parse_input,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0.5%", 0.005),
        ("15%", 0.15),
        (0.08, 0.08),
        pytest.param(1.5, 0.015, id="plain-number-over-one-is-treated-as-percent-style-input"),
        ("150%", 1.5),
    ],
)
def test_coerce_number_ratio_inputs(value, expected):
    assert _coerce_number(value, "ratio_field", ratio=True) == pytest.approx(expected)


def test_parse_input_applies_defaults_and_quality_flags_for_missing_assumptions():
    valuation_input = parse_input(
        {
            "historical": {"revenue": {"2025": 100}, "ebit": {"2025": 10}},
            "forecast": {"revenue_growth": {"2026": 0.1}},
            "assumptions": {"wacc": 0.09, "diluted_shares": 10},
        }
    )

    assert valuation_input.assumptions.tax_rate == pytest.approx(DEFAULT_TAX_RATE)
    assert valuation_input.assumptions.terminal_growth == pytest.approx(DEFAULT_TERMINAL_GROWTH)
    assert valuation_input.quality_flags == ("default_tax_rate", "default_terminal_growth")


def test_parse_input_respects_explicit_zero_nol_utilization():
    valuation_input = parse_input(
        {
            "historical": {"revenue": {"2025": 100}, "ebit": {"2025": 10}},
            "forecast": {"revenue_growth": {"2026": 0.1}},
            "assumptions": {
                "wacc": 0.09,
                "diluted_shares": 10,
                "tax_rate": 0.0,
                "terminal_growth": 0.0,
                "nol_utilization_pct": 0.0,
            },
        }
    )

    assert valuation_input.assumptions.tax_rate == pytest.approx(0.0)
    assert valuation_input.assumptions.terminal_growth == pytest.approx(0.0)
    assert valuation_input.assumptions.nol_utilization_pct == pytest.approx(0.0)
    assert valuation_input.quality_flags == ()
