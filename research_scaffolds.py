from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

COMMON_DCF_OUTPUT_SCHEMA = (
    "historical_financials",
    "forecast_drivers",
    "terminal_assumptions",
    "risk_assumptions",
    "implied_irr",
    "quality_flags",
    "sources",
)


@dataclass(frozen=True)
class QuarterlyFundamentalSnapshot:
    ticker: str
    fiscal_period: str
    period_end: str
    filed: str | None
    revenue: float | None = None
    gross_profit: float | None = None
    operating_cash_flow: float | None = None
    capex: float | None = None
    total_assets: float | None = None
    total_debt: float | None = None
    cash: float | None = None
    shares: float | None = None


def ttm_rollup(
    snapshots: Sequence[QuarterlyFundamentalSnapshot],
    field_name: str,
    *,
    periods: int = 4,
) -> float | None:
    ordered = sorted(snapshots, key=lambda snapshot: snapshot.period_end)
    values: list[float] = []
    for snapshot in reversed(ordered):
        value = getattr(snapshot, field_name)
        if value is None:
            continue
        values.append(float(value))
        if len(values) == periods:
            break
    if len(values) < periods:
        return None
    return float(sum(values))


@dataclass(frozen=True)
class UniverseAssumption:
    universe_id: str
    source_path: str
    point_in_time_membership: bool
    survivor_bias_warning: str


def survivor_only_universe_assumption(tickers_file: str | Path) -> UniverseAssumption:
    path = Path(tickers_file)
    return UniverseAssumption(
        universe_id=path.stem,
        source_path=path.as_posix(),
        point_in_time_membership=False,
        survivor_bias_warning=(
            "Universe is derived from a current ticker file and is not CRSP/Compustat-quality "
            "point-in-time membership."
        ),
    )


@dataclass(frozen=True)
class SectorTemplate:
    name: str
    aliases: tuple[str, ...]
    prompt_focus: tuple[str, ...]
    forecast_driver_fields: tuple[str, ...]
    terminal_assumption_focus: tuple[str, ...]
    risk_checks: tuple[str, ...]
    source_requirements: tuple[str, ...]
    output_schema: tuple[str, ...] = COMMON_DCF_OUTPUT_SCHEMA


SECTOR_DCF_TEMPLATES: Mapping[str, SectorTemplate] = {
    "software": SectorTemplate(
        name="Software",
        aliases=("software", "saas", "systems software", "application software", "cloud"),
        prompt_focus=("ARR/revenue durability", "margin path", "stock-based compensation"),
        forecast_driver_fields=("recurring revenue growth", "net retention/churn", "sales efficiency", "SBC dilution"),
        terminal_assumption_focus=("normalized FCF margin", "mature revenue growth", "SBC cash-adjustment treatment"),
        risk_checks=("customer concentration", "AI/platform disruption", "capitalized software or SBC distortion"),
        source_requirements=("latest 10-K revenue recognition note", "segment or product revenue disclosure", "latest earnings release"),
    ),
    "semiconductors": SectorTemplate(
        name="Semiconductors",
        aliases=("semiconductor", "semiconductors", "chip", "chips"),
        prompt_focus=("cycle normalization", "gross margin", "capex intensity"),
        forecast_driver_fields=("unit/content growth", "cycle-normal revenue", "gross margin by mix", "fabless/foundry capex needs"),
        terminal_assumption_focus=("through-cycle margin", "cycle-normal growth", "maintenance capex intensity"),
        risk_checks=("inventory correction", "customer concentration", "export controls or supply-chain concentration"),
        source_requirements=("latest 10-K segment/end-market disclosure", "recent quarterly inventory commentary", "capex or manufacturing footprint disclosure"),
    ),
    "energy": SectorTemplate(
        name="Energy",
        aliases=("energy", "oil", "gas", "exploration", "production", "midstream"),
        prompt_focus=("commodity deck", "maintenance capex", "reserve life"),
        forecast_driver_fields=("production/volume path", "realized commodity price deck", "lifting/transport cost", "maintenance capex"),
        terminal_assumption_focus=("mid-cycle commodity assumptions", "reserve-life fade", "reinvestment rate"),
        risk_checks=("commodity sensitivity", "reserve replacement", "environmental/regulatory liabilities"),
        source_requirements=("latest reserves or production disclosure", "latest capex budget", "commodity sensitivity or hedging disclosure"),
    ),
    "industrials": SectorTemplate(
        name="Industrials",
        aliases=("industrial", "industrials", "machinery", "aerospace", "transportation"),
        prompt_focus=("order backlog", "cycle risk", "working capital"),
        forecast_driver_fields=("orders/backlog conversion", "volume/mix", "price/cost spread", "working-capital normalization"),
        terminal_assumption_focus=("mid-cycle margin", "maintenance capex", "normalized working capital"),
        risk_checks=("cycle downturn", "supply-chain execution", "project/accounting contract risk"),
        source_requirements=("latest backlog/order disclosure", "segment margin disclosure", "working-capital or cash-conversion commentary"),
    ),
    "healthcare": SectorTemplate(
        name="Healthcare",
        aliases=("healthcare", "health care", "pharma", "biotech", "medical", "managed care", "medtech"),
        prompt_focus=("pipeline durability", "patent cliffs", "reimbursement risk"),
        forecast_driver_fields=("volume/script growth", "pricing/reimbursement", "pipeline or product-cycle contribution", "R&D/sales mix"),
        terminal_assumption_focus=("post-patent or post-cycle growth", "normalized R&D burden", "regulatory risk premium"),
        risk_checks=("patent/exclusivity cliffs", "trial or approval risk", "payer/reimbursement pressure"),
        source_requirements=("latest 10-K product/segment disclosure", "pipeline or regulatory update", "latest reimbursement or payer commentary"),
    ),
    "utilities": SectorTemplate(
        name="Utilities",
        aliases=("utility", "utilities", "electric utilities", "multi-utilities", "water utilities"),
        prompt_focus=("rate base", "allowed ROE", "regulatory lag"),
        forecast_driver_fields=("rate-base growth", "allowed ROE/equity layer", "customer load growth", "regulated capex plan"),
        terminal_assumption_focus=("allowed ROE sustainability", "regulated asset growth", "capital structure"),
        risk_checks=("rate-case lag", "storm/wildfire liability", "financing and regulatory disallowance"),
        source_requirements=("latest rate-base/capex plan", "recent rate-case order or filing", "financing plan or credit metrics"),
    ),
    "consumer": SectorTemplate(
        name="Consumer",
        aliases=("consumer", "retail", "restaurant", "staples", "discretionary", "brand"),
        prompt_focus=("same-store sales", "gross margin", "brand durability"),
        forecast_driver_fields=("same-store sales or volume", "price/mix", "gross margin", "store/unit growth"),
        terminal_assumption_focus=("steady-state unit economics", "brand moat durability", "normalized reinvestment"),
        risk_checks=("demand elasticity", "input-cost pressure", "channel or private-label pressure"),
        source_requirements=("latest traffic/same-store sales or volume disclosure", "gross margin bridge", "brand/channel commentary"),
    ),
    "general": SectorTemplate(
        name="General",
        aliases=("general", "unknown"),
        prompt_focus=("revenue growth", "margin path", "capital intensity"),
        forecast_driver_fields=("revenue growth", "EBIT margin", "D&A", "capex", "working capital"),
        terminal_assumption_focus=("terminal growth", "normalized EBIT margin", "maintenance capital intensity"),
        risk_checks=("cyclicality", "competitive pressure", "balance-sheet risk"),
        source_requirements=("latest annual report", "latest earnings release", "current price source"),
    ),
}


def _normalize_template_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def sector_template_for(sector_name: str | None) -> SectorTemplate:
    if not sector_name:
        return SECTOR_DCF_TEMPLATES["general"]
    normalized = _normalize_template_key(sector_name)
    for key, template in SECTOR_DCF_TEMPLATES.items():
        if _normalize_template_key(key) in normalized:
            return template
        if any(_normalize_template_key(alias) in normalized for alias in template.aliases):
            return template
    return SECTOR_DCF_TEMPLATES["general"]


def required_sector_template_names() -> tuple[str, ...]:
    return ("Software", "Semiconductors", "Energy", "Industrials", "Healthcare", "Utilities", "Consumer", "General")


def sector_template_catalog() -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "name": template.name,
            "aliases": list(template.aliases),
            "prompt_focus": list(template.prompt_focus),
            "forecast_driver_fields": list(template.forecast_driver_fields),
            "terminal_assumption_focus": list(template.terminal_assumption_focus),
            "risk_checks": list(template.risk_checks),
            "source_requirements": list(template.source_requirements),
            "output_schema": list(template.output_schema),
        }
        for key, template in SECTOR_DCF_TEMPLATES.items()
    ]


def render_sector_dcf_prompt_context(sector_name: str | None) -> str:
    template = sector_template_for(sector_name)
    lines = [
        "",
        "Sector-specific DCF template context:",
        f"- Selected template: {template.name}",
        "- Keep the common output schema stable. The JSON fields still map to:",
        "  " + ", ".join(template.output_schema),
        "- Give extra attention to these sector forecast drivers:",
    ]
    lines.extend(f"  - {item}" for item in template.forecast_driver_fields)
    lines.append("- Terminal assumption focus:")
    lines.extend(f"  - {item}" for item in template.terminal_assumption_focus)
    lines.append("- Sector risk checks:")
    lines.extend(f"  - {item}" for item in template.risk_checks)
    lines.append("- Preferred source evidence:")
    lines.extend(f"  - {item}" for item in template.source_requirements)
    lines.append("Do not add sector-specific JSON fields; express sector detail inside the existing forecast, assumptions, presentation, sources, and notes fields.")
    return "\n".join(lines)


@dataclass(frozen=True)
class TextFeatureExperimentConfig:
    experiment_id: str = "experiment_c_llm_text_features"
    allowed_sources: tuple[str, ...] = ("10-K MD&A", "10-K risk factors", "10-Q MD&A")
    feature_names: tuple[str, ...] = (
        "tone_change",
        "uncertainty_change",
        "risk_factor_novelty",
        "management_hedging",
        "capital_allocation_discipline",
        "competitive_pressure",
        "pricing_power",
        "demand_weakness",
        "supply_chain_stress",
        "regulatory_pressure",
        "accounting_aggressiveness",
        "guidance_credibility",
    )
    separate_from_dcf_labels: bool = True

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)
