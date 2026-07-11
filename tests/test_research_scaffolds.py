from research_scaffolds import (
    COMMON_DCF_OUTPUT_SCHEMA,
    QuarterlyFundamentalSnapshot,
    TextFeatureExperimentConfig,
    render_sector_dcf_prompt_context,
    required_sector_template_names,
    sector_template_catalog,
    sector_template_for,
    survivor_only_universe_assumption,
    ttm_rollup,
)


def test_quarterly_ttm_rollup_requires_four_available_quarters():
    snapshots = [
        QuarterlyFundamentalSnapshot("TST", "2025Q1", "2025-03-31", "2025-05-01", revenue=10),
        QuarterlyFundamentalSnapshot("TST", "2025Q2", "2025-06-30", "2025-08-01", revenue=11),
        QuarterlyFundamentalSnapshot("TST", "2025Q3", "2025-09-30", "2025-11-01", revenue=12),
        QuarterlyFundamentalSnapshot("TST", "2025Q4", "2025-12-31", "2026-02-01", revenue=13),
    ]

    assert ttm_rollup(snapshots, "revenue") == 46.0
    assert ttm_rollup(snapshots[:3], "revenue") is None


def test_sector_templates_keep_common_output_schema():
    template = sector_template_for("Information Technology - Software")

    assert template.name == "Software"
    assert template.output_schema == COMMON_DCF_OUTPUT_SCHEMA
    assert "recurring revenue growth" in template.forecast_driver_fields
    assert sector_template_for("Unknown").name == "General"


def test_sector_template_catalog_covers_required_sectors_with_stable_schema():
    catalog = sector_template_catalog()
    names = {entry["name"] for entry in catalog}

    assert names == set(required_sector_template_names())
    for entry in catalog:
        assert tuple(entry["output_schema"]) == COMMON_DCF_OUTPUT_SCHEMA
        assert entry["forecast_driver_fields"]
        assert entry["risk_checks"]
        assert entry["source_requirements"]


def test_sector_prompt_context_preserves_common_schema_without_new_json_fields():
    context = render_sector_dcf_prompt_context("Health Care - Biotechnology")

    assert "Selected template: Healthcare" in context
    assert "Do not add sector-specific JSON fields" in context
    assert "historical_financials" in context
    assert "pipeline or product-cycle contribution" in context


def test_universe_assumption_records_survivor_bias():
    assumption = survivor_only_universe_assumption("tickers.txt")

    assert assumption.point_in_time_membership is False
    assert "not CRSP/Compustat-quality" in assumption.survivor_bias_warning


def test_text_feature_config_is_separate_experiment_c():
    config = TextFeatureExperimentConfig()

    assert config.experiment_id == "experiment_c_llm_text_features"
    assert config.separate_from_dcf_labels is True
    assert "risk_factor_novelty" in config.feature_names
