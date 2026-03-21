from distill import ModelMetrics, decile_labels_from_percentiles, determine_verdict, fractional_percentile_rank


def test_fractional_percentile_rank_spans_zero_to_one():
    ranks = fractional_percentile_rank([10.0, 20.0, 30.0, 40.0, 50.0])

    assert ranks.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_decile_labels_clip_top_bucket_to_nine():
    labels = decile_labels_from_percentiles([0.0, 0.1, 0.5, 0.95, 1.0])

    assert labels.tolist() == [0, 1, 5, 9, 9]


def test_determine_verdict_uses_spearman_as_primary_metric():
    elasticnet_metrics = ModelMetrics(spearman=0.20, r2=0.05, mae=0.20)
    xgb_metrics = ModelMetrics(spearman=0.24, r2=-0.10, mae=0.30)

    assert determine_verdict(elasticnet_metrics, xgb_metrics) == "Verdict: XGBoost Ranker wins."
