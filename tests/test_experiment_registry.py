from pathlib import Path

from experiment_registry import ROOT, build_experiment_metadata


def test_experiment_metadata_uses_repo_relative_artifact_paths():
    metadata = build_experiment_metadata(
        experiment_id="test",
        feature_config={},
        model_config={},
        universe_config={},
        backtest_config={},
        artifacts={"requirements": ROOT / "requirements.txt"},
        data_snapshot_paths={"requirements": ROOT / "requirements.txt"},
        label_snapshot_paths={},
    )

    assert metadata["artifacts"]["requirements"] == "requirements.txt"
    assert metadata["data_snapshots"]["requirements"] == "requirements.txt"
    assert not Path(metadata["artifacts"]["requirements"]).is_absolute()

