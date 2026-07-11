from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def repo_relative(path: str | Path, *, root: Path = ROOT) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError:
        return candidate.name


def file_sha256(path: str | Path) -> str | None:
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_git_commit(root: Path = ROOT) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def build_experiment_metadata(
    *,
    experiment_id: str,
    feature_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    universe_config: Mapping[str, Any],
    backtest_config: Mapping[str, Any],
    data_snapshot_paths: Mapping[str, str | Path] | None = None,
    label_snapshot_paths: Mapping[str, str | Path] | None = None,
    artifacts: Mapping[str, str | Path] | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    data_paths = data_snapshot_paths or {}
    label_paths = label_snapshot_paths or {}
    artifact_paths = artifacts or {}
    data_hashes = {name: file_sha256(path) for name, path in data_paths.items()}
    label_hashes = {name: file_sha256(path) for name, path in label_paths.items()}
    return {
        "experiment_id": experiment_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": current_git_commit(root),
        "data_snapshot_hash": payload_sha256(data_hashes),
        "label_snapshot_hash": payload_sha256(label_hashes),
        "data_snapshots": {name: repo_relative(path, root=root) for name, path in data_paths.items()},
        "label_snapshots": {name: repo_relative(path, root=root) for name, path in label_paths.items()},
        "feature_config": dict(feature_config),
        "model_config": dict(model_config),
        "universe_config": dict(universe_config),
        "backtest_config": dict(backtest_config),
        "artifacts": {name: repo_relative(path, root=root) for name, path in artifact_paths.items()},
    }


def write_experiment_metadata(path: str | Path, metadata: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(metadata), indent=2, sort_keys=True), encoding="utf-8")

