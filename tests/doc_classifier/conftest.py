from __future__ import annotations

import pytest

import traenslenzor.doc_classifier.utils.schemas as schemas

if not hasattr(schemas, "MetricName") and hasattr(schemas, "Metric"):
    # Backwards compatibility shim for renamed Metric enum.
    schemas.MetricName = schemas.Metric

from traenslenzor.doc_classifier.configs.path_config import PathConfig


@pytest.fixture
def project_root(tmp_path):
    """Provide an isolated project root and reset the PathConfig singleton."""

    root = tmp_path / "project"
    root.mkdir()
    PathConfig._instances.pop(PathConfig, None)
    try:
        yield root
    finally:
        PathConfig._instances.pop(PathConfig, None)


@pytest.fixture
def fresh_path_config(project_root) -> PathConfig:
    """Return a PathConfig instance rooted at the isolated project directory."""

    config = PathConfig()
    config.root = project_root
    config.data_root = project_root / ".data"
    config.hf_cache = config.data_root / "hf_cache"
    logs_dir = project_root / ".logs"
    config.checkpoints = logs_dir / "checkpoints"
    config.wandb = logs_dir / "wandb"
    config.configs_dir = project_root / ".configs"
    return config
