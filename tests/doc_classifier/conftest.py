from __future__ import annotations

from pathlib import Path

import pytest

from traenslenzor.doc_classifier.configs.path_config import PathConfig

_DEFAULT_CKPT_NAME = "alexnet-scratch-epoch=epoch=1-val_loss=val/loss=0.84.ckpt"


@pytest.fixture
def running_file_server(file_server: None) -> None:
    """Reuse the session-wide file server fixture from tests/conftest.py."""
    yield


@pytest.fixture(autouse=True)
def ensure_dummy_checkpoint() -> Path:
    """Create a stub checkpoint so DocClassifierMCPConfig validation passes."""
    cfg = PathConfig()
    ckpt_path = cfg.checkpoints / _DEFAULT_CKPT_NAME
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.touch(exist_ok=True)
    return ckpt_path


@pytest.fixture
def project_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary project root for path-related tests."""
    return tmp_path_factory.mktemp("project_root")


@pytest.fixture
def fresh_path_config(project_root: Path) -> PathConfig:
    """Return a PathConfig instance rooted at a temporary directory."""
    cfg = PathConfig()
    cfg.root = project_root
    cfg.data_root = project_root / ".cache"
    cfg.hf_cache = cfg.data_root / "hf_cache"
    cfg.checkpoints = project_root / ".artifacts" / "ckpts"
    cfg.wandb = project_root / ".artifacts" / "wandb"
    cfg.configs_dir = project_root / ".configs"
    for path in (cfg.data_root, cfg.hf_cache, cfg.checkpoints, cfg.wandb, cfg.configs_dir):
        path.mkdir(parents=True, exist_ok=True)
    # Ensure the default checkpoint path exists for components that validate it eagerly
    default_ckpt = cfg.checkpoints / _DEFAULT_CKPT_NAME
    default_ckpt.parent.mkdir(parents=True, exist_ok=True)
    default_ckpt.touch(exist_ok=True)
    return cfg
