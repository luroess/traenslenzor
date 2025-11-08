from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from traenslenzor.doc_classifier.configs import ExperimentConfig, OptunaConfig
from traenslenzor.doc_classifier.utils import Stage

EXPERIMENT_MODULE = "traenslenzor.doc_classifier.configs.experiment_config"


def test_migrate_legacy_keys_and_ckpt_resolution(tmp_path, fresh_path_config):
    ckpt = fresh_path_config.checkpoints / "legacy.ckpt"
    ckpt.touch()

    data = ExperimentConfig._migrate_legacy_keys({"from_ckpt": ckpt.name})
    assert data["ckpt_path"] == ckpt.name

    cfg = ExperimentConfig(paths=fresh_path_config, ckpt_path=ckpt.as_posix())
    assert cfg.ckpt_path == ckpt


def test_stage_validator_rejects_unknown_value():
    with pytest.raises(ValueError):
        ExperimentConfig(stage="unknown")


def test_resolve_ckpt_path_requires_existing_file(tmp_path, fresh_path_config):
    missing = fresh_path_config.checkpoints / "missing.ckpt"
    with pytest.raises(FileNotFoundError):
        ExperimentConfig(paths=fresh_path_config, ckpt_path=missing)


def test_setup_target_invokes_components(monkeypatch, fresh_path_config):
    config = ExperimentConfig(paths=fresh_path_config)

    trainer = MagicMock()
    module = MagicMock()
    module.config = SimpleNamespace(num_classes=2)

    class DummyDataModule:
        def __init__(self):
            self.train_ds = SimpleNamespace(num_classes=2)
            self.setup_stage = None

        def setup(self, stage=None):
            self.setup_stage = stage

    datamodule = DummyDataModule()

    monkeypatch.setattr(f"{EXPERIMENT_MODULE}.TrainerFactoryConfig.setup_target", lambda self: trainer)
    monkeypatch.setattr(f"{EXPERIMENT_MODULE}.DocClassifierConfig.setup_target", lambda self: module)
    monkeypatch.setattr(f"{EXPERIMENT_MODULE}.DocDataModuleConfig.setup_target", lambda self: datamodule)

    trainer_obj, module_obj, dm_obj = config.setup_target(setup_stage="val")
    assert trainer_obj is trainer
    assert module_obj is module
    assert dm_obj is datamodule
    assert dm_obj.setup_stage == Stage.VAL


def test_setup_target_and_run_respects_stage(monkeypatch, fresh_path_config, tmp_path):
    config = ExperimentConfig(paths=fresh_path_config)
    ckpt_file = fresh_path_config.checkpoints / "state.ckpt"
    ckpt_file.touch()
    config.ckpt_path = ckpt_file

    trainer = MagicMock()
    trainer.fit = MagicMock()
    trainer.validate = MagicMock()
    trainer.test = MagicMock()

    monkeypatch.setattr(
        ExperimentConfig,
        "setup_target",
        lambda self, setup_stage=None: (trainer, MagicMock(), MagicMock()),
    )

    config.setup_target_and_run(stage=Stage.TRAIN)
    trainer.fit.assert_called_once()

    config.setup_target_and_run(stage=Stage.VAL)
    trainer.validate.assert_called_once()

    config.setup_target_and_run(stage=Stage.TEST)
    trainer.test.assert_called_once()


def test_from_puml_file_uses_configs_dir(tmp_path, fresh_path_config):
    cfg = ExperimentConfig(paths=fresh_path_config)
    path = cfg.export_puml("relative.puml")

    loaded = ExperimentConfig.from_puml_file("relative.puml", path_config=fresh_path_config)
    assert loaded.run_name == cfg.run_name
    assert path.exists()


def test_run_optuna_study_executes_trials(monkeypatch, fresh_path_config):
    cfg = ExperimentConfig(paths=fresh_path_config, optuna_config=OptunaConfig(n_trials=1))

    class DummyTrainer:
        def __init__(self):
            self.callback_metrics = {"val/loss": 1.0}

        def fit(self, *_, **__):
            return None

    dummy_trainer = DummyTrainer()

    def fake_setup_target(self, setup_stage=None, trial=None):
        return dummy_trainer, MagicMock(), MagicMock()

    monkeypatch.setattr(ExperimentConfig, "setup_target", fake_setup_target)

    class DummyTrial:
        number = 0
        params = {"lr": 0.1}

    class DummyStudy:
        def optimize(self, objective, n_trials):
            objective(DummyTrial())

    monkeypatch.setattr(f"{EXPERIMENT_MODULE}.OptunaConfig.setup_target", lambda self: DummyStudy())
    monkeypatch.setattr(
        "traenslenzor.doc_classifier.configs.experiment_config.wandb",
        SimpleNamespace(run=None, finish=lambda: None),
    )

    cfg.run_optuna_study()
    assert cfg.trainer_config.wandb_config.group == "optuna"
    assert cfg.trainer_config.wandb_config.job_type.startswith("Opt:")
