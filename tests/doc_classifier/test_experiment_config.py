from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import traenslenzor.doc_classifier.lightning.lit_module as lit_module
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


def test_migrate_legacy_keys_passthrough():
    assert ExperimentConfig._migrate_legacy_keys("raw-string") == "raw-string"


def test_resolve_ckpt_path_requires_existing_file(tmp_path, fresh_path_config):
    missing = fresh_path_config.checkpoints / "missing.ckpt"
    with pytest.raises(FileNotFoundError):
        ExperimentConfig(paths=fresh_path_config, ckpt_path=missing)


def test_stage_validator_handles_none(monkeypatch):
    monkeypatch.setattr(
        "traenslenzor.doc_classifier.configs.experiment_config.Stage.from_str",
        lambda value: None,
    )
    with pytest.raises(ValueError):
        ExperimentConfig._parse_stage("custom")


def test_resolve_ckpt_path_helper(tmp_path, fresh_path_config):
    ckpt = fresh_path_config.checkpoints / "helper.ckpt"
    ckpt.touch()

    info = SimpleNamespace(data={"paths": fresh_path_config})
    resolved = ExperimentConfig._resolve_ckpt_path("helper.ckpt", info)
    assert resolved == ckpt


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

    monkeypatch.setattr(
        f"{EXPERIMENT_MODULE}.TrainerFactoryConfig.setup_target", lambda self: trainer
    )
    monkeypatch.setattr(
        f"{EXPERIMENT_MODULE}.DocClassifierConfig.setup_target", lambda self: module
    )
    monkeypatch.setattr(
        f"{EXPERIMENT_MODULE}.DocDataModuleConfig.setup_target", lambda self: datamodule
    )

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


def test_setup_target_loads_checkpoint(monkeypatch, fresh_path_config, tmp_path):
    ckpt = fresh_path_config.checkpoints / "load_me.ckpt"
    ckpt.touch()

    class Loader(lit_module.DocClassifierModule):
        called_with: dict[str, Any] = {}

        def __init__(self, config):
            super().__init__(config)

        @classmethod
        def load_from_checkpoint(cls, checkpoint_path, params):
            cls.called_with = {"path": checkpoint_path, "params": params}
            return MagicMock(config=SimpleNamespace(num_classes=2))

    trainer = MagicMock()

    class DummyDataModule:
        def __init__(self):
            self.train_ds = SimpleNamespace(num_classes=2)

        def setup(self, stage=None):
            self.stage = stage

    datamodule = DummyDataModule()

    monkeypatch.setattr(
        f"{EXPERIMENT_MODULE}.TrainerFactoryConfig.setup_target", lambda self: trainer
    )
    monkeypatch.setattr(
        f"{EXPERIMENT_MODULE}.DocDataModuleConfig.setup_target", lambda self: datamodule
    )

    # Use a module config with the Loader class as target
    module_config = lit_module.DocClassifierConfig()
    # Can't set target directly due to validation, so we patch the class method instead
    original_load = lit_module.DocClassifierModule.load_from_checkpoint
    monkeypatch.setattr(
        lit_module.DocClassifierModule, "load_from_checkpoint", Loader.load_from_checkpoint
    )

    cfg = ExperimentConfig(
        paths=fresh_path_config, ckpt_path=ckpt.as_posix(), module_config=module_config
    )
    trainer_obj, module_obj, datamodule_obj = cfg.setup_target()

    # Restore original
    monkeypatch.setattr(lit_module.DocClassifierModule, "load_from_checkpoint", original_load)

    assert trainer_obj is trainer
    assert datamodule_obj is datamodule
    assert Loader.called_with["path"] == ckpt.as_posix()


def test_run_optuna_requires_config(fresh_path_config):
    cfg = ExperimentConfig(paths=fresh_path_config, optuna_config=None)
    with pytest.raises(ValueError):
        cfg.run_optuna_study()


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
    finish_called = SimpleNamespace(count=0)

    def finish():
        finish_called.count += 1

    monkeypatch.setattr(
        "traenslenzor.doc_classifier.configs.experiment_config.wandb",
        SimpleNamespace(run=True, finish=finish),
    )

    cfg.run_optuna_study()
    assert finish_called.count == 1
    assert cfg.trainer_config.wandb_config.group == "optuna"
    assert cfg.trainer_config.wandb_config.job_type.startswith("Opt:")


def test_flags_propagate_to_nested_configs(fresh_path_config):
    cfg = ExperimentConfig(paths=fresh_path_config, is_debug=False, verbose=False)
    assert hasattr(cfg.datamodule_config, "is_debug") and cfg.datamodule_config.is_debug is False
    assert (
        getattr(cfg.trainer_config, "verbose", True) is False
        if hasattr(cfg.trainer_config, "verbose")
        else True
    )


def test_setup_target_requires_ckpt_loader(monkeypatch, fresh_path_config, tmp_path):
    ckpt = fresh_path_config.checkpoints / "bad.ckpt"
    ckpt.touch()

    class BadLoader(lit_module.DocClassifierModule):
        def __init__(self, config):
            super().__init__(config)

    cfg = ExperimentConfig(paths=fresh_path_config, ckpt_path=ckpt.as_posix())
    cfg.module_config.__dict__["target"] = object()  # Bypass validation to simulate missing loader
    monkeypatch.setattr(
        f"{EXPERIMENT_MODULE}.TrainerFactoryConfig.setup_target", lambda self: MagicMock()
    )

    class DummyDataModule:
        def __init__(self):
            self.train_ds = SimpleNamespace(num_classes=None)

        def setup(self, stage=None):
            self.stage = stage

    monkeypatch.setattr(
        f"{EXPERIMENT_MODULE}.DocDataModuleConfig.setup_target", lambda self: DummyDataModule()
    )

    with pytest.raises(RuntimeError):
        cfg.setup_target()
