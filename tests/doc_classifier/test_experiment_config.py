from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import traenslenzor.doc_classifier.lightning.lit_module as lit_module
from traenslenzor.doc_classifier.configs import ExperimentConfig, OptunaConfig
from traenslenzor.doc_classifier.utils import Optimizable, Stage

EXPERIMENT_MODULE = "traenslenzor.doc_classifier.configs.experiment_config"


def test_stage_validator_rejects_unknown_value():
    with pytest.raises(ValueError):
        ExperimentConfig(stage="unknown")


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
    config.from_ckpt = ckpt_file

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


def test_run_optuna_requires_config(fresh_path_config):
    cfg = ExperimentConfig(paths=fresh_path_config, optuna_config=None)
    with pytest.raises(ValueError):
        cfg.run_optuna_study()


def test_run_optuna_study_executes_trials(monkeypatch, fresh_path_config):
    cfg = ExperimentConfig(paths=fresh_path_config, optuna_config=OptunaConfig(n_trials=1))

    optim_field = cfg.module_config.optimizer.__class__.model_fields["learning_rate"]
    original_extra = dict(optim_field.json_schema_extra or {})
    optim_field.json_schema_extra = {
        **original_extra,
        "optimizable": Optimizable.continuous(
            low=1e-4,
            high=1e-2,
            name="module_config.optimizer.learning_rate",
        ),
    }

    class DummyTrainer:
        def __init__(self):
            self.callback_metrics = {"val/loss": 1.0}

        def fit(self, *_, **__):
            return None

    dummy_trainer = DummyTrainer()

    def fake_setup_target(self, setup_stage=None, trial=None):
        assert pytest.approx(self.module_config.optimizer.learning_rate, rel=1e-9) == 0.002
        return dummy_trainer, MagicMock(), MagicMock()

    monkeypatch.setattr(ExperimentConfig, "setup_target", fake_setup_target)

    class DummyTrial:
        def __init__(self):
            self.number = 0
            self.params = {"module_config.optimizer.learning_rate": 0.002}

        def suggest_float(self, name, low, high, log=False):
            return self.params[name]

        def suggest_categorical(self, name, choices):
            return choices[0]

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

    try:
        cfg.run_optuna_study()
    finally:
        optim_field.json_schema_extra = original_extra

    assert finish_called.count == 1
    assert cfg.trainer_config.wandb_config.group == "optuna"
    assert cfg.trainer_config.wandb_config.job_type.startswith("Opt:")
    assert cfg.optuna_config is not None
    assert (
        pytest.approx(
            cfg.optuna_config.suggested_params["module_config.optimizer.learning_rate"],
            rel=1e-9,
        )
        == 0.002
    )


def test_flags_propagate_to_nested_configs(fresh_path_config):
    cfg = ExperimentConfig(paths=fresh_path_config, is_debug=False, verbose=False)
    assert hasattr(cfg.datamodule_config, "is_debug") and cfg.datamodule_config.is_debug is False
    assert (
        getattr(cfg.trainer_config, "verbose", True) is False
        if hasattr(cfg.trainer_config, "verbose")
        else True
    )
