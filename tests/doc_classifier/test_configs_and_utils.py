from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from datasets import Split

import traenslenzor.doc_classifier.configs.optuna_config as optuna_config_module
import traenslenzor.doc_classifier.configs.wandb_config as wandb_config_module
import traenslenzor.doc_classifier.lightning.lit_trainer_factory as trainer_factory_module
from traenslenzor.doc_classifier.configs import ExperimentConfig, PathConfig, WandbConfig
from traenslenzor.doc_classifier.configs.optuna_config import OptunaConfig
from traenslenzor.doc_classifier.lightning import TrainerCallbacksConfig, TrainerFactoryConfig
from traenslenzor.doc_classifier.utils import MetricName, Stage


def test_stage_conversion_and_split():
    assert Stage.from_str("train") is Stage.TRAIN
    assert Stage.from_str("fit") is Stage.TRAIN
    assert Stage.from_str("validate") is Stage.VAL
    assert Stage.from_str(Stage.TEST) is Stage.TEST
    assert str(Stage.VAL) == "val"
    assert Stage.VAL.to_split() is Split.VALIDATION
    assert Stage.TEST.to_split() is Split.TEST
    assert str(MetricName.TRAIN_LOSS) == "train/loss"

    with pytest.raises(ValueError):
        Stage.from_str("unknown-stage")


def test_path_config_creates_required_directories(project_root):
    config = PathConfig()
    config.root = project_root
    config.data_root = project_root / ".cache"
    config.hf_cache = config.data_root / "hf"
    config.checkpoints = project_root / ".artifacts/ckpts"
    config.wandb = project_root / ".artifacts/wandb"
    config.configs_dir = project_root / ".configs"

    assert config.data_root == project_root / ".cache"
    assert config.hf_cache == project_root / ".cache/hf"
    assert config.checkpoints == project_root / ".artifacts/ckpts"
    assert config.wandb == project_root / ".artifacts/wandb"
    assert config.configs_dir == project_root / ".configs"

    for field_name in ("data_root", "checkpoints", "wandb", "configs_dir"):
        assert getattr(config, field_name).is_dir()


def test_path_config_requires_existing_root(tmp_path):
    PathConfig._instances.pop(PathConfig, None)
    with pytest.raises(ValueError):
        PathConfig(root=tmp_path / "missing-root")


def test_default_data_root_helper():
    from traenslenzor.doc_classifier.configs.path_config import _default_data_root

    assert _default_data_root().name == ".data"


def test_wandb_config_setup_target_uses_custom_directory(tmp_path, monkeypatch):
    captured_kwargs: dict[str, str] = {}

    class DummyLogger(wandb_config_module.WandbLogger):
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            # Skip WandB initialisation
            self.kwargs = kwargs

    monkeypatch.setattr(wandb_config_module, "WandbLogger", DummyLogger)

    target_dir = tmp_path / "wandb"
    config = WandbConfig(name="unit-test", prefix="exp", save_dir=target_dir)

    logger = config.setup_target()

    assert isinstance(logger, DummyLogger)
    assert captured_kwargs["name"] == "unit-test"
    assert captured_kwargs["prefix"] == "exp"
    assert captured_kwargs["save_dir"] == target_dir.as_posix()
    assert target_dir.is_dir()


def test_trainer_callbacks_config_builds_requested_callbacks(tmp_path):
    cfg = TrainerCallbacksConfig(
        use_model_checkpoint=True,
        checkpoint_dir=tmp_path / "ckpts",
        use_early_stopping=True,
        use_lr_monitor=True,
    )

    callbacks = cfg.setup_target()

    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

    assert any(isinstance(cb, ModelCheckpoint) for cb in callbacks)
    assert any(isinstance(cb, EarlyStopping) for cb in callbacks)
    assert any(isinstance(cb, LearningRateMonitor) for cb in callbacks)
    assert (tmp_path / "ckpts").is_dir()


def test_trainer_factory_updates_wandb_config_metadata(tmp_path):
    cfg = TrainerFactoryConfig()
    cfg.wandb_config.tags = ["existing"]

    experiment = SimpleNamespace(
        run_name="demo-run",
        stage=Stage.TEST,
        paths=SimpleNamespace(wandb=tmp_path / "wandb"),
    )

    cfg.update_wandb_config(experiment)

    assert cfg.wandb_config.name == "demo-run"
    assert "test" in cfg.wandb_config.tags
    assert cfg.wandb_config.save_dir == experiment.paths.wandb


def test_trainer_factory_update_wandb_disabled():
    cfg = TrainerFactoryConfig(use_wandb=False)
    cfg.update_wandb_config(SimpleNamespace(run_name="ignored"))
    assert cfg.wandb_config.name is None


def test_trainer_factory_setup_target_invokes_dependencies(monkeypatch):
    cfg = TrainerFactoryConfig()

    monkeypatch.setattr(
        f"{trainer_factory_module.__name__}.WandbConfig.setup_target",
        lambda self: "logger",
    )
    monkeypatch.setattr(
        f"{trainer_factory_module.__name__}.TrainerCallbacksConfig.setup_target",
        lambda self: ["cb"],
    )

    class DummyTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(trainer_factory_module.pl, "Trainer", DummyTrainer)

    trainer = cfg.setup_target()
    assert trainer.kwargs["logger"] == "logger"
    assert trainer.kwargs["callbacks"] == ["cb"]


def test_experiment_config_propagates_flags_and_stage(fresh_path_config):
    config = ExperimentConfig(
        paths=fresh_path_config,
        is_debug=True,
        verbose=False,
        stage="val",
    )

    assert config.stage is Stage.VAL
    assert config.is_debug is True
    assert config.trainer_config.fast_dev_run is True


def test_experiment_config_export_puml_writes_diagram(fresh_path_config):
    config = ExperimentConfig(paths=fresh_path_config)

    output = config.export_puml("unit_test.puml")

    assert output.exists()
    assert output.parent == fresh_path_config.configs_dir
    assert "@startuml" in output.read_text()


def test_experiment_config_resolves_relative_checkpoint(fresh_path_config):
    ckpt_file = fresh_path_config.checkpoints / "model.ckpt"
    ckpt_file.touch()

    config = ExperimentConfig(paths=fresh_path_config, ckpt_path=ckpt_file)

    assert config.ckpt_path == ckpt_file


def test_optuna_config_setup_target_uses_selected_strategies(monkeypatch):
    recorded_kwargs: dict[str, object] = {}

    def fake_create_study(**kwargs):
        recorded_kwargs.update(kwargs)
        return "study"

    monkeypatch.setattr(optuna_config_module.optuna, "create_study", fake_create_study)

    cfg = OptunaConfig(sampler="random", pruner="successive_halving", direction="maximize")
    study = cfg.setup_target()

    assert study == "study"
    assert recorded_kwargs["direction"] == "maximize"
    assert recorded_kwargs["sampler"].__class__.__name__ == "RandomSampler"
    assert recorded_kwargs["pruner"].__class__.__name__ == "SuccessiveHalvingPruner"


def test_optuna_config_pruning_and_wandb_logging(monkeypatch):
    cfg = OptunaConfig(monitor="val/acc")
    trial = MagicMock()

    callback = cfg.get_pruning_callback(trial)
    assert callback.monitor == "val/acc"

    mock_wandb = SimpleNamespace(
        run=object(),
        config=MagicMock(),
    )
    monkeypatch.setattr(optuna_config_module, "wandb", mock_wandb)

    cfg.suggested_params = {"lr": 0.1}
    cfg.log_to_wandb()

    mock_wandb.config.update.assert_called_once_with({"lr": 0.1}, allow_val_change=True)


def test_optuna_config_setup_optimizables_is_noop():
    cfg = OptunaConfig()
    cfg.setup_optimizables(MagicMock(), MagicMock())
