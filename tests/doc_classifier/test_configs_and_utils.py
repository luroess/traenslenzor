from enum import Enum
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import optuna
import pytest
from datasets import Split
from pydantic import Field

import traenslenzor.doc_classifier.configs.optuna_config as optuna_config_module
import traenslenzor.doc_classifier.configs.wandb_config as wandb_config_module
import traenslenzor.doc_classifier.lightning.lit_trainer_factory as trainer_factory_module
from traenslenzor.doc_classifier.configs import ExperimentConfig, PathConfig, WandbConfig
from traenslenzor.doc_classifier.configs.optuna_config import OptunaConfig
from traenslenzor.doc_classifier.lightning import TrainerCallbacksConfig, TrainerFactoryConfig
from traenslenzor.doc_classifier.utils import (
    BaseConfig,
    Metric,
    Optimizable,
    Stage,
    optimizable_field,
)


def test_stage_conversion_and_split():
    assert Stage.from_str("train") is Stage.TRAIN
    assert Stage.from_str("fit") is Stage.TRAIN
    assert Stage.from_str("validate") is Stage.VAL
    assert Stage.from_str(Stage.TEST) is Stage.TEST
    assert str(Stage.VAL) == "val"
    assert Stage.TRAIN.to_split() is Split.TRAIN
    assert Stage.VAL.to_split() is Split.VALIDATION
    assert Stage.TEST.to_split() is Split.TEST
    assert str(Metric.TRAIN_LOSS) == "train/loss"

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
    with pytest.raises(ValueError):
        PathConfig._validate_root(tmp_path / "missing-root")


def test_default_data_root_helper():
    from traenslenzor.doc_classifier.configs.path_config import _default_data_root

    assert _default_data_root().name == ".data"



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
    class Dummy(BaseConfig[None]):
        value: int = 1

    cfg = OptunaConfig()
    cfg.setup_optimizables(Dummy(), optuna.trial.FixedTrial({}))


class _DummyEnum(Enum):
    A = "a"
    B = "b"


class _LeafConfig(BaseConfig[None]):
    scalar: float = optimizable_field(
        default=0.1,
        optimizable=Optimizable.continuous(low=0.01, high=0.2),
    )
    choice: _DummyEnum = optimizable_field(
        default=_DummyEnum.A,
        optimizable=Optimizable(
            target=_DummyEnum,
            categories=[member.value for member in _DummyEnum],
        ),
    )


class _ContainerConfig(BaseConfig[None]):
    leaf: _LeafConfig = Field(default_factory=_LeafConfig)
    values: list[Any] = Field(default_factory=lambda: [Optimizable.discrete(low=1, high=3)])


def test_optuna_config_applies_metadata() -> None:
    config = _LeafConfig()
    opt_cfg = OptunaConfig()
    trial = optuna.trial.FixedTrial({"scalar": 0.05, "choice": "b"})

    opt_cfg.setup_optimizables(config, trial)

    assert config.scalar == pytest.approx(0.05)
    assert config.choice == _DummyEnum.B
    assert opt_cfg.suggested_params["scalar"] == pytest.approx(0.05)
    assert opt_cfg.suggested_params["choice"] == "b"


def test_optuna_config_handles_nested_structures() -> None:
    config = _ContainerConfig()
    opt_cfg = OptunaConfig()
    trial = optuna.trial.FixedTrial(
        {
            "leaf.scalar": 0.08,
            "leaf.choice": "a",
            "values[0]": 3,
        }
    )

    opt_cfg.setup_optimizables(config, trial)

    assert config.leaf.scalar == pytest.approx(0.08)
    assert config.values[0] == 3
    assert opt_cfg.suggested_params["values[0]"] == 3
