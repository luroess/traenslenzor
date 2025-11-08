"""High-level experiment orchestration for the document classifier."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna
import wandb
from pydantic import Field, ValidationInfo, field_validator, model_validator
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from typing_extensions import Self

from ..lightning import DocClassifierConfig, DocDataModuleConfig, TrainerFactoryConfig
from ..utils import BaseConfig, Console, Stage
from .optuna_config import OptunaConfig
from .path_config import PathConfig


class ExperimentConfig(BaseConfig[Trainer]):
    """Compose trainer, module, and datamodule factories for a single run."""

    seed: int | None = Field(
        default=42,
    )
    """Random seed applied via Lightning's `seed_everything`."""
    is_debug: bool = Field(True)
    """Enable debug mode for nested configs. When True this may enable faster dev runs and
    more verbose diagnostics for components that honour the flag."""

    verbose: bool = Field(True)
    """Toggle verbose logging output across nested components and the console."""

    run_name: str = Field(
        default_factory=lambda: datetime.now().strftime("R%Y-%m-%d_%H:%M:%S"),
    )
    """Run name forwarded to experiment loggers such as Weights & Biases (W&B)."""
    stage: Stage = Field(
        default=Stage.TRAIN,
    )
    """Primary stage to run when invoking `run()` (train/val/test)."""
    ckpt_path: Path | None = Field(
        default=None,
    )
    """Checkpoint file to restore before training or evaluation. May be relative to
    `paths.checkpoints` and will be resolved during validation."""

    paths: PathConfig = Field(default_factory=PathConfig)
    trainer_config: TrainerFactoryConfig = Field(default_factory=TrainerFactoryConfig)
    module_config: DocClassifierConfig = Field(default_factory=DocClassifierConfig)
    datamodule_config: DocDataModuleConfig = Field(default_factory=DocDataModuleConfig)

    optuna_config: OptunaConfig | None = Field(
        default=None,
    )
    """Optional Optuna configuration enabling hyperparameter searches and trial
    orchestration when set."""

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_keys(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Accept legacy keys such as `from_ckpt` from older configs."""
        if not isinstance(data, dict):
            return data
        legacy = data.pop("from_ckpt", None)
        if legacy is not None and "ckpt_path" not in data:
            data["ckpt_path"] = legacy
        return data

    @field_validator("stage", mode="before")
    @classmethod
    def _parse_stage(cls, value: Stage | str | None) -> Stage:
        if isinstance(value, Stage):
            return value
        stage = Stage.from_str(value)
        if stage is None:
            raise ValueError(f"Unsupported stage '{value}'.")
        return stage

    @field_validator("ckpt_path", mode="before")
    @classmethod
    def _resolve_ckpt_path(
        cls,
        value: str | Path | None,
        info: ValidationInfo,
    ) -> Path | None:
        if value in (None, ""):
            return None
        path = Path(value)
        paths_cfg = info.data.get("paths")
        if not path.is_absolute() and isinstance(paths_cfg, PathConfig):
            path = paths_cfg.checkpoints / path
        path = path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path '{path}' does not exist.")
        return path

    @model_validator(mode="after")
    def _propagate_common_flags(self) -> Self:
        """Sync debug/verbose flags with nested configs where supported."""
        for child in (self.module_config, self.datamodule_config, self.trainer_config):
            if hasattr(child, "is_debug"):
                setattr(child, "is_debug", self.is_debug)
            if hasattr(child, "verbose"):
                setattr(child, "verbose", self.verbose)
        if self.is_debug and hasattr(self.trainer_config, "fast_dev_run"):
            self.trainer_config.fast_dev_run = bool(self.trainer_config.fast_dev_run or True)
        return self

    @model_validator(mode="after")
    def _apply_seed(self) -> Self:
        if self.seed is not None:
            Console.with_prefix(self.__class__.__name__, "seed").set_verbose(self.verbose).log(
                f"Setting random seed to {self.seed} (Lightning seed_everything).",
            )
            seed_everything(self.seed, workers=True)
        return self

    @model_validator(mode="after")
    def _sync_wandb(self) -> Self:
        if hasattr(self.trainer_config, "update_wandb_config"):
            self.trainer_config.update_wandb_config(self)
        return self

    def setup_target(
        self,
        setup_stage: Stage | str = Stage.TRAIN,
    ) -> tuple[Trainer, LightningModule, LightningDataModule]:
        """Create trainer, module, and datamodule instances."""
        console = Console.with_prefix(self.__class__.__name__, "setup_target")
        console.set_verbose(self.verbose).set_debug(self.is_debug)

        trainer = self.trainer_config.setup_target()

        resolved_stage = Stage.from_str(setup_stage)

        if self.ckpt_path is not None:
            console.log(f"Loading model from checkpoint: {self.ckpt_path}")
            try:
                module_cls = getattr(self.module_config, "target", None)
                if module_cls is None or not hasattr(module_cls, "load_from_checkpoint"):
                    raise AttributeError("Configured module target cannot load from checkpoints.")
                lit_module = module_cls.load_from_checkpoint(
                    checkpoint_path=self.ckpt_path.as_posix(),
                    params=self.module_config,
                )
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"Failed to load checkpoint: {exc}") from exc
        else:
            lit_module = self.module_config.setup_target()

        lit_datamodule = self.datamodule_config.setup_target()

        lit_datamodule.setup(stage=resolved_stage)
        dataset = getattr(lit_datamodule, "train_ds", None)
        data_classes = getattr(dataset, "num_classes", None)
        module_classes = getattr(getattr(lit_module, "config", None), "num_classes", data_classes)
        if data_classes is not None and module_classes is not None:
            assert data_classes == module_classes, (
                f"Configured dataset and module disagree on num_classes: {data_classes} (data) vs "
                f"{module_classes} (module)."
            )

        console.log("Experiment setup complete.")

        return trainer, lit_module, lit_datamodule

    def setup_target_and_run(
        self,
        stage: Stage | str | None = None,
    ) -> Trainer:
        """Instantiate components and execute the requested stage."""
        resolved_stage = stage or self.stage

        trainer, lit_module, lit_datamodule = self.setup_target(
            setup_stage=resolved_stage,
        )

        console = Console.with_prefix(self.__class__.__name__, str(resolved_stage))
        console.set_verbose(self.verbose).set_debug(self.is_debug)

        ckpt_input = str(self.ckpt_path) if self.ckpt_path is not None else None
        if resolved_stage is Stage.TRAIN:
            console.log("Starting training (fit)...")
            trainer.fit(lit_module, datamodule=lit_datamodule, ckpt_path=ckpt_input)
        elif resolved_stage is Stage.VAL:
            console.log("Starting validation...")
            trainer.validate(lit_module, datamodule=lit_datamodule, ckpt_path=ckpt_input)
        elif resolved_stage is Stage.TEST:
            console.log("Starting testing...")
            trainer.test(lit_module, datamodule=lit_datamodule, ckpt_path=ckpt_input)

        return trainer

    def run_optuna_study(self) -> None:
        """Integrate Optuna for hyperparameter optimisation."""
        if self.optuna_config is None:
            raise ValueError("OptunaConfig is not set!")

        console = Console.with_prefix(self.__class__.__name__, "optuna")
        console.set_verbose(self.verbose).set_debug(self.is_debug)

        def objective(trial: optuna.Trial) -> float:
            experiment_config_copy = deepcopy(self)

            if experiment_config_copy.trainer_config.use_wandb:
                experiment_config_copy.trainer_config.wandb_config.name = (
                    f"{self.run_name}_T{trial.number}"
                )
            experiment_config_copy.optuna_config.setup_optimizables(
                experiment_config_copy,
                trial,
            )

            trainer, lit_module, lit_datamodule = experiment_config_copy.setup_target(
                setup_stage=self.stage,
                trial=trial,
            )
            trainer.fit(lit_module, datamodule=lit_datamodule)

            monitor = experiment_config_copy.optuna_config.monitor
            raw_metric = trainer.callback_metrics.get(monitor)
            metric = (
                float(raw_metric.item())
                if hasattr(raw_metric, "item")
                else float(raw_metric or float("inf"))
            )
            console.log(
                f"Trial {trial.number} finished with {monitor}: {metric}\nparams: {trial.params}",
            )
            experiment_config_copy.optuna_config.suggested_params.update(trial.params)
            experiment_config_copy.optuna_config.log_to_wandb()

            if wandb.run is not None:
                wandb.finish()

            return metric

        if self.trainer_config.use_wandb:
            self.trainer_config.wandb_config.group = "optuna"
            self.trainer_config.wandb_config.job_type = f"Opt:{self.run_name}"

        study = self.optuna_config.setup_target()
        study.optimize(objective, n_trials=self.optuna_config.n_trials)

    def run(self) -> Trainer:
        """Convenience CLI entry point."""
        return self.setup_target_and_run(self.stage)
