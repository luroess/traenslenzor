"""Trainer factory with W&B integration.

Provides a configurable wrapper to instantiate PyTorch Lightning trainers
following the Config-as-Factory pattern established in UniTraj.
"""

from collections.abc import Sequence
from typing import Any

import pytorch_lightning as pl
import torch
from pydantic import Field, field_validator, model_validator
from pytorch_lightning.loggers import Logger
from pytorch_lightning.trainer.connectors.accelerator_connector import _PRECISION_INPUT
from typing_extensions import Self

from ..configs.wandb_config import WandbConfig
from ..utils import BaseConfig, Console
from .lit_trainer_callbacks import TrainerCallbacksConfig


class TrainerFactoryConfig(BaseConfig[pl.Trainer]):
    """Configuration for constructing a PyTorch Lightning trainer."""

    target: type[pl.Trainer] = Field(default_factory=lambda: pl.Trainer, exclude=True)

    accelerator: str = "auto"
    devices: int | str | Sequence[int] = "auto"
    strategy: str | None = "auto"
    max_epochs: int | None = 10
    precision: _PRECISION_INPUT = "32-true"
    """Floating-point precision for training. See PyTorch Lightning documentation for available options."""
    gradient_clip_val: float | None = None
    accumulate_grad_batches: int = 1
    log_every_n_steps: int = 50
    fast_dev_run: bool = False
    deterministic: bool | str | None = None
    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = None
    check_val_every_n_epoch: int = 1

    callbacks: TrainerCallbacksConfig = Field(default_factory=TrainerCallbacksConfig)
    use_wandb: bool = True
    wandb_config: WandbConfig = Field(default_factory=WandbConfig)

    @field_validator("precision")
    @classmethod
    def _set_matmul_precision(cls, value: _PRECISION_INPUT) -> _PRECISION_INPUT:
        if value in {"32-true", 32, "32"}:
            torch.set_float32_matmul_precision("medium")
        return value

    @model_validator(mode="after")
    def _debug_defaults(self) -> Self:
        if self.fast_dev_run:
            Console.with_prefix(self.__class__.__name__).warn(
                "Fast dev run enabled; trainer will use a single batch per split.",
            )
        return self

    def update_wandb_config(self, experiment: "BaseConfig") -> None:
        """Propagate experiment metadata into the W&B logger config."""
        if not self.use_wandb:
            return
        if not self.wandb_config.name:
            self.wandb_config.name = experiment.run_name  # type: ignore[attr-defined]
        stage = getattr(experiment, "stage", None)
        if stage is not None:
            tags = set(self.wandb_config.tags or [])
            tags.add(str(stage))
            self.wandb_config.tags = sorted(tags)
        if hasattr(experiment, "paths"):
            self.wandb_config.save_dir = experiment.paths.wandb  # type: ignore[attr-defined]

    def setup_target(self, **_: Any) -> pl.Trainer:
        """Instantiate the configured trainer."""
        logger: Logger | None = None
        if self.use_wandb:
            logger = self.wandb_config.setup_target()

        callbacks = self.callbacks.setup_target()

        return pl.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            strategy=self.strategy,
            max_epochs=self.max_epochs,
            precision=self.precision,
            gradient_clip_val=self.gradient_clip_val,
            accumulate_grad_batches=self.accumulate_grad_batches,
            log_every_n_steps=self.log_every_n_steps,
            fast_dev_run=self.fast_dev_run,
            deterministic=self.deterministic,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            callbacks=callbacks,
            logger=logger,
        )
