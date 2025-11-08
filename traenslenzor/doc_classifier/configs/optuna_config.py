"""Optuna integration helpers for experiment orchestration."""

from __future__ import annotations

from typing import Any, Literal

import optuna
import wandb
from optuna import pruners, samplers
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from pydantic import Field
from pytorch_lightning import Callback

from ..utils import BaseConfig


class OptunaConfig(BaseConfig):
    """Configure an Optuna study used by :class:`ExperimentConfig`."""

    study_name: str = "doc-classifier"
    direction: Literal["minimize", "maximize"] = "minimize"
    n_trials: int = 20
    monitor: str = "val/loss"
    storage: str | None = None
    load_if_exists: bool = True

    sampler: Literal["tpe", "random"] = "tpe"
    pruner: Literal["median", "successive_halving"] = "median"

    suggested_params: dict[str, Any] = Field(default_factory=dict, exclude=True)

    def setup_target(self) -> optuna.Study:
        """Create or load an Optuna study."""
        sampler = samplers.TPESampler() if self.sampler == "tpe" else samplers.RandomSampler()
        pruner = (
            pruners.MedianPruner() if self.pruner == "median" else pruners.SuccessiveHalvingPruner()
        )
        return optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=self.load_if_exists,
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
        )

    def setup_optimizables(self, experiment_config: BaseConfig, trial: optuna.Trial) -> None:
        """Hook to mutate `experiment_config` before each trial.

        The default implementation is a no-op; override this by subclassing or by
        injecting callable wrappers in the config tree.
        """
        _ = experiment_config, trial

    def log_to_wandb(self) -> None:
        """Send the most recent suggestions to W&B."""
        if wandb.run is not None and self.suggested_params:
            wandb.config.update(self.suggested_params, allow_val_change=True)

    def get_pruning_callback(self, trial: optuna.Trial) -> Callback:
        """Return a PyTorch Lightning pruning callback for the configured monitor."""
        return PyTorchLightningPruningCallback(trial=trial, monitor=self.monitor)
