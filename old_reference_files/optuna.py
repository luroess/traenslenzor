from enum import Enum
from typing import Any, Dict, Literal, Optional, Type

import optuna
import wandb
from optuna import Study, Trial, pruners, samplers
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from pydantic import Field
from pytorch_lightning import Callback

from ..utils import CONSOLE, BaseConfig
from .optuna_optimizable import Optimizable
from .paths import PathConfig


class TPESamplerConfig(BaseConfig["samplers.TPESampler"]):
    n_startup_trials: int = 5
    """"Number of initial trials using random sampling before TPE starts."""
    n_ei_candidates: int = 24
    """Number of candidate samples to evaluate for expected improvement."""
    multivariate: bool = False
    """Enable multivariate TPE to consider correlations between parameters."""
    group: bool = False
    """Group parameters in multivariate TPE for more efficient sampling."""
    warn_independent_sampling: bool = True
    """Warn when independent sampling is used despite multivariate being enabled."""
    seed: Optional[int] = None
    target: Type[samplers.TPESampler] = Field(
        default_factory=lambda: samplers.TPESampler, exclude=True
    )


class MedianPrunerConfig(BaseConfig["pruners.MedianPruner"]):
    n_startup_trials: int = 5
    n_warmup_steps: int = 0
    interval_steps: int = 1
    target: Type[pruners.MedianPruner] = Field(
        default_factory=lambda: pruners.MedianPruner, exclude=True
    )


class SamplerType(Enum):
    TPE = "tpe"
    RANDOM = "random"

    def setup_target(self, **kwargs) -> samplers.BaseSampler:
        match self:
            case self.TPE:
                return samplers.TPESampler(**kwargs)
            case self.RANDOM:
                return samplers.RandomSampler(**kwargs)
            case _:
                raise ValueError(f"Unknown sampler type: {self}")


class PrunerType(Enum):
    MEDIAN = "median"
    SUCCESSIVE_HALVING = "successive_halving"

    def setup_target(self, **kwargs) -> pruners.BasePruner:
        match self:
            case self.MEDIAN:
                return pruners.MedianPruner(**kwargs)
            case self.SUCCESSIVE_HALVING:
                return pruners.SuccessiveHalvingPruner(**kwargs)
            case _:
                raise ValueError(f"Unknown pruner type: {self}")


class OptunaConfig(BaseConfig):
    """Configuration for Optuna hyperparameter optimization"""

    target: Type[Study] = Field(default_factory=lambda: Study)

    # Study settings
    study_name: str = "optimization_study"
    n_trials: int = 100
    direction: Literal["minimize", "maximize"] = "minimize"
    load_if_exists: bool = True

    # Optimization strategy
    monitor: str = "val_loss"
    sampler_type: SamplerType = SamplerType.TPE
    pruner_type: PrunerType = PrunerType.MEDIAN

    # Sampler and Pruner configurations
    tpe_params: TPESamplerConfig = Field(default_factory=TPESamplerConfig)
    median_params: MedianPrunerConfig = Field(default_factory=MedianPrunerConfig)

    # Storage configuration
    storage_template: str = Field(default_factory=lambda: PathConfig().optuna_study_uri)

    suggested_params: Dict[str, Any] = Field(default_factory=dict)

    def setup_target(self) -> Study:
        """Create or load an Optuna study"""

        sampler_params = {
            SamplerType.TPE: self.tpe_params.model_dump(),
            SamplerType.RANDOM: {},
        }

        pruner_params = {
            PrunerType.MEDIAN: self.median_params.model_dump(),
            PrunerType.SUCCESSIVE_HALVING: {},
        }

        return optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_template.format(study_name=self.study_name),
            load_if_exists=self.load_if_exists,
            direction=self.direction,
            sampler=self.sampler_type.setup_target(**sampler_params[self.sampler_type]),
            pruner=self.pruner_type.setup_target(**pruner_params[self.pruner_type]),
        )

    def setup_optimizables(
        self, experiment_config: "ExperimentConfig", trial: optuna.Trial  # type: ignore
    ) -> None:
        """Setup optimizable fields in the experiment configuration"""

        def setup_field(config: BaseConfig, field_name: str, field_value: Any) -> None:
            if isinstance(field_value, Optimizable):
                # Call setup_target on Optimizable fields
                suggested_value = field_value.setup_target(field_name, trial)
                CONSOLE.log(f"Setting {field_name} to {suggested_value}")
                # log param to wandb
                self.suggested_params.update({field_name: suggested_value})
                setattr(
                    config,
                    field_name,
                    suggested_value,
                )
            elif isinstance(field_value, BaseConfig):
                # Recurse into BaseConfig fields
                for sub_field_name, sub_field_value in field_value:
                    setup_field(field_value, sub_field_name, sub_field_value)

        # Iterate through all fields of experiment_config
        for field_name, field_value in experiment_config:
            setup_field(experiment_config, field_name, field_value)

    def log_to_wandb(self) -> None:
        """Log suggested parameters to Weights & Biases"""
        wandb.config.update(self.suggested_params)

    def get_pruning_callback(self, trial: Trial) -> PyTorchLightningPruningCallback:
        """Get the PyTorch Lightning pruning callback"""
        return OptunaPruningCallback(trial=trial, monitor=self.monitor)

    def get_best_trial(self) -> Optional[optuna.Trial]:
        """Retrieve the best trial from the study"""
        study = self.setup_target()
        return study.best_trial if study.trials else None


class OptunaPruningCallback(PyTorchLightningPruningCallback, Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
