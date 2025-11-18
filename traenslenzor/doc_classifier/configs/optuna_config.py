"""Optuna integration helpers for experiment orchestration."""

from typing import Any, Callable, Literal

import optuna
import wandb
from optuna import pruners, samplers
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from pydantic import Field
from pytorch_lightning import Callback

from ..utils import BaseConfig, Metric, Optimizable
from .path_config import PathConfig

Setter = Callable[[Any], None]


class OptunaConfig(BaseConfig):
    """Configure an Optuna study used by :class:`ExperimentConfig`."""

    study_name: str = "doc-classifier"
    direction: Literal["minimize", "maximize"] = "minimize"
    n_trials: int = 20
    monitor: str = Metric.VAL_LOSS
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
            storage=PathConfig().optuna_study_uri.format(study_name=self.study_name),
            load_if_exists=self.load_if_exists,
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
        )

    def setup_optimizables(self, experiment_config: BaseConfig, trial: optuna.Trial) -> None:
        """Apply Optimizable hints embedded in the config tree."""
        if not isinstance(experiment_config, BaseConfig):
            return
        self.suggested_params.clear()

        def join(prefix: str | None, suffix: str) -> str:
            """Compose dotted paths for nested fields."""
            return f"{prefix}.{suffix}" if prefix else suffix

        def visit(
            value: Any,
            *,
            path: str | None,
            setter: Setter,
            field_info=None,
        ) -> None:
            """Walk config values and apply Optuna suggestions when encountered."""
            optimizable = _extract_optimizable(field_info, value)
            if optimizable is not None:
                suggestion = optimizable.suggest(trial, path or optimizable.name or "param")
                setter(suggestion)
                self.suggested_params[path or optimizable.name or "param"] = optimizable.serialize(
                    suggestion
                )
                return

            if isinstance(value, BaseConfig):
                for child_name, child_field in value.__class__.model_fields.items():
                    child_value = getattr(value, child_name)
                    visit(
                        child_value,
                        path=join(path, child_name),
                        setter=lambda new_value, obj=value, attr=child_name: setattr(
                            obj, attr, new_value
                        ),
                        field_info=child_field,
                    )
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    visit(
                        item,
                        path=f"{path}[{idx}]" if path else f"[{idx}]",
                        setter=lambda new_value, seq=value, index=idx: seq.__setitem__(
                            index, new_value
                        ),
                        field_info=None,
                    )
            elif isinstance(value, dict):
                for key, item in value.items():
                    visit(
                        item,
                        path=join(path, str(key)),
                        setter=lambda new_value,
                        mapping=value,
                        mapping_key=key: mapping.__setitem__(mapping_key, new_value),
                        field_info=None,
                    )

        def _extract_optimizable(field, current_value: Any) -> Optimizable | None:
            """Retrieve an Optimizable definition from the field metadata or value."""
            if isinstance(current_value, Optimizable):
                return current_value
            if field is not None:
                extras = field.json_schema_extra or {}
                opt = extras.get("optimizable")
                if isinstance(opt, Optimizable):
                    return opt
            return None

        for name, field in experiment_config.__class__.model_fields.items():
            value = getattr(experiment_config, name)
            visit(
                value,
                path=name,
                setter=lambda new_value, obj=experiment_config, attr=name: setattr(
                    obj, attr, new_value
                ),
                field_info=field,
            )

    def log_to_wandb(self) -> None:
        """Send the most recent suggestions to W&B."""
        if wandb.run is not None and self.suggested_params:
            wandb.config.update(self.suggested_params, allow_val_change=True)

    def get_pruning_callback(self, trial: optuna.Trial) -> Callback:
        """Return a PyTorch Lightning pruning callback for the configured monitor."""
        return PyTorchLightningPruningCallback(trial=trial, monitor=self.monitor)
