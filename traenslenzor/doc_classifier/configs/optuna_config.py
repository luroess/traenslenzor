"""Optuna integration helpers for experiment orchestration."""

from collections.abc import Mapping
from typing import Any, Callable, Literal

import optuna
import wandb
from optuna import pruners, samplers
from optuna_integration import PyTorchLightningPruningCallback
from pydantic import Field
from pytorch_lightning import Callback

from ..utils import BaseConfig, Console, Metric, Optimizable
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

    search_space: dict[str, Any] = Field(default_factory=dict)
    """Optional Optuna search-space declared in config files (e.g. TOML).

    Supports either:
    - A flat mapping of dotted config paths -> Optimizable definitions, or
    - A nested mapping mirroring the experiment config tree (will be flattened internally).
    """

    suggested_params: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @staticmethod
    def _flatten_search_space(search_space: Mapping[str, Any]) -> dict[str, Optimizable]:
        """Flatten nested search space mappings into dotted paths."""

        optimizable_keys = set(Optimizable.model_fields)
        leaf_keys = optimizable_keys | {"choices"}

        def join(prefix: str | None, suffix: str) -> str:
            return f"{prefix}.{suffix}" if prefix else suffix

        def parse_leaf(value: object) -> Optimizable | None:
            if isinstance(value, Optimizable):
                return value
            if not isinstance(value, Mapping):
                return None

            keys = set(value.keys())
            if not (keys & leaf_keys):
                return None
            if not keys.issubset(leaf_keys):
                return None

            payload = dict(value)
            if "choices" in payload and "categories" not in payload:
                payload["categories"] = payload.pop("choices")

            return Optimizable.model_validate(payload)

        flattened: dict[str, Optimizable] = {}

        def walk(value: object, *, path: str | None) -> None:
            leaf = parse_leaf(value)
            if leaf is not None:
                if path is None:
                    raise ValueError("Optuna search_space leaf requires a path.")
                flattened[path] = leaf
                return

            if isinstance(value, Mapping):
                for key, child in value.items():
                    walk(child, path=join(path, str(key)))

        walk(search_space, path=None)
        return flattened

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

    def setup_optimizables(
        self,
        experiment_config: BaseConfig,
        trial: optuna.Trial,
        *,
        console: Console | None = None,
    ) -> None:
        """Apply Optimizable hints embedded in the config tree."""
        if not isinstance(experiment_config, BaseConfig):
            return
        self.suggested_params.clear()

        opt_console = console or Console.with_prefix(self.__class__.__name__, "setup_optimizables")
        opt_console.set_verbose(getattr(experiment_config, "verbose", False)).set_debug(
            getattr(experiment_config, "is_debug", False)
        )

        def join(prefix: str | None, suffix: str) -> str:
            return f"{prefix}.{suffix}" if prefix else suffix

        search_space = self._flatten_search_space(self.search_space) if self.search_space else {}

        def apply_optimizable(
            optimizable: Optimizable, *, path: str | None, setter: Setter
        ) -> None:
            param_name = optimizable.name or path or "param"
            suggestion = optimizable.suggest(trial, param_name)
            setter(suggestion)
            serialized = optimizable.serialize(suggestion)
            self.suggested_params[param_name] = serialized
            opt_console.log(f"Optuna suggest {param_name}={serialized} (trial {trial.number})")
            if optimizable.description:
                opt_console.log(f"Optuna param note: {param_name} - {optimizable.description}")

        def visit(
            *,
            path: str | None,
            getter: Callable[[], Any],
            setter: Setter,
            field_info=None,
        ) -> None:
            """Walk config values and apply Optuna suggestions when encountered."""
            value = getter()

            if path is not None and (optimizable := search_space.get(path)) is not None:
                apply_optimizable(optimizable, path=path, setter=setter)
                value = getter()
            else:
                optimizable = _extract_optimizable(field_info, value)
                if optimizable is not None:
                    apply_optimizable(optimizable, path=path, setter=setter)
                    value = getter()

            if isinstance(value, BaseConfig):
                for child_name, child_field in value.__class__.model_fields.items():
                    visit(
                        path=join(path, child_name),
                        getter=lambda obj=value, attr=child_name: getattr(obj, attr),
                        setter=lambda new_value, obj=value, attr=child_name: setattr(
                            obj, attr, new_value
                        ),
                        field_info=child_field,
                    )
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    visit(
                        path=f"{path}[{idx}]" if path else f"[{idx}]",
                        getter=lambda seq=value, index=idx: seq[index],
                        setter=lambda new_value, seq=value, index=idx: seq.__setitem__(
                            index, new_value
                        ),
                        field_info=None,
                    )
            elif isinstance(value, dict):
                for key, item in value.items():
                    visit(
                        path=join(path, str(key)),
                        getter=lambda mapping=value, mapping_key=key: mapping[mapping_key],
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
            visit(
                path=name,
                getter=lambda obj=experiment_config, attr=name: getattr(obj, attr),
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
