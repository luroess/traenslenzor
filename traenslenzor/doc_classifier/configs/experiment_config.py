"""High-level experiment orchestration for the document classifier."""

import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import optuna
import wandb
from pydantic import Field, ValidationInfo, field_validator, model_validator
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from typing_extensions import Self

from ..lightning import (
    DocClassifierConfig,
    DocDataModuleConfig,
    TrainerFactoryConfig,
    TunerConfig,
)
from ..utils import BaseConfig, Console, Stage
from .optuna_config import OptunaConfig
from .path_config import PathConfig


class ExperimentConfig(BaseConfig[Trainer]):
    """Compose trainer, module, and datamodule factories for a single run."""

    seed: int | None = Field(
        default=42,
    )
    """Random seed applied via Lightning's `seed_everything`."""
    is_debug: bool = Field(False)
    """Enable debug mode for nested configs. When True this may enable faster dev runs and
    more verbose diagnostics for components that honour the flag."""

    verbose: bool = Field(True)
    """Toggle verbose logging output across nested components and the console."""

    compute_stats: bool = Field(False)
    """Compute dataset grayscale mean/std before running the requested stage.
    >>> uv run traenslenzor/doc_classifier/run.py --compute_stats True
    """

    run_name: str = Field(
        default_factory=lambda: datetime.now().strftime("R%Y-%m-%d_%H:%M:%S"),
    )
    """Run name forwarded to experiment loggers such as Weights & Biases (W&B)."""

    stage: Stage = Field(
        default=Stage.TRAIN,
    )
    """Primary stage to run when invoking `run()` (train/val/test)."""
    from_ckpt: Path | None = Field(
        default=None,
    )
    """Checkpoint file to restore before training or evaluation. May be relative to
    `paths.checkpoints` and will be resolved during validation."""

    paths: PathConfig = Field(default_factory=PathConfig)
    trainer_config: TrainerFactoryConfig = Field(default_factory=TrainerFactoryConfig)
    module_config: DocClassifierConfig = Field(default_factory=DocClassifierConfig)
    datamodule_config: DocDataModuleConfig = Field(default_factory=DocDataModuleConfig)
    tuner_config: TunerConfig = Field(default_factory=TunerConfig)
    """Configuration for hyperparameter tuning (batch size and learning rate optimization)."""

    optuna_config: OptunaConfig | None = Field(
        default=None,
    )
    """Optional Optuna configuration enabling hyperparameter searches and trial
    orchestration when set."""

    @property
    def default_config_path(self) -> Path:
        """Return the default path for saving/loading TOML config files."""
        return self.paths.configs_dir / f"{self.run_name}.toml"

    def save_config(
        self,
        path: Path | str | None = None,
        *,
        include_comments: bool = True,
        include_type_hints: bool = True,
    ) -> Path:
        """Save the experiment configuration to a TOML file.

        Args:
            path: Path to save the config. If None, uses default_config_path.
            include_comments: Include docstring comments in TOML output.
            include_type_hints: Include type hints in TOML comments.

        Returns:
            Path to the saved config file.
        """
        target_path = Path(path) if path is not None else self.default_config_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        return self.save_toml(
            path=target_path,
            include_comments=include_comments,
            include_type_hints=include_type_hints,
        )

    @field_validator("stage", mode="before")
    @classmethod
    def _parse_stage(cls, value: Stage | str | None) -> Stage:
        if isinstance(value, Stage):
            return value
        stage = Stage.from_str(value)
        if stage is None:
            raise ValueError(f"Unsupported stage '{value}'.")
        return stage

    @field_validator("from_ckpt", mode="before")
    @classmethod
    def _resolve_ckpt_path(
        cls,
        value: str | Path | None,
        info: ValidationInfo,
    ) -> Path | None:
        if isinstance(paths_cfg := info.data.get("paths"), PathConfig):
            return paths_cfg.resolve_checkpoint_path(value)

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
        console = Console.with_prefix(self.__class__.__name__, "_sync_wandb")
        console.set_verbose(self.verbose)
        if hasattr(self.trainer_config, "update_wandb_config"):
            console.log("Syncing W&B configuration with experiment metadata")
            self.trainer_config.update_wandb_config(self)
        return self

    def setup_target(  # type: ignore[override]
        self,
        setup_stage: Stage | str = Stage.TRAIN,
        *,
        trial: optuna.Trial | None = None,
    ) -> tuple[Trainer, LightningModule, LightningDataModule]:
        """Create trainer, module, and datamodule instances."""
        console = Console.with_prefix(self.__class__.__name__, "setup_target")
        console.set_verbose(self.verbose).set_debug(self.is_debug)

        console.log(f"Setting up experiment: {self.run_name}")
        console.log(f"Stage: {setup_stage}, Seed: {self.seed}")

        console.log("Creating Trainer...")
        trainer = self.trainer_config.setup_target(self, trial=trial)

        resolved_stage = Stage.from_str(setup_stage)

        if self.from_ckpt is not None:
            console.log(f"Loading model from checkpoint: {self.from_ckpt}")
            try:
                lit_module = self.module_config.target.load_from_checkpoint(
                    checkpoint_path=self.from_ckpt,
                    params=self.module_config,
                    weights_only=False,
                )
                console.log(f"Successfully loaded checkpoint: {lit_module.__class__.__name__}")
            except Exception as exc:
                console.error(f"Failed to load checkpoint: {exc}")
                raise RuntimeError(f"Failed to load checkpoint: {exc}") from exc
        else:
            console.log("Creating new model from config...")
            lit_module = self.module_config.setup_target()
            console.log(f"Model created: {lit_module.__class__.__name__}")

        console.log("Creating DataModule...")
        lit_datamodule = self.datamodule_config.setup_target()

        lit_datamodule.setup(stage=resolved_stage)
        dataset = getattr(lit_datamodule, "train_ds", None)
        data_classes = getattr(dataset, "num_classes", None)
        module_classes = getattr(getattr(lit_module, "config", None), "num_classes", data_classes)
        if data_classes is not None and module_classes is not None:
            console.log(f"Verified num_classes: {data_classes} (data) == {module_classes} (model)")
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
        """Instantiate components and execute the requested stage.

        Args:
            stage: Training stage (train/val/test).
        """
        resolved_stage = stage or self.stage

        console = Console.with_prefix(self.__class__.__name__, "setup_target_and_run")
        console.set_verbose(self.verbose).set_debug(self.is_debug)

        console.log(f"Running experiment: {self.run_name} (stage={resolved_stage})")

        trainer, lit_module, lit_datamodule = self.setup_target(
            setup_stage=resolved_stage,
        )

        if self.compute_stats:
            console.log("Computing dataset grayscale statistics.")
            lit_datamodule.compute_grayscale_mean_std()
            console.log("Dataset statistics computation complete. Exiting.")
            return trainer

        # Run tuning before training if requested
        if (
            self.tuner_config.use_batch_size_tuning or self.tuner_config.use_learning_rate_tuning
        ) and resolved_stage is Stage.TRAIN:
            console.log("Running hyperparameter tuning...")
            self.tuner_config.run_tuning(trainer, lit_module, lit_datamodule)

        stage_console = Console.with_prefix(self.__class__.__name__, str(resolved_stage))
        stage_console.set_verbose(self.verbose).set_debug(self.is_debug)

        ckpt_input = str(self.from_ckpt) if self.from_ckpt is not None else None
        try:
            if resolved_stage is Stage.TRAIN:
                stage_console.log("Starting training (fit)...")
                trainer.fit(lit_module, datamodule=lit_datamodule, ckpt_path=ckpt_input)
                stage_console.log("Training completed")
            elif resolved_stage is Stage.VAL:
                stage_console.log("Starting validation...")
                trainer.validate(lit_module, datamodule=lit_datamodule, ckpt_path=ckpt_input)
                stage_console.log("Validation completed")
            elif resolved_stage is Stage.TEST:
                stage_console.log("Starting testing...")
                trainer.test(lit_module, datamodule=lit_datamodule, ckpt_path=ckpt_input)
                stage_console.log("Testing completed")
        except KeyboardInterrupt:
            stage_console.warn("Keyboard interrupt received. Shutting down.")
            if wandb.run is not None:
                wandb.finish()
            return trainer

        return trainer

    def compute_grayscale_mean_std(self) -> tuple[float, float]:
        """Compute grayscale mean/std for the configured training split.

        Returns:
            Tuple of (mean, std) in [0, 1] for a single grayscale channel.
        """
        console = Console.with_prefix(self.__class__.__name__, "compute_grayscale_mean_std")
        console.set_verbose(self.verbose).set_debug(self.is_debug)

        console.log("Creating DataModule for dataset statistics.")
        datamodule = self.datamodule_config.setup_target()
        mean, std = datamodule.compute_grayscale_mean_std()
        return mean, std

    def run_optuna_study(self) -> None:
        """Integrate Optuna for hyperparameter optimisation."""
        if self.optuna_config is None:
            raise ValueError("OptunaConfig is not set!")

        console = Console.with_prefix(self.__class__.__name__, "optuna")
        console.set_verbose(self.verbose).set_debug(self.is_debug)

        console.log(f"Starting Optuna study: {self.optuna_config.study_name}")
        console.log(f"Number of trials: {self.optuna_config.n_trials}")
        console.log(f"Monitoring metric: {self.optuna_config.monitor}")

        def _coerce_metric(raw_metric: object | None) -> float | None:
            if raw_metric is None:
                return None
            try:
                value = (
                    float(raw_metric.item()) if hasattr(raw_metric, "item") else float(raw_metric)
                )
            except (TypeError, ValueError):
                return None
            if not math.isfinite(value):
                return None
            return value

        def _fallback_metric(direction: str) -> float:
            return float("inf") if direction == "minimize" else float("-inf")

        def _get_metric_from_trainer(
            trainer: Trainer | None,
            monitor: str,
        ) -> tuple[float | None, str | None]:
            if trainer is None:
                return None, None
            monitor_key = str(monitor)
            for source_name in ("callback_metrics", "logged_metrics", "progress_bar_metrics"):
                metrics = getattr(trainer, source_name, None)
                if metrics and monitor_key in metrics:
                    metric_value = _coerce_metric(metrics.get(monitor_key))
                    if metric_value is not None:
                        return metric_value, source_name
            ckpt_callback = getattr(trainer, "checkpoint_callback", None)
            if ckpt_callback is not None:
                metric_value = _coerce_metric(getattr(ckpt_callback, "best_model_score", None))
                if metric_value is not None:
                    return metric_value, "checkpoint_callback.best_model_score"
            return None, None

        def objective(trial: optuna.Trial) -> float:
            trial_console = Console.with_prefix(self.__class__.__name__, f"trial_{trial.number}")
            trial_console.set_verbose(self.verbose)

            trial_console.log(f"Starting trial {trial.number}")

            experiment_config_copy = deepcopy(self)
            assert experiment_config_copy.optuna_config is not None

            if experiment_config_copy.trainer_config.use_wandb:
                experiment_config_copy.trainer_config.wandb_config.name = (
                    f"{self.run_name}_T{trial.number}"
                )
                trial_console.log(
                    f"W&B run name: {experiment_config_copy.trainer_config.wandb_config.name}"
                )

            experiment_config_copy.optuna_config.setup_optimizables(
                experiment_config_copy,
                trial,
            )

            trainer: Trainer | None = None
            monitor = str(experiment_config_copy.optuna_config.monitor)
            metric = _fallback_metric(experiment_config_copy.optuna_config.direction)
            metric_source = "fallback"

            try:
                trainer, lit_module, lit_datamodule = experiment_config_copy.setup_target(
                    setup_stage=self.stage,
                    trial=trial,
                )
                trainer.fit(lit_module, datamodule=lit_datamodule)

                if experiment_config_copy.trainer_config.callbacks.use_optuna_pruning:
                    _check_pruned(trainer, trial_console)

                extracted_metric, source = _get_metric_from_trainer(trainer, monitor)
                if extracted_metric is None:
                    trial_console.warn(
                        f"No '{monitor}' metric found after training; using fallback {metric:.4f}."
                    )
                else:
                    metric = extracted_metric
                    metric_source = source or "unknown"
            except optuna.TrialPruned as exc:
                trial_console.warn(f"Trial {trial.number} pruned: {exc}")
                raise
            except KeyboardInterrupt:
                trial_console.warn("Keyboard interrupt received. Stopping Optuna study.")
                raise
            except Exception as exc:
                trial_console.error(f"Trial {trial.number} failed: {exc}")
                extracted_metric, source = _get_metric_from_trainer(trainer, monitor)
                if extracted_metric is not None:
                    metric = extracted_metric
                    metric_source = f"failed:{source}"
                    trial_console.warn(f"Recovered '{monitor}' from {source}: {metric:.4f}.")
                else:
                    metric_source = "failed:fallback"
                    trial_console.warn(
                        f"No '{monitor}' metric available after failure; using fallback "
                        f"{metric:.4f}."
                    )
                trial.set_user_attr("failed_reason", repr(exc))
            finally:
                experiment_config_copy.optuna_config.log_to_wandb()

                if self.optuna_config is not None:
                    self.optuna_config.suggested_params = (
                        experiment_config_copy.optuna_config.suggested_params.copy()
                    )

                if wandb.run is not None:
                    wandb.finish()

            trial.set_user_attr("metric_source", metric_source)
            trial_console.log(
                f"Trial {trial.number} finished with {monitor}: {metric:.4f}",
            )
            console.dbg(f"Trial {trial.number} params: {trial.params}")
            return metric

        def _check_pruned(trainer: Trainer, trial_console: Console) -> None:
            """Raise TrialPruned if Optuna requested pruning."""
            for callback in trainer.callbacks:
                check_pruned = getattr(callback, "check_pruned", None)
                if callable(check_pruned):
                    trial_console.log("Optuna pruning check executed.")
                    check_pruned()
                    return
            trial_console.warn("Optuna pruning enabled but no pruning callback was found.")

        if self.trainer_config.use_wandb:
            self.trainer_config.wandb_config.group = "optuna"
            self.trainer_config.wandb_config.job_type = f"Opt:{self.run_name}"
            console.log("W&B configured for Optuna study")

        study = self.optuna_config.setup_target()
        console.log(f"Running optimization with {self.optuna_config.n_trials} trials...")
        try:
            study.optimize(objective, n_trials=self.optuna_config.n_trials)
        except KeyboardInterrupt:
            console.warn("Keyboard interrupt received. Stopping Optuna study.")
            return

        if hasattr(study, "best_value") and hasattr(study, "best_params"):
            console.log(f"Optuna study completed. Best value: {study.best_value:.4f}")
            console.log(f"Best params: {study.best_params}")
        else:
            console.log("Optuna study completed")
