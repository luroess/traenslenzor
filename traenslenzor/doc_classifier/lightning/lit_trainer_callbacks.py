from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import model_validator
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    Timer,
    TQDMProgressBar,
)
from typing_extensions import Self

from traenslenzor.doc_classifier.utils.console import Console

from ..configs.path_config import PathConfig
from ..data_handling.huggingface_rvl_cdip_ds import make_transform_fn
from ..data_handling.transforms import TrainHeavyTransformConfig, TransformConfig
from ..utils import BaseConfig, Metric
from .finetune_callback import OneCycleBackboneFinetuning

if TYPE_CHECKING:
    from optuna import Trial

    from ..configs.optuna_config import OptunaConfig


class CustomTQDMProgressBar(TQDMProgressBar):
    """Custom TQDM progress bar that hides the version number (v_num)."""

    def get_metrics(self, *args, **kwargs):
        """Get metrics to display in progress bar, excluding version number.

        Returns:
            dict[str, float]: Metrics dictionary with v_num removed.
        """
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items


class CustomRichProgressBar(RichProgressBar):
    """Custom Rich progress bar that hides the version number (v_num)."""

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items


class TrainTransformSwitchCallback(Callback):
    """Switch training transforms to the heavy pipeline at a target epoch."""

    def __init__(self, switch_epoch: int = 8) -> None:
        super().__init__()
        self.switch_epoch = switch_epoch
        self._switched = False

    @staticmethod
    def _build_heavy_config(base: TransformConfig | None) -> TrainHeavyTransformConfig:
        if isinstance(base, TrainHeavyTransformConfig):
            return base
        if base is None:
            return TrainHeavyTransformConfig()
        base_data = base.model_dump()
        base_data["transform_type"] = "train_heavy"
        filtered = {
            key: value
            for key, value in base_data.items()
            if key in TrainHeavyTransformConfig.model_fields
        }
        return TrainHeavyTransformConfig(**filtered)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if self._switched or trainer.current_epoch < self.switch_epoch:
            return

        console = Console.with_prefix(self.__class__.__name__, "on_train_epoch_start")
        datamodule = trainer.datamodule
        if datamodule is None or not hasattr(datamodule, "config"):
            console.warn("No datamodule with config found; cannot switch transforms.")
            return

        train_cfg = getattr(datamodule.config, "train_ds", None)
        if train_cfg is None:
            console.warn("No train_ds config found; cannot switch transforms.")
            return

        base_cfg = getattr(train_cfg, "transform_config", None)
        if isinstance(base_cfg, TrainHeavyTransformConfig):
            self._switched = True
            console.log("Training transforms already set to TrainHeavyTransformConfig.")
            return

        heavy_cfg = self._build_heavy_config(base_cfg)
        train_cfg.transform_config = heavy_cfg

        try:
            dataset = datamodule.train_ds
            transform_pipeline = heavy_cfg.setup_target()
            dataset.set_transform(make_transform_fn(transform_pipeline))
            self._switched = True
            console.log(
                f"Switched training transforms to {heavy_cfg.__class__.__name__} "
                f"(epoch={trainer.current_epoch})."
            )
        except Exception as exc:
            console.warn(f"Failed to switch training transforms: {exc}")


class TrainerCallbacksConfig(BaseConfig[list[Callback]]):
    """Configuration for standard trainer callbacks."""

    use_model_checkpoint: bool = True
    checkpoint_monitor: Metric = Metric.VAL_LOSS
    """Metric to monitor for model checkpointing."""
    checkpoint_mode: str = "min"
    """Mode for checkpoint monitor ('min' or 'max')."""
    checkpoint_dir: Path | None = None
    """Directory to save checkpoints. If None, uses PathConfig().checkpoints."""
    checkpoint_filename: str | None = None
    """Filename template for checkpoints. If None, derives a safe template from ``checkpoint_monitor``."""
    checkpoint_save_top_k: int = 1
    """Number of best models to save."""
    checkpoint_auto_insert_metric_name: bool = False
    """Whether Lightning should auto-prefix metric names in the filename (avoid when names include '/')."""

    use_early_stopping: bool = False
    """Enable early stopping based on validation metrics."""
    early_stopping_monitor: Metric = Metric.VAL_LOSS
    """Metric to monitor for early stopping."""
    early_stopping_mode: str = "min"
    """Mode for early stopping monitor ('min' or 'max')."""
    early_stopping_patience: int = 5
    """Number of epochs with no improvement after which training stops."""

    use_lr_monitor: bool = True
    lr_logging_interval: str = "step"

    use_optuna_pruning: bool = False
    """Enable Optuna pruning callback for hyperparameter optimization runs."""

    use_rich_progress_bar: bool = False
    """Enable Rich progress bar for enhanced terminal output. Mutually exclusive with use_tqdm_progress_bar."""

    use_tqdm_progress_bar: bool = True
    """Enable TQDM progress bar. Mutually exclusive with use_rich_progress_bar."""
    tqdm_refresh_rate: int = 1
    """How often to refresh the TQDM progress bar (in batches)."""

    use_rich_model_summary: bool = True
    """Enable rich model summary using the Rich library for better visualization."""
    rich_summary_max_depth: int = 1
    """Maximum depth for the rich model summary tree."""

    use_backbone_finetuning: bool = False
    """Enable OneCycleLR-safe backbone finetuning callback for transfer learning."""
    backbone_unfreeze_at_epoch: int = 10
    """Epoch at which to unfreeze the backbone for finetuning."""
    backbone_lambda_func: str | None = None
    """Deprecated: unused by the OneCycleLR-safe finetuning callback."""
    backbone_train_bn: bool = True
    """Whether to train batch normalization layers during backbone finetuning."""

    use_train_heavy_switch: bool = False
    """Switch training transforms to TrainHeavyTransformConfig after a target epoch."""
    train_heavy_switch_epoch: int = 8
    """Epoch to activate the heavy training transform pipeline (0-based)."""

    use_timer: bool = False
    """Enable timer callback to track training duration."""
    timer_duration: dict[str, int] | None = None
    """Maximum training duration as dict (e.g., {'hours': 2, 'minutes': 30})."""
    timer_interval: str = "step"
    """Timer interval ('step' or 'epoch')."""

    @model_validator(mode="after")
    def _validate_progress_bars_mutually_exclusive(self) -> Self:
        """Ensure only one progress bar type is enabled."""
        if self.use_rich_progress_bar and self.use_tqdm_progress_bar:
            raise ValueError(
                "use_rich_progress_bar and use_tqdm_progress_bar are mutually exclusive. "
                "Enable only one progress bar type."
            )
        return self

    @staticmethod
    def _sanitize_metric_name(metric_name: str) -> str:
        """Return a filename-safe label for a metric key."""
        return metric_name.replace("/", "_")

    @classmethod
    def _default_checkpoint_filename(cls, monitor_name: str) -> str:
        """Build a safe default filename template from the monitor metric."""
        safe_name = cls._sanitize_metric_name(monitor_name)
        return f"epoch={{epoch}}-{safe_name}=" + "{" + monitor_name + ":.2f}"

    def setup_target(  # type: ignore[override]
        self,
        model_name: str | None = None,
        *,
        trial: "Trial | None" = None,
        optuna_config: "OptunaConfig | None" = None,
    ) -> list[Callback]:
        console = Console.with_prefix(self.__class__.__name__)
        callbacks: list[Callback] = []

        if trial:
            object.__setattr__(self, "use_model_checkpoint", False)
            object.__setattr__(self, "use_early_stopping", False)
            if self.use_optuna_pruning is False:
                console.warn(
                    "Optuna trial provided but use_optuna_pruning is False. "
                    "Enabling use_optuna_pruning."
                )
                object.__setattr__(self, "use_optuna_pruning", True)

        if self.use_model_checkpoint:
            dirpath = (
                self.checkpoint_dir if self.checkpoint_dir is not None else PathConfig().checkpoints
            )
            dirpath.mkdir(parents=True, exist_ok=True)

            monitor_name = str(self.checkpoint_monitor)
            auto_insert_metric_name = self.checkpoint_auto_insert_metric_name
            if auto_insert_metric_name and "/" in monitor_name:
                console.warn(
                    "checkpoint_auto_insert_metric_name=True with a '/'-delimited metric name "
                    "creates subdirectories. Forcing auto_insert_metric_name=False."
                )
                auto_insert_metric_name = False

            filename_template = (
                self.checkpoint_filename
                if self.checkpoint_filename not in (None, "")
                else self._default_checkpoint_filename(monitor_name)
            )
            ckpt_fn = f"{model_name}-" + filename_template if model_name else filename_template
            callbacks.append(
                ModelCheckpoint(
                    monitor=monitor_name,
                    mode=self.checkpoint_mode,
                    save_top_k=self.checkpoint_save_top_k,
                    filename=ckpt_fn,
                    auto_insert_metric_name=auto_insert_metric_name,
                    dirpath=dirpath.as_posix(),
                ),
            )
            console.log(f"ModelCheckpoint active, saving to: {dirpath}/{ckpt_fn}")

        if self.use_early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor=self.early_stopping_monitor,
                    mode=self.early_stopping_mode,
                    patience=self.early_stopping_patience,
                ),
            )

        if self.use_lr_monitor:
            callbacks.append(
                LearningRateMonitor(logging_interval=self.lr_logging_interval),
            )

        if self.use_rich_progress_bar:
            callbacks.append(
                CustomRichProgressBar(),
            )

        if self.use_tqdm_progress_bar:
            callbacks.append(
                CustomTQDMProgressBar(refresh_rate=self.tqdm_refresh_rate),
            )

        if self.use_rich_model_summary:
            callbacks.append(
                RichModelSummary(max_depth=self.rich_summary_max_depth),
            )

        if self.use_backbone_finetuning:
            if self.backbone_lambda_func:
                console.warn(
                    "backbone_lambda_func is ignored by OneCycleBackboneFinetuning. "
                    "Use OneCycleLR param-group scaling instead."
                )
            callbacks.append(
                OneCycleBackboneFinetuning(
                    unfreeze_backbone_at_epoch=self.backbone_unfreeze_at_epoch,
                    train_bn=self.backbone_train_bn,
                )
            )

        if self.use_train_heavy_switch:
            callbacks.append(
                TrainTransformSwitchCallback(
                    switch_epoch=self.train_heavy_switch_epoch,
                )
            )
            console.log(f"Train transforms will switch at epoch {self.train_heavy_switch_epoch}.")

        if self.use_timer:
            callbacks.append(
                Timer(
                    duration=timedelta(**self.timer_duration) if self.timer_duration else None,
                    interval=self.timer_interval,
                ),
            )

        if self.use_optuna_pruning:
            if optuna_config is None:
                raise ValueError("optuna_config is required when use_optuna_pruning is True.")
            if trial is None:
                raise ValueError("trial is required when use_optuna_pruning is True.")
            callbacks.append(optuna_config.get_pruning_callback(trial))
            console.log(f"Optuna pruning active (monitor={optuna_config.monitor})")

        return callbacks
