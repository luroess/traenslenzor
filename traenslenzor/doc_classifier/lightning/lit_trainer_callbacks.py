from datetime import timedelta
from pathlib import Path

from pytorch_lightning.callbacks import (
    BackboneFinetuning,
    BatchSizeFinder,
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    Timer,
)

from ..configs.path_config import PathConfig
from ..utils import BaseConfig, Metric


class TrainerCallbacksConfig(BaseConfig[list[Callback]]):
    """Configuration for standard trainer callbacks."""

    use_model_checkpoint: bool = True
    checkpoint_monitor: Metric = Metric.VAL_LOSS
    """Metric to monitor for model checkpointing."""
    checkpoint_mode: str = "min"
    """Mode for checkpoint monitor ('min' or 'max')."""
    checkpoint_dir: Path | None = None
    """Directory to save checkpoints. If None, uses PathConfig().checkpoints."""
    checkpoint_filename: str = f"epoch={{epoch}}-val_loss={{{Metric.VAL_LOSS}:.2f}}"
    """Filename template for checkpoints."""
    checkpoint_save_top_k: int = 1
    """Number of best models to save."""

    use_early_stopping: bool = False
    """Enable early stopping based on validation metrics."""
    early_stopping_monitor: Metric = Metric.VAL_LOSS
    """Metric to monitor for early stopping."""
    early_stopping_mode: str = "min"
    """Mode for early stopping monitor ('min' or 'max')."""
    early_stopping_patience: int = 5
    """Number of epochs with no improvement after which training stops."""

    use_lr_monitor: bool = True
    lr_logging_interval: str = "epoch"

    use_rich_model_summary: bool = True
    """Enable rich model summary using the Rich library for better visualization."""
    rich_summary_max_depth: int = 1
    """Maximum depth for the rich model summary tree."""

    use_batch_size_finder: bool = False
    """Enable automatic batch size finder to find optimal batch size."""
    batch_size_mode: str = "power"
    """Mode for batch size finder ('power' or 'binsearch')."""
    batch_size_init_val: int = 2
    """Initial batch size value for the finder."""
    batch_size_max_trials: int = 25
    """Maximum number of trials for batch size finder."""

    use_backbone_finetuning: bool = False
    """Enable backbone finetuning callback for transfer learning."""
    backbone_unfreeze_at_epoch: int = 10
    """Epoch at which to unfreeze the backbone for finetuning."""
    backbone_lambda_func: str | None = None
    """Optional lambda function for custom backbone parameter unfreezing logic."""
    backbone_train_bn: bool = True
    """Whether to train batch normalization layers during backbone finetuning."""

    use_timer: bool = False
    """Enable timer callback to track training duration."""
    timer_duration: dict[str, int] | None = None
    """Maximum training duration as dict (e.g., {'hours': 2, 'minutes': 30})."""
    timer_interval: str = "step"
    """Timer interval ('step' or 'epoch')."""

    def setup_target(self) -> list[Callback]:
        callbacks: list[Callback] = []

        if self.use_model_checkpoint:
            dirpath = (
                self.checkpoint_dir if self.checkpoint_dir is not None else PathConfig().checkpoints
            )
            dirpath.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    monitor=self.checkpoint_monitor,
                    mode=self.checkpoint_mode,
                    save_top_k=self.checkpoint_save_top_k,
                    filename=self.checkpoint_filename,
                    dirpath=dirpath.as_posix(),
                ),
            )

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

        if self.use_rich_model_summary:
            callbacks.append(
                RichModelSummary(max_depth=self.rich_summary_max_depth),
            )

        if self.use_batch_size_finder:
            callbacks.append(
                BatchSizeFinder(
                    mode=self.batch_size_mode,
                    init_val=self.batch_size_init_val,
                    max_trials=self.batch_size_max_trials,
                ),
            )

        if self.use_backbone_finetuning:
            callbacks.append(
                BackboneFinetuning(
                    unfreeze_backbone_at_epoch=self.backbone_unfreeze_at_epoch,
                    lambda_func=eval(self.backbone_lambda_func) if self.backbone_lambda_func else None,
                    backbone_initial_ratio_lr=0.1,
                    should_align=True,
                    train_bn=self.backbone_train_bn,
                ),
            )

        if self.use_timer:
            callbacks.append(
                Timer(
                    duration=timedelta(**self.timer_duration) if self.timer_duration else None,
                    interval=self.timer_interval,
                ),
            )

        return callbacks
