from datetime import timedelta
from pathlib import Path

from pydantic import model_validator
from pytorch_lightning.callbacks import (
    BackboneFinetuning,
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

from ..configs.path_config import PathConfig
from ..utils import BaseConfig, Metric


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

    @model_validator(mode="after")
    def _validate_progress_bars_mutually_exclusive(self) -> Self:
        """Ensure only one progress bar type is enabled."""
        if self.use_rich_progress_bar and self.use_tqdm_progress_bar:
            raise ValueError(
                "use_rich_progress_bar and use_tqdm_progress_bar are mutually exclusive. "
                "Enable only one progress bar type."
            )
        return self

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
                    filename=self.checkpoint_filename.replace("/", "-"),
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
            callbacks.append(
                BackboneFinetuning(
                    unfreeze_backbone_at_epoch=self.backbone_unfreeze_at_epoch,
                    lambda_func=eval(self.backbone_lambda_func)
                    if self.backbone_lambda_func
                    else None,
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
