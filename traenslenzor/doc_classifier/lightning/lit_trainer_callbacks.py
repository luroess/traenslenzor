from pathlib import Path

from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from ..configs.path_config import PathConfig
from ..utils import BaseConfig, Metric


# TODO: add RichModelSummary, BatchSizeFinder, BackboneFinetuning, Timer after getting the respective library docs (#get-library-docs), probably with id /lightning-ai/pytorch-lightning
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

    def setup_target(self) -> list[Callback]:
        callbacks: list[Callback] = []

        if self.use_model_checkpoint:
            dirpath = (
                self.checkpoint_dir
                if self.checkpoint_dir is not None
                else PathConfig().checkpoints
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

        return callbacks
