from pathlib import Path

from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from ..utils import BaseConfig


class TrainerCallbacksConfig(BaseConfig[list[Callback]]):
    """Configuration for standard trainer callbacks."""

    use_model_checkpoint: bool = True
    checkpoint_monitor: str = "val/loss"
    checkpoint_mode: str = "min"
    checkpoint_dir: Path | None = None
    checkpoint_filename: str = "epoch={epoch}-val_loss={val/loss:.2f}"
    checkpoint_save_top_k: int = 1

    use_early_stopping: bool = False
    early_stopping_monitor: str = "val/loss"
    early_stopping_mode: str = "min"
    early_stopping_patience: int = 5

    use_lr_monitor: bool = True
    lr_logging_interval: str = "epoch"

    def setup_target(self) -> list[Callback]:
        callbacks: list[Callback] = []

        if self.use_model_checkpoint:
            dirpath = (
                self.checkpoint_dir
                if self.checkpoint_dir is not None
                else Path.cwd() / "checkpoints"
            )
            dirpath.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    monitor=self.checkpoint_monitor,
                    mode=self.checkpoint_mode,
                    save_top_k=self.checkpoint_save_top_k,
                    filename=self.checkpoint_filename,
                    dirpath=str(dirpath),
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
