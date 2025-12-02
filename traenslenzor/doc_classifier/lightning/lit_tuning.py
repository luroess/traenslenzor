"""Lightning Tuner configuration and execution for hyperparameter optimization.

This module provides automated tuning for batch size and learning rate using
PyTorch Lightning's Tuner class. The Tuner runs BEFORE training begins to find
optimal hyperparameters.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import wandb
from matplotlib import pyplot as plt
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner import Tuner

from ..utils import BaseConfig, Console

if TYPE_CHECKING:
    from ..lightning import DocClassifierModule


class TunerConfig(BaseConfig[Tuner]):
    """Configuration for [PyTorch Lightning Tuner](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.tuner.tuning.Tuner.html) (batch size and learning rate optimization).

    The Tuner provides two main methods:
    1. `scale_batch_size()`: Find optimal (max) batch size based on available GPU memory
    2. `lr_find()`: Run LR Range Test to find optimal learning rate based on the [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)

    """

    use_batch_size_tuning: bool = False
    """Enable automatic batch size tuning to find optimal batch size that fits in GPU memory."""

    batch_size_mode: str = "binsearch"
    """Batch size search mode: 'power' (powers of 2) or 'binsearch' (binary search).
    'binsearch' is recommended as it's more efficient and won't exceed dataset size as easily."""

    batch_size_init_val: int = 32
    """Initial batch size value for the finder. Start with a reasonable value (32-64) for faster convergence."""

    batch_size_steps_per_trial: int = 20
    """Number of training steps to run per batch size trial.
    Higher values (15-25) ensure proper GPU memory allocation and more accurate OOM detection.
    Must be high enough for PyTorch to fully allocate memory (cumulative over forward+backward passes)."""

    batch_size_max_trials: int = 15
    """Maximum number of trials for batch size finder. Reduced from 25 since binsearch is more efficient."""

    # ---------------------------------------------------------------------- LR Tuning
    use_learning_rate_tuning: bool = False
    """Enable learning rate finder to run LR Range Test (Leslie Smith method)."""

    lr_attr_name: str = "optimizer.learning_rate"
    """Attribute name for learning rate in model.hparams (must match what's saved in save_hyperparameters)."""

    lr_min: float = 1e-6
    """Minimum learning rate for LR finder range test."""

    lr_max: float = 1e-1
    """Maximum learning rate for LR finder range test."""

    lr_num_training: int = 100
    """Number of training iterations for LR finder (trains while exponentially increasing LR)."""

    lr_plot_path: Path | None = None
    """Deprecated. Kept for backward compatibility; LR plots are logged to WandB only."""

    update_model_lr: bool = True
    """Whether to automatically update the model's learning rate with the suggested value."""

    verbose: bool = True
    """Enable verbose logging for tuning operations."""

    is_debug: bool = False
    """Enable debug mode for additional diagnostic output."""

    def setup_target(self) -> Tuner:
        """Create Tuner instance (requires Trainer, created separately)."""
        raise NotImplementedError(
            "TunerConfig.setup_target() requires a Trainer instance. "
            "Use run_tuning() instead, which accepts trainer, module, and datamodule."
        )

    def run_tuning(
        self,
        trainer: Trainer,
        lit_module: "DocClassifierModule",
        lit_datamodule: LightningDataModule,
    ) -> tuple[int | None, float | None]:
        """Run batch size and/or learning rate tuning before training.

        Args:
            trainer: PyTorch Lightning Trainer instance.
            lit_module: Lightning module to tune hyperparameters for.
            lit_datamodule: Lightning data module providing training data.

        Returns:
            Tuple of (optimal_batch_size, suggested_learning_rate).
            Values are None if corresponding tuning is disabled.
        """
        console = Console.with_prefix(self.__class__.__name__, "run_tuning")
        console.set_verbose(self.verbose).set_debug(self.is_debug)

        if not self.use_batch_size_tuning and not self.use_learning_rate_tuning:
            console.log(
                "Tuning disabled (both use_batch_size_tuning and use_learning_rate_tuning are False)"
            )
            return None, None

        lit_module._is_tuning = True

        console.log(
            f"Initializing PyTorch Lightning Tuner to tune {'batch size' if self.use_batch_size_tuning else ''}{' and ' if self.use_batch_size_tuning and self.use_learning_rate_tuning else ''}{'learning rate' if self.use_learning_rate_tuning else ''}."
        )
        tuner = Tuner(trainer)

        optimal_batch_size = (
            self._tune_batch_size(
                tuner=tuner,
                trainer=trainer,
                lit_module=lit_module,
                lit_datamodule=lit_datamodule,
                console=console,
            )
            if self.use_batch_size_tuning
            else None
        )

        suggested_lr = (
            self._tune_learning_rate(
                tuner=tuner,
                trainer=trainer,
                lit_module=lit_module,
                lit_datamodule=lit_datamodule,
                console=console,
            )
            if self.use_learning_rate_tuning
            else None
        )

        return optimal_batch_size, suggested_lr

    def _tune_batch_size(
        self,
        tuner: Tuner,
        trainer: Trainer,
        lit_module: "DocClassifierModule",
        lit_datamodule: LightningDataModule,
        console: Console,
    ) -> int | None:
        """Run Lightning's batch size finder and validate the result."""
        lit_datamodule.setup("fit")
        dataset_size = len(lit_datamodule.train_ds)

        console.log("Running batch size tuning.")
        console.plog(
            {
                "mode": self.batch_size_mode,
                "init_val": self.batch_size_init_val,
                "steps_per_trial": self.batch_size_steps_per_trial,
                "max_trials": self.batch_size_max_trials,
                "dataset_size": dataset_size,
            }
        )

        try:
            tuner.scale_batch_size(
                lit_module,
                datamodule=lit_datamodule,
                mode=self.batch_size_mode,
                steps_per_trial=self.batch_size_steps_per_trial,
                init_val=self.batch_size_init_val,
                max_trials=self.batch_size_max_trials,
            )
        except Exception as exc:
            console.error(f"Batch size tuning failed: {exc}")
            console.warn("Continuing with original batch size.")
            return None

        optimal_batch_size = int(lit_datamodule.hparams.batch_size)

        console.log(f"Optimal batch size: '{optimal_batch_size}'")
        ratio = optimal_batch_size / dataset_size if dataset_size else 0.0
        self._log_metrics(
            trainer,
            {
                "tuner/optimal_batch_size": float(optimal_batch_size),
                "tuner/dataset_size": float(dataset_size),
                "tuner/batch_to_dataset_ratio": ratio,
            },
        )
        return optimal_batch_size

    def _tune_learning_rate(
        self,
        tuner: Tuner,
        trainer: Trainer,
        lit_module: "DocClassifierModule",
        lit_datamodule: LightningDataModule,
        console: Console,
    ) -> float | None:
        """Run Leslie Smith's LR range test via Lightning's ``lr_find``.

        The method traverses one epoch, while linearly or exponentially increasing the
        learning rate from ``lr_min`` to ``lr_max`` on every step. Tracking the loss
        curve reveals the steepest stable descent before divergence, which is a
        reliable upper bound.
        """
        console.log(
            "Running LR range test "
            f"[{self.lr_min:.2e}, {self.lr_max:.2e}] over {self.lr_num_training} steps."
        )

        try:
            lr_finder = tuner.lr_find(
                lit_module,
                datamodule=lit_datamodule,
                min_lr=self.lr_min,
                max_lr=self.lr_max,
                num_training=self.lr_num_training,
                attr_name=self.lr_attr_name,
            )
        except Exception as exc:
            console.error(f"Learning rate finder failed: {exc}")
            console.warn("Continuing with original learning rate.")
            return None

        suggested_lr = float(lr_finder.suggestion())
        console.log(f"Suggested learning rate: {suggested_lr:.2e}")

        self._log_metrics(
            trainer,
            {
                "tuner/suggested_learning_rate": suggested_lr,
                "tuner/lr_min": self.lr_min,
                "tuner/lr_max": self.lr_max,
            },
        )

        optimizer_params = getattr(lit_module, "params", None)
        if (
            self.update_model_lr
            and optimizer_params is not None
            and hasattr(optimizer_params.optimizer, "learning_rate")
        ):
            old_lr = optimizer_params.optimizer.learning_rate
            optimizer_params.optimizer.learning_rate = suggested_lr
            console.log(f"Updated optimizer LR: {old_lr:.2e} → {suggested_lr:.2e}")

        self._log_lr_plot(lr_finder, console, trainer)
        return suggested_lr

    def _log_lr_plot(self, lr_finder, console: Console, trainer: Trainer) -> None:
        """Log the LR finder plot to WandB when available."""
        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            console.log("No experiment logger attached; skipping LR finder plot.")
            return

        matplotlib.use("Agg")
        fig = lr_finder.plot(suggest=True)

        if isinstance(logger, WandbLogger):
            logger.experiment.log({"tuner/lr_range_test": wandb.Image(fig)})
            console.log("LR finder plot logged to WandB.")

        plt.close(fig)

    @staticmethod
    def _log_metrics(trainer: Trainer, metrics: dict[str, float]) -> None:
        """Log tuner metrics through the trainer logger if available."""
        if trainer.logger is None:
            return
        trainer.logger.log_metrics(metrics, step=0)
