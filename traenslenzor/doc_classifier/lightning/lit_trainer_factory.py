"""Trainer factory with W&B integration.

Provides a configurable wrapper to instantiate PyTorch Lightning trainers.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import pytorch_lightning as pl
import torch
from pydantic import Field, model_validator
from pytorch_lightning.trainer.connectors.accelerator_connector import _PRECISION_INPUT
from typing_extensions import Self

from ..configs.wandb_config import WandbConfig
from ..utils import BaseConfig, Console
from .lit_trainer_callbacks import TrainerCallbacksConfig

if TYPE_CHECKING:
    from ..configs.experiment_config import ExperimentConfig


class TrainerFactoryConfig(BaseConfig):
    """Configuration for constructing a PyTorch Lightning trainer."""

    target: type[pl.Trainer] = Field(default_factory=lambda: pl.Trainer, exclude=True)

    is_debug: bool = False
    """Set fast_dev_run to True, use CPU, set num_workers to 0, don't create model_checkpoints if True"""

    fast_dev_run: bool = False
    """Runs n if set to n (int) else 1 if set to True batch(es) of train, val and test to find any bugs (ie: a sort of unit test). Default: False."""

    accelerator: str = "auto"
    """Supports passing different accelerator types ("cpu", "gpu", "tpu", "hpu", "mps", "auto") as well as custom accelerator instances."""

    devices: int | str | Sequence[int] = "auto"
    """The devices to use. Can be set to a positive number (int or str), a sequence of device indices (list or str), the value -1 to indicate all available devices should be used, or "auto" for automatic selection based on the chosen accelerator. Default: "auto"."""

    strategy: str | None = "auto"
    """Supports different training strategies with aliases as well custom strategies. Default: "auto"."""

    max_epochs: int | None = 10
    """Stop training once this number of epochs is reached. Disabled by default (None). If both max_epochs and max_steps are not specified, defaults to max_epochs = 1000. To enable infinite training, set max_epochs = -1."""

    precision: _PRECISION_INPUT = "32-true"
    """Controls the dtype of model weights and activations. Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'), 16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed'). Can be used on CPU, GPU, TPUs, or HPUs. Default: '32-true'."""

    tf32_matmul_precision: str | None = "medium"
    """TensorFloat-32 matmul precision for CUDA devices with Tensor Cores.

    Options:
        - 'highest': Full IEEE FP32 (no TF32) - Most accurate, slowest
        - 'high': TF32 enabled - Good balance of speed and accuracy (~2-3x faster)
        - 'medium': TF32 enabled - Maximum speed, slightly less precise than 'high'
        - None: Use PyTorch default (typically allows TF32)

    NOTE: This setting ONLY affects FP32 operations on Tensore Cores (precision='32-true' or '64-true').
    When using mixed precision (precision='16-mixed' or 'bf16-mixed'), this setting
    has no effect because operations use FP16/BF16 natively, not FP32.

    Recommended combinations:
        - precision='32-true' + tf32_matmul_precision='medium': Fast training (default)
        - precision='32-true' + tf32_matmul_precision='highest': Maximum accuracy

    Default: 'medium'."""

    gradient_clip_val: float | None = None
    """The value at which to clip gradients. Passing gradient_clip_val=None disables gradient clipping. If using Automatic Mixed Precision (AMP), the gradients will be unscaled before. Default: None."""

    accumulate_grad_batches: int = 1
    """Accumulates gradients over k batches before stepping the optimizer. Default: 1."""

    log_every_n_steps: int = 50
    """How often to log within steps. Default: 50."""

    deterministic: bool | str | None = None
    """If True, sets whether PyTorch operations must use deterministic algorithms. Set to "warn" to use deterministic algorithms whenever possible, throwing warnings on operations that don't support deterministic mode. If not set, defaults to False. Default: None."""

    limit_train_batches: int | float | None = None
    """How much of training dataset to check (float = fraction, int = num_batches). Value is per device. Default: 1.0."""

    limit_val_batches: int | float | None = None
    """How much of validation dataset to check (float = fraction, int = num_batches). Value is per device. Default: 1.0."""

    check_val_every_n_epoch: int = 1
    """Perform a validation loop after every N training epochs. If None, validation will be done solely based on the number of training batches, requiring val_check_interval to be an integer value. Default: 1."""

    callbacks: TrainerCallbacksConfig = Field(default_factory=TrainerCallbacksConfig)
    """Add a callback or list of callbacks. Default: None."""

    use_wandb: bool = True
    """Whether to enable W&B logging. When True, wandb_config is used to instantiate the logger. Default: True."""

    wandb_config: WandbConfig = Field(default_factory=WandbConfig)
    """W&B logger configuration. Used to instantiate the W&B logger when use_wandb is True."""

    @model_validator(mode="after")
    def _debug_defaults(self) -> Self:
        """Apply debug-mode defaults when is_debug=True.

        This validator runs after initialization and when is_debug is propagated
        from parent configs via setattr(), ensuring debug settings are applied.

        Uses object.__setattr__() to avoid retriggering validation (which would
        cause infinite recursion).
        """
        console = Console.with_prefix(self.__class__.__name__, "_debug_defaults")

        if self.is_debug:
            object.__setattr__(self, "fast_dev_run", True)
            object.__setattr__(self, "accelerator", "cpu")
            object.__setattr__(self, "devices", 1)
            object.__setattr__(self.callbacks, "use_model_checkpoint", False)
            torch.autograd.set_detect_anomaly(True)
            console.log(
                "Debug settings: fast_dev_run=True, accelerator=cpu, devices=1, "
                "checkpointing disabled, anomaly detection enabled"
            )

        if self.fast_dev_run:
            Console.with_prefix(self.__class__.__name__).log(
                "Fast dev run enabled; trainer will use a single batch per split.",
            )
        return self

    def update_wandb_config(self, experiment: "ExperimentConfig") -> None:
        """Propagate experiment metadata into the W&B logger config."""
        if not self.use_wandb:
            return

        console = (
            Console.with_prefix(self.__class__.__name__, "update_wandb_config")
            .set_verbose(experiment.verbose)
            .set_debug(experiment.is_debug)
        )

        if not self.wandb_config.name:
            self.wandb_config.name = experiment.run_name
            console.log(f"Set W&B run name: {experiment.run_name}")

        stage = getattr(experiment, "stage", None)
        if stage is not None:
            tags = set(self.wandb_config.tags or [])
            tags.add(str(stage))
            self.wandb_config.tags = sorted(tags)
            console.log(f"Added stage tag to W&B: {stage}")

    def setup_target(self, experiment: Optional["ExperimentConfig"] = None) -> pl.Trainer:
        """Instantiate the configured trainer.

        Args:
            experiment: Optional experiment config (ignored currently, kept for API compatibility).
        """
        console = Console.with_prefix(self.__class__.__name__, "setup_target")
        if experiment is not None:
            console.set_verbose(getattr(experiment, "verbose", False)).set_debug(
                getattr(experiment, "is_debug", False)
            )

        # Configure TF32 matmul precision for Tensor Cores (Ampere+ GPUs)
        if self.tf32_matmul_precision is not None:
            try:
                torch.set_float32_matmul_precision(self.tf32_matmul_precision)
                console.log(f"Set TF32 matmul precision to '{self.tf32_matmul_precision}'")
            except Exception as e:
                console.warn(f"Failed to set TF32 matmul precision: {e}")

        console.log(f"Creating Trainer with accelerator={self.accelerator}, devices={self.devices}")
        console.log(f"Max epochs: {self.max_epochs}, precision: {self.precision}")

        callbacks = self.callbacks.setup_target(
            model_name=experiment.module_config.backbone if experiment else None
        )
        console.log(f"Configured {len(callbacks)} callbacks: ")
        console.plog(list(map(lambda cb: type(cb).__name__, callbacks)))

        logger = None
        if self.is_debug:
            logger = True
            console.log("Using default logger (debug mode)")
        elif self.use_wandb:
            logger = self.wandb_config.setup_target()
            console.log(f"Using W&B logger: {self.wandb_config.name}")
        else:
            console.log("No logger configured")

        return pl.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            strategy=self.strategy,
            max_epochs=self.max_epochs,
            precision=self.precision,
            gradient_clip_val=self.gradient_clip_val,
            accumulate_grad_batches=self.accumulate_grad_batches,
            log_every_n_steps=self.log_every_n_steps,
            fast_dev_run=self.fast_dev_run,
            deterministic=self.deterministic,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            callbacks=callbacks,
            logger=logger,
        )
