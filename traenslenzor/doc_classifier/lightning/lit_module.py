"""Lightning module for the document-class detector.

Provides AlexNet, ResNet-50, and ViT-B/16 backbones with optional head-only
training and OneCycle learning-rate scheduling.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal

import pytorch_lightning as pl
import torchmetrics
from jaxtyping import Float, Int64
from pydantic import Field
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import models

from ..models.alexnet import AlexNetParams
from ..utils import BaseConfig, Metric

if TYPE_CHECKING:
    pass

# Type aliases for tensor shapes using jaxtyping
# B = batch, C = channels, H = height, W = width, N = num_classes
ImageBatch = Float[Tensor, "B C H W"]
Logits = Float[Tensor, "B N"]
Targets = Int64[Tensor, "B"]
ScalarLoss = Float[Tensor, ""]


class OptimizerConfig(BaseConfig[Optimizer]):
    """AdamW optimizer configuration tailored for document classification."""

    learning_rate: float = 5e-4
    """Learning rate for AdamW optimizer."""

    weight_decay: float = 1e-4
    """L2 regularization weight decay."""

    def setup_target(self, params: Iterable[Tensor]) -> Optimizer:
        """Instantiate the AdamW optimizer.

        Args:
            params: Iterable of model parameters to optimize.

        Returns:
            Optimizer: Configured AdamW optimizer instance.
        """
        return AdamW(params=params, lr=self.learning_rate, weight_decay=self.weight_decay)


class OneCycleSchedulerConfig(BaseConfig[OneCycleLR]):
    """OneCycle learning-rate scheduler configuration."""

    max_lr: float = 0.01
    """Maximum learning rate in the cycle."""

    base_momentum: float = 0.85
    """Lower momentum boundary in the cycle."""

    max_momentum: float = 0.95
    """Upper momentum boundary in the cycle."""

    div_factor: float = 25.0
    """Initial learning rate = max_lr / div_factor."""

    final_div_factor: float = 1e4
    """Final learning rate = max_lr / (div_factor * final_div_factor)."""

    pct_start: float = 0.3
    """Percentage of cycle spent increasing learning rate."""

    anneal_strategy: Literal["cos", "linear"] = "cos"
    """Annealing strategy: 'cos' or 'linear'."""

    def setup_target(
        self,
        optimizer: Optimizer,
        *,
        total_steps: int,
    ) -> OneCycleLR:
        """Instantiate the OneCycle scheduler.

        Args:
            optimizer: Optimizer instance to schedule.
            total_steps: Total number of training steps (batches).

        Returns:
            OneCycleLR: Configured OneCycle scheduler instance.
        """
        return OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=total_steps,
            pct_start=self.pct_start,
            anneal_strategy=self.anneal_strategy,
            cycle_momentum=True,
            base_momentum=self.base_momentum,
            max_momentum=self.max_momentum,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
        )


class BackboneType(Enum):
    """Supported vision backbones for document classification."""

    ALEXNET = auto()
    RESNET50 = auto()
    VIT_B16 = auto()

    def build(self, num_classes: int, train_head_only: bool, use_pretrained: bool) -> nn.Module:
        """Instantiate the selected backbone with the requested number of output classes.

        Args:
            num_classes: Number of output classes for classification head.
            train_head_only: If True, freeze backbone weights and train only the head.
            use_pretrained: If True, load ImageNet pretrained weights (ignored for AlexNet).

        Returns:
            nn.Module: Configured model with classification head for num_classes.
        """
        match self:
            case BackboneType.ALEXNET:
                return AlexNetParams(num_classes=num_classes).setup_target()

            case BackboneType.RESNET50:
                # Instantiate ResNet50 with ImageNet weights (V2: acc@1 80.86)
                model = models.resnet50(
                    weights=models.ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
                )
                if train_head_only:
                    for param in model.parameters():
                        param.requires_grad = False
                # Replace classification head: fc = nn.Linear(2048, num_classes)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                return model

            case BackboneType.VIT_B16:
                # Instantiate Vision Transformer with ImageNet weights
                model = models.vit_b_16(
                    weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if use_pretrained else None
                )
                if train_head_only:
                    for param in model.parameters():
                        param.requires_grad = False
                # Replace classification head: heads = nn.Linear(hidden_dim, num_classes)
                model.heads = nn.Linear(model.hidden_dim, num_classes)
                return model


class DocClassifierConfig(BaseConfig["DocClassifierModule"]):
    """Lightning module configuration for document classification."""

    target: type["DocClassifierModule"] = Field(
        default_factory=lambda: DocClassifierModule, exclude=True
    )

    num_classes: int = 16
    """Number of document classes (RVL-CDIP has 16)."""

    backbone: BackboneType = BackboneType.ALEXNET
    """Vision backbone architecture."""

    train_head_only: bool = False
    """If True, freeze backbone and train only classification head."""

    use_pretrained: bool = True
    """If True, initialize backbone with ImageNet pretrained weights."""

    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    """Optimizer configuration."""

    scheduler: OneCycleSchedulerConfig = Field(default_factory=OneCycleSchedulerConfig)
    """Learning rate scheduler configuration."""


class DocClassifierModule(pl.LightningModule):
    """Lightning module implementing training/validation/test loops for document classification.

    Supports three backbones (AlexNet, ResNet-50, ViT-B/16) with optional head-only training.
    Uses CrossEntropyLoss and torchmetrics for accuracy tracking.
    """

    def __init__(self, config: DocClassifierConfig):
        """Initialize the module with the given configuration.

        Args:
            config: Configuration specifying backbone, optimizer, scheduler, etc.
        """
        super().__init__()
        self.config = config
        self.model = config.backbone.build(
            num_classes=config.num_classes,
            train_head_only=config.train_head_only,
            use_pretrained=config.use_pretrained,
        )
        self.loss_fn = nn.CrossEntropyLoss()

        # Torch Metrics
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=config.num_classes
        )
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=config.num_classes)
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=config.num_classes
        )

    # ---------------------------------------------------------------------- steps
    def forward(self, batch: ImageBatch) -> Logits:
        """Run a forward pass through the backbone.

        Args:
            batch: Batch of input images with shape (B, C, H, W) where:
                   B = batch size, C = channels, H = height, W = width.

        Returns:
            Raw logits for each class with shape (B, N) where N = num_classes.
        """
        return self.model(batch)

    def training_step(self, batch: tuple[ImageBatch, Targets], batch_idx: int) -> ScalarLoss:
        """Compute the training loss and log metrics.

        Args:
            batch: Tuple of (inputs, targets) where:
                   - inputs: ImageBatch with shape (B, C, H, W)
                   - targets: Targets with shape (B,) containing class indices
            batch_idx: Index of the current batch.

        Returns:
            Scalar training loss.
        """
        inputs, targets = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        self.train_accuracy(logits, targets)
        self.log(Metric.TRAIN_LOSS, loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            Metric.TRAIN_ACCURACY,
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: tuple[ImageBatch, Targets], batch_idx: int) -> None:
        """Log validation loss and accuracy.

        Args:
            batch: Tuple of (inputs, targets) where:
                   - inputs: ImageBatch with shape (B, C, H, W)
                   - targets: Targets with shape (B,) containing class indices
            batch_idx: Index of the current batch.
        """
        inputs, targets = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        self.val_accuracy(logits, targets)
        self.log(Metric.VAL_LOSS, loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            Metric.VAL_ACCURACY,
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch: tuple[ImageBatch, Targets], batch_idx: int) -> None:
        """Log test metrics.

        Args:
            batch: Tuple of (inputs, targets) where:
                   - inputs: ImageBatch with shape (B, C, H, W)
                   - targets: Targets with shape (B,) containing class indices
            batch_idx: Index of the current batch.
        """
        inputs, targets = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        self.test_accuracy(logits, targets)
        self.log(Metric.TEST_LOSS, loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            Metric.TEST_ACCURACY,
            self.test_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    # ---------------------------------------------------------------- optimizers
    def configure_optimizers(self) -> dict[str, Any]:
        """Instantiate optimizer and scheduler using their configs.

        Returns:
            dict: Lightning optimizer configuration with scheduler.
        """
        trainable_params = (p for p in self.parameters() if p.requires_grad)
        optimizer = self.config.optimizer.setup_target(trainable_params)
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps is None or total_steps <= 0:
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())  # type: ignore[union-attr]
            max_epochs = getattr(self.trainer, "max_epochs", 1) or 1
            total_steps = steps_per_epoch * max_epochs
        scheduler = self.config.scheduler.setup_target(optimizer, total_steps=total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "val/loss",
            },
        }

    def on_train_start(self) -> None:
        """Log hyperparameters to the attached logger."""
        if hasattr(self.logger, "log_hyperparams"):
            self.logger.log_hyperparams(self.config.model_dump())
