"""Lightning module for the document-class detector.

Provides AlexNet, ResNet-50, and ViT-B/16 backbones with optional head-only
training and OneCycle learning-rate scheduling. The implementation is adapted
from the previous ADL ingredient classifier but updated to the UniTraj style
guide and Config-as-Factory pattern.
"""

from collections.abc import Iterable
from enum import Enum, auto
from typing import Any

import pytorch_lightning as pl
import torchmetrics
from pydantic import Field
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import models

from ..models.alexnet import AlexNetParams
from ..utils import BaseConfig


class OptimizerConfig(BaseConfig[Optimizer]):
    """AdamW optimizer configuration tailored for document classification."""

    learning_rate: float = 5e-4
    weight_decay: float = 1e-4

    def setup_target(self, params: Iterable[Tensor]) -> Optimizer:
        """Instantiate the AdamW optimizer."""
        return AdamW(params=params, lr=self.learning_rate, weight_decay=self.weight_decay)


class OneCycleSchedulerConfig(BaseConfig[OneCycleLR]):
    """OneCycle learning-rate scheduler configuration."""

    max_lr: float = 0.01
    base_momentum: float = 0.85
    max_momentum: float = 0.95
    div_factor: float = 25.0
    final_div_factor: float = 1e4
    pct_start: float = 0.3
    anneal_strategy: str = "cos"

    def setup_target(
        self,
        optimizer: Optimizer,
        *,
        total_steps: int,
    ) -> OneCycleLR:
        """Instantiate the OneCycle scheduler."""
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
    """Supported vision backbones."""

    ALEXNET = auto()
    RESNET50 = auto()
    VIT_B16 = auto()

    def build(self, num_classes: int, train_head_only: bool, use_pretrained: bool) -> nn.Module:
        """Instantiate the selected backbone with the requested number of output classes."""
        if self is BackboneType.ALEXNET:
            return AlexNetParams(num_classes=num_classes).setup_target()
        elif self is BackboneType.RESNET50:
            # Instantiate ResNet50 with ImageNet weights (V1: acc@1 76.13, V2: acc@1 80.86)
            model = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
            )
            if train_head_only:
                for param in model.parameters():
                    param.requires_grad = False

            # self.fc = nn.Linear(512 * block.expansion, num_classes)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif self is BackboneType.VIT_B16:
            # Instantiate Vision Transformer with ImageNet weights
            model = models.vit_b_16(
                weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if use_pretrained else None
            )
            # if representation_size is None:
            #     heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
            # ...
            # self.heads = nn.Sequential(heads_layers)
            # Use representation_size = None, as it is the default value
            if train_head_only:
                for param in model.parameters():
                    param.requires_grad = False
            model.heads = nn.Linear(model.hidden_dim, num_classes)
            return model


class DocClassifierConfig(BaseConfig["DocClassifierModule"]):
    """Lightning module configuration for document classification."""

    target: type["DocClassifierModule"] = Field(default_factory=lambda: DocClassifierModule)

    num_classes: int = 16
    backbone: BackboneType = BackboneType.ALEXNET
    train_head_only: bool = False
    use_pretrained: bool = True
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: OneCycleSchedulerConfig = Field(default_factory=OneCycleSchedulerConfig)


class DocClassifierModule(pl.LightningModule):
    """Lightning module implementing training/validation/test loops for document classification."""

    def __init__(self, config: DocClassifierConfig):
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
    def forward(self, batch: Tensor) -> Tensor:
        """Run a forward pass through the backbone."""
        return self.model(batch)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Compute the training loss and log metrics."""
        inputs, targets = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        self.train_accuracy(logits, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/accuracy",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Log validation loss and accuracy."""
        inputs, targets = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        self.val_accuracy(logits, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/accuracy",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Log test metrics."""
        inputs, targets = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        self.test_accuracy(logits, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/accuracy",
            self.test_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    # ---------------------------------------------------------------- optimizers
    def configure_optimizers(self) -> dict[str, Any]:
        """Instantiate optimizer and scheduler using their configs."""
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
