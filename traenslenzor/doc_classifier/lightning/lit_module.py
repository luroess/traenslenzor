"""Lightning module for the document-class detector.

Provides AlexNet, ResNet-50, and ViT-B/16 backbones with optional head-only
training and OneCycle learning-rate scheduling.
"""

from collections.abc import Iterable
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Sequence

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from jaxtyping import Float, Int64
from pydantic import Field
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import models

from ..models.alexnet import AlexNetParams
from ..utils import BaseConfig, Console, Metric

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


class BackboneType(StrEnum):
    """Supported vision backbones for document classification."""

    ALEXNET = "alexnet"
    RESNET50 = "resnet50"
    VIT_B16 = "vit_b_16"

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

    def __init__(self, params: DocClassifierConfig):
        """Initialize the module with the given configuration.

        Args:
            config: Configuration specifying backbone, optimizer, scheduler, etc.
        """
        super().__init__()
        self.params = params

        # Save hyperparameters for Lightning and logging
        hparams = params.model_dump()
        hparams["optimizer.learning_rate"] = params.optimizer.learning_rate  # Flatten for LR Finder
        self.save_hyperparameters(hparams)

        self.model = params.backbone.build(
            num_classes=params.num_classes,
            train_head_only=params.train_head_only,
            use_pretrained=params.use_pretrained,
        )
        self.loss_fn = nn.CrossEntropyLoss()

        # Torch Metrics
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=params.num_classes
        )
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=params.num_classes)
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=params.num_classes
        )
        self.confusion_matrix = torchmetrics.ConfusionMatrix(
            num_classes=self.params.num_classes,
            task="multiclass",
        )

        self._is_tuning: bool = False

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

    @torch.no_grad()
    def predict_proba(self, batch: ImageBatch) -> Float[Tensor, "B N"]:
        """Inference helper that returns softmax probabilities.

        Keeps the module in eval mode and avoids gradient tracking so it can be
        safely used from serving code (e.g. the MCP runtime).
        """

        was_training = self.training
        self.eval()
        logits = self(batch)
        probs = F.softmax(logits, dim=-1)
        if was_training:
            self.train()
        return probs

    @torch.no_grad()
    def classify_batch(
        self,
        batch: ImageBatch,
        *,
        class_names: Sequence[str] | None = None,
        top_k: int = 3,
    ) -> list[list[dict[str, float | int | str]]]:
        """Convert model outputs into top-k label dictionaries.

        Args:
            batch: Normalized tensor batch with shape (B, C, H, W).
            class_names: Optional list of label names matching the class indices.
            top_k: Number of highest-probability classes to return per item.

        Returns:
            List (per item) of dictionaries containing label metadata suitable for
            MCP serialization.
        """

        probs = self.predict_proba(batch)
        top_probs, top_indices = probs.topk(k=top_k, dim=-1)

        results: list[list[dict[str, float | int | str]]] = []
        for row_indices, row_probs in zip(top_indices, top_probs):
            item_preds: list[dict[str, float | int | str]] = []
            for idx, prob in zip(row_indices, row_probs):
                class_idx = int(idx)
                label = class_names[class_idx] if class_names is not None else f"class_{class_idx}"
                item_preds.append(
                    {
                        "label": label,
                        "index": class_idx,
                        "probability": float(prob),
                    }
                )
            results.append(item_preds)

        return results

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
            on_step=True,
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
        self.confusion_matrix(logits, targets)

        self.log(Metric.VAL_LOSS, loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            Metric.VAL_ACCURACY,
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self) -> None:
        """Log confusion matrix plot and table to WandB at the end of validation epoch."""
        if self.logger is None:
            return

        # Compute confusion matrix
        confmat = self.confusion_matrix.compute()

        # Get class names from the datamodule
        class_names = self._get_class_names()

        # Log confusion matrix plot
        self._log_confusion_matrix_plot(confmat, class_names)

        # Log confusion matrix as table
        self._log_confusion_matrix_table(confmat, class_names)

        # Reset confusion matrix for next epoch
        self.confusion_matrix.reset()

    def _log_confusion_matrix_plot(
        self,
        confmat: Float[Tensor, "N N"],
        class_names: list[str] | None,
    ) -> None:
        """Create and log confusion matrix plot to WandB.

        Args:
            confmat (Tensor['N N', float]): Computed confusion matrix tensor.
            class_names: List of class names for axis labels, or None.
        """
        if not hasattr(self.logger, "experiment"):
            return

        import matplotlib.pyplot as plt
        import wandb

        # Create confusion matrix plot using torchmetrics
        fig, ax = self.confusion_matrix.plot(val=confmat)

        # Customize the plot with class names if available
        if class_names is not None:
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(class_names, rotation=0, fontsize=8)
            ax.set_xlabel("Predicted Class", fontsize=10)
            ax.set_ylabel("True Class", fontsize=10)
            ax.set_title(
                f"Confusion Matrix - Epoch {self.current_epoch}",
                fontsize=12,
                fontweight="bold",
            )

        # Log to WandB
        self.logger.experiment.log(
            {
                Metric.VAL_CONFUSION_MATRIX: wandb.Image(fig),
                "epoch": self.current_epoch,
            }
        )

        # Close the figure to free memory
        plt.close(fig)

    def _log_confusion_matrix_table(
        self,
        confmat: Float[Tensor, "N N"],
        class_names: list[str] | None,
    ) -> None:
        """Log confusion matrix as WandB Table for interactive exploration.

        Args:
            confmat (Tensor['N N', float]): Computed confusion matrix tensor.
            class_names: List of class names for table labels, or None.
        """
        if not hasattr(self.logger, "experiment"):
            return

        import wandb

        # Convert tensor to numpy for table creation
        confmat_np = confmat.cpu().numpy()

        # Use class names if available, otherwise use indices
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(confmat_np))]

        # Create WandB Table with confusion matrix data
        # Columns: "Actual" + predicted class names
        columns = ["Actual"] + class_names
        data = []

        for i, actual_class in enumerate(class_names):
            row = [actual_class] + confmat_np[i].tolist()
            data.append(row)

        table = wandb.Table(columns=columns, data=data)

        # Log table to WandB
        self.logger.experiment.log(
            {
                f"{Metric.VAL_CONFUSION_MATRIX}_table": table,
                "epoch": self.current_epoch,
            }
        )

    def _get_class_names(self) -> list[str] | None:
        """Get class names from the dataset.

        Returns:
            List of class names if available, None otherwise.
        """
        try:
            # Access the datamodule's validation dataset
            if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
                val_ds = self.trainer.datamodule.val_ds

                # Extract class names from HuggingFace dataset features
                if hasattr(val_ds, "features") and "label" in val_ds.features:
                    label_feature = val_ds.features["label"]
                    if hasattr(label_feature, "names"):
                        return label_feature.names

        except Exception as e:
            console = Console.with_prefix(self.__class__.__name__, "_get_class_names")
            console.warn(f"Failed to get class names from dataset: {e}")

        return None

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

    def on_train_start(self):
        if self.logger is not None and Console._pl_logger is None:
            console = Console.integrate_with_logger(self.logger, self.global_step).with_prefix(
                self.__class__.__name__, "train_start"
            )
            console.log(
                f"Integrated Console with Lightning logger '{self.logger.__class__.__name__}'."
            )
        else:
            console = Console.with_prefix(self.__class__.__name__, "train_start")

        if not self._is_tuning:
            hparams = self.hparams
            hparams.update({"batch_size": self.trainer.datamodule.config.batch_size})
            console.plog(hparams)

    # ---------------------------------------------------------------- optimizers
    def configure_optimizers(self) -> dict[str, Any]:
        """Instantiate optimizer and scheduler using their configs.

        Returns:
            dict: Lightning optimizer configuration with scheduler.
        """
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())

        optimizer = self.params.optimizer.setup_target(trainable_params)

        if (
            total_steps := getattr(self.trainer, "estimated_stepping_batches", None)
        ) is None or total_steps <= 0:
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            total_steps = steps_per_epoch * getattr(self.trainer, "max_epochs", 1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.params.scheduler.setup_target(optimizer, total_steps=total_steps),
                "interval": "step",
                "monitor": Metric.VAL_LOSS,
            },
        }
