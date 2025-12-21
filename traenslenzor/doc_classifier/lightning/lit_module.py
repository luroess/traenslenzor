"""Lightning module for the document-class detector.

Provides AlexNet, ResNet-50, and ViT-B/16 backbones with optional head-only
training and OneCycle learning-rate scheduling.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Sequence

import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns  # type: ignore[import-untyped]
import torch
import torch.nn.functional as F
import torchmetrics
import wandb
from jaxtyping import Float, Int64
from pydantic import Field
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import models

from ..interpretability import AttributionEngine, InterpretabilityConfig
from ..models.alexnet import AlexNetParams
from ..utils import BaseConfig, Console, Metric

# B = batch, C = channels, H = height, W = width, N = num_classes
ImageBatch_BCHW = Float[Tensor, "B C H W"]
Logits_BN = Float[Tensor, "B N"]
Targets_B = Int64[Tensor, "B"]
ScalarLoss = Float[Tensor, ""]


@dataclass
class BackboneSpec:
    """Container holding backbone and classification head modules."""

    model: nn.Module
    """Full model including both backbone and head."""

    backbone: nn.Module
    """Feature extractor without the task-specific head."""

    head: nn.Module
    """Task-specific classification head producing logits."""


def _set_requires_grad(module: nn.Module, *, requires_grad: bool) -> None:
    """Toggle ``requires_grad`` on all parameters of ``module``."""

    for param in module.parameters():
        param.requires_grad = requires_grad


class OptimizerConfig(BaseConfig[Optimizer]):
    """AdamW optimizer configuration tailored for document classification."""

    learning_rate: float = 5e-4
    """Learning rate for AdamW optimizer."""

    weight_decay: float = 1e-4
    """L2 regularization weight decay."""

    backbone_lr_scale: float = 1.0
    """Scale factor for backbone learning rate relative to head (1.0 = same LR)."""

    def setup_target(self, params: Iterable[Tensor] | list[dict[str, Any]]) -> Optimizer:  # type: ignore[override]
        """Instantiate the AdamW optimizer.

        Args:
            params: Iterable of model parameters or param-group dicts.

        Returns:
            Optimizer: Configured AdamW optimizer instance.
        """
        return AdamW(params=params, lr=self.learning_rate, weight_decay=self.weight_decay)


class OneCycleSchedulerConfig(BaseConfig[OneCycleLR]):
    """OneCycle learning-rate scheduler configuration."""

    max_lr: float = 1e-3
    """Maximum learning rate in the cycle (can be overridden per param group)."""

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

    def setup_target(  # type: ignore[override]
        self,
        optimizer: Optimizer,
        *,
        total_steps: int,
        max_lr: float | list[float] | None = None,
    ) -> OneCycleLR:
        """Instantiate the OneCycle scheduler.

        Args:
            optimizer: Optimizer instance to schedule.
            total_steps: Total number of training steps (batches).
            max_lr: Optional override for maximum learning rate(s). Use a list for per-group values.

        Returns:
            OneCycleLR: Configured OneCycle scheduler instance.
        """
        return OneCycleLR(
            optimizer,
            max_lr=self.max_lr if max_lr is None else max_lr,
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

    def build(self, num_classes: int, train_head_only: bool, use_pretrained: bool) -> BackboneSpec:
        """Instantiate the selected backbone and head pair.

        Args:
            num_classes: Number of output classes for classification head.
            train_head_only: If True, freeze backbone weights and train only the head.
            use_pretrained: If True, load ImageNet pretrained weights (ignored for AlexNet).

        Returns:
            BackboneSpec: Backbone/head modules.
        """
        match self:
            case BackboneType.ALEXNET:
                alexnet = AlexNetParams(num_classes=num_classes).setup_target()
                backbone = alexnet.features
                if train_head_only:
                    _set_requires_grad(backbone, requires_grad=False)
                return BackboneSpec(alexnet, backbone, head=alexnet.classifier)

            case BackboneType.RESNET50:
                model = models.resnet50(
                    weights=models.ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
                )
                in_features = model.fc.in_features
                model.fc = nn.Identity()
                backbone = model  # keeps class name ResNet
                head = nn.Linear(in_features, num_classes)
                if train_head_only:
                    _set_requires_grad(backbone, requires_grad=False)
                return BackboneSpec(model=model, backbone=backbone, head=head)

            case BackboneType.VIT_B16:
                vit = models.vit_b_16(
                    weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if use_pretrained else None
                )
                vit.heads = nn.Identity()
                backbone = vit  # class name VisionTransformer
                head = nn.Linear(vit.hidden_dim, num_classes)
                if train_head_only:
                    _set_requires_grad(backbone, requires_grad=False)
                return BackboneSpec(model=vit, backbone=backbone, head=head)


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

    interpretability: InterpretabilityConfig | None = None
    """Optional Captum interpretability configuration."""


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

        Console.with_prefix(self.__class__.__name__, "__init__").log(
            f"Using backbone: {params.backbone} with head-only training: {params.train_head_only} and pretrained: {params.use_pretrained}"
        )

        spec = params.backbone.build(
            num_classes=params.num_classes,
            train_head_only=params.train_head_only,
            use_pretrained=params.use_pretrained,
        )

        # self.model = spec.model
        self.backbone = spec.backbone
        self.head = spec.head

        # Loss function
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
        self.attribution_engine: AttributionEngine | None = None

        if params.interpretability is not None:
            self.attribution_engine = params.interpretability.setup_target(self)

    # ---------------------------------------------------------------------- steps
    def forward(self, batch: ImageBatch_BCHW) -> Logits_BN:
        """Run a forward pass through the backbone.

        Args:
            batch: Batch of input images with shape (B, C, H, W) where:
                   B = batch size, C = channels, H = height, W = width.

        Returns:
            Raw logits for each class with shape (B, N) where N = num_classes.
        """
        features = self.backbone(batch)
        logits = self.head(features)
        return logits

    @torch.no_grad()
    def predict_proba(self, batch: ImageBatch_BCHW) -> Float[Tensor, "B N"]:
        """Inference helper that returns softmax probabilities.

        Keeps the module in eval mode and avoids gradient tracking so it can be
        safely used from serving code (e.g. the MCP runtime).
        """

        was_training = self.training
        self.eval()
        logits = self.forward(batch)
        probs = F.softmax(logits, dim=-1)
        if was_training:
            self.train()
        return probs

    @torch.no_grad()
    def classify_batch(
        self,
        batch: ImageBatch_BCHW,
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

    def attribute_batch(
        self,
        batch: ImageBatch_BCHW,
        *,
        target: Targets_B | int | None = None,
        additional_forward_args: Sequence[Tensor] | Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute interpretability heatmaps for a batch using the configured engine.

        Args:
            batch: Normalised tensor batch with shape (B, C, H, W).
            target: Optional class indices to attribute. Defaults to argmax per sample.
            additional_forward_args: Extra tensors forwarded to the model.

        Returns:
            Dictionary containing ``heatmap`` (Tensor['B H W']) and ``raw`` attribution.

        Raises:
            RuntimeError: If no interpretability engine is configured.
        """
        if self.attribution_engine is None:
            raise RuntimeError("Interpretability is not configured for this module.")

        result = self.attribution_engine.attribute(
            batch,
            target=target,
            additional_forward_args=additional_forward_args,
        )
        return {
            "heatmap": result.heatmap,
            "raw": result.raw_attribution,
        }

    def training_step(self, batch: tuple[ImageBatch_BCHW, Targets_B], batch_idx: int) -> ScalarLoss:
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

    def validation_step(self, batch: tuple[ImageBatch_BCHW, Targets_B], batch_idx: int) -> None:
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

        confmat = self.confusion_matrix.compute()
        class_names = self._get_class_names()
        self._log_confusion_matrix_plot(confmat, class_names)
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

        # Force headless backend to avoid X11 dependency during tests/CI

        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg", force=True)

        confmat_cpu = confmat.detach().float().cpu()

        # Row-normalise to show proportions per true class and display fractions
        row_sums = confmat_cpu.sum(dim=1, keepdim=True).clamp_min(1e-9)
        confmat_frac = confmat_cpu / row_sums

        fig, ax = plt.subplots(figsize=(16, 12))
        sns.heatmap(
            confmat_frac.numpy(),
            annot=True,
            fmt=".2f",
            cmap="viridis",
            linewidths=0.5,
            cbar_kws={"label": "Recall"},
            square=True,
            ax=ax,
            vmin=0.0,
            vmax=1.0,
        )

        if class_names is not None:
            ax.set_xticklabels(
                list(map(lambda x: x.title(), class_names)), rotation=35, ha="right", fontsize=8
            )
            ax.set_yticklabels(list(map(lambda x: x.title(), class_names)), rotation=0, fontsize=8)

        ax.set_xlabel("Predicted Class", fontsize=11)
        ax.set_ylabel("True Class", fontsize=11)
        ax.set_title(
            f"Confusion Matrix - Epoch {self.current_epoch}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # Log to WandB
        self.logger.experiment.log(
            {
                Metric.VAL_CONFUSION_MATRIX: wandb.Image(fig),
                "epoch": self.current_epoch,
            }
        )

        plt.close(fig)

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

    def test_step(self, batch: tuple[ImageBatch_BCHW, Targets_B], batch_idx: int) -> None:
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
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.head.parameters())
        backbone_lr = self.params.optimizer.learning_rate * self.params.optimizer.backbone_lr_scale
        head_lr = self.params.optimizer.learning_rate

        param_groups = [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]

        optimizer = self.params.optimizer.setup_target(param_groups)

        if (total_steps := self.trainer.estimated_stepping_batches) <= 0:
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            total_steps = steps_per_epoch * getattr(self.trainer, "max_epochs", 1)

        max_lr = self.params.scheduler.max_lr
        max_lrs = [max_lr * self.params.optimizer.backbone_lr_scale, max_lr]

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.params.scheduler.setup_target(
                    optimizer,
                    total_steps=total_steps,
                    max_lr=max_lrs,
                ),
                "interval": "step",
                "monitor": Metric.VAL_LOSS,
            },
        }
