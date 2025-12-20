"""Custom backbone finetuning callback compatible with OneCycleLR.

This callback freezes the backbone before training and unfreezes it at a
configured epoch without altering optimizer param groups. Keeping the
param-group layout fixed avoids conflicts with schedulers like OneCycleLR.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm

from ..utils import Console

if TYPE_CHECKING:
    from pytorch_lightning import LightningModule, Trainer
    from torch.nn import Parameter
    from torch.optim import Optimizer


class OneCycleBackboneFinetuning(BaseFinetuning):
    """Freeze/unfreeze backbone without changing optimizer param groups.

    Designed for schedulers like OneCycleLR that require a fixed param-group
    layout for the entire training run.
    """

    def __init__(
        self,
        unfreeze_backbone_at_epoch: int = 10,
        *,
        train_bn: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the finetuning callback.

        Args:
            unfreeze_backbone_at_epoch (int): Epoch at which the backbone becomes trainable.
            train_bn (bool): Whether batch-norm parameters remain trainable during finetuning.
            verbose (bool): Whether to emit verbose logs.
        """
        super().__init__()
        self.unfreeze_backbone_at_epoch = unfreeze_backbone_at_epoch
        self.train_bn = train_bn
        self.verbose = verbose
        self._console = Console.with_prefix(self.__class__.__name__).set_verbose(verbose)

    def on_fit_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        """Validate configuration and optimizer compatibility.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance.
            pl_module (LightningModule): Model being trained.

        Raises:
            MisconfigurationException: If the module does not define a backbone.
        """
        if not hasattr(pl_module, "backbone") or not isinstance(pl_module.backbone, Module):
            raise MisconfigurationException(
                "The LightningModule should have a nn.Module `backbone` attribute."
            )

        super().on_fit_start(trainer, pl_module)

        for optimizer in trainer.optimizers:
            if not self._optimizer_has_backbone(optimizer, pl_module.backbone):
                self._console.warn(
                    "Backbone parameters are missing from the optimizer param groups. "
                    "Unfreezing will not update backbone weights; ensure configure_optimizers "
                    "includes backbone params even if frozen."
                )

    def freeze_before_training(self, pl_module: "LightningModule") -> None:
        """Freeze the backbone before the optimizer is configured.

        Args:
            pl_module (LightningModule): Model being trained.
        """
        self.freeze(pl_module.backbone, train_bn=self.train_bn)
        if self.verbose:
            self._console.log("Froze backbone parameters for head-only warmup.")

    def finetune_function(
        self,
        pl_module: "LightningModule",
        epoch: int,
        optimizer: "Optimizer",
    ) -> None:
        """Unfreeze the backbone at the configured epoch.

        Args:
            pl_module (LightningModule): Model being trained.
            epoch (int): Current training epoch.
            optimizer (Optimizer): Optimizer used for training (unused).
        """
        del optimizer

        if epoch < self.unfreeze_backbone_at_epoch:
            return

        backbone_params = list(
            self._iter_backbone_params(pl_module.backbone, include_bn=self.train_bn)
        )
        if backbone_params and all(param.requires_grad for param in backbone_params):
            return

        self._set_backbone_trainable(pl_module.backbone)
        if self.verbose:
            self._console.log(f"Unfroze backbone at epoch {epoch} (train_bn={self.train_bn}).")

    def _set_backbone_trainable(self, backbone: Module) -> None:
        """Set backbone parameters to trainable, honoring batch-norm settings.

        Args:
            backbone (Module): Backbone module to unfreeze.
        """
        for module in self.flatten_modules(backbone):
            if isinstance(module, _BatchNorm) and not self.train_bn:
                self.freeze_module(module)
            else:
                self.make_trainable(module)

    def _iter_backbone_params(
        self,
        backbone: Module,
        *,
        include_bn: bool,
    ) -> Iterable["Parameter"]:
        """Yield backbone parameters, optionally excluding batch-norm parameters.

        Args:
            backbone (Module): Backbone module to scan.
            include_bn (bool): Whether to include batch-norm parameters.

        Returns:
            Iterable[Parameter]: Parameters for the backbone modules.
        """
        for module in self.flatten_modules(backbone):
            if isinstance(module, _BatchNorm) and not include_bn:
                continue
            for param in module.parameters(recurse=False):
                yield param

    @staticmethod
    def _optimizer_has_backbone(optimizer: "Optimizer", backbone: Module) -> bool:
        """Check whether the optimizer contains all backbone parameters.

        Args:
            optimizer (Optimizer): Optimizer to inspect.
            backbone (Module): Backbone module whose params should be present.

        Returns:
            bool: True if all backbone parameters are present in optimizer groups.
        """
        backbone_ids = {id(param) for param in backbone.parameters()}
        optimizer_ids = {
            id(param) for group in optimizer.param_groups for param in group.get("params", [])
        }
        return backbone_ids.issubset(optimizer_ids)
