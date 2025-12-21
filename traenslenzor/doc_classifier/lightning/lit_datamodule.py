"""LightningDataModule wrapper around RVL-CDIP with configurable transforms."""

import os
import warnings
from typing import Any, Callable

import numpy as np
import pytorch_lightning as pl
import torch
from datasets import Dataset as HFDataset
from pydantic import Field, model_validator
from torch.utils.data import DataLoader
from typing_extensions import Self

from ..data_handling.huggingface_rvl_cdip_ds import RVLCDIPConfig
from ..data_handling.transforms import TrainTransformConfig, ValTransformConfig
from ..utils import BaseConfig, Console, Stage

# Suppress PyTorch's internal pin_memory deprecation warning (PyTorch 2.9+)
# This warning comes from DataLoader's internal implementation
warnings.filterwarnings(
    "ignore",
    message=r"The argument 'device' of Tensor\.pin_memory\(\) is deprecated",
    category=DeprecationWarning,
    module=r"torch\.utils\.data\._utils\.pin_memory",
)


def collate_hf_batch(batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate HuggingFace dataset batches into (images, labels) tuple.

    Converts list of dicts from HF Dataset into tuple format expected by Lightning.

    Args:
        batch: List of dicts with 'image' (Tensor) and 'label' (int) keys.

    Returns:
        Tuple of (images, labels) where:
            - images: Stacked image tensors with shape (B, C, H, W)
            - labels: Label tensor with shape (B,)
    """
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return images, labels


def _collate_grayscale_stats(batch: list[dict[str, Any]]) -> torch.Tensor:
    """Collate grayscale images into a float tensor for statistics.

    Args:
        batch: List of dicts with an "image" key containing a grayscale image.

    Returns:
        Float tensor with shape (B, 1, H, W) in [0, 1].
    """
    images: list[torch.Tensor] = []
    for item in batch:
        image = item["image"]
        if isinstance(image, torch.Tensor):
            tensor = image
        else:
            tensor = torch.as_tensor(np.asarray(image))

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            if tensor.shape[0] == 1:
                pass
            elif tensor.shape[-1] == 1:
                tensor = tensor.permute(2, 0, 1)
            else:
                raise ValueError(
                    "Expected grayscale images with a single channel, "
                    f"got shape={tuple(tensor.shape)}."
                )
        else:
            raise ValueError(f"Unsupported image shape {tuple(tensor.shape)}.")

        images.append(tensor)

    return torch.stack(images)


def _default_train_ds() -> RVLCDIPConfig:
    return RVLCDIPConfig(
        split=Stage.TRAIN,
        transform_config=TrainTransformConfig(),
    )


def _default_ds(split: Stage) -> RVLCDIPConfig:
    return RVLCDIPConfig(
        split=split,
        transform_config=ValTransformConfig(),
    )


class DocDataModuleConfig(BaseConfig["DocDataModule"]):
    """Configuration for :class:`DocDataModule`."""

    target: type["DocDataModule"] = Field(default_factory=lambda: DocDataModule, exclude=True)

    batch_size: int = 32
    """Batch size for DataLoaders."""

    num_workers: int = Field(default_factory=lambda: os.cpu_count() or 0)
    """Number of worker processes for data loading (defaults to logical CPU count)."""

    persistent_workers: bool = True
    """Whether to keep DataLoader worker processes alive between epochs."""

    limit_num_samples: float | None = Field(default=None, gt=0, lt=1.0)
    """Limit number of samples per dataset for debugging.
    If int, uses that many samples. If float between 0 and 1, uses that fraction of the dataset."""

    is_debug: bool = False
    verbose: bool = True

    train_ds: RVLCDIPConfig = Field(default_factory=_default_train_ds)
    """Configuration for training dataset with transforms."""

    val_ds: RVLCDIPConfig = Field(
        default_factory=lambda: _default_ds(Stage.VAL),
    )
    """Configuration for validation dataset with transforms."""

    test_ds: RVLCDIPConfig = Field(
        default_factory=lambda: _default_ds(Stage.TEST),
    )
    """Configuration for test dataset with transforms."""

    @model_validator(mode="after")
    def _debug_defaults(self) -> Self:
        """Apply debug-mode defaults when is_debug=True.

        Uses object.__setattr__() to avoid retriggering validation (which would
        cause infinite recursion).
        """
        console = Console.with_prefix(self.__class__.__name__, "_debug_defaults")

        if self.is_debug:
            object.__setattr__(self, "num_workers", 0)
            console.log("Debug settings: num_workers=0 applied for DataModule")

        return self


class DocDataModule(pl.LightningDataModule):
    """LightningDataModule that exposes RVL-CDIP datasets to the trainer.

    This module integrates HuggingFace datasets with PyTorch Lightning,
    supporting the new transform system via RVLCDIPConfig.
    """

    def __init__(self, config: DocDataModuleConfig):
        super().__init__()

        self.config = config
        self.batch_size = config.batch_size
        self.save_hyperparameters(config.model_dump())

        self._train_ds: HFDataset | None = None
        self._val_ds: HFDataset | None = None
        self._test_ds: HFDataset | None = None

        self._stage_attr_map = {
            Stage.TRAIN: ("_train_ds", self.config.train_ds),
            Stage.VAL: ("_val_ds", self.config.val_ds),
            Stage.TEST: ("_test_ds", self.config.test_ds),
        }

    # --------------------------------------------------------------------- setup
    def setup(self, stage: Stage | str | None = None) -> None:
        requested_stage = (
            stage if isinstance(stage, Stage) else Stage.from_str(stage) if stage else None
        )
        if requested_stage is None:
            self._ensure_all_datasets()
            return

        if requested_stage in self._stage_attr_map:
            self._ensure_dataset(requested_stage)
        else:
            raise ValueError(f"Unsupported stage '{stage}'.")

    def _ensure_all_datasets(self) -> None:
        for stage in self._stage_attr_map:
            self._ensure_dataset(stage)

    # ------------------------------------------------------------------ loaders
    def train_dataloader(self) -> DataLoader:
        self._ensure_dataset(Stage.TRAIN)
        return self._build_dataloader(
            dataset=self._train_ds,
            shuffle=True,
            collate_fn=collate_hf_batch,
        )

    def val_dataloader(self) -> DataLoader:
        self._ensure_dataset(Stage.VAL)
        return self._build_dataloader(
            dataset=self._val_ds,
            shuffle=False,
            collate_fn=collate_hf_batch,
        )

    def test_dataloader(self) -> DataLoader:
        self._ensure_dataset(Stage.TEST)
        return self._build_dataloader(
            dataset=self._test_ds,
            shuffle=False,
            collate_fn=collate_hf_batch,
        )

    # ---------------------------------------------------------------- properties
    @property
    def train_ds(self) -> HFDataset:
        """Training dataset."""
        self._ensure_dataset(Stage.TRAIN)
        return self._train_ds

    @property
    def val_ds(self) -> HFDataset:
        """Validation dataset."""
        self._ensure_dataset(Stage.VAL)
        return self._val_ds

    @property
    def test_ds(self) -> HFDataset:
        """Test dataset."""
        self._ensure_dataset(Stage.TEST)
        return self._test_ds

    def compute_grayscale_mean_std(self) -> tuple[float, float]:
        """Compute grayscale mean/std on the training split.

        Returns:
            Tuple of (mean, std) in [0, 1] for a single grayscale channel.


        ```Grayscale mean/std: mean=0.911966, std=0.241507 on 16056269824 pixels, 10000 batches, batch_size=32 ```
        """

        console = Console.with_prefix(self.__class__.__name__, "compute_grayscale_mean_std")
        console.set_verbose(self.config.verbose).set_debug(self.config.is_debug)
        console.log("Computing grayscale stats for train-split.")

        total_sum = 0.0
        total_sumsq = 0.0
        total_pixels = 0

        data_loader = self._build_dataloader(
            dataset=self.config.train_ds.model_copy(
                deep=True,
                update={
                    "transform_config": ValTransformConfig(
                        apply_normalization=False, convert_to_rgb=False
                    )
                },
            ).setup_target(),
            shuffle=False,
            collate_fn=_collate_grayscale_stats,
        )
        num_batches = len(data_loader)

        for batch in data_loader:
            assert isinstance(batch, torch.Tensor)
            total_sum += float(batch.sum().item())
            total_sumsq += float(torch.square(batch).sum().item())
            total_pixels += batch.numel()

        if total_pixels == 0:
            raise ValueError("No pixels found while computing dataset statistics.")

        mean = total_sum / total_pixels
        variance = total_sumsq / total_pixels - mean**2
        std = float(torch.sqrt(torch.tensor(variance)).item())

        console.log(
            f"Grayscale mean/std: mean={mean:.6f}, std={std:.6f} on {total_pixels} pixels, {num_batches} batches, batch_size={data_loader.batch_size}."
        )
        return mean, std

    def _build_dataloader(
        self,
        *,
        dataset: HFDataset,
        shuffle: bool,
        collate_fn: Callable[[list[dict[str, Any]]], Any],
    ) -> DataLoader:
        loader_kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "num_workers": self.config.num_workers,
            "persistent_workers": self.config.persistent_workers,
            "shuffle": shuffle,
            "collate_fn": collate_fn,
        }
        if self.config.num_workers > 0:
            loader_kwargs["multiprocessing_context"] = "spawn"
        return DataLoader(dataset, **loader_kwargs)

    def _ensure_dataset(self, stage: Stage) -> None:
        attr_name, cfg = self._stage_attr_map[stage]
        dataset = getattr(self, attr_name)
        if dataset is None:
            dataset = cfg.setup_target()

            if self.config.limit_num_samples is not None:
                dataset = self._apply_sample_limit(dataset, stage)

            setattr(self, attr_name, dataset)

    def _apply_sample_limit(self, dataset: HFDataset, stage: Stage) -> HFDataset:
        """Apply limit_num_samples to reduce dataset size for debugging.

        Args:
            dataset: Full HuggingFace dataset.
            stage: Dataset stage (train/val/test) for logging.

        Returns:
            HFDataset: Limited dataset with specified number/fraction of samples.
        """
        limit = self.config.limit_num_samples
        original_size = len(dataset)
        target_size = int(original_size * limit)

        # Select subset
        limited_dataset = dataset.select(range(target_size))

        # Log the limitation
        console = Console.with_prefix(self.__class__.__name__, "_apply_sample_limit")
        console.set_verbose(self.config.verbose).set_debug(self.config.is_debug)
        console.log(
            f"Limited {stage} dataset: {original_size} → {target_size} samples "
            f"({target_size / original_size * 100:.1f}%)"
        )

        return limited_dataset
