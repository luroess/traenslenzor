"""LightningDataModule wrapper around RVL-CDIP with configurable transforms."""

import warnings
from typing import Any

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
# This warning comes from DataLoader's internal implementation, not our code
# Will be fixed in future PyTorch releases
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


def _default_num_workers() -> int:
    return torch.get_num_threads()


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


def _pin_memory(pin_memory: bool = True) -> bool:
    # Only pin memory if CUDA is available
    return pin_memory and torch.cuda.is_available()


class DocDataModuleConfig(BaseConfig["DocDataModule"]):
    """Configuration for :class:`DocDataModule`."""

    target: type["DocDataModule"] = Field(default_factory=lambda: DocDataModule, exclude=True)

    batch_size: int = 32
    """Batch size for DataLoaders."""

    num_workers: int = Field(default_factory=_default_num_workers)
    """Number of worker processes for data loading."""

    pin_memory: bool = True
    """Whether to pin memory in DataLoaders for faster GPU transfer."""

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

        This validator runs after initialization and when is_debug is propagated
        from parent configs via setattr(), ensuring debug settings are applied.

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
        return DataLoader(
            self._train_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            pin_memory=_pin_memory(self.config.pin_memory),
            collate_fn=collate_hf_batch,
        )

    def val_dataloader(self) -> DataLoader:
        self._ensure_dataset(Stage.VAL)
        return DataLoader(
            self._val_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=_pin_memory(self.config.pin_memory),
            collate_fn=collate_hf_batch,
        )

    def test_dataloader(self) -> DataLoader:
        self._ensure_dataset(Stage.TEST)
        return DataLoader(
            self._test_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=_pin_memory(self.config.pin_memory),
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

    def _ensure_dataset(self, stage: Stage) -> None:
        attr_name, cfg = self._stage_attr_map[stage]
        dataset = getattr(self, attr_name)
        if dataset is None:
            dataset = cfg.setup_target()
            setattr(self, attr_name, dataset)
