"""Lightning data module for the document-class detector.

The module wires our RVL-CDIP dataset configs into PyTorch Lightning's training
loop while adhering to the Config-as-Factory conventions.
"""

from __future__ import annotations

import os

import pytorch_lightning as pl
from datasets import Dataset as HFDataset
from pydantic import Field
from torch.utils.data import DataLoader

from ..data_handling.huggingface_rvl_cdip_ds import RVLCDIPConfig
from ..data_handling.transforms import TrainTransformConfig, ValTransformConfig
from ..utils import BaseConfig, Stage


def _default_num_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def _default_train_dataset() -> RVLCDIPConfig:
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

    target: type[DocDataModule] = Field(default_factory=lambda: DocDataModule, exclude=True)

    batch_size: int = 32
    """Batch size for DataLoaders."""

    num_workers: int = Field(default_factory=_default_num_workers)
    """Number of worker processes for data loading."""

    pin_memory: bool = True
    """Whether to pin memory in DataLoaders for faster GPU transfer."""

    train_dataset: RVLCDIPConfig = Field(default_factory=_default_train_dataset)
    """Configuration for training dataset with transforms."""

    val_dataset: RVLCDIPConfig = Field(
        default_factory=lambda: _default_ds(Stage.VAL),
    )
    """Configuration for validation dataset with transforms."""

    test_dataset: RVLCDIPConfig = Field(
        default_factory=lambda: _default_ds(Stage.TEST),
    )
    """Configuration for test dataset with transforms."""


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

        # All datasets for easy access
        self._all_ds: list[HFDataset] | None = None

    # --------------------------------------------------------------------- setup
    def setup(self, stage: Stage | str | None = None) -> None:
        requested_stage = (
            stage if isinstance(stage, Stage) else Stage.from_str(stage) if stage else None
        )
        if requested_stage is None:
            self._ensure_all_datasets()
            return

        # Setup dataset based on requested stage using match-case pattern
        match requested_stage:
            case Stage.TRAIN:
                self._train_ds = self.config.train_dataset.setup_target()
            case Stage.VAL:
                self._val_ds = self.config.val_dataset.setup_target()
            case Stage.TEST:
                self._test_ds = self.config.test_dataset.setup_target()

    def _ensure_all_datasets(self) -> None:
        if self._train_ds is None:
            self._train_ds = self.config.train_dataset.setup_target()
        if self._val_ds is None:
            self._val_ds = self.config.val_dataset.setup_target()
        if self._test_ds is None:
            self._test_ds = self.config.test_dataset.setup_target()

    # ------------------------------------------------------------------ loaders
    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            self.setup(str(Stage.TRAIN))
        return DataLoader(
            self._train_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            pin_memory=self.config.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_ds is None:
            self.setup(str(Stage.VAL))
        return DataLoader(
            self._val_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        if self._test_ds is None:
            self.setup(str(Stage.TEST))
        return DataLoader(
            self._test_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    # ---------------------------------------------------------------- properties
    @property
    def train_ds(self) -> HFDataset:
        """Training dataset."""
        if self._train_ds is None:
            self.setup(str(Stage.TRAIN))
        return self._train_ds

    @property
    def val_ds(self) -> HFDataset:
        """Validation dataset."""
        if self._val_ds is None:
            self.setup(str(Stage.VAL))
        return self._val_ds

    @property
    def test_ds(self) -> HFDataset:
        """Test dataset."""
        if self._test_ds is None:
            self.setup(str(Stage.TEST))
        return self._test_ds
