"""Lightning data module for the document-class detector.

The module wires our RVL-CDIP dataset configs into PyTorch Lightning's training
loop while adhering to the Config-as-Factory conventions.
"""

from __future__ import annotations

import os

import pytorch_lightning as pl
from pydantic import Field
from torch.utils.data import DataLoader

from ..configs import BaseConfig, Stage
from ..data_handling.rvl_cdip_dataset import RVLCDIPDataset, RVLCDIPDatasetConfig


def _default_num_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count))


class DocDataModuleConfig(BaseConfig["DocDataModule"]):
    """Configuration for :class:`DocDataModule`."""

    target: type[DocDataModule] = Field(default_factory=lambda: DocDataModule)

    batch_size: int = 32
    num_workers: int = Field(default_factory=_default_num_workers)
    pin_memory: bool = True

    train_dataset: RVLCDIPDatasetConfig
    val_dataset: RVLCDIPDatasetConfig
    test_dataset: RVLCDIPDatasetConfig


class DocDataModule(pl.LightningDataModule):
    """LightningDataModule that exposes RVL-CDIP datasets to the trainer."""

    def __init__(self, config: DocDataModuleConfig):
        super().__init__()
        self.config = config
        self._train_ds: RVLCDIPDataset | None = None
        self._val_ds: RVLCDIPDataset | None = None
        self._test_ds: RVLCDIPDataset | None = None

    # --------------------------------------------------------------------- setup
    def setup(self, stage: Stage | str | None = None) -> None:
        requested_stage = (
            stage if isinstance(stage, Stage) else Stage.from_str(stage) if stage else None
        )
        if requested_stage is None:
            self._ensure_all_datasets()
            return

        if requested_stage is Stage.TRAIN:
            self._train_ds = self.config.train_dataset.setup_target()
        elif requested_stage is Stage.VAL:
            self._val_ds = self.config.val_dataset.setup_target()
        elif requested_stage is Stage.TEST:
            self._test_ds = self.config.test_dataset.setup_target()
        else:  # pragma: no cover - defensive branch
            raise ValueError(f"Unsupported stage '{stage}'.")

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
    def train_ds(self) -> RVLCDIPDataset | None:
        return self._train_ds

    @property
    def val_ds(self) -> RVLCDIPDataset | None:
        return self._val_ds

    @property
    def test_ds(self) -> RVLCDIPDataset | None:
        return self._test_ds
