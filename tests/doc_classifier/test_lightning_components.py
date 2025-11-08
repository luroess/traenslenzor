from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset

import traenslenzor.doc_classifier.data_handling.huggingface_rvl_cdip_ds as rvl_module
import traenslenzor.doc_classifier.lightning.lit_module as lit_module
from traenslenzor.doc_classifier.data_handling.huggingface_rvl_cdip_ds import RVLCDIPConfig
from traenslenzor.doc_classifier.data_handling.transforms import (
    FineTuneTransformConfig,
    TrainTransformConfig,
    ValTransformConfig,
)
from traenslenzor.doc_classifier.lightning import (
    BackboneType,
    DocClassifierConfig,
    DocDataModuleConfig,
    OneCycleSchedulerConfig,
    OptimizerConfig,
)
from traenslenzor.doc_classifier.models.alexnet import AlexNet, AlexNetParams
from traenslenzor.doc_classifier.utils import Stage


def test_optimizer_config_builds_adamw():
    param = nn.Parameter(torch.ones(1, requires_grad=True))
    optimizer = OptimizerConfig(learning_rate=1e-3, weight_decay=1e-2).setup_target([param])

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-2)


def test_one_cycle_scheduler_config_builds_scheduler():
    param = nn.Parameter(torch.ones(1, requires_grad=True))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    scheduler = OneCycleSchedulerConfig(max_lr=0.01).setup_target(
        optimizer,
        total_steps=10,
    )

    assert isinstance(scheduler, OneCycleLR)
    assert scheduler.total_steps == 10


def test_alexnet_params_build_model():
    params = AlexNetParams(num_classes=5, dropout=0.1)
    model = params.setup_target()

    assert isinstance(model, AlexNet)
    assert model.classifier[-1].out_features == 5


def test_doc_classifier_module_forward_pass_returns_logits():
    config = DocClassifierConfig(
        num_classes=4,
        backbone=BackboneType.ALEXNET,
        use_pretrained=False,
    )
    module = config.setup_target()

    batch = torch.randn(2, 3, 224, 224)
    logits = module(batch)

    assert logits.shape == (2, 4)


def test_train_head_only_freezes_backbone(monkeypatch):
    class DummyResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(8, 4)
            self.fc = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(self.layer(torch.flatten(x, 1)))

    monkeypatch.setattr(lit_module.models, "resnet50", lambda **_: DummyResNet())

    config = DocClassifierConfig(
        num_classes=3,
        backbone=BackboneType.RESNET50,
        train_head_only=True,
        use_pretrained=False,
    )
    module = config.setup_target()

    non_head_params = [
        param.requires_grad
        for name, param in module.model.named_parameters()
        if not name.startswith("fc.")
    ]
    assert non_head_params
    assert all(flag is False for flag in non_head_params)

    assert module.model.fc.out_features == 3
    assert all(param.requires_grad for param in module.model.fc.parameters())


class _DummyDataset(Dataset):
    def __init__(self, label_offset: int):
        self.label_offset = label_offset

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int):
        return {
            "image": torch.zeros(1, 8, 8),
            "label": self.label_offset + idx,
        }


def test_doc_data_module_loaders_use_patched_datasets(monkeypatch):
    datasets = {
        Stage.TRAIN: _DummyDataset(0),
        Stage.VAL: _DummyDataset(100),
        Stage.TEST: _DummyDataset(200),
    }

    def fake_setup_target(self):
        return datasets[self.split]

    monkeypatch.setattr(RVLCDIPConfig, "setup_target", fake_setup_target)

    config = DocDataModuleConfig(batch_size=2, num_workers=0)
    datamodule = config.setup_target()

    datamodule.setup(Stage.TRAIN)
    train_batch = next(iter(datamodule.train_dataloader()))
    assert train_batch["image"].shape[0] == 2

    val_batch = next(iter(datamodule.val_dataloader()))
    assert val_batch["label"][0].item() >= 100

    test_batch = next(iter(datamodule.test_dataloader()))
    assert test_batch["label"][0].item() >= 200


def test_doc_data_module_setup_without_stage(monkeypatch):
    datasets = {
        Stage.TRAIN: _DummyDataset(0),
        Stage.VAL: _DummyDataset(10),
        Stage.TEST: _DummyDataset(20),
    }

    def fake_setup_target(self):
        return datasets[self.split]

    monkeypatch.setattr(RVLCDIPConfig, "setup_target", fake_setup_target)

    datamodule = DocDataModuleConfig(batch_size=1, num_workers=0).setup_target()
    datamodule.setup(None)

    assert datamodule.train_ds is datasets[Stage.TRAIN]
    assert datamodule.val_ds is datasets[Stage.VAL]
    assert datamodule.test_ds is datasets[Stage.TEST]


def test_doc_data_module_invalid_stage(monkeypatch):
    monkeypatch.setattr(RVLCDIPConfig, "setup_target", lambda self: _DummyDataset(0))
    datamodule = DocDataModuleConfig(batch_size=1, num_workers=0).setup_target()

    with pytest.raises(ValueError):
        datamodule.setup("invalid")


@pytest.mark.parametrize(
    "config_cls",
    [TrainTransformConfig, FineTuneTransformConfig, ValTransformConfig],
)
def test_transform_configs_generate_tensors(config_cls):
    config = config_cls(img_size=32)
    pipeline = config.setup_target()

    sample = np.random.randint(0, 255, size=(64, 64, 1), dtype=np.uint8)
    result = pipeline(image=sample)

    assert "image" in result
    assert torch.is_tensor(result["image"])
    assert tuple(result["image"].shape[-2:]) == (32, 32)


def test_rvlcdip_config_setup_target_applies_transform(monkeypatch, fresh_path_config):
    captured_kwargs: dict[str, str] = {}

    class DummyHF:
        def __init__(self):
            self.applied_transform = None

        def set_transform(self, fn):
            self.applied_transform = fn

    dataset = DummyHF()

    def fake_load_dataset(path, split, cache_dir, streaming, num_proc):
        captured_kwargs.update(
            {
                "path": path,
                "split": split,
                "cache_dir": cache_dir,
                "streaming": streaming,
                "num_proc": num_proc,
            }
        )
        return dataset

    monkeypatch.setattr(rvl_module, "load_dataset", fake_load_dataset)

    cfg = rvl_module.RVLCDIPConfig(
        num_workers=2,
        streaming=True,
        transform_config=TrainTransformConfig(img_size=16),
    )
    returned = cfg.setup_target()

    assert returned is dataset
    assert dataset.applied_transform is not None
    assert captured_kwargs["path"] == "chainyo/rvl-cdip"
    assert captured_kwargs["split"] == "train"
    assert captured_kwargs["cache_dir"] == fresh_path_config.hf_cache.as_posix()
    assert captured_kwargs["streaming"] is True
    assert captured_kwargs["num_proc"] == 2


def test_vit_backbone_replacement(monkeypatch):
    class DummyViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_dim = 8
            self.heads = nn.Linear(8, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.heads(torch.flatten(x, 1))

    monkeypatch.setattr(lit_module.models, "vit_b_16", lambda **_: DummyViT())

    config = DocClassifierConfig(
        num_classes=5,
        backbone=BackboneType.VIT_B16,
        train_head_only=True,
        use_pretrained=False,
    )
    module = config.setup_target()

    assert module.model.heads.out_features == 5
    frozen = [param.requires_grad for param in module.model.parameters() if param is not module.model.heads.weight]
    assert frozen
    assert not any(frozen)


def test_training_validation_test_steps_execute(monkeypatch):
    config = DocClassifierConfig(num_classes=2, backbone=BackboneType.ALEXNET, use_pretrained=False)
    module = config.setup_target()

    batch = (torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))

    loss = module.training_step(batch, 0)
    assert loss.item() >= 0

    module.validation_step(batch, 0)
    module.test_step(batch, 0)


def test_configure_optimizers_and_on_train_start(monkeypatch):
    config = DocClassifierConfig(num_classes=2, backbone=BackboneType.ALEXNET, use_pretrained=False)
    module = config.setup_target()

    dataloader = [0, 1]
    module.trainer = SimpleNamespace(
        estimated_stepping_batches=None,
        datamodule=SimpleNamespace(train_dataloader=lambda: dataloader),
        max_epochs=2,
    )

    optimizers = module.configure_optimizers()
    assert "optimizer" in optimizers
    assert optimizers["lr_scheduler"]["interval"] == "step"

    logger = MagicMock()
    module.logger = logger
    module.on_train_start()
    logger.log_hyperparams.assert_called_once()
