from __future__ import annotations

import torch
from torch import nn

from traenslenzor.doc_classifier.interpretability import (
    AttributionMethod,
    InterpretabilityConfig,
)
from traenslenzor.doc_classifier.lightning import (
    BackboneType,
    DocClassifierConfig,
    DocClassifierModule,
)
from traenslenzor.doc_classifier.lightning.lit_module import BackboneSpec


class _TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        pooled = self.pool(feats).flatten(1)
        return self.head(pooled)


def test_grad_cam_returns_heatmap_aligned_to_input() -> None:
    model = _TinyCNN()
    cfg = InterpretabilityConfig(method=AttributionMethod.GRAD_CAM)
    engine = cfg.setup_target(model)

    x = torch.randn(2, 3, 16, 16)
    target = torch.tensor([0, 1])

    result = engine.attribute(x, target=target)

    assert result.heatmap.shape == (2, 16, 16)
    assert torch.isfinite(result.heatmap).all()
    assert 0.0 <= float(result.heatmap.min().detach()) <= 1.0
    assert 0.0 <= float(result.heatmap.max().detach()) <= 1.0


def test_integrated_gradients_matches_input_shape() -> None:
    model = _TinyCNN()
    cfg = InterpretabilityConfig(method=AttributionMethod.INTEGRATED_GRADIENTS, n_steps=4)
    engine = cfg.setup_target(model)

    x = torch.randn(1, 3, 8, 8)
    target = torch.tensor([1])
    result = engine.attribute(x, target=target)

    assert result.raw_attribution.shape == x.shape
    assert result.heatmap.shape[-2:] == (8, 8)


def test_occlusion_uses_window_and_stride_shapes() -> None:
    model = _TinyCNN()
    cfg = InterpretabilityConfig(
        method=AttributionMethod.OCCLUSION,
        occlusion_window=(4, 4),
        occlusion_stride=(4, 4),
    )
    engine = cfg.setup_target(model)

    x = torch.randn(1, 3, 12, 12)
    target = torch.tensor([0])
    result = engine.attribute(x, target=target)

    assert result.heatmap.shape == (1, 12, 12)
    # Occlusion output is non-negative after min-max normalisation
    assert 0.0 <= float(result.heatmap.min().detach()) <= 1.0
    assert 0.0 <= float(result.heatmap.max().detach()) <= 1.0


def test_doc_classifier_exposes_attribute_batch(monkeypatch) -> None:
    def tiny_backbone(self, num_classes: int, train_head_only: bool, use_pretrained: bool):
        model = _TinyCNN()
        model.head = nn.Linear(4, num_classes)
        backbone = nn.Sequential(model.features, model.pool)
        head = nn.Sequential(nn.Flatten(1), model.head)
        combined = nn.Sequential(backbone, head)
        return BackboneSpec(model=combined, backbone=backbone, head=head)

    monkeypatch.setattr(BackboneType, "build", tiny_backbone, raising=True)

    config = DocClassifierConfig(
        num_classes=2,
        backbone=BackboneType.ALEXNET,
        interpretability=InterpretabilityConfig(method=AttributionMethod.GRAD_CAM),
        train_head_only=False,
        use_pretrained=False,
    )

    module = DocClassifierModule(config)

    batch = torch.randn(2, 3, 16, 16)
    result = module.attribute_batch(batch)

    assert set(result.keys()) == {"heatmap", "raw"}
    assert result["heatmap"].shape[-2:] == (16, 16)
    assert result["heatmap"].shape[0] == 2


class _TinyViT(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv_proj = nn.Conv2d(3, 16, kernel_size=16, stride=16)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.heads = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.conv_proj(x)
        pooled = self.pool(feats).flatten(1)
        return self.heads(pooled)


def test_attribute_batch_supports_all_backbones(monkeypatch) -> None:
    def fake_build(self, num_classes: int, train_head_only: bool, use_pretrained: bool):
        if self is BackboneType.VIT_B16:
            vit = _TinyViT(num_classes)
            backbone = nn.Sequential(vit.conv_proj, vit.pool)
            head = nn.Sequential(nn.Flatten(1), vit.heads)
            combined = nn.Sequential(backbone, head)
            return BackboneSpec(model=combined, backbone=backbone, head=head)

        cnn = _TinyCNN()
        cnn.head = nn.Linear(4, num_classes)
        backbone = nn.Sequential(cnn.features, cnn.pool)
        head = nn.Sequential(nn.Flatten(1), cnn.head)
        combined = nn.Sequential(backbone, head)
        return BackboneSpec(model=combined, backbone=backbone, head=head)

    monkeypatch.setattr(BackboneType, "build", fake_build, raising=True)

    for backbone in BackboneType:
        config = DocClassifierConfig(
            num_classes=3,
            backbone=backbone,
            interpretability=InterpretabilityConfig(method=AttributionMethod.GRAD_CAM),
            train_head_only=False,
            use_pretrained=False,
        )
        module = DocClassifierModule(config)
        batch = torch.randn(2, 3, 32, 32)
        result = module.attribute_batch(batch)
        assert result["heatmap"].shape == (2, 32, 32)
        assert torch.isfinite(result["heatmap"]).all()
