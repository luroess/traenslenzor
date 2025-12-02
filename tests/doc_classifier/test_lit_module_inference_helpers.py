from __future__ import annotations

import pytest
import torch

from traenslenzor.doc_classifier.lightning import (
    BackboneType,
    DocClassifierConfig,
    DocClassifierModule,
)
from traenslenzor.doc_classifier.lightning.lit_module import BackboneSpec


def test_classify_batch_preserves_training_state(monkeypatch):
    # Replace heavy backbone with a tiny linear head for fast unit tests
    def tiny_backbone(self, num_classes: int, train_head_only: bool, use_pretrained: bool):
        backbone = torch.nn.Identity()
        head = torch.nn.Linear(8, num_classes)
        model = torch.nn.Sequential(backbone, head)
        return BackboneSpec(model=model, backbone=backbone, head=head)

    monkeypatch.setattr(BackboneType, "build", tiny_backbone, raising=True)

    config = DocClassifierConfig(
        num_classes=4,
        backbone=BackboneType.ALEXNET,
        train_head_only=False,
        use_pretrained=False,
    )

    module = DocClassifierModule(config)
    module.train(True)

    batch = torch.randn(2, 8)
    preds = module.classify_batch(batch, class_names=["a", "b", "c", "d"], top_k=2)

    # Training mode should be restored after the no-grad call
    assert module.training is True

    assert len(preds) == 2
    assert all(len(sample) == 2 for sample in preds)
    assert set(preds[0][0].keys()) == {"label", "index", "probability"}

    probs = [p["probability"] for p in preds[0]]
    assert probs == pytest.approx(sorted(probs, reverse=True))
    assert all(0.0 <= p <= 1.0 for p in probs)
