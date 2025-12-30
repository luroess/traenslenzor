import torch

from traenslenzor.doc_classifier.lightning.lit_module import BackboneType, DocClassifierConfig


def test_alexnet_backbone_head_split_supports_grayscale_forward() -> None:
    config = DocClassifierConfig(
        num_classes=16,
        backbone=BackboneType.ALEXNET,
        train_head_only=False,
        use_pretrained=False,
    )
    module = config.setup_target()

    batch = torch.randn(2, 1, 224, 224)
    logits = module(batch)

    assert logits.shape == (2, 16)

