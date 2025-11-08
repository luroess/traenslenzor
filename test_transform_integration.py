"""Quick test of transform system integration."""

from traenslenzor.doc_classifier.data_handling.huggingface_rvl_cdip_ds import RVLCDIPConfig
from traenslenzor.doc_classifier.data_handling.transforms import (
    FineTuneTransformConfig,
    TrainTransformConfig,
    ValTransformConfig,
)
from traenslenzor.doc_classifier.utils import Stage


def test_transform_configs():
    """Test that transform configs can be created and setup."""
    print("Testing transform configs...")

    # Create configs
    train_cfg = TrainTransformConfig(img_size=224)
    finetune_cfg = FineTuneTransformConfig(img_size=256)
    val_cfg = ValTransformConfig(img_size=224)

    print(f"✅ Train config: {train_cfg.img_size}px, mean={train_cfg.grayscale_mean}")
    print(f"✅ FineTune config: {finetune_cfg.img_size}px")
    print(f"✅ Val config: {val_cfg.img_size}px")

    # Test setup_target()
    train_pipeline = train_cfg.setup_target()
    print(f"✅ Train pipeline: {len(train_pipeline.transforms)} transforms")

    finetune_pipeline = finetune_cfg.setup_target()
    print(f"✅ FineTune pipeline: {len(finetune_pipeline.transforms)} transforms")

    val_pipeline = val_cfg.setup_target()
    print(f"✅ Val pipeline: {len(val_pipeline.transforms)} transforms")


def test_dataset_integration():
    """Test that RVLCDIPConfig accepts transform configs."""
    print("\nTesting dataset integration...")

    # Create dataset config with transforms
    dataset_cfg = RVLCDIPConfig(
        split=Stage.VAL,
        transform_config=ValTransformConfig(img_size=224),
        verbose=True,
    )

    print(
        f"✅ Dataset config created with transform_config={dataset_cfg.transform_config.__class__.__name__}"
    )
    print(f"   Split: {dataset_cfg.split}")
    print(f"   Transform img_size: {dataset_cfg.transform_config.img_size}")

    # Note: We won't actually load the dataset to avoid network calls
    print("✅ Config ready for dataset loading (setup_target() would apply transforms)")


if __name__ == "__main__":
    test_transform_configs()
    test_dataset_integration()
    print("\n✨ All integration tests passed!")
