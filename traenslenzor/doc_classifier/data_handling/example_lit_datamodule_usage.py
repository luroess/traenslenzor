"""Example usage of DocDataModule with transform configurations.

This script demonstrates how to configure the Lightning DataModule with different
transform pipelines for training, validation, and testing.
"""

from traenslenzor.doc_classifier.data_handling.huggingface_rvl_cdip_ds import (
    RVLCDIPConfig,
)
from traenslenzor.doc_classifier.data_handling.transforms import (
    FineTuneTransformConfig,
    TrainTransformConfig,
    ValTransformConfig,
)
from traenslenzor.doc_classifier.lightning.lit_datamodule import DocDataModuleConfig
from traenslenzor.doc_classifier.utils import Stage


def main():
    """Demonstrate DataModule configuration with transforms."""

    # ==========================================================================
    # Example 1: Training from scratch with heavy augmentation
    # ==========================================================================
    print("=" * 80)
    print("Example 1: Training from Scratch Configuration")
    print("=" * 80)

    train_from_scratch_config = DocDataModuleConfig(
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        train_dataset=RVLCDIPConfig(
            split=Stage.TRAIN,
            transform_config=TrainTransformConfig(
                img_size=224,
                grayscale_mean=0.485,
                grayscale_std=0.229,
            ),
            verbose=True,
        ),
        val_dataset=RVLCDIPConfig(
            split=Stage.VAL,
            transform_config=ValTransformConfig(
                img_size=224,
                grayscale_mean=0.485,
                grayscale_std=0.229,
            ),
            verbose=True,
        ),
        test_dataset=RVLCDIPConfig(
            split=Stage.TEST,
            transform_config=ValTransformConfig(
                img_size=224,
                grayscale_mean=0.485,
                grayscale_std=0.229,
            ),
            verbose=True,
        ),
    )

    print("\n✅ Configuration created:")
    print(f"   Batch size: {train_from_scratch_config.batch_size}")
    print(f"   Num workers: {train_from_scratch_config.num_workers}")
    print(
        f"   Train transforms: {train_from_scratch_config.train_dataset.transform_config.__class__.__name__}"
    )
    print(
        f"   Val transforms: {train_from_scratch_config.val_dataset.transform_config.__class__.__name__}"
    )
    print(
        f"   Test transforms: {train_from_scratch_config.test_dataset.transform_config.__class__.__name__}"
    )

    # ==========================================================================
    # Example 2: Fine-tuning pretrained model with light augmentation
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Example 2: Fine-Tuning Configuration")
    print("=" * 80)

    finetune_config = DocDataModuleConfig(
        batch_size=128,  # Larger batch for fine-tuning
        num_workers=8,
        pin_memory=True,
        train_dataset=RVLCDIPConfig(
            split=Stage.TRAIN,
            transform_config=FineTuneTransformConfig(
                img_size=256,  # ResNet-50/ViT often use 224 or 256
            ),
        ),
        val_dataset=RVLCDIPConfig(
            split=Stage.VAL,
            transform_config=ValTransformConfig(img_size=256),
        ),
        test_dataset=RVLCDIPConfig(
            split=Stage.TEST,
            transform_config=ValTransformConfig(img_size=256),
        ),
    )

    print("\n✅ Fine-tuning configuration created:")
    print(f"   Batch size: {finetune_config.batch_size}")
    print(f"   Image size: {finetune_config.train_dataset.transform_config.img_size}")
    print("   Train transforms: FineTuneTransformConfig (light augmentation)")

    # ==========================================================================
    # Example 3: Inference/Testing only (no training)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Example 3: Inference-Only Configuration")
    print("=" * 80)

    inference_config = DocDataModuleConfig(
        batch_size=256,  # Large batch for inference
        num_workers=8,
        pin_memory=True,
        # Use val dataset as dummy for train/val (won't be used)
        train_dataset=RVLCDIPConfig(
            split=Stage.VAL,
            transform_config=ValTransformConfig(img_size=224),
        ),
        val_dataset=RVLCDIPConfig(
            split=Stage.VAL,
            transform_config=ValTransformConfig(img_size=224),
        ),
        test_dataset=RVLCDIPConfig(
            split=Stage.TEST,
            transform_config=ValTransformConfig(img_size=224),
        ),
    )

    print("\n✅ Inference configuration created:")
    print(f"   Batch size: {inference_config.batch_size}")
    print("   All datasets use deterministic ValTransformConfig")

    # ==========================================================================
    # Example 4: Instantiate DataModule and inspect
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Example 4: DataModule Instantiation")
    print("=" * 80)

    # Create the actual DataModule
    datamodule = train_from_scratch_config.setup_target()

    print("\n✅ DocDataModule instantiated:")
    print(f"   Class: {datamodule.__class__.__name__}")
    print(f"   Batch size: {datamodule.config.batch_size}")
    print(
        f"   Datasets initialized: train={datamodule.train_ds is not None}, "
        f"val={datamodule.val_ds is not None}, test={datamodule.test_ds is not None}"
    )

    print("\n" + "=" * 80)
    print("Usage with PyTorch Lightning Trainer:")
    print("=" * 80)
    print("""
    from pytorch_lightning import Trainer
    from traenslenzor.doc_classifier.lightning.lit_module import DocClassifier

    # Create model
    model = DocClassifier(model_config=...)

    # Create datamodule
    datamodule_config = DocDataModuleConfig(...)
    datamodule = datamodule_config.setup_target()

    # Train
    trainer = Trainer(max_epochs=10, accelerator="gpu", devices=1)
    trainer.fit(model, datamodule=datamodule)

    # Test
    trainer.test(model, datamodule=datamodule)
    """)


if __name__ == "__main__":
    main()
