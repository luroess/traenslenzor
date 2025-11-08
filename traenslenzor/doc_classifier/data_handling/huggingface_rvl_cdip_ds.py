from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from datasets import Dataset as HFDataset
from datasets import load_dataset
from pydantic import Field

from traenslenzor.doc_classifier.configs.path_config import PathConfig

from ..utils import BaseConfig, Console, Stage
from .transforms import TransformConfig

if TYPE_CHECKING:
    from albumentations import Compose


class RVLCDIPConfig(BaseConfig[HFDataset]):
    """Configuration for loading the RVL-CDIP document classification dataset."""

    target: type[HFDataset] = Field(default=HFDataset, exclude=True)

    hf_hub_name: str = Field(default="chainyo/rvl-cdip")
    """Hugging Face Hub dataset identifier for RVL-CDIP (uses modern Parquet format)."""

    split: Stage = Field(default=Stage.TRAIN)
    """Dataset split to load: TRAIN, VAL, or TEST."""

    transform_config: TransformConfig | None = Field(default=None)
    """Optional transform configuration. If provided, transforms will be applied via .set_transform()."""

    streaming: bool = Field(default=False)
    """Enable streaming mode."""

    num_workers: int = Field(default=4)
    """Number of worker processes for data loading."""

    is_debug: bool = Field(default=False)
    """Enable debug mode for verbose logging."""

    verbose: bool = Field(default=False)
    """Toggle verbose console output."""

    def setup_target(self) -> HFDataset:
        """Load the RVL-CDIP HuggingFace Dataset.

        Returns:
            HFDataset: Hugging Face Dataset object with 16 document classes.
                      If transform_config is provided, transforms are applied via .set_transform().

        Example:
            ```python
            from traenslenzor.doc_classifier.data_handling.transforms import TrainTransformConfig
            from traenslenzor.doc_classifier.utils import Stage

            # With transforms
            config = RVLCDIPConfig(
                split=Stage.TRAIN,
                transform_config=TrainTransformConfig(img_size=224)
            )
            dataset = config.setup_target()  # Transforms already applied

            # Without transforms (raw dataset)
            config = RVLCDIPConfig(split=Stage.VAL)
            dataset = config.setup_target()
            ```
        """
        console = (
            Console.with_prefix(self.__class__.__name__, "setup_target")
            .set_verbose(self.verbose)
            .set_debug(self.is_debug)
        )

        console.log(f"Synchronizing RVL-CDIP split='{self.split}' with local cache.")

        hf_dataset = load_dataset(
            path=self.hf_hub_name,
            split=str(self.split),
            cache_dir=PathConfig().hf_cache.as_posix(),
            streaming=self.streaming,
            num_proc=self.num_workers,
        )

        # Apply transforms if config provided
        if self.transform_config is not None:
            console.log(f"Applying {self.transform_config.__class__.__name__} transforms.")
            transform_pipeline = self.transform_config.setup_target()
            hf_dataset.set_transform(make_transform_fn(transform_pipeline))

        return hf_dataset


def make_transform_fn(
    albumentations_transform: "Compose",
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a transform function compatible with HuggingFace Dataset.set_transform().

    This factory function wraps an Albumentations pipeline for use with HF datasets.

    Args:
        albumentations_transform: Albumentations Compose pipeline.

    Returns:
        Transform function that takes a batch dict and returns transformed batch.
        Expects input dict with 'image' and 'label' keys.
        Outputs dict with 'image' (tensor) and 'label' (int).

    Example:
        ```python
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        # Define Albumentations pipeline
        transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2(),
        ])

        # Create HF-compatible transform function
        transform_fn = make_transform_fn(transform)

        # Apply to dataset
        dataset = config.setup_target()
        dataset.set_transform(transform_fn)
        ```
    """

    def transform_batch(examples: dict[str, Any]) -> dict[str, Any]:
        """Apply Albumentations transform to a batch of examples.

        Args:
            examples: Batch dictionary with 'image' (list of PIL Images) and
                     'label' (list of ints) from HuggingFace Dataset.

        Returns:
            Transformed batch with:
                - 'image': List of tensors (C, H, W) after Albumentations
                - 'label': Original labels (unchanged)
        """

        def transform_single(image):
            """PIL → numpy → Albumentations → tensor."""
            return albumentations_transform(image=np.array(image))["image"]

        return {
            "image": list(map(transform_single, examples["image"])),
            "label": examples["label"],
        }

    return transform_batch
