from typing import TYPE_CHECKING, Annotated, Any, Callable

import numpy as np
from datasets import Dataset as HFDataset
from datasets import load_dataset
from pydantic import Field, Tag

from traenslenzor.doc_classifier.configs.path_config import PathConfig

from ..utils import BaseConfig, Console, Stage
from .transforms import (
    FineTunePlusTransformConfig,
    FineTuneTransformConfig,
    TrainTransformConfig,
    ValTransformConfig,
)

if TYPE_CHECKING:
    from albumentations import Compose


TransformConfigUnion = Annotated[
    Annotated[TrainTransformConfig, Tag("train")]
    | Annotated[FineTuneTransformConfig, Tag("finetune")]
    | Annotated[FineTunePlusTransformConfig, Tag("finetune_plus")]
    | Annotated[ValTransformConfig, Tag("val")],
    Field(discriminator="transform_type"),
]


class RVLCDIPConfig(BaseConfig[HFDataset]):
    """Configuration for loading the RVL-CDIP document classification dataset."""

    target: type[HFDataset] = Field(default=HFDataset, exclude=True)

    hf_hub_name: str = Field(default="chainyo/rvl-cdip")
    """Hugging Face Hub dataset identifier for RVL-CDIP (uses modern Parquet format)."""

    split: Stage = Field(default=Stage.TRAIN)
    """Dataset split to load: TRAIN, VAL, or TEST."""

    transform_config: TransformConfigUnion | None = Field(default=None)
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


class _TransformApplier:
    """Pickle-able Albumentations batch applier for HF datasets."""

    def __init__(self, albumentations_transform: "Compose") -> None:
        self.albumentations_transform = albumentations_transform

    def __call__(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Apply Albumentations transform to a batch of examples."""

        images = [
            self.albumentations_transform(image=np.array(image))["image"]
            for image in examples["image"]
        ]
        return {"image": images, "label": examples["label"]}


def make_transform_fn(
    albumentations_transform: "Compose",
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a pickleable transform callable for Dataset.set_transform()."""

    return _TransformApplier(albumentations_transform)
