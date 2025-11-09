"""Albumentations transform configurations for document classification.

This module provides Config-as-Factory pattern for creating Albumentations
pipelines optimized for grayscale document images (RVL-CDIP dataset).

Three pipeline types are provided:
- TrainTransformConfig: Heavy augmentation for training from scratch
- FineTuneTransformConfig: Light augmentation for fine-tuning pretrained models
- ValTransformConfig: Deterministic transforms for validation/testing
"""

from typing import TYPE_CHECKING

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pydantic import Field

from ..utils import BaseConfig

if TYPE_CHECKING:
    pass


class TransformConfig(BaseConfig[A.Compose]):
    """Base configuration for Albumentations transforms.

    All transform configs should inherit from this and implement setup_target().
    """

    target: type[A.Compose] = Field(default=A.Compose, exclude=True)

    img_size: int = Field(default=224)
    """Target image size (height and width) after transformation."""

    grayscale_mean: float = Field(default=0.485)
    """Mean value for grayscale normalization (single channel)."""

    grayscale_std: float = Field(default=0.229)
    """Standard deviation for grayscale normalization (single channel)."""

    def setup_target(self) -> A.Compose:
        """Create and return an Albumentations Compose pipeline.

        Returns:
            A.Compose: Configured augmentation pipeline.
        """
        raise NotImplementedError("Subclasses must implement setup_target()")


class TrainTransformConfig(TransformConfig):
    """Heavy augmentation pipeline for training from scratch.

    Includes geometric transforms, noise, blur, and contrast adjustments
    suitable for learning robust features from limited data.
    """

    def setup_target(self) -> A.Compose:
        """Create training augmentation pipeline.

        Returns:
            A.Compose: Training pipeline with heavy augmentation.
        """
        return A.Compose(
            [
                # Resize to slightly larger size for random cropping
                A.SmallestMaxSize(max_size=int(self.img_size * 1.1)),
                # Geometric augmentations (document-friendly)
                A.RandomResizedCrop(
                    size=(self.img_size, self.img_size),
                    scale=(0.8, 1.0),  # Moderate cropping
                    ratio=(0.9, 1.1),  # Keep aspect ratio close to square
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=5, border_mode=0, p=0.3),  # Small rotation for scanned docs
                A.Perspective(scale=(0.02, 0.05), p=0.3),  # Simulate camera perspective
                # Document-specific degradations
                A.GaussNoise(std_range=(0.02, 0.08), p=0.3),  # Scanner noise
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),  # Slight blur
                A.MotionBlur(blur_limit=3, p=0.2),  # Motion artifacts
                # Brightness/contrast (common in scanned documents)
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5,
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                # Elastic deformation (subtle document warping)
                A.ElasticTransform(
                    alpha=30,
                    sigma=5,
                    p=0.2,
                ),
                # Convert grayscale to RGB (repeat channel 3 times for pretrained models)
                A.ToRGB(p=1.0),
                # Normalization and tensor conversion (ImageNet stats for RGB)
                A.Normalize(
                    mean=[self.grayscale_mean, self.grayscale_mean, self.grayscale_mean],
                    std=[self.grayscale_std, self.grayscale_std, self.grayscale_std],
                ),
                ToTensorV2(),
            ]
        )


class FineTuneTransformConfig(TransformConfig):
    """Light augmentation pipeline for fine-tuning pretrained models.

    Uses minimal augmentation to preserve features learned during pretraining
    while adapting to the document classification task.
    """

    def setup_target(self) -> A.Compose:
        """Create fine-tuning augmentation pipeline.

        Returns:
            A.Compose: Fine-tuning pipeline with light augmentation.
        """
        return A.Compose(
            [
                # Minimal geometric augmentation
                A.SmallestMaxSize(max_size=self.img_size),
                A.RandomResizedCrop(
                    size=(self.img_size, self.img_size),
                    scale=(0.9, 1.0),  # Gentle cropping
                    ratio=(0.95, 1.05),  # Nearly square
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                # Light quality adjustments
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3,
                ),
                # Convert grayscale to RGB
                A.ToRGB(p=1.0),
                # Normalization and tensor conversion
                A.Normalize(
                    mean=[self.grayscale_mean, self.grayscale_mean, self.grayscale_mean],
                    std=[self.grayscale_std, self.grayscale_std, self.grayscale_std],
                ),
                ToTensorV2(),
            ]
        )


class ValTransformConfig(TransformConfig):
    """Deterministic transforms for validation and testing.

    No augmentation - only resize, center crop, normalize, and tensorize.
    """

    def setup_target(self) -> A.Compose:
        """Create validation/test transformation pipeline.

        Returns:
            A.Compose: Validation pipeline with no augmentation.
        """
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=self.img_size),
                A.CenterCrop(height=self.img_size, width=self.img_size),
                # Convert grayscale to RGB
                A.ToRGB(p=1.0),
                A.Normalize(
                    mean=[self.grayscale_mean, self.grayscale_mean, self.grayscale_mean],
                    std=[self.grayscale_std, self.grayscale_std, self.grayscale_std],
                ),
                ToTensorV2(),
            ]
        )
