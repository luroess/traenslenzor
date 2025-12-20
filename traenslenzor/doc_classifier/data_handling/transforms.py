"""Albumentations transform configurations for document classification.

This module provides Config-as-Factory pattern for creating Albumentations
pipelines optimized for grayscale document images (RVL-CDIP dataset).

Three pipeline types are provided:
- TrainTransformConfig: Heavy augmentation for training from scratch
- FineTuneTransformConfig: Light augmentation for fine-tuning pretrained models
- FineTunePlusTransformConfig: Moderate augmentation for fine-tuning with extra regularization
- ValTransformConfig: Deterministic transforms for validation/testing
"""

from typing import TYPE_CHECKING, Literal

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pydantic import Field

from ..utils import BaseConfig

if TYPE_CHECKING:
    pass


# FIXME: conversion to RGB before noramlization !?
class TransformConfig(BaseConfig[A.Compose]):
    """Base configuration for Albumentations transforms.

    All transform configs should inherit from this and implement setup_target().
    """

    target: type[A.Compose] = Field(default=A.Compose, exclude=True)

    transform_type: Literal["base"] = Field(default="base")
    """Discriminator field for identifying transform type in serialized configs."""

    img_size: int = Field(default=224)
    """Target image size (height and width) after transformation."""

    grayscale_mean: float = Field(default=0.911966)
    """Mean value for grayscale normalization (single channel)."""

    grayscale_std: float = Field(default=0.241507)
    """Standard deviation for grayscale normalization (single channel)."""

    apply_normalization: bool = Field(default=True)
    """When True, apply normalization using grayscale_mean and grayscale_std; when False, skip normalization. Used for computing dataset stats."""

    convert_to_rgb: bool = Field(default=True)
    """When True, replicate grayscale to 3 channels; when False, keep single channel (uses ToGray)."""

    def setup_target(self) -> A.Compose:
        """Create and return an Albumentations Compose pipeline.

        Subclasses must override this method to provide their specific transform pipeline.

        Raises:
            NotImplementedError: If called on the base class directly.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement setup_target() to provide a transform pipeline."
        )


class TrainTransformConfig(TransformConfig):
    """Heavy augmentation pipeline for training from scratch.

    Includes geometric transforms, noise, blur, and contrast adjustments
    suitable for learning robust features from limited data.
    """

    transform_type: Literal["train"] = Field(default="train")
    """Discriminator field identifying this as a training transform config."""

    def setup_target(self) -> A.Compose:
        """Create training augmentation pipeline.

        Returns:
            A.Compose: Training pipeline with heavy augmentation.
        """
        mean = (
            [self.grayscale_mean, self.grayscale_mean, self.grayscale_mean]
            if self.convert_to_rgb
            else [self.grayscale_mean]
        )
        std = (
            [self.grayscale_std, self.grayscale_std, self.grayscale_std]
            if self.convert_to_rgb
            else [self.grayscale_std]
        )

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
                # Channel handling
                (A.ToRGB(p=1.0) if self.convert_to_rgb else A.ToGray(p=1.0, num_output_channels=1)),
                A.Normalize(
                    mean=mean,
                    std=std,
                ),
                ToTensorV2(),
            ]
        )


class FineTuneTransformConfig(TransformConfig):
    """Light augmentation pipeline for fine-tuning pretrained models.

    Uses minimal augmentation to preserve features learned during pretraining
    while adapting to the document classification task.
    """

    transform_type: Literal["finetune"] = Field(default="finetune")
    """Discriminator field identifying this as a fine-tuning transform config."""

    def setup_target(self) -> A.Compose:
        """Create fine-tuning augmentation pipeline.

        Returns:
            A.Compose: Fine-tuning pipeline with light augmentation.
        """
        mean = (
            [self.grayscale_mean, self.grayscale_mean, self.grayscale_mean]
            if self.convert_to_rgb
            else [self.grayscale_mean]
        )
        std = (
            [self.grayscale_std, self.grayscale_std, self.grayscale_std]
            if self.convert_to_rgb
            else [self.grayscale_std]
        )

        return A.Compose(
            [
                # Geometric augmentation
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
                # Convert channels
                (A.ToRGB(p=1.0) if self.convert_to_rgb else A.ToGray(p=1.0, num_output_channels=1)),
                # Normalization and tensor conversion
                A.Normalize(
                    mean=mean,
                    std=std,
                ),
                ToTensorV2(),
            ]
        )


class FineTunePlusTransformConfig(TransformConfig):
    """Moderate augmentation pipeline for fine-tuning pretrained models.

    Adds mild geometric and noise augmentations to improve generalization
    without overwhelming pretrained features.
    """

    transform_type: Literal["finetune_plus"] = Field(default="finetune_plus")
    """Discriminator field identifying this as a moderate fine-tuning transform config."""

    def setup_target(self) -> A.Compose:
        """Create moderate fine-tuning augmentation pipeline.

        Returns:
            A.Compose: Fine-tuning pipeline with moderate augmentation.
        """
        mean = (
            [self.grayscale_mean, self.grayscale_mean, self.grayscale_mean]
            if self.convert_to_rgb
            else [self.grayscale_mean]
        )
        std = (
            [self.grayscale_std, self.grayscale_std, self.grayscale_std]
            if self.convert_to_rgb
            else [self.grayscale_std]
        )

        return A.Compose(
            [
                A.SmallestMaxSize(max_size=int(self.img_size * 1.05)),
                A.RandomResizedCrop(
                    size=(self.img_size, self.img_size),
                    scale=(0.85, 1.0),
                    ratio=(0.9, 1.1),
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=3, border_mode=0, p=0.2),
                A.Perspective(scale=(0.01, 0.03), p=0.2),
                A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
                A.GaussianBlur(blur_limit=(3, 3), p=0.1),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    p=0.4,
                ),
                A.RandomGamma(gamma_limit=(90, 110), p=0.2),
                (A.ToRGB(p=1.0) if self.convert_to_rgb else A.ToGray(p=1.0, num_output_channels=1)),
                A.Normalize(
                    mean=mean,
                    std=std,
                ),
                ToTensorV2(),
            ]
        )


class ValTransformConfig(TransformConfig):
    """Deterministic transforms for validation and testing.

    No augmentation - only resize, center crop, normalize, and tensorize.
    """

    transform_type: Literal["val"] = Field(default="val")
    """Discriminator field identifying this as a validation/test transform config."""

    def setup_target(self) -> A.Compose:
        """Create validation/test transformation pipeline.

        Returns:
            A.Compose: Validation pipeline with no augmentation.
        """
        mean = (
            [self.grayscale_mean, self.grayscale_mean, self.grayscale_mean]
            if self.convert_to_rgb
            else [self.grayscale_mean]
        )
        std = (
            [self.grayscale_std, self.grayscale_std, self.grayscale_std]
            if self.convert_to_rgb
            else [self.grayscale_std]
        )

        return A.Compose(
            [
                A.SmallestMaxSize(max_size=self.img_size),
                A.CenterCrop(height=self.img_size, width=self.img_size),
                # A.Resize(height=self.img_size, width=self.img_size),
                # Convert grayscale to desired channel count
                (A.ToRGB(p=1.0) if self.convert_to_rgb else A.ToGray(p=1.0, num_output_channels=1)),
                (
                    A.Normalize(
                        mean=mean,
                        std=std,
                    )
                    if self.apply_normalization
                    else A.NoOp()
                ),
                ToTensorV2(),
            ]
        )
