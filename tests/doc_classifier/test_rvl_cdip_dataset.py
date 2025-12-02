"""Unit tests for RVL-CDIP dataset integration."""

from unittest.mock import MagicMock

import albumentations as A
import numpy as np
import pytest
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from traenslenzor.doc_classifier.data_handling.huggingface_rvl_cdip_ds import (
    RVLCDIPConfig,
    make_transform_fn,
)


class TestRVLCDIPConfig:
    """Test suite for RVLCDIPConfig."""

    def test_default_values(self):
        """Test that default configuration values are set correctly."""
        config = RVLCDIPConfig()

        assert config.hf_hub_name == "chainyo/rvl-cdip"
        assert config.num_workers == 4
        assert config.is_debug is False
        assert config.verbose is False  # Updated default


class TestMakeTransformFn:
    """Test suite for make_transform_fn factory."""

    def test_transform_fn_creation(self):
        """Test that make_transform_fn creates a callable."""
        mock_transform = MagicMock()
        transform_fn = make_transform_fn(mock_transform)

        assert callable(transform_fn)

    def test_transform_fn_with_albumentations(self):
        """Test transform function with real Albumentations pipeline."""
        transform = A.Compose(
            [
                A.Resize(224, 224),
                ToTensorV2(),
            ]
        )

        transform_fn = make_transform_fn(transform)

        mock_images = [Image.new("L", (100, 100), color=128) for _ in range(4)]
        mock_batch = {
            "image": mock_images,
            "label": [0, 1, 2, 3],
        }

        result = transform_fn(mock_batch)

        assert "image" in result
        assert "label" in result
        assert len(result["image"]) == 4
        assert len(result["label"]) == 4

        for img_tensor in result["image"]:
            assert isinstance(img_tensor, torch.Tensor)
            assert img_tensor.shape == (1, 224, 224)

    def test_transform_fn_preserves_labels(self):
        """Test that transform function doesn't modify labels."""
        mock_transform = MagicMock()
        mock_transform.side_effect = lambda image: {"image": torch.zeros(3, 224, 224)}

        transform_fn = make_transform_fn(mock_transform)

        mock_batch = {
            "image": [Image.new("RGB", (100, 100)) for _ in range(3)],
            "label": [5, 10, 15],
        }

        result = transform_fn(mock_batch)

        assert result["label"] == [5, 10, 15]

    def test_transform_fn_numpy_conversion(self):
        """Test that PIL images are converted to numpy arrays."""
        calls = []

        def tracking_transform(image):
            calls.append(type(image))
            return {"image": torch.zeros(1, 224, 224)}

        mock_compose = MagicMock()
        mock_compose.side_effect = tracking_transform

        transform_fn = make_transform_fn(mock_compose)

        mock_batch = {
            "image": [Image.new("L", (100, 100))],
            "label": [0],
        }

        transform_fn(mock_batch)

        assert calls[0] == np.ndarray


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
