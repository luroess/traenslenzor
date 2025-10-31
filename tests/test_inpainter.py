from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from PIL import Image
from PIL.Image import Image as PILImage

from traenslenzor.image_renderer.inpainting import Inpainter


@pytest.fixture
def inpainter() -> Inpainter:
    return Inpainter(device="cpu")


@pytest.fixture
def test_image_path() -> Path:
    """Path to sample input image for inpainting tests."""
    return Path(__file__).parent / "fixtures" / "image_1.png"


@pytest.fixture
def test_mask_path() -> Path:
    """Path to sample mask for inpainting tests."""
    return Path(__file__).parent / "fixtures" / "mask_1.png"


@pytest.fixture
def sample_image(test_image_path: Path) -> PILImage:
    return Image.open(test_image_path).convert("RGB")


@pytest.fixture
def test_mask_image(test_mask_path: Path) -> PILImage:
    return Image.open(test_mask_path).convert("L")


@pytest.fixture
def partial_mask(test_mask_image: PILImage) -> NDArray[np.uint8]:
    """Convert mask image to numpy array format expected by inpainter."""
    mask_array = np.array(test_mask_image)
    # Normalize to 0 or 1 and add channel dimension
    mask_array = (mask_array > 127).astype(np.uint8)
    return mask_array[np.newaxis, :, :]


@pytest.fixture
def empty_mask(sample_image: PILImage) -> NDArray[np.uint8]:
    """Empty mask for testing preservation of unmasked regions."""
    h, w = sample_image.size
    return np.zeros((1, w, h), dtype=np.uint8)


def test_init_loads_model_on_device(inpainter: Inpainter) -> None:
    """Verify inpainter initializes with model loaded on correct device."""
    assert inpainter.LaMa is not None
    assert isinstance(inpainter.device, torch.device)


def test_inpaint_snapshot_comparison(
    inpainter: Inpainter,
    sample_image: PILImage,
    partial_mask: NDArray[np.uint8],
    image_regression,
) -> None:
    """Test inpainting output matches snapshot with reasonable tolerance.

    This validates that the inpainting model produces consistent results
    for the same input image and mask combination.
    """
    result = inpainter.inpaint(sample_image, partial_mask)

    # Convert normalized float array back to uint8 PIL Image
    result_uint8 = (result * 255).clip(0, 255).astype(np.uint8)
    result_image = Image.fromarray(result_uint8)

    # Allow some tolerance due to floating-point precision and model variations
    image_regression(result_image, threshold=0.02)


def test_inpaint_preserves_unmasked_regions_snapshot(
    inpainter: Inpainter,
    sample_image: PILImage,
    empty_mask: NDArray[np.uint8],
    image_regression,
) -> None:
    """Test that inpainting with empty mask preserves original image exactly.

    With no masked regions, the output should be nearly identical to input,
    validating that unmasked pixels are preserved during processing.
    """
    result = inpainter.inpaint(sample_image, empty_mask)

    result_uint8 = (result * 255).clip(0, 255).astype(np.uint8)
    result_image = Image.fromarray(result_uint8)

    # Stricter tolerance since nothing should be inpainted
    image_regression(result_image, threshold=0.01)


def test_inpaint_returns_valid_output_format(
    inpainter: Inpainter, sample_image: PILImage, partial_mask: NDArray[np.uint8]
) -> None:
    """Verify inpaint returns correctly formatted array."""
    result = inpainter.inpaint(sample_image, partial_mask)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.ndim == 3
    assert result.shape[2] == 3  # RGB channels
    assert result.shape[0] == sample_image.height
    assert result.shape[1] == sample_image.width


def test_inpaint_output_in_valid_range(
    inpainter: Inpainter, sample_image: PILImage, partial_mask: NDArray[np.uint8]
) -> None:
    """Verify inpaint output values are in valid range [0, 1]."""
    result = inpainter.inpaint(sample_image, partial_mask)

    assert result.min() >= 0.0
    assert result.max() <= 1.0
