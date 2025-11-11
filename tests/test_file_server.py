from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from traenslenzor.file_server.client import FileClient
from traenslenzor.image_utils.image_utils import pil_to_numpy


@pytest.fixture
def temp_file(tmp_path: Path) -> Path:
    """Create a temporary test file with some content."""
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("Hello, file server!")
    return file_path


@pytest.fixture
def temp_image(tmp_path: Path) -> Path:
    """Create a temporary test image file."""
    file_path = tmp_path / "test_image.png"
    # Create a simple 100x100 RGB image with a red square
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(file_path)
    return file_path


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Create a sample PIL image for testing."""
    # Create a 50x50 blue image
    return Image.new("RGB", (50, 50), color=(0, 0, 255))


@pytest.mark.anyio
async def test_put_uploads_file_successfully(file_server: None, temp_file: Path) -> None:
    """Test that put() uploads a file and returns a UUID."""
    file_id = await FileClient.put(str(temp_file))

    assert file_id is not None
    assert isinstance(file_id, str)
    assert len(file_id) == 36  # UUID length with hyphens


@pytest.mark.anyio
async def test_put_raises_on_nonexistent_file(file_server: None) -> None:
    """Test that put() raises FileNotFoundError for nonexistent files."""
    with pytest.raises(FileNotFoundError):
        await FileClient.put("/nonexistent/path/file.txt")


@pytest.mark.anyio
async def test_put_img_uploads_pil_image(file_server: None, sample_pil_image: Image.Image) -> None:
    """Test that put_img() uploads a PIL image and returns a UUID."""
    file_id = await FileClient.put_img("test.png", sample_pil_image)

    assert file_id is not None
    assert isinstance(file_id, str)
    assert len(file_id) == 36


@pytest.mark.anyio
async def test_get_raw_bytes_downloads_file(file_server: None, temp_file: Path) -> None:
    """Test that get_raw_bytes() downloads file content correctly."""
    # Upload file first
    file_id = await FileClient.put(str(temp_file))
    assert file_id is not None

    # Download as bytes
    file_bytes = await FileClient.get_raw_bytes(file_id)

    assert file_bytes is not None
    assert isinstance(file_bytes, bytes)
    assert file_bytes == b"Hello, file server!"


@pytest.mark.anyio
async def test_get_raw_bytes_returns_none_for_404(file_server: None) -> None:
    """Test that get_raw_bytes() returns None for non-existent file."""
    result = await FileClient.get_raw_bytes("00000000-0000-0000-0000-000000000000")
    assert result is None


@pytest.mark.anyio
async def test_get_image_downloads_as_pil(file_server: None, sample_pil_image: Image.Image) -> None:
    """Test that get_image() downloads and returns a PIL Image."""
    # Upload image first
    file_id = await FileClient.put_img("test.png", sample_pil_image)
    assert file_id is not None

    # Download as PIL Image
    downloaded_img = await FileClient.get_image(file_id)

    assert downloaded_img is not None
    assert isinstance(downloaded_img, Image.Image)
    assert downloaded_img.size == (50, 50)
    assert downloaded_img.mode == "RGB"


@pytest.mark.anyio
async def test_get_image_returns_none_for_404(file_server: None) -> None:
    """Test that get_image() returns None for non-existent file."""
    result = await FileClient.get_image("00000000-0000-0000-0000-000000000000")
    assert result is None


@pytest.mark.anyio
async def test_get_image_as_numpy_downloads_as_array(
    file_server: None, sample_pil_image: Image.Image
) -> None:
    """Test that get_image_as_numpy() downloads and returns numpy array."""
    # Upload image first
    file_id = await FileClient.put_img("test.png", sample_pil_image)
    assert file_id is not None

    # Download as numpy array
    img_array = await FileClient.get_image_as_numpy(file_id)

    assert img_array is not None
    assert isinstance(img_array, np.ndarray)
    assert img_array.shape == (50, 50, 3)
    assert img_array.dtype == np.float32


@pytest.mark.anyio
async def test_get_image_as_numpy_returns_none_for_404(file_server: None) -> None:
    """Test that get_image_as_numpy() returns None for non-existent file."""
    result = await FileClient.get_image_as_numpy("00000000-0000-0000-0000-000000000000")
    assert result is None


@pytest.mark.anyio
async def test_rem_deletes_file_successfully(file_server: None, temp_file: Path) -> None:
    """Test that rem() deletes a file and returns True."""
    # Upload file first
    file_id = await FileClient.put(str(temp_file))
    assert file_id is not None

    # Delete the file
    deleted = await FileClient.rem(file_id)
    assert deleted is True

    # Verify file is gone
    result = await FileClient.get_raw_bytes(file_id)
    assert result is None


@pytest.mark.anyio
async def test_rem_returns_false_for_404(file_server: None) -> None:
    """Test that rem() returns False for non-existent file."""
    result = await FileClient.rem("00000000-0000-0000-0000-000000000000")
    assert result is False


@pytest.mark.anyio
async def test_round_trip_image_preserves_data(
    file_server: None, sample_pil_image: Image.Image
) -> None:
    """Test that uploading and downloading an image preserves pixel data."""
    # Upload original image
    file_id = await FileClient.put_img("original.png", sample_pil_image)
    assert file_id is not None

    # Download as numpy array
    downloaded_array = await FileClient.get_image_as_numpy(file_id)
    assert downloaded_array is not None

    # Convert original to numpy for comparison
    original_array = pil_to_numpy(sample_pil_image)

    # Verify pixel-perfect match
    assert downloaded_array.shape == original_array.shape
    np.testing.assert_array_equal(downloaded_array, original_array)


@pytest.mark.anyio
async def test_multiple_files_stored_independently(file_server: None, tmp_path: Path) -> None:
    """Test that multiple files can be stored and retrieved independently."""
    # Create multiple files
    file1 = tmp_path / "file1.txt"
    file1.write_text("Content 1")
    file2 = tmp_path / "file2.txt"
    file2.write_text("Content 2")

    # Upload both
    id1 = await FileClient.put(str(file1))
    id2 = await FileClient.put(str(file2))

    if id1 is None or id2 is None:
        raise RuntimeError("Failed to upload test files to file server")

    assert id1 != id2

    # Download and verify
    bytes1 = await FileClient.get_raw_bytes(id1)
    bytes2 = await FileClient.get_raw_bytes(id2)

    assert bytes1 == b"Content 1"
    assert bytes2 == b"Content 2"
