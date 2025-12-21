"""Tests for FontNameDetector model."""

from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from traenslenzor.font_detector.font_name_detector import FontNameDetector
from traenslenzor.font_detector.font_size_model.data_gen import FONT_CONFIGS


# Helper to get font path
def get_font_path(font_name: str) -> str:
    if font_name not in FONT_CONFIGS:
        return None
    for path in FONT_CONFIGS[font_name]:
        if Path(path).exists():
            return path
    return None


@pytest.fixture
def detector():
    """Fixture to load detector once."""
    return FontNameDetector()


def create_test_image(text: str, font_name: str, font_size: int, output_path: Path):
    """Create a test image with known font."""
    font_path = get_font_path(font_name)
    if not font_path:
        return False

    # Fixed canvas size
    canvas_width, canvas_height = 800, 400
    img = Image.new("RGB", (canvas_width, canvas_height), color="white")
    draw = ImageDraw.Draw(img)

    # Load font
    font = ImageFont.truetype(font_path, size=font_size)

    # Draw text
    draw.text((10, 10), text, fill="black", font=font)
    img.save(output_path)
    return True


@pytest.mark.parametrize("font_name", ["Roboto-Regular", "Inter-Regular"])
def test_font_detection_run(detector, font_name, tmp_path):
    """Test that font detection runs without error on generated images."""
    if not get_font_path(font_name):
        pytest.skip(f"Font {font_name} not found")

    image_path = tmp_path / f"test_{font_name}.png"
    success = create_test_image("Hello World", font_name, 24, image_path)

    if not success:
        pytest.skip("Could not create test image")

    # Run detection
    detected_font = detector.detect(str(image_path))

    # Assert we got a string back
    assert isinstance(detected_font, str)
    assert len(detected_font) > 0

    # We don't strictly assert accuracy here as it depends on the model and synthetic data quality
    # But we can check if it returns one of the known classes if possible
    # (The model might return other fonts too)


def test_detector_invalid_path(detector):
    """Test detector behavior with invalid path."""
    with pytest.raises(Exception):
        detector.detect("non_existent_image.png")
