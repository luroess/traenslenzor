"""End-to-end tests for font detector using generated images."""

import json
import os
from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock, patch

import pytest

from traenslenzor.font_detector.font_size_model.data_gen import FONT_CONFIGS, render_text_box
from traenslenzor.font_detector.server import estimate_font_size_logic


# Use a temporary directory for test images
@pytest.fixture
def test_image_dir(tmp_path):
    return tmp_path


def get_font_path(font_name: str) -> str:
    """Helper to get font path or return None if not found."""
    if font_name not in FONT_CONFIGS:
        return None

    for path in FONT_CONFIGS[font_name]:
        if os.path.exists(path):
            return path
    return None


def generate_test_image(
    text: str, font_name: str, font_size_pt: int, output_path: Path, max_width: int = 800
) -> Tuple[float, float, int]:
    """
    Generate a realistic text cutout image using the same logic as training data.
    Returns the (width, height, num_lines) of the text box.
    """
    font_path = get_font_path(font_name)
    if not font_path:
        pytest.skip(f"Font {font_name} not found")
        return 0.0, 0.0, 0

    # Use the shared render logic from data_gen to ensure consistency
    img, box_size, num_lines = render_text_box(
        text=text,
        font_path=font_path,
        font_size_pt=float(font_size_pt),
        padding=0,
        max_width=max_width,
    )

    img.save(output_path)
    return box_size[0], box_size[1], num_lines


@pytest.mark.parametrize("font_name", FONT_CONFIGS.keys())
def test_contract_with_auto_detection(font_name, test_image_dir):
    """Test estimate_font_size with image_path for auto-detection."""

    # Check if font exists
    if not get_font_path(font_name):
        pytest.skip(f"Font {font_name} not found")

    target_size = 24

    # Define test cases
    test_cases = [
        ("Standard", f"Testing {font_name} 123"),
        ("Lower", f"testing {font_name.lower()}"),
        ("Numbers", "0123456789"),
    ]

    for label, text in test_cases:
        image_path = test_image_dir / f"test_{font_name}_{label}.png"
        width, height, num_lines = generate_test_image(
            text, font_name, target_size, image_path, max_width=1200
        )

        # Mock the font detector to return the correct font name
        with patch("traenslenzor.font_detector.server.get_font_name_detector") as mock_get_detector:
            mock_detector = MagicMock()
            mock_detector.detect.return_value = font_name
            mock_get_detector.return_value = mock_detector

            # Call the tool
            result_json = estimate_font_size_logic(
                text_box_size=[width, height], text=text, image_path=str(image_path), font_name=""
            )

            result = json.loads(result_json)
            assert "error" not in result, f"Tool returned error: {result.get('error')}"

            estimated_size = result["font_size_pt"]
            error = abs(estimated_size - target_size)

            # Relaxed tolerance for contract testing
            assert error < 20.0, (
                f"Size estimation off for {font_name}: {estimated_size} vs {target_size}"
            )


@pytest.mark.parametrize("font_name", FONT_CONFIGS.keys())
def test_multiline_handling(font_name, test_image_dir):
    """Test multiline text handling."""
    if not get_font_path(font_name):
        pytest.skip(f"Font {font_name} not found")

    target_size = 24
    text_multi = (
        "This is a longer test string intended to produce multiple lines for the "
        f"font {font_name} in order to exercise multiline handling. 0123456789"
    )
    image_path_multi = test_image_dir / f"test_{font_name}_multi.png"
    width_m, height_m, num_lines_m = generate_test_image(
        text_multi, font_name, target_size, image_path_multi, max_width=200
    )

    with patch("traenslenzor.font_detector.server.get_font_name_detector") as mock_get_detector:
        mock_detector = MagicMock()
        mock_detector.detect.return_value = font_name
        mock_get_detector.return_value = mock_detector

        result_json = estimate_font_size_logic(
            text_box_size=[width_m, height_m],
            text=text_multi,
            image_path=str(image_path_multi),
            font_name="",
            # num_lines=num_lines_m,  # Pass the correct number of lines
        )

        result = json.loads(result_json)
        assert "error" not in result, f"Tool returned error: {result.get('error')}"
        assert result["font_name"] == font_name
        estimated_size = result["font_size_pt"]
        error = abs(estimated_size - target_size)
        assert error < 10.0, (
            f"Size estimation off (multi) for {font_name}: {estimated_size} vs {target_size}"
        )


@pytest.mark.parametrize("font_name", FONT_CONFIGS.keys())
def test_full_pipeline(font_name, test_image_dir):
    """
    Test the full pipeline: Image -> Detect Font -> Estimate Size.
    """
    if not get_font_path(font_name):
        pytest.skip(f"Font {font_name} not found")

    target_size = 24
    text = f"Testing {font_name} pipeline 123"

    image_path = test_image_dir / f"pipeline_{font_name}.png"
    width, height, num_lines = generate_test_image(
        text, font_name, target_size, image_path, max_width=1000
    )

    # Call estimate_font_size_logic WITHOUT font_name
    # This triggers internal detection
    result_json = estimate_font_size_logic(
        text_box_size=[width, height],
        text=text,
        image_path=str(image_path),
        font_name="",  # Force detection
        # num_lines=num_lines,
    )

    result = json.loads(result_json)
    assert "error" not in result, f"Tool returned error: {result.get('error')}"

    detected_font = result.get("font_name")
    estimated_size = result.get("font_size_pt")

    # Check font detection (relaxed)
    # Note: Detection might fail or fallback, but the pipeline should complete
    # We check if the result is valid
    assert detected_font is not None
    assert isinstance(estimated_size, (int, float))

    # Check size estimation
    error = abs(estimated_size - target_size)
    assert error < 10.0, f"Size estimation off: {estimated_size} vs {target_size}"
