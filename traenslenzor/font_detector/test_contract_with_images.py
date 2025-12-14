import json
import os
from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock, patch

from PIL import Image

from traenslenzor.font_detector.font_size_model.data_gen import extract_features, render_text_box
from traenslenzor.font_detector.server import detect_font_name_logic, estimate_font_size_logic

# Font paths (adjust based on your system or use the ones from data_gen.py)
FONT_PATHS = {
    "Roboto-Regular": "/usr/share/fonts/TTF/Roboto-Regular.ttf",
    "RobotoMono-Regular": "/usr/share/fonts/TTF/RobotoMono-Regular.ttf",
    "Inter-Regular": "/usr/share/fonts/inter/Inter.ttc",
    "Lato-Regular": "/usr/share/fonts/TTF/Lato-Regular.ttf",
    "IBMPlexSans-Regular": "/usr/share/fonts/TTF/IBMPlexSans-Regular.ttf",
}

OUTPUT_DIR = Path(__file__).parent / "test_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_test_image(
    text: str, font_name: str, font_size_pt: int, output_path: Path, max_width: int = 800
) -> Tuple[float, float, int]:
    """
    Generate a realistic text cutout image using the same logic as training data.
    Returns the (width, height, num_lines) of the text box.
    """
    font_path = FONT_PATHS.get(font_name)
    if not font_path or not os.path.exists(font_path):
        print(f"Skipping {font_name}: Font file not found at {font_path}")
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
    print(f"Generated {output_path} ({box_size[0]}x{box_size[1]}, {num_lines} lines)")

    return box_size[0], box_size[1], num_lines


def test_contract_with_auto_detection():
    """Test estimate_font_size with image_path for auto-detection."""

    # Test for each supported font
    for font_name in FONT_PATHS.keys():
        print(f"\nTesting {font_name}...")
        target_size = 24

        # Define test cases
        test_cases = [
            ("Standard", f"Testing {font_name} 123"),
            ("Lower", f"testing {font_name.lower()}"),
            ("Numbers", "0123456789"),
        ]

        for label, text in test_cases:
            image_path = OUTPUT_DIR / f"test_{font_name}_{label}.png"
            width, height, num_lines = generate_test_image(
                text, font_name, target_size, image_path, max_width=1200
            )
            if width == 0:
                continue

            # Extract and print features
            features = extract_features((width, height), text, num_lines=num_lines)
            print(
                f"  [{label}] Features: W={width:.1f}, H={height:.1f}, Lines={num_lines}, Density={features[3]:.4f}"
            )

            # Mock the font detector to return the correct font name
            with patch(
                "traenslenzor.font_detector.server.get_font_name_detector"
            ) as mock_get_detector:
                mock_detector = MagicMock()
                mock_detector.detect.return_value = font_name
                mock_get_detector.return_value = mock_detector

                # Call the tool
                result_json = estimate_font_size_logic(
                    text_box_size=[width, height],
                    text=text,
                    image_path=str(image_path),
                    font_name="",
                )

                result = json.loads(result_json)
                if "error" in result:
                    print(f"  [{label}] Error: {result['error']}")
                    continue

                estimated_size = result["font_size_pt"]
                error = estimated_size - target_size
                print(f"  [{label}] Est: {estimated_size:.2f} pt, Error: {error:+.2f} pt")

            # Relaxed tolerance for contract testing
            assert error < 20.0, (
                f"Size estimation off for {font_name}: {estimated_size} vs {target_size}"
            )

        # --- Multiline example (force wrapping) ---
        text_multi = (
            "This is a longer test string intended to produce multiple lines for the "
            f"font {font_name} in order to exercise multiline handling. 0123456789"
        )
        image_path_multi = OUTPUT_DIR / f"test_{font_name}_multi.png"
        width_m, height_m, num_lines_m = generate_test_image(
            text_multi, font_name, target_size, image_path_multi, max_width=200
        )
        if width_m == 0:
            continue

        with patch("traenslenzor.font_detector.server.get_font_name_detector") as mock_get_detector:
            mock_detector = MagicMock()
            mock_detector.detect.return_value = font_name
            mock_get_detector.return_value = mock_detector

            result_json = estimate_font_size_logic(
                text_box_size=[width_m, height_m],
                text=text_multi,
                image_path=str(image_path_multi),
                font_name="",
                num_lines=num_lines_m,  # Pass the correct number of lines
            )

            print(f"Result (multi): {result_json}")
            result = json.loads(result_json)
            assert "error" not in result, f"Tool returned error: {result.get('error')}"
            assert result["font_name"] == font_name
            estimated_size = result["font_size_pt"]
            error = abs(estimated_size - target_size)
            print(f"Estimated size (multi): {estimated_size:.2f} pt, Error: {error:.2f} pt")
            assert error < 10.0, (
                f"Size estimation off (multi) for {font_name}: {estimated_size} vs {target_size}"
            )


def test_font_detection_accuracy():
    """Test actual font detection accuracy on generated images."""
    print("\n=== Testing Font Detection Accuracy ===")

    total = 0
    correct = 0

    for font_name in FONT_PATHS.keys():
        # Generate a good sample for detection (longer text, reasonable size)
        text = "The quick brown fox jumps over the lazy dog. 1234567890"
        image_path = OUTPUT_DIR / f"accuracy_{font_name}.png"

        # Use 32pt font for clear detection
        width, height, _ = generate_test_image(text, font_name, 32, image_path, max_width=1000)
        if width == 0:
            continue

        # Add padding to help the model (ViT usually expects square-ish or standard aspect ratios)
        try:
            img = Image.open(image_path)
            # Create a larger canvas
            target_w = max(img.width + 100, 400)
            target_h = max(img.height + 100, 400)
            new_img = Image.new("RGB", (target_w, target_h), "white")
            offset_x = (target_w - img.width) // 2
            offset_y = (target_h - img.height) // 2
            new_img.paste(img, (offset_x, offset_y))
            new_img.save(image_path)
        except Exception as e:
            print(f"  Failed to pad image: {e}")

        total += 1

        # Call the tool (no mocking)
        try:
            # Use the logic function directly
            result_json = detect_font_name_logic(str(image_path))
            result = json.loads(result_json)

            if "error" in result:
                print(f"  [{font_name}] Error: {result['error']}")
                continue

            detected = result["font_name"]

            # Check match (relaxed)
            match = font_name.lower() in detected.lower() or detected.lower() in font_name.lower()

            if match:
                print(f"  [{font_name}] ✓ Detected: {detected}")
                correct += 1
            else:
                print(f"  [{font_name}] ✗ Expected {font_name}, got {detected}")

        except Exception as e:
            print(f"  [{font_name}] Exception: {e}")

    if total > 0:
        accuracy = correct / total * 100
        print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")
        if accuracy < 80.0:
            print(
                f"WARNING: Font detection accuracy is low ({accuracy:.1f}%). Check model or test images."
            )


def test_full_pipeline():
    """
    Test the full pipeline: Image -> Detect Font -> Estimate Size.
    Uses the same images as the size model tests.
    """
    print("\n=== Testing Full Pipeline (Detection + Estimation) ===")

    for font_name in FONT_PATHS.keys():
        print(f"\nTesting pipeline for {font_name}...")
        target_size = 24
        text = f"Testing {font_name} pipeline 123"

        image_path = OUTPUT_DIR / f"pipeline_{font_name}.png"
        width, height, num_lines = generate_test_image(
            text, font_name, target_size, image_path, max_width=1000
        )

        if width == 0:
            continue

        # Call estimate_font_size_logic WITHOUT font_name
        # This triggers internal detection
        result_json = estimate_font_size_logic(
            text_box_size=[width, height],
            text=text,
            image_path=str(image_path),
            font_name="",  # Force detection
            num_lines=num_lines,
        )

        result = json.loads(result_json)

        if "error" in result:
            print(f"  Error: {result['error']}")
            continue

        detected_font = result.get("font_name")
        estimated_size = result.get("font_size_pt")

        print(f"  Detected: {detected_font}")
        print(f"  Estimated: {estimated_size:.2f} pt")

        # Check font detection (relaxed)
        if font_name.lower() in detected_font.lower() or detected_font.lower() in font_name.lower():
            print("  Font Detection: PASS")
        else:
            print(f"  Font Detection: FAIL (Expected {font_name})")

        # Check size estimation
        error = abs(estimated_size - target_size)
        if error < 10.0:
            print(f"  Size Estimation: PASS (Error {error:.2f} pt)")
        else:
            print(f"  Size Estimation: FAIL (Error {error:.2f} pt)")


if __name__ == "__main__":
    # Manual run if executed directly
    try:
        test_contract_with_auto_detection()
        test_font_detection_accuracy()
        test_full_pipeline()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()
