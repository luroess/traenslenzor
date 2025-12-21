#!/usr/bin/env python3
"""Test script for font name detection."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from traenslenzor.font_detector.font_name_detector import FontNameDetector
from traenslenzor.font_detector.font_size_model.data_gen import get_font_path


def create_test_image(text: str, font_name: str, font_size: int, output_path: str):
    """Create a test image with known font for testing - matching HF model training format."""
    try:
        font_path = get_font_path(font_name)

        # Match the HuggingFace training format:
        # - 800x400 canvas
        # - Random offsets
        # - Line wrapping based on font width
        # - Random line spacing

        import random

        from PIL import Image, ImageDraw, ImageFont

        # Fixed canvas size (matching HF training)
        canvas_width, canvas_height = 800, 400
        img = Image.new("RGB", (canvas_width, canvas_height), color="white")
        draw = ImageDraw.Draw(img)

        # Load font
        font = ImageFont.truetype(font_path, size=font_size)

        # Calculate words per line based on font width (matching HF approach)
        font_avg_char_width = font.getbbox("x")[2]
        words_per_line = int(canvas_width / (font_avg_char_width * 5))
        words_per_line = max(words_per_line, 3)  # At least 3 words per line

        # Wrap text
        words = text.split()
        lines = []
        for i in range(0, len(words), words_per_line):
            line = " ".join(words[i : i + words_per_line])
            lines.append(line)

        wrapped_text = "\n".join(lines)

        # Random offsets (matching HF training)
        offset_x = random.randint(-20, 10)
        offset_y = random.randint(-20, 10)

        # Random line spacing (matching HF training: 0 to 1.25x font size)
        line_spacing = random.uniform(0, 1.25) * font_size

        # Draw text with spacing
        draw.text((offset_x, offset_y), wrapped_text, fill="black", font=font, spacing=line_spacing)

        img.save(output_path)

        print(f"  {Path(output_path).name} ({font_name}, {font_size}pt)")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def test_font_detection(image_path: str, expected_font: str = None):
    """Test font detection on an image."""
    try:
        detector = FontNameDetector()
        detected_font = detector.detect(image_path)

        if expected_font:
            match = (
                "✓"
                if expected_font.lower() in detected_font.lower()
                or detected_font.lower() in expected_font.lower()
                else "✗"
            )
            print(f"  {match} {expected_font:<24} -> {detected_font}")
        else:
            print(f"  Detected: {detected_font}")

        return detected_font

    except Exception as e:
        print(f"  Failed: {e}")
        return None


def main():
    """Run font detection tests."""
    print("\nCreating test images...")

    # Create test images directory
    test_dir = Path(__file__).parent / "test_images"
    test_dir.mkdir(exist_ok=True)

    # Create test images for available fonts (matching HuggingFace model classes)
    # Use longer prose-like text similar to the model's training data (inaugural speeches)
    test_cases = [
        (
            "We hold these truths to be self evident that all men are created equal. They are endowed by their Creator with certain unalienable Rights including Life Liberty and the pursuit of Happiness. To secure these rights Governments are instituted among Men deriving their just powers from the consent of the governed. Whenever any Form of Government becomes destructive of these ends it is the Right of the People to alter or to abolish it and to institute new Government laying its foundation on such principles and organizing its powers in such form as to them shall seem most likely to effect their Safety and Happiness.",
            "Roboto-Regular",
            24,
            "roboto_regular_test.png",
        ),
        (
            "Four score and seven years ago our fathers brought forth on this continent a new nation conceived in Liberty and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war testing whether that nation or any nation so conceived and so dedicated can long endure. We are met on a great battle field of that war. We have come to dedicate a portion of that field as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this.",
            "RobotoMono-Regular",
            20,
            "robotomono_regular_test.png",
        ),
        (
            "In the beginning God created the heaven and the earth. And the earth was without form and void and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters. And God said Let there be light and there was light. And God saw the light that it was good and God divided the light from the darkness. And God called the light Day and the darkness he called Night. And the evening and the morning were the first day.",
            "Inter-Regular",
            22,
            "inter_regular_test.png",
        ),
        (
            "It was the best of times it was the worst of times it was the age of wisdom it was the age of foolishness it was the epoch of belief it was the epoch of incredulity it was the season of Light it was the season of Darkness it was the spring of hope it was the winter of despair we had everything before us we had nothing before us we were all going direct to Heaven we were all going direct the other way.",
            "Lato-Regular",
            22,
            "lato_regular_test.png",
        ),
        (
            "Call me Ishmael. Some years ago never mind how long precisely having little or no money in my purse and nothing particular to interest me on shore I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation. Whenever I find myself growing grim about the mouth whenever it is a damp drizzly November in my soul.",
            "IBMPlexSans-Regular",
            22,
            "ibmplex_regular_test.png",
        ),
    ]

    created_images = []
    for text, font_name, size, filename in test_cases:
        output_path = test_dir / filename
        if create_test_image(text, font_name, size, str(output_path)):
            created_images.append((str(output_path), font_name))
        print()

    if not created_images:
        print("\nNo test images created.")
        return

    print("\nDetecting fonts...")

    results = []
    for image_path, expected_font in created_images:
        detected = test_font_detection(image_path, expected_font)
        if detected:
            results.append((expected_font, detected))
        print()

    # Summary
    print("\nResults:")

    if results:
        for expected, detected in results:
            match = (
                "+"
                if expected.lower() in detected.lower() or detected.lower() in expected.lower()
                else "?"
            )
            print(f"  {match} {expected:<20} -> {detected}")

    print(f"\nTest images saved in: {test_dir}")
    print("Test with custom image: python test_font_detection.py /path/to/image.png\n")


def test_custom_image(image_path: str):
    """Test font detection on a custom image."""
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        return

    print(f"\nTesting: {image_path}")
    detected = test_font_detection(image_path)
    print(detected)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test custom image
        test_custom_image(sys.argv[1])
    else:
        # Run full test suite
        main()
