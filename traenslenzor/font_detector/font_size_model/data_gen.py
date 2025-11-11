"""Dataset generator for font size regression."""

import argparse
import csv
import random
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .features import extract_features

# Font configurations by following criteria:
# 1. Detectabale by bygaborcselle/font-identifier model
# 2. Available on Linux
FONT_CONFIGS = {
    "Roboto-Regular": [
        "/usr/share/fonts/TTF/Roboto-Regular.ttf",
    ],
    "RobotoMono-Regular": [
        "/usr/share/fonts/TTF/RobotoMono-Regular.ttf",
    ],
    "Inter-Regular": [
        "/usr/share/fonts/inter/Inter.ttc",
    ],
    "Lato-Regular": [
        "/usr/share/fonts/TTF/Lato-Regular.ttf",
    ],
    "IBMPlexSans-Regular": [
        "/usr/share/fonts/TTF/IBMPlexSans-Regular.ttf",
    ],
}

# Sample texts for rendering
SAMPLE_TEXTS = [
    "Hello World",
    "The quick brown fox",
    "Lorem ipsum",
    "Font Size Test",
    "Machine Learning",
    "Python Programming",
    "Data Science",
    "Artificial Intelligence",
    "Deep Learning",
    "Computer Vision",
    "Natural Language",
    "Neural Networks",
    "OpenAI GPT",
    "Transformer Model",
    "ABCDEFGHIJKLM",
    "NOPQRSTUVWXYZ",
    "abcdefghijklm",
    "nopqrstuvwxyz",
    "0123456789",
    "Sample Text",
]


def get_font_path(font_name: str) -> str:
    """
    Get font file path, trying multiple locations.

    Args:
        font_name: Font name

    Returns:
        Font file path

    Raises:
        ValueError: If font not found
    """
    if font_name not in FONT_CONFIGS:
        raise ValueError(f"Unknown font: {font_name}. Available: {list(FONT_CONFIGS.keys())}")

    font_paths = FONT_CONFIGS[font_name]

    # Try each path until we find one that exists
    for font_path in font_paths:
        if Path(font_path).exists():
            return font_path

    # None found
    raise FileNotFoundError(
        f"Font file not found for {font_name}. Tried paths:\n"
        + "\n".join(f"  - {p}" for p in font_paths)
    )


def render_text_box(
    text: str,
    font_path: str,
    font_size_pt: float,
    padding: int = 10,
    max_width: int = 800,
) -> Tuple[Image.Image, Tuple[float, float]]:
    """
    Render text with line wrapping and return cropped bounding box.

    Args:
        text: Text to render (can contain multiple sentences)
        font_path: Path to font file
        font_size_pt: Font size in points
        padding: Padding around text in pixels
        max_width: Maximum width before wrapping (pixels)

    Returns:
        (image, (width_px, height_px)) tuple
    """
    # Load font
    font = ImageFont.truetype(font_path, size=int(font_size_pt))

    # Create temporary image to get text size
    temp_img = Image.new("RGB", (1, 1), color="white")
    temp_draw = ImageDraw.Draw(temp_img)

    # Word wrap the text
    words = text.split()
    lines = []
    current_line: list[str] = []

    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = temp_draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]

        if line_width <= max_width - 2 * padding:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                # Single word too long, add it anyway
                lines.append(word)
                current_line = []

    if current_line:
        lines.append(" ".join(current_line))

    # Calculate dimensions
    line_height = temp_draw.textbbox((0, 0), "Ay", font=font)[3]
    line_spacing = int(line_height * 0.2)  # 20% line spacing

    max_line_width: float = 0
    for line in lines:
        bbox = temp_draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        max_line_width = max(max_line_width, line_width)

    total_height: float = len(lines) * line_height + (len(lines) - 1) * line_spacing

    # Create actual image with padding
    img_width: float = max_line_width + 2 * padding
    img_height: float = total_height + 2 * padding

    img = Image.new("RGB", (int(img_width), int(img_height)), color="white")
    draw = ImageDraw.Draw(img)

    # Draw each line
    y_offset: float = padding
    for line in lines:
        draw.text((padding, int(y_offset)), line, font=font, fill="black")
        y_offset += line_height + line_spacing

    # Return image and box size
    box_size = (img_width, img_height)

    return img, box_size


def generate_sample(
    font_name: str,
    font_path: str,
    min_size: float = 8.0,
    max_size: float = 42.0,
) -> dict:
    """
    Generate a single training sample.

    Args:
        font_name: Name of the font
        font_path: Path to font file
        min_size: Minimum font size in points
        max_size: Maximum font size in points

    Returns:
        Dictionary with sample data
    """
    # Randomly select text and font size
    text = random.choice(SAMPLE_TEXTS)
    font_size_pt = random.uniform(min_size, max_size)

    # Render text
    img, box_size = render_text_box(text, font_path, font_size_pt)

    # Extract features
    features = extract_features(box_size, text)

    return {
        "font_name": font_name,
        "text": text,
        "font_size_pt": font_size_pt,
        "width_px": box_size[0],
        "height_px": box_size[1],
        "features": features,
    }


def generate_dataset(
    font_name: str,
    n_train: int = 10000,
    n_val: int = 1000,
    n_test: int = 1000,
    min_size: float = 8.0,
    max_size: float = 42.0,
    seed: int = 42,
    output_dir: str = "data",
) -> dict:
    """
    Generate dataset for a single font.

    Args:
        font_name: Name of the font
        n_train: Number of training samples
        n_val: Number of validation samples
        n_test: Number of test samples
        min_size: Minimum font size
        max_size: Maximum font size
        seed: Random seed
        output_dir: Output directory

    Returns:
        Dictionary with dataset statistics
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Get font path
    font_path = get_font_path(font_name)

    # Create output directory
    output_path = Path(output_dir) / font_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating dataset for {font_name}...")
    print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")
    print(f"  Font size range: [{min_size}, {max_size}] pt")

    # Generate samples for each split
    splits = {
        "train": n_train,
        "val": n_val,
        "test": n_test,
    }

    stats = {}

    for split_name, n_samples in splits.items():
        samples = []

        print(f"  Generating {split_name} split...")
        for i in range(n_samples):
            if (i + 1) % 1000 == 0:
                print(f"    {i + 1}/{n_samples}")

            sample = generate_sample(font_name, font_path, min_size, max_size)
            samples.append(sample)

        # Save to CSV
        csv_path = output_path / f"{split_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            header = ["font_name", "text", "font_size_pt", "width_px", "height_px"]
            feature_names = [f"feat_{i}" for i in range(30)]
            header.extend(feature_names)
            writer.writerow(header)

            # Data
            for sample in samples:
                row = [
                    sample["font_name"],
                    sample["text"],
                    sample["font_size_pt"],
                    sample["width_px"],
                    sample["height_px"],
                ]
                row.extend(sample["features"].tolist())
                writer.writerow(row)

        print(f"    Saved to {csv_path}")

        # Compute statistics
        sizes = [s["font_size_pt"] for s in samples]
        stats[split_name] = {
            "n_samples": n_samples,
            "size_mean": np.mean(sizes),
            "size_std": np.std(sizes),
            "size_min": np.min(sizes),
            "size_max": np.max(sizes),
        }

    print(f"Dataset generation complete for {font_name}!")

    return stats


def main():
    """CLI for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate font size regression dataset")
    parser.add_argument(
        "--font",
        type=str,
        required=True,
        choices=list(FONT_CONFIGS.keys()),
        help="Font name",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=10000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--n-val",
        type=int,
        default=1000,
        help="Number of validation samples",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=1000,
        help="Number of test samples",
    )
    parser.add_argument(
        "--min-size",
        type=float,
        default=8.0,
        help="Minimum font size in points",
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=42.0,
        help="Maximum font size in points",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: module's data directory)",
    )

    args = parser.parse_args()

    # Set default output dir if not provided
    if args.output_dir is None:
        args.output_dir = str(Path(__file__).parent.parent / "data")

    stats = generate_dataset(
        font_name=args.font,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        min_size=args.min_size,
        max_size=args.max_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print("\nDataset Statistics:")
    for split_name, split_stats in stats.items():
        print(f"\n{split_name.upper()}:")
        for key, value in split_stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
