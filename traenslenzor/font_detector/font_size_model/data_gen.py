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
FONT_DIR = Path(__file__).parent.parent / "fonts"
FONT_CONFIGS = {
    "Roboto-Regular": [
        str(FONT_DIR / "Roboto-Regular.ttf"),
    ],
    "RobotoMono-Regular": [
        str(FONT_DIR / "RobotoMono-Regular.otf"),
    ],
    "Inter-Regular": [
        str(FONT_DIR / "Inter-Regular.otf"),
    ],
    "Lato-Regular": [
        str(FONT_DIR / "Lato-Regular.ttf"),
    ],
    "IBMPlexSans-Regular": [
        str(FONT_DIR / "IBMPlexSans-Regular.ttf"),
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
    # Longer texts for multiline generation
    "The quick brown fox jumps over the lazy dog. This is a classic pangram that contains every letter of the alphabet.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "Font detection is a challenging problem in computer vision. It involves identifying the typeface used in an image of text.",
    "Machine learning models can be trained to estimate font size by analyzing the dimensions and content of a text box.",
    "Deep learning has revolutionized the field of natural language processing and computer vision in recent years.",
    "Python is a versatile programming language that is widely used for web development, data analysis, and artificial intelligence.",
]


def generate_random_text(min_words: int = 1, max_words: int = 30) -> str:
    """Generate random text from a simple vocabulary."""
    vocab = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "lorem",
        "ipsum",
        "dolor",
        "sit",
        "amet",
        "consectetur",
        "adipiscing",
        "elit",
        "sed",
        "do",
        "eiusmod",
        "tempor",
        "incididunt",
        "ut",
        "labore",
        "et",
        "dolore",
        "magna",
        "aliqua",
        "ut",
        "enim",
        "ad",
        "minim",
        "veniam",
        "quis",
        "nostrud",
        "exercitation",
        "ullamco",
        "laboris",
        "nisi",
        "ut",
        "aliquip",
        "ex",
        "ea",
        "commodo",
        "consequat",
        "duis",
        "aute",
        "irure",
        "dolor",
        "in",
        "reprehenderit",
        "in",
        "voluptate",
        "velit",
        "esse",
        "cillum",
        "dolore",
        "eu",
        "fugiat",
        "nulla",
        "pariatur",
        "excepteur",
        "sint",
        "occaecat",
        "cupidatat",
        "non",
        "proident",
        "sunt",
        "in",
        "culpa",
        "qui",
        "officia",
        "deserunt",
        "mollit",
        "anim",
        "id",
        "est",
        "laborum",
        "hello",
        "world",
        "python",
        "programming",
        "code",
        "data",
        "science",
        "machine",
        "learning",
        "artificial",
        "intelligence",
        "neural",
        "network",
        "deep",
        "computer",
        "vision",
        "image",
        "processing",
        "text",
        "font",
        "size",
        "estimation",
        "detection",
        "recognition",
        "analysis",
        "system",
        "model",
        "algorithm",
        "function",
        "variable",
        "class",
        "object",
        "method",
        "interface",
        "implementation",
        "design",
        "pattern",
        "architecture",
        "software",
        "engineering",
        "development",
        "testing",
        "debugging",
        "deployment",
        "cloud",
        "server",
        "client",
        "database",
        "storage",
        "network",
        "security",
        "privacy",
        "authentication",
        "authorization",
        "encryption",
        "decryption",
        "compression",
        "optimization",
        "performance",
        "scalability",
        "reliability",
        "availability",
        "maintainability",
        "usability",
        "accessibility",
        "internationalization",
        "localization",
        "documentation",
        "support",
        "community",
        "open",
        "source",
        "license",
        "copyright",
        "trademark",
        "patent",
        "legal",
        "ethical",
        "social",
        "economic",
        "political",
        "environmental",
        "impact",
        "future",
        "trends",
        "challenges",
        "opportunities",
        "risks",
        "benefits",
        "conclusion",
        "summary",
        "introduction",
        "background",
        "methodology",
        "results",
        "discussion",
        "references",
        "appendix",
        "glossary",
        "index",
        "table",
        "figure",
        "chart",
        "graph",
        "diagram",
        "illustration",
        "image",
        "picture",
        "photo",
        "video",
        "audio",
        "music",
        "sound",
        "voice",
        "speech",
        "language",
        "translation",
    ]

    num_words = random.randint(min_words, max_words)
    words = [random.choice(vocab) for _ in range(num_words)]

    # Capitalize first word
    words[0] = words[0].capitalize()

    # Add punctuation
    text = " ".join(words)
    if random.random() < 0.8:
        text += "."

    # Occasionally add numbers (to match test distribution)
    if random.random() < 0.3:
        text += f" {random.randint(0, 999)}"

    return text


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
    padding: int = 0,
    max_width: int = 800,
) -> Tuple[Image.Image, Tuple[float, float], int]:
    """
    Render text with line wrapping and return cropped bounding box.

    Args:
        text: Text to render (can contain multiple sentences)
        font_path: Path to font file
        font_size_pt: Font size in points
        padding: Padding around text in pixels
        max_width: Maximum width before wrapping (pixels)

    Returns:
        (image, (width_px, height_px), num_lines) tuple
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

    # Calculate dimensions using tight bounding box of the whole text block
    # First, draw everything onto a temporary canvas large enough to hold it
    # Standard line height for spacing, but crop tightly at the end

    # Standard line height for spacing calculation
    standard_bbox = temp_draw.textbbox((0, 0), "Ay", font=font)
    standard_height = standard_bbox[3] - standard_bbox[1]
    line_spacing = int(standard_height * 0.2)

    # Estimate max dimensions
    max_w = max_width + padding * 2
    est_h = (standard_height + line_spacing) * len(lines) + padding * 2

    # Create temp image with transparent background for accurate bbox
    temp_full = Image.new("RGBA", (int(max_w), int(est_h)), color=(255, 255, 255, 0))
    draw_full = ImageDraw.Draw(temp_full)

    # Draw text
    current_y = padding
    for line in lines:
        draw_full.text((padding, current_y), line, font=font, fill="black")
        current_y += standard_height + line_spacing

    # Get tight bounding box of the content (alpha channel is non-zero for text)
    bbox = temp_full.getbbox()
    if bbox:
        left, top, right, bottom = bbox
        # Calculate crop dimensions with padding
        crop_w = (right - left) + 2 * padding
        crop_h = (bottom - top) + 2 * padding

        # Create final image with white background
        img = Image.new("RGB", (int(crop_w), int(crop_h)), color="white")

        # Draw text again on the final image
        # We need to adjust coordinates relative to the crop
        # Or simpler: just paste the crop from temp_full onto white background?
        # But temp_full has transparent background.
        # Let's just redraw to be safe and clean.

        # Actually, simpler:
        # We know the shift is `left - padding`.
        # So new x = old x - (left - padding).
        # But we drew at `padding`.
        # So `left` is likely `padding` (or slightly more if first char has bearing).

        # Let's just paste the relevant part of temp_full onto a white background.
        # Crop from temp_full
        text_crop = temp_full.crop((left, top, right, bottom))

        # Paste onto center of new white image
        img.paste(text_crop, (padding, padding), mask=text_crop)

        box_size = (float(img.width), float(img.height))
    else:
        # Empty image fallback
        img = Image.new("RGB", (padding * 2, padding * 2), color="white")
        box_size = (float(padding * 2), float(padding * 2))

    return img, box_size, len(lines)


def generate_sample(
    font_name: str,
    font_path: str,
    min_size: float,
    max_size: float,
) -> dict:
    """
    Generate a single sample.

    Args:
        font_name: Font name
        font_path: Font file path
        min_size: Min font size
        max_size: Max font size

    Returns:
        Dictionary with sample data
    """
    # Randomly select text and font size
    # 50% chance to use predefined sample texts, 50% chance to generate random text
    if random.random() < 0.5:
        text = random.choice(SAMPLE_TEXTS)
    else:
        # Generate random text length based on whether we want multiline or not
        # But we decide multiline later based on max_width.
        # Let's just generate a random length string.
        text = generate_random_text(min_words=1, max_words=30)

    # Use integer font size to avoid mismatch between label and rendering
    # ImageFont.truetype truncates float sizes to int
    font_size_pt = float(random.randint(int(min_size), int(max_size)))

    # Decide whether this sample should be multiline (50% chance)
    if random.random() < 0.5:
        # multiline: choose a relatively small max_width to force wrapping
        # Width should be related to font size to ensure reasonable wrapping
        # e.g. 5-15 chars wide
        approx_char_width = font_size_pt * 0.6
        max_width = int(random.randint(5, 20) * approx_char_width)
        # Ensure min width
        max_width = max(max_width, int(font_size_pt * 2))
    else:
        # single line: large max_width so no wrapping occurs
        max_width = 2000

    # Render text with chosen max_width
    img, box_size, num_lines = render_text_box(
        text, font_path, font_size_pt, padding=0, max_width=max_width
    )

    # Extract features
    features = extract_features(box_size, text)  # Removed num_lines

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
            feature_names = [f"feat_{i}" for i in range(34)]
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
