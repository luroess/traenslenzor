"""Inference module for font size estimation."""

import argparse
from pathlib import Path
from typing import Optional, Tuple

from .features import FeatureNormalizer, extract_features
from .model import FontSizeRegressorMLP


class FontSizeEstimator:
    """Font size estimator with lazy model loading."""

    def __init__(self, checkpoints_dir: Optional[str] = None):
        """
        Initialize estimator.

        Args:
            checkpoints_dir: Directory containing model checkpoints.
                           If None, uses default location relative to this module.
        """
        if checkpoints_dir is None:
            # Use checkpoints in the font_detector directory
            checkpoints_dir = str(Path(__file__).parent.parent / "checkpoints")

        self.checkpoints_dir = Path(checkpoints_dir)
        # Cache for loaded models
        self.models: dict[str, FontSizeRegressorMLP] = {}
        # Cache for loaded normalizers
        self.normalizers: dict[str, FeatureNormalizer] = {}

    def _load_model(self, font_name: str):
        """
        Lazy load model and normalizer for a font.

        Args:
            font_name: Font name

        Raises:
            FileNotFoundError: If model files not found
        """
        if font_name in self.models:
            return

        model_dir = self.checkpoints_dir / font_name
        model_path = model_dir / "best.json"
        norm_path = model_dir / "norm.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found for font '{font_name}' at {model_path}")

        if not norm_path.exists():
            raise FileNotFoundError(f"Normalizer not found for font '{font_name}' at {norm_path}")

        # Load model and normalizer
        self.models[font_name] = FontSizeRegressorMLP.load(str(model_path))
        self.normalizers[font_name] = FeatureNormalizer.load(str(norm_path))

    def estimate(
        self,
        text_box_size: Tuple[float, float],
        text: str,
        font_name: str,
        num_lines: int = 1,
    ) -> float:
        """
        Estimate font size.

        Args:
            text_box_size: (width_px, height_px) tuple
            text: Text content
            font_name: Font name
            num_lines: Number of lines (default: 1)

        Returns:
            Estimated font size in points
        """

        # Load model if needed
        self._load_model(font_name)

        # Extract and normalize features
        features = extract_features(text_box_size, text, num_lines=num_lines)
        features_norm = self.normalizers[font_name].normalize(features)

        # Predict
        model = self.models[font_name]
        pred = model.forward(features_norm, training=False)

        # Return scalar prediction (handle both array and scalar)
        if hasattr(pred, "item"):
            result: float = float(pred.item())
            return result
        return float(pred)

    def list_available_fonts(self) -> list:
        """
        List available fonts (models that have been trained).

        Returns:
            List of font names
        """
        if not self.checkpoints_dir.exists():
            return []

        fonts = []
        for font_dir in self.checkpoints_dir.iterdir():
            if font_dir.is_dir():
                model_path = font_dir / "best.json"
                norm_path = font_dir / "norm.json"
                if model_path.exists() and norm_path.exists():
                    fonts.append(font_dir.name)

        return sorted(fonts)


def estimate_font_size(
    text_box_size: Tuple[float, float],
    text: str,
    font_name: Optional[str] = None,
    checkpoints_dir: Optional[str] = None,
    num_lines: int = 1,
) -> dict:
    """
    Estimate font size (MCP tool interface).

    Args:
        text_box_size: (width_px, height_px) tuple
        text: Text content
        font_name: Optional font name hint
        checkpoints_dir: Directory containing checkpoints (uses default if None)
        num_lines: Number of lines (default: 1)

    Returns:
        Dictionary with 'font_size_pt' key

    Raises:
        ValueError: If font_name not provided and can't be detected
    """
    # If no font name provided, try to detect
    if font_name is None:
        # Would call detect_font_name here, but for now require font_name
        raise ValueError(
            "font_name is required for estimation (detection not implemented in this context)"
        )

    # Create estimator and run inference
    estimator = FontSizeEstimator(checkpoints_dir)
    font_size_pt = estimator.estimate(text_box_size, text, font_name, num_lines=num_lines)

    return {"font_size_pt": font_size_pt}


def main():
    """CLI for inference."""
    parser = argparse.ArgumentParser(description="Estimate font size")
    parser.add_argument(
        "--font",
        type=str,
        required=True,
        help="Font name",
    )
    parser.add_argument(
        "--w",
        type=float,
        required=True,
        help="Text box width in pixels",
    )
    parser.add_argument(
        "--h",
        type=float,
        required=True,
        help="Text box height in pixels",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text content",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=1,
        help="Number of lines (default: 1)",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=None,
        help="Checkpoints directory (default: module's checkpoints directory)",
    )

    args = parser.parse_args()

    result = estimate_font_size(
        text_box_size=(args.w, args.h),
        text=args.text,
        font_name=args.font,
        checkpoints_dir=args.checkpoints_dir,
        num_lines=args.lines,
    )

    print(f"Estimated font size: {result['font_size_pt']:.2f} pt")


if __name__ == "__main__":
    main()
