"""Font name detector using custom ResNet18 model."""

from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class FontNameDetector:
    """Detect font name from image using custom ResNet18 model."""

    def __init__(
        self,
        device: Optional[str] = None,
    ):
        """
        Initialize font name detector with local model.

        Args:
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = (
            Path(__file__).parent / "checkpoints" / "classifier" / "resnet18_fonts.pth"
        )

        # Lazy loading - only load when first needed
        self._model: Optional[Any] = None
        self._transform: Optional[Any] = None
        self._classes: Optional[List[str]] = None

    def _load_model(self) -> None:
        """Lazy load the model."""
        if self._model is not None:
            return

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {self.checkpoint_path}. Please run train_classifier.py first."
            )

        print(f"Loading custom font classifier from {self.checkpoint_path} on {self.device}...")

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self._classes = checkpoint["classes"]
        num_classes = len(self._classes)

        # Recreate model structure
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        self._model = model

        # Define transform (matches training)
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        print(f"Model loaded successfully! Classes: {self._classes}")

    def _prepare_image(self, img: Image.Image) -> Image.Image:
        """
        Load and prepare image for inference.
        Strategy: Center crop 224x224 if larger, pad if smaller. No resizing/scaling.
        """
        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        target_size = 224

        # Create white canvas
        new_img = Image.new("RGB", (target_size, target_size), "white")

        # If image is larger than target, crop center
        if w > target_size or h > target_size:
            # Calculate crop box
            left = max(0, (w - target_size) // 2)
            top = max(0, (h - target_size) // 2)
            right = min(w, left + target_size)
            bottom = min(h, top + target_size)

            img = img.crop((left, top, right, bottom))
            w, h = img.size

        # Paste in center
        offset_x = (target_size - w) // 2
        offset_y = (target_size - h) // 2
        new_img.paste(img, (offset_x, offset_y))

        return new_img

    def detect(self, img: Image.Image) -> str:
        """
        Detect font name from image.

        Args:
            img: PIL Image object

        Returns:
            Detected font name as string
        """
        # Load model if not already loaded
        self._load_model()
        assert self._model is not None and self._transform is not None and self._classes is not None

        # Prepare image
        img = self._prepare_image(img)

        # Transform to tensor
        input_tensor = self._transform(img).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self._model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

            predicted_class = predicted_idx.item()
            conf_val = confidence.item()

        # Get label
        font_name = self._classes[predicted_class]

        print(f"Detected: {font_name} (confidence: {conf_val:.2%})")

        return font_name

    def detect_top_k(self, img: Image.Image, k: int = 5) -> List[Tuple[str, float]]:
        """
        Detect top-k most likely font names from image.

        Args:
            img: PIL Image object
            k: Number of top predictions to return

        Returns:
            List of (font_name, confidence) tuples, sorted by confidence
        """
        # Load model if not already loaded
        self._load_model()
        assert self._model is not None and self._transform is not None and self._classes is not None

        # Prepare image
        img = self._prepare_image(img)

        # Transform to tensor
        input_tensor = self._transform(img).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self._model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

            # Get top-k predictions
            top_k_probs, top_k_indices = torch.topk(probs, k=min(k, len(probs)))

        # Build results
        results = []
        for prob, idx in zip(top_k_probs.tolist(), top_k_indices.tolist()):
            font_name = self._classes[idx]
            results.append((font_name, prob))

        return results


def detect_font_name(image_path: str) -> dict:
    """
    Detect font name from image (MCP tool interface).

    Args:
        image_path: Path to image file containing text

    Returns:
        Dictionary with 'font_name' key
    """
    detector = FontNameDetector()
    font_name = detector.detect(image_path)
    return {"font_name": font_name}
