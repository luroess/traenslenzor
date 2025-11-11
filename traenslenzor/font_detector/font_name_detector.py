"""Font name detector using local HuggingFace model."""

from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


class FontNameDetector:
    """Detect font name from image using gaborcselle/font-identifier model."""

    def __init__(
        self,
        model_name: str = "gaborcselle/font-identifier",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize font name detector with local model.

        Args:
            model_name: HuggingFace model identifier (default: gaborcselle/font-identifier)
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        # Force CPU mode for compatibility (CUDA requires sm_70+)
        self.device = device or "cpu"
        self.cache_dir = cache_dir

        # Lazy loading - only load when first needed
        self._processor: Optional[Any] = None
        self._model: Optional[Any] = None

    def _load_model(self) -> None:
        """Lazy load the model and processor."""
        if self._model is None:
            print(f"Loading font identifier model '{self.model_name}' on {self.device}...")
            self._processor = AutoImageProcessor.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
            self._model = AutoModelForImageClassification.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
            assert self._model is not None  # type narrowing
            self._model.to(self.device)
            self._model.eval()
            print(
                f"Model loaded successfully! Can classify {len(self._model.config.id2label)} fonts."
            )

    def _prepare_image(self, image_path: str) -> Image.Image:
        """
        Load and prepare image for inference.

        Args:
            image_path: Path to image file

        Returns:
            PIL Image object
        """
        img = Image.open(image_path)

        # Convert to RGB if needed
        if img.mode != "RGB":
            return img.convert("RGB")

        return img

    def detect(self, image_path: str) -> str:
        """
        Detect font name from image.

        Args:
            image_path: Path to image file containing text

        Returns:
            Detected font name as string

        Raises:
            FileNotFoundError: If image file does not exist
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load model if not already loaded
        self._load_model()
        assert self._processor is not None and self._model is not None  # type narrowing

        # Prepare image
        img = self._prepare_image(image_path)

        # Preprocess image
        inputs = self._processor(img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

            # Get top prediction
            predicted_class = logits.argmax(-1).item()
            confidence = torch.softmax(logits, dim=-1)[0][predicted_class].item()

        # Get label
        font_name: str = self._model.config.id2label[predicted_class]

        print(f"Detected: {font_name} (confidence: {confidence:.2%})")

        return font_name

    def detect_top_k(self, image_path: str, k: int = 5) -> list[tuple[str, float]]:
        """
        Detect top-k most likely font names from image.

        Args:
            image_path: Path to image file containing text
            k: Number of top predictions to return

        Returns:
            List of (font_name, confidence) tuples, sorted by confidence
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load model if not already loaded
        self._load_model()
        assert self._processor is not None and self._model is not None  # type narrowing

        # Prepare image
        img = self._prepare_image(image_path)

        # Preprocess image
        inputs = self._processor(img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]

            # Get top-k predictions
            top_k_probs, top_k_indices = torch.topk(probs, k=min(k, len(probs)))

        # Build results
        results = []
        for prob, idx in zip(top_k_probs.tolist(), top_k_indices.tolist()):
            font_name: str = self._model.config.id2label[idx]
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
