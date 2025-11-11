"""Feature extraction for font size estimation."""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def extract_letter_histogram(text: str) -> np.ndarray:
    """
    Extract normalized letter distribution histogram over a-z.

    Args:
        text: Input text string

    Returns:
        26-dimensional array with normalized counts (sum=1), or zeros if no letters
    """
    # Initialize histogram for a-z
    hist = np.zeros(26, dtype=np.float32)

    # Count lowercase letters
    text_lower = text.lower()
    letter_count = 0

    for char in text_lower:
        if "a" <= char <= "z":
            idx = ord(char) - ord("a")
            hist[idx] += 1
            letter_count += 1

    # Normalize to sum=1 (or leave as zeros if no letters)
    if letter_count > 0:
        hist = (hist / letter_count).astype(np.float32)

    return hist


def extract_features(
    text_box_size: Tuple[float, float],
    text: str,
) -> np.ndarray:
    """
    Extract features for font size estimation.

    Features (28-dimensional):
    - width_px (1)
    - height_px (1)
    - text_len (1)
    - letter_histogram (26)
    - length_to_box_ratio (1) - additional derived feature

    Args:
        text_box_size: (width_px, height_px) tuple
        text: Text string

    Returns:
        Feature vector as numpy array (28-dimensional)

    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if not isinstance(text_box_size, (tuple, list)) or len(text_box_size) != 2:
        raise ValueError(f"text_box_size must be a 2-element tuple/list, got: {text_box_size}")

    width_px, height_px = text_box_size

    if width_px <= 0 or height_px <= 0:
        raise ValueError(f"Box dimensions must be positive, got: {text_box_size}")

    if not isinstance(text, str):
        raise ValueError(f"text must be a string, got: {type(text)}")

    # Extract features
    text_len = len(text)
    letter_hist = extract_letter_histogram(text)

    # Additional derived feature: character density
    text_len_float = float(text_len)
    box_area = width_px * height_px
    char_density = text_len_float / box_area if box_area > 0 else 0.0

    # Combine features
    features = np.array(
        [
            width_px,
            height_px,
            text_len_float,
            char_density,
        ],
        dtype=np.float32,
    )

    # Concatenate letter histogram
    features = np.concatenate([features, letter_hist])

    return features


class FeatureNormalizer:
    """Normalize features using stored mean and std."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        """
        Initialize normalizer.

        Args:
            mean: Mean values for each feature
            std: Standard deviation for each feature
        """
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

        # Avoid division by zero
        self.std = np.where(self.std < 1e-7, 1.0, self.std)

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features.

        Args:
            features: Feature vector

        Returns:
            Normalized feature vector
        """
        return (features - self.mean) / self.std

    def denormalize(self, features: np.ndarray) -> np.ndarray:
        """
        Denormalize features.

        Args:
            features: Normalized feature vector

        Returns:
            Original scale feature vector
        """
        return features * self.std + self.mean

    def save(self, path: str):
        """
        Save normalizer to JSON file.

        Args:
            path: Output file path
        """
        data = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FeatureNormalizer":
        """
        Load normalizer from JSON file.

        Args:
            path: Input file path

        Returns:
            FeatureNormalizer instance
        """
        with open(path, "r") as f:
            data = json.load(f)

        mean = np.array(data["mean"], dtype=np.float32)
        std = np.array(data["std"], dtype=np.float32)

        return cls(mean, std)

    @classmethod
    def fit(cls, features_list: List[np.ndarray]) -> "FeatureNormalizer":
        """
        Fit normalizer from list of feature vectors.

        Args:
            features_list: List of feature vectors

        Returns:
            Fitted FeatureNormalizer instance
        """
        features_array = np.array(features_list, dtype=np.float32)
        mean = np.mean(features_array, axis=0)
        std = np.std(features_array, axis=0)

        return cls(mean, std)


def validate_features(features: np.ndarray, expected_dim: int = 30) -> bool:
    """
    Validate feature vector.

    Args:
        features: Feature vector
        expected_dim: Expected dimensionality

    Returns:
        True if valid

    Raises:
        ValueError: If features are invalid
    """
    if not isinstance(features, np.ndarray):
        raise ValueError(f"Features must be numpy array, got: {type(features)}")

    if features.shape != (expected_dim,):
        raise ValueError(f"Features must have shape ({expected_dim},), got: {features.shape}")

    if not np.all(np.isfinite(features)):
        raise ValueError("Features contain NaN or Inf values")

    # Check letter histogram sums to ~1 (last 26 elements)
    letter_hist = features[-26:]
    hist_sum = np.sum(letter_hist)
    if hist_sum > 0 and not np.isclose(hist_sum, 1.0, atol=0.01):
        raise ValueError(f"Letter histogram should sum to 1.0, got: {hist_sum}")

    return True
