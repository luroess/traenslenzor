"""Test feature extraction functions."""

import numpy as np
import pytest

from traenslenzor.font_detector.font_size_model.features import (
    FeatureNormalizer,
    extract_features,
    extract_letter_histogram,
    validate_features,
)


class TestLetterHistogram:
    """Test letter histogram extraction."""

    def test_simple_text(self):
        """Test with simple text."""
        hist = extract_letter_histogram("hello")

        # Check shape
        assert hist.shape == (26,)

        # Check sum
        assert np.abs(np.sum(hist) - 1.0) < 1e-6

        # Check counts: h=1, e=1, l=2, o=1 -> total=5
        expected = np.zeros(26)
        expected[ord("h") - ord("a")] = 1 / 5  # h
        expected[ord("e") - ord("a")] = 1 / 5  # e
        expected[ord("l") - ord("a")] = 2 / 5  # l (appears twice)
        expected[ord("o") - ord("a")] = 1 / 5  # o

        assert np.allclose(hist, expected)

    def test_empty_string(self):
        """Test with empty string."""
        hist = extract_letter_histogram("")
        assert hist.shape == (26,)
        assert np.all(hist == 0)

    def test_no_letters(self):
        """Test with no letters (only numbers/symbols)."""
        hist = extract_letter_histogram("123 !@#")
        assert hist.shape == (26,)
        assert np.all(hist == 0)

    def test_mixed_case(self):
        """Test with mixed case (should be case-insensitive)."""
        hist1 = extract_letter_histogram("Hello")
        hist2 = extract_letter_histogram("HELLO")
        hist3 = extract_letter_histogram("hello")

        assert np.allclose(hist1, hist2)
        assert np.allclose(hist1, hist3)

    def test_with_numbers_and_symbols(self):
        """Test with letters, numbers, and symbols."""
        hist = extract_letter_histogram("abc123!@#xyz")

        # Should only count letters: a, b, c, x, y, z (6 letters)
        expected = np.zeros(26)
        expected[0] = 1 / 6  # a
        expected[1] = 1 / 6  # b
        expected[2] = 1 / 6  # c
        expected[23] = 1 / 6  # x
        expected[24] = 1 / 6  # y
        expected[25] = 1 / 6  # z

        assert np.allclose(hist, expected)


class TestExtractFeatures:
    """Test feature extraction."""

    def test_basic_features(self):
        """Test basic feature extraction."""
        features = extract_features((100, 50), "Hello World")

        # Check shape (36 features: 10 basic + 26 letter hist)
        assert features.shape == (36,)

        # Check basic features
        assert features[0] == 100  # width
        assert features[1] == 50  # height
        assert features[2] == 11  # text length
        assert features[3] > 0  # char density

        # Check letter histogram sums to ~1
        letter_hist = features[10:]
        assert np.abs(np.sum(letter_hist) - 1.0) < 1e-6

    def test_invalid_box_size(self):
        """Test with invalid box size."""
        with pytest.raises(ValueError):
            extract_features((100,), "text")  # wrong length

        with pytest.raises(ValueError):
            extract_features((-100, 50), "text")  # negative width

        with pytest.raises(ValueError):
            extract_features((100, -50), "text")  # negative height

    def test_invalid_text(self):
        """Test with invalid text."""
        with pytest.raises(ValueError):
            extract_features((100, 50), 123)  # not a string

    def test_empty_text(self):
        """Test with empty text."""
        features = extract_features((100, 50), "")

        assert features[2] == 0  # text length
        assert np.all(features[10:] == 0)  # letter hist all zeros

    def test_no_letters(self):
        """Test with text containing no letters."""
        features = extract_features((100, 50), "123 !@#")

        assert features[2] == 7  # text length
        assert np.all(features[10:] == 0)  # letter hist all zeros


class TestFeatureNormalizer:
    """Test feature normalizer."""

    def test_normalize_denormalize(self):
        """Test normalization and denormalization."""
        # Create normalizer
        mean = np.array([10.0, 20.0, 30.0])
        std = np.array([2.0, 4.0, 6.0])
        normalizer = FeatureNormalizer(mean, std)

        # Test normalization
        features = np.array([12.0, 24.0, 36.0])
        normalized = normalizer.normalize(features)

        expected = (features - mean) / std
        assert np.allclose(normalized, expected)

        # Test denormalization
        denormalized = normalizer.denormalize(normalized)
        assert np.allclose(denormalized, features)

    def test_fit(self):
        """Test fitting normalizer from data."""
        # Create sample data
        data = [
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 4.0, 6.0]),
            np.array([3.0, 6.0, 9.0]),
        ]

        normalizer = FeatureNormalizer.fit(data)

        # Check mean and std
        expected_mean = np.array([2.0, 4.0, 6.0])
        assert np.allclose(normalizer.mean, expected_mean)

    def test_save_load(self, tmp_path):
        """Test saving and loading normalizer."""
        # Create normalizer
        mean = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        std = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        normalizer = FeatureNormalizer(mean, std)

        # Save
        save_path = tmp_path / "norm.json"
        normalizer.save(str(save_path))

        # Load
        loaded = FeatureNormalizer.load(str(save_path))

        # Check equality
        assert np.allclose(loaded.mean, normalizer.mean)
        assert np.allclose(loaded.std, normalizer.std)

    def test_zero_std_protection(self):
        """Test that zero std is handled."""
        mean = np.array([10.0, 20.0])
        std = np.array([2.0, 0.0])  # zero std

        normalizer = FeatureNormalizer(mean, std)

        # Should replace zero with 1.0
        assert normalizer.std[1] == 1.0

        # Normalization should not divide by zero
        features = np.array([12.0, 20.0])
        normalized = normalizer.normalize(features)
        assert np.all(np.isfinite(normalized))


class TestValidateFeatures:
    """Test feature validation."""

    def test_valid_features(self):
        """Test with valid features."""
        features = extract_features((100, 50), "Hello")
        assert validate_features(features)

    def test_wrong_type(self):
        """Test with wrong type."""
        with pytest.raises(ValueError, match="must be numpy array"):
            validate_features([1, 2, 3])

    def test_wrong_shape(self):
        """Test with wrong shape."""
        with pytest.raises(ValueError, match="must have shape"):
            validate_features(np.array([1, 2, 3]))

    def test_nan_values(self):
        """Test with NaN values."""
        features = np.ones(36)
        features[0] = np.nan

        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_features(features)

    def test_inf_values(self):
        """Test with Inf values."""
        features = np.ones(36)
        features[0] = np.inf

        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_features(features)
