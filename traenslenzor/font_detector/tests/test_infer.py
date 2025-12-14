"""Test inference module."""

import json

import numpy as np
import pytest

from traenslenzor.font_detector.font_size_model.features import FeatureNormalizer
from traenslenzor.font_detector.font_size_model.infer import FontSizeEstimator
from traenslenzor.font_detector.font_size_model.model import FontSizeRegressorMLP


@pytest.fixture
def dummy_checkpoint(tmp_path):
    """Create a dummy checkpoint for testing."""
    # Create checkpoint directory
    checkpoint_dir = tmp_path / "checkpoints" / "TestFont"
    checkpoint_dir.mkdir(parents=True)

    # Create dummy model
    model = FontSizeRegressorMLP(input_dim=36, hidden1=64, hidden2=32)
    model.save(str(checkpoint_dir / "best.json"))

    # Create dummy normalizer
    mean = np.random.randn(36).astype(np.float32)
    std = np.ones(36, dtype=np.float32)
    normalizer = FeatureNormalizer(mean, std)
    normalizer.save(str(checkpoint_dir / "norm.json"))

    return tmp_path / "checkpoints"


class TestFontSizeEstimator:
    """Test FontSizeEstimator class."""

    def test_init(self, tmp_path):
        """Test initialization."""
        estimator = FontSizeEstimator(str(tmp_path))
        assert estimator.checkpoints_dir == tmp_path
        assert len(estimator.models) == 0
        assert len(estimator.normalizers) == 0

    def test_lazy_loading(self, dummy_checkpoint):
        """Test that models are loaded lazily."""
        estimator = FontSizeEstimator(str(dummy_checkpoint))

        # Should not be loaded yet
        assert len(estimator.models) == 0

        # Estimate will trigger loading
        result = estimator.estimate((100, 50), "Hello", "TestFont")

        # Now should be loaded
        assert "TestFont" in estimator.models
        assert "TestFont" in estimator.normalizers
        assert isinstance(result, float)

    def test_estimate(self, dummy_checkpoint):
        """Test font size estimation."""
        estimator = FontSizeEstimator(str(dummy_checkpoint))

        # Run estimation
        result = estimator.estimate((100, 50), "Hello World", "TestFont")

        # Check result
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_missing_model(self, tmp_path):
        """Test error when model is missing."""
        estimator = FontSizeEstimator(str(tmp_path))

        with pytest.raises(FileNotFoundError, match="Model not found"):
            estimator.estimate((100, 50), "Hello", "NonexistentFont")

    def test_list_available_fonts(self, dummy_checkpoint):
        """Test listing available fonts."""
        estimator = FontSizeEstimator(str(dummy_checkpoint))

        fonts = estimator.list_available_fonts()
        assert "TestFont" in fonts
        assert isinstance(fonts, list)

    def test_list_available_fonts_empty(self, tmp_path):
        """Test listing fonts when directory is empty."""
        estimator = FontSizeEstimator(str(tmp_path / "empty"))
        fonts = estimator.list_available_fonts()
        assert fonts == []

    def test_multiple_fonts(self, tmp_path):
        """Test with multiple fonts."""
        # Create multiple checkpoints
        for font_name in ["Font1", "Font2", "Font3"]:
            checkpoint_dir = tmp_path / "checkpoints" / font_name
            checkpoint_dir.mkdir(parents=True)

            model = FontSizeRegressorMLP(input_dim=36, hidden1=64, hidden2=32)
            model.save(str(checkpoint_dir / "best.json"))

            mean = np.random.randn(36).astype(np.float32)
            std = np.ones(36, dtype=np.float32)
            normalizer = FeatureNormalizer(mean, std)
            normalizer.save(str(checkpoint_dir / "norm.json"))

        estimator = FontSizeEstimator(str(tmp_path / "checkpoints"))

        # List fonts
        fonts = estimator.list_available_fonts()
        assert len(fonts) == 3
        assert set(fonts) == {"Font1", "Font2", "Font3"}

        # Estimate with each font
        for font_name in fonts:
            result = estimator.estimate((100, 50), "Test", font_name)
            assert isinstance(result, float)

    def test_caching(self, dummy_checkpoint):
        """Test that models are cached after first load."""
        estimator = FontSizeEstimator(str(dummy_checkpoint))

        # First estimation
        _ = estimator.estimate((100, 50), "Hello", "TestFont")
        model1 = estimator.models["TestFont"]

        # Second estimation
        _ = estimator.estimate((100, 50), "Hello", "TestFont")
        model2 = estimator.models["TestFont"]

        # Should be the same model instance (cached)
        assert model1 is model2

    def test_different_inputs(self, dummy_checkpoint):
        """Test with different input combinations."""
        estimator = FontSizeEstimator(str(dummy_checkpoint))

        test_cases = [
            ((100, 50), "Hello"),
            ((200, 100), "World"),
            ((50, 25), "A"),
            ((500, 200), "The quick brown fox jumps over the lazy dog"),
            ((10, 5), ""),  # empty text
            ((1000, 1000), "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        ]

        for box_size, text in test_cases:
            result = estimator.estimate(box_size, text, "TestFont")
            assert isinstance(result, float)
            assert np.isfinite(result)


class TestEstimateFontSizeFunction:
    """Test estimate_font_size wrapper function."""

    def test_with_font_name(self, dummy_checkpoint):
        """Test with font name provided."""
        from traenslenzor.font_detector.font_size_model.infer import estimate_font_size

        result = estimate_font_size(
            text_box_size=(100, 50),
            text="Hello",
            font_name="TestFont",
            checkpoints_dir=str(dummy_checkpoint),
        )

        assert "font_size_pt" in result
        assert isinstance(result["font_size_pt"], float)

    def test_without_font_name(self, dummy_checkpoint):
        """Test without font name (should raise error for now)."""
        from traenslenzor.font_detector.font_size_model.infer import estimate_font_size

        with pytest.raises(ValueError, match="font_name is required"):
            estimate_font_size(
                text_box_size=(100, 50),
                text="Hello",
                font_name=None,
                checkpoints_dir=str(dummy_checkpoint),
            )

    def test_return_format(self, dummy_checkpoint):
        """Test that return format matches schema."""
        from traenslenzor.font_detector.font_size_model.infer import estimate_font_size

        result = estimate_font_size(
            text_box_size=(100, 50),
            text="Hello",
            font_name="TestFont",
            checkpoints_dir=str(dummy_checkpoint),
        )

        # Should be a dict with exactly one key
        assert isinstance(result, dict)
        assert "font_size_pt" in result
        assert len(result) == 1

        # Value should be numeric
        assert isinstance(result["font_size_pt"], (int, float))

        # Should be JSON serializable
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed == result
