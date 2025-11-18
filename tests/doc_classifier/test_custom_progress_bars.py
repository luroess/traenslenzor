"""Test custom progress bars that hide v_num."""

from unittest.mock import Mock

from traenslenzor.doc_classifier.lightning.lit_trainer_callbacks import (
    CustomRichProgressBar,
    CustomTQDMProgressBar,
)
from traenslenzor.doc_classifier.utils.schemas import Metric


class TestCustomProgressBars:
    """Test suite for custom progress bar implementations."""

    def test_custom_tqdm_removes_v_num(self):
        """Test that CustomTQDMProgressBar removes v_num from metrics."""
        progress_bar = CustomTQDMProgressBar()

        # Mock the parent's get_metrics to return metrics with v_num
        mock_metrics = {
            "train/loss_step": 1.010,
            "v_num": "2hcj",
            "train/loss_epoch": 0.846,
        }

        # Patch super().get_metrics() to return our mock metrics
        original_get_metrics = progress_bar.__class__.__bases__[0].get_metrics
        progress_bar.__class__.__bases__[0].get_metrics = (
            lambda self, *args, **kwargs: mock_metrics.copy()
        )

        try:
            result = progress_bar.get_metrics()

            # v_num should be removed
            assert "v_num" not in result
            # Other metrics should remain
            assert "train/loss_step" in result
            assert "train/loss_epoch" in result
            assert result["train/loss_step"] == 1.010
            assert result["train/loss_epoch"] == 0.846
        finally:
            # Restore original method
            progress_bar.__class__.__bases__[0].get_metrics = original_get_metrics

    def test_custom_rich_removes_v_num(self):
        """Test that CustomRichProgressBar removes v_num from metrics."""
        progress_bar = CustomRichProgressBar()

        # Mock the parent's get_metrics to return metrics with v_num
        mock_trainer = Mock()
        mock_module = Mock()

        mock_metrics = {
            "train/loss_step": 1.010,
            "v_num": "2hcj",
            "train/loss_epoch": 0.846,
        }

        # Patch super().get_metrics() to return our mock metrics
        original_get_metrics = progress_bar.__class__.__bases__[0].get_metrics
        progress_bar.__class__.__bases__[0].get_metrics = (
            lambda self, *args, **kwargs: mock_metrics.copy()
        )

        try:
            result = progress_bar.get_metrics(mock_trainer, mock_module)

            # v_num should be removed
            assert "v_num" not in result
            # Other metrics should remain
            assert "train/loss_step" in result
            assert "train/loss_epoch" in result
            assert result["train/loss_step"] == 1.010
            assert result["train/loss_epoch"] == 0.846
        finally:
            # Restore original method
            progress_bar.__class__.__bases__[0].get_metrics = original_get_metrics

    def test_custom_tqdm_handles_missing_v_num(self):
        """Test that CustomTQDMProgressBar gracefully handles missing v_num."""
        progress_bar = CustomTQDMProgressBar()

        # Mock metrics without v_num
        mock_metrics = {
            "train/loss": 0.5,
            "val/acc": 0.85,
        }

        original_get_metrics = progress_bar.__class__.__bases__[0].get_metrics
        progress_bar.__class__.__bases__[0].get_metrics = (
            lambda self, *args, **kwargs: mock_metrics.copy()
        )

        try:
            result = progress_bar.get_metrics()

            # Should work fine even without v_num
            assert len(result) == 2
            assert result["train/loss"] == 0.5
            assert result["val/acc"] == 0.85
        finally:
            progress_bar.__class__.__bases__[0].get_metrics = original_get_metrics

    def test_custom_rich_handles_missing_v_num(self):
        """Test that CustomRichProgressBar gracefully handles missing v_num."""
        progress_bar = CustomRichProgressBar()

        mock_trainer = Mock()
        mock_module = Mock()

        # Mock metrics without v_num
        mock_metrics = {
            "train/loss": 0.5,
            "val/acc": 0.85,
        }

        original_get_metrics = progress_bar.__class__.__bases__[0].get_metrics
        progress_bar.__class__.__bases__[0].get_metrics = (
            lambda self, *args, **kwargs: mock_metrics.copy()
        )

        try:
            result = progress_bar.get_metrics(mock_trainer, mock_module)

            # Should work fine even without v_num
            assert len(result) == 2
            assert result["train/loss"] == 0.5
            assert result["val/acc"] == 0.85
        finally:
            progress_bar.__class__.__bases__[0].get_metrics = original_get_metrics

    def test_metric_enum_integration(self):
        """Test that progress bars work correctly with Metric enum."""
        progress_bar = CustomTQDMProgressBar()

        # Mock metrics using Metric enum (as they would appear in real usage)
        mock_metrics = {
            Metric.TRAIN_LOSS: 1.234,
            Metric.TRAIN_ACCURACY: 0.567,
            Metric.VAL_LOSS: 0.890,
            "v_num": "2hcj",  # This should be removed
        }

        original_get_metrics = progress_bar.__class__.__bases__[0].get_metrics
        progress_bar.__class__.__bases__[0].get_metrics = (
            lambda self, *args, **kwargs: mock_metrics.copy()
        )

        try:
            result = progress_bar.get_metrics()

            # v_num should be removed
            assert "v_num" not in result

            # Metric enum values should be preserved
            assert Metric.TRAIN_LOSS in result
            assert Metric.TRAIN_ACCURACY in result
            assert Metric.VAL_LOSS in result

            # Values should match
            assert result[Metric.TRAIN_LOSS] == 1.234
            assert result[Metric.TRAIN_ACCURACY] == 0.567
            assert result[Metric.VAL_LOSS] == 0.890
        finally:
            progress_bar.__class__.__bases__[0].get_metrics = original_get_metrics
