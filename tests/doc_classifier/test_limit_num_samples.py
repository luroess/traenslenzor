"""Test limit_num_samples functionality in DocDataModule."""

from unittest.mock import Mock, patch

import pytest

from traenslenzor.doc_classifier.data_handling.huggingface_rvl_cdip_ds import RVLCDIPConfig
from traenslenzor.doc_classifier.lightning.lit_datamodule import (
    DocDataModule,
    DocDataModuleConfig,
)
from traenslenzor.doc_classifier.utils import Stage


class TestLimitNumSamples:
    """Test suite for limit_num_samples feature."""

    def test_no_limit_loads_full_dataset(self):
        """Test that without limit_num_samples, full dataset is loaded."""
        config = DocDataModuleConfig(limit_num_samples=None)
        datamodule = DocDataModule(config)

        # Mock the RVLCDIPConfig.setup_target to return a fake dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)

        with patch.object(RVLCDIPConfig, "setup_target", return_value=mock_dataset) as mock_setup:
            # Ensure dataset
            datamodule._ensure_dataset(Stage.TRAIN)

            # Verify setup_target was called
            assert mock_setup.called

            # Verify no select was called (full dataset)
            assert not mock_dataset.select.called

            # Verify dataset is set
            assert datamodule._train_ds == mock_dataset

    def test_absolute_limit_with_int(self):
        """Test limiting dataset to absolute number of samples."""
        config = DocDataModuleConfig(limit_num_samples=100)
        datamodule = DocDataModule(config)

        # Mock dataset with 1000 samples
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)
        mock_limited = Mock()
        mock_dataset.select = Mock(return_value=mock_limited)

        with patch.object(RVLCDIPConfig, "setup_target", return_value=mock_dataset):
            datamodule._ensure_dataset(Stage.TRAIN)

            # Verify select was called with range(100)
            mock_dataset.select.assert_called_once()
            call_args = mock_dataset.select.call_args[0][0]
            assert list(call_args) == list(range(100))

            # Verify limited dataset is set
            assert datamodule._train_ds == mock_limited

    def test_fractional_limit_with_float(self):
        """Test limiting dataset to fraction of samples."""
        config = DocDataModuleConfig(limit_num_samples=0.1)  # 10%
        datamodule = DocDataModule(config)

        # Mock dataset with 1000 samples
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)
        mock_limited = Mock()
        mock_dataset.select = Mock(return_value=mock_limited)

        with patch.object(RVLCDIPConfig, "setup_target", return_value=mock_dataset):
            datamodule._ensure_dataset(Stage.TRAIN)

            # Verify select was called with range(100) (10% of 1000)
            mock_dataset.select.assert_called_once()
            call_args = mock_dataset.select.call_args[0][0]
            assert list(call_args) == list(range(100))

            assert datamodule._train_ds == mock_limited

    def test_limit_larger_than_dataset(self):
        """Test that limit larger than dataset size uses full dataset."""
        config = DocDataModuleConfig(limit_num_samples=2000)
        datamodule = DocDataModule(config)

        # Mock dataset with 1000 samples
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)
        mock_limited = Mock()
        mock_dataset.select = Mock(return_value=mock_limited)

        with patch.object(RVLCDIPConfig, "setup_target", return_value=mock_dataset):
            datamodule._ensure_dataset(Stage.TRAIN)

            # Verify select was called with range(1000) (full dataset)
            mock_dataset.select.assert_called_once()
            call_args = mock_dataset.select.call_args[0][0]
            assert list(call_args) == list(range(1000))

    def test_invalid_float_limit_raises_error(self):
        """Test that invalid float values raise ValueError."""
        # Float > 1.0
        with pytest.raises(ValueError, match="must be in \\(0, 1\\]"):
            config = DocDataModuleConfig(limit_num_samples=1.5)
            datamodule = DocDataModule(config)
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=1000)

            with patch.object(RVLCDIPConfig, "setup_target", return_value=mock_dataset):
                datamodule._ensure_dataset(Stage.TRAIN)

        # Float <= 0
        with pytest.raises(ValueError, match="must be in \\(0, 1\\]"):
            config = DocDataModuleConfig(limit_num_samples=0.0)
            datamodule = DocDataModule(config)
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=1000)

            with patch.object(RVLCDIPConfig, "setup_target", return_value=mock_dataset):
                datamodule._ensure_dataset(Stage.TRAIN)

    def test_zero_limit_raises_error(self):
        """Test that zero limit raises ValueError."""
        with pytest.raises(ValueError, match="Invalid target_size"):
            config = DocDataModuleConfig(limit_num_samples=0)
            datamodule = DocDataModule(config)
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=1000)

            with patch.object(RVLCDIPConfig, "setup_target", return_value=mock_dataset):
                datamodule._ensure_dataset(Stage.TRAIN)

    def test_negative_limit_raises_error(self):
        """Test that negative limit raises ValueError."""
        with pytest.raises(ValueError, match="Invalid target_size"):
            config = DocDataModuleConfig(limit_num_samples=-10)
            datamodule = DocDataModule(config)
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=1000)

            with patch.object(RVLCDIPConfig, "setup_target", return_value=mock_dataset):
                datamodule._ensure_dataset(Stage.TRAIN)

    def test_limit_applies_to_all_splits(self):
        """Test that limit_num_samples applies to train, val, and test."""
        config = DocDataModuleConfig(limit_num_samples=50)
        datamodule = DocDataModule(config)

        for stage, attr_name in [
            (Stage.TRAIN, "_train_ds"),
            (Stage.VAL, "_val_ds"),
            (Stage.TEST, "_test_ds"),
        ]:
            # Mock dataset
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=1000)
            mock_limited = Mock()
            mock_dataset.select = Mock(return_value=mock_limited)

            with patch.object(RVLCDIPConfig, "setup_target", return_value=mock_dataset):
                datamodule._ensure_dataset(stage)

                # Verify select was called with range(50)
                mock_dataset.select.assert_called_once()
                call_args = mock_dataset.select.call_args[0][0]
                assert list(call_args) == list(range(50))

                # Verify limited dataset is set
                assert getattr(datamodule, attr_name) == mock_limited

    def test_apply_sample_limit_logging(self):
        """Test that _apply_sample_limit logs the limitation."""
        config = DocDataModuleConfig(limit_num_samples=100, verbose=True)
        datamodule = DocDataModule(config)

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)
        mock_limited = Mock()
        mock_dataset.select = Mock(return_value=mock_limited)

        result = datamodule._apply_sample_limit(mock_dataset, Stage.TRAIN)

        # Verify result
        assert result == mock_limited

        # Verify select was called correctly
        mock_dataset.select.assert_called_once()
        call_args = mock_dataset.select.call_args[0][0]
        assert list(call_args) == list(range(100))

    def test_fractional_limit_rounding(self):
        """Test that fractional limits round down correctly."""
        config = DocDataModuleConfig(limit_num_samples=0.15)  # 15%
        datamodule = DocDataModule(config)

        # Mock dataset with 1000 samples
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)
        mock_limited = Mock()
        mock_dataset.select = Mock(return_value=mock_limited)

        with patch.object(RVLCDIPConfig, "setup_target", return_value=mock_dataset):
            datamodule._ensure_dataset(Stage.TRAIN)

            # Verify select was called with range(150) (15% of 1000)
            mock_dataset.select.assert_called_once()
            call_args = mock_dataset.select.call_args[0][0]
            assert list(call_args) == list(range(150))

    def test_small_fractional_limit(self):
        """Test very small fractional limits work correctly."""
        config = DocDataModuleConfig(limit_num_samples=0.001)  # 0.1%
        datamodule = DocDataModule(config)

        # Mock dataset with 10000 samples
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10000)
        mock_limited = Mock()
        mock_dataset.select = Mock(return_value=mock_limited)

        with patch.object(RVLCDIPConfig, "setup_target", return_value=mock_dataset):
            datamodule._ensure_dataset(Stage.TRAIN)

            # Verify select was called with range(10) (0.1% of 10000)
            mock_dataset.select.assert_called_once()
            call_args = mock_dataset.select.call_args[0][0]
            assert list(call_args) == list(range(10))

    def test_dataset_cached_after_first_load(self):
        """Test that dataset is only loaded once and cached."""
        config = DocDataModuleConfig(limit_num_samples=100)
        datamodule = DocDataModule(config)

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)
        mock_limited = Mock()
        mock_dataset.select = Mock(return_value=mock_limited)

        with patch.object(RVLCDIPConfig, "setup_target", return_value=mock_dataset) as mock_setup:
            # First call
            datamodule._ensure_dataset(Stage.TRAIN)
            assert mock_setup.call_count == 1

            # Second call - should use cached dataset
            datamodule._ensure_dataset(Stage.TRAIN)
            assert mock_setup.call_count == 1  # Still 1, not called again

            # Verify the same limited dataset is used
            assert datamodule._train_ds == mock_limited
