"""Test confusion matrix logging to WandB."""

from unittest.mock import Mock, patch

import torch
import wandb

from traenslenzor.doc_classifier.lightning.lit_module import (
    DocClassifierConfig,
    DocClassifierModule,
)
from traenslenzor.doc_classifier.utils.schemas import Metric


class TestConfusionMatrixLogging:
    """Test suite for confusion matrix logging functionality."""

    def test_get_class_names_from_dataset(self):
        """Test that class names can be retrieved from the dataset."""
        # Create a mock trainer with datamodule
        config = DocClassifierConfig(num_classes=3)
        module = DocClassifierModule(config)

        # Mock the trainer and datamodule
        mock_trainer = Mock()
        mock_datamodule = Mock()
        mock_val_ds = Mock()

        # Mock dataset features with class names
        mock_label_feature = Mock()
        mock_label_feature.names = ["class_a", "class_b", "class_c"]
        mock_val_ds.features = {"label": mock_label_feature}
        mock_datamodule.val_ds = mock_val_ds
        mock_trainer.datamodule = mock_datamodule

        module.trainer = mock_trainer

        # Get class names
        class_names = module._get_class_names()

        assert class_names is not None
        assert len(class_names) == 3
        assert class_names == ["class_a", "class_b", "class_c"]

    def test_get_class_names_returns_none_when_unavailable(self):
        """Test that _get_class_names returns None when dataset is unavailable."""
        config = DocClassifierConfig(num_classes=3)
        module = DocClassifierModule(config)

        # Mock trainer with no datamodule
        mock_trainer = Mock()
        mock_trainer.datamodule = None
        module.trainer = mock_trainer

        # Get class names
        class_names = module._get_class_names()

        assert class_names is None

    def test_on_validation_epoch_end_with_wandb_logger(self):
        """Test that confusion matrix plot and table are logged to WandB correctly."""
        config = DocClassifierConfig(num_classes=3)
        module = DocClassifierModule(config)

        # Mock the trainer
        mock_trainer = Mock()
        mock_datamodule = Mock()
        mock_val_ds = Mock()

        # Mock dataset features with class names
        mock_label_feature = Mock()
        mock_label_feature.names = ["class_a", "class_b", "class_c"]
        mock_val_ds.features = {"label": mock_label_feature}
        mock_datamodule.val_ds = mock_val_ds
        mock_trainer.datamodule = mock_datamodule
        mock_trainer.current_epoch = 5  # Mock current_epoch on trainer

        # Mock the WandB logger
        mock_logger = Mock()
        mock_experiment = Mock()
        mock_logger.experiment = mock_experiment
        mock_trainer.logger = mock_logger

        module.trainer = mock_trainer

        # Update confusion matrix with some fake predictions
        fake_preds = torch.tensor([0, 1, 2, 0, 1, 2])
        fake_targets = torch.tensor([0, 1, 2, 1, 2, 0])
        module.confusion_matrix.update(fake_preds, fake_targets)

        # Call on_validation_epoch_end
        with patch("matplotlib.pyplot.close") as mock_close:
            module.on_validation_epoch_end()

            # Verify WandB logging was called twice (plot + table)
            assert mock_experiment.log.call_count == 2

            # Check first call (plot)
            first_call_args = mock_experiment.log.call_args_list[0][0][0]
            assert Metric.VAL_CONFUSION_MATRIX in first_call_args
            assert "epoch" in first_call_args
            assert first_call_args["epoch"] == 5
            assert isinstance(first_call_args[Metric.VAL_CONFUSION_MATRIX], wandb.Image)

            # Check second call (table)
            second_call_args = mock_experiment.log.call_args_list[1][0][0]
            assert f"{Metric.VAL_CONFUSION_MATRIX}_table" in second_call_args
            assert "epoch" in second_call_args
            assert second_call_args["epoch"] == 5
            assert isinstance(second_call_args[f"{Metric.VAL_CONFUSION_MATRIX}_table"], wandb.Table)

            # Verify figure was closed
            assert mock_close.called

    def test_on_validation_epoch_end_without_logger(self):
        """Test that on_validation_epoch_end handles missing logger gracefully."""
        config = DocClassifierConfig(num_classes=3)
        module = DocClassifierModule(config)

        # Mock trainer with no logger
        mock_trainer = Mock()
        mock_trainer.logger = None
        module.trainer = mock_trainer

        # Update confusion matrix
        fake_preds = torch.tensor([0, 1, 2])
        fake_targets = torch.tensor([0, 1, 2])
        module.confusion_matrix.update(fake_preds, fake_targets)

        # Should not raise an error
        module.on_validation_epoch_end()

    def test_confusion_matrix_reset_after_epoch(self):
        """Test that confusion matrix is reset after each validation epoch."""
        config = DocClassifierConfig(num_classes=3)
        module = DocClassifierModule(config)

        # Mock the trainer with logger
        mock_trainer = Mock()
        mock_logger = Mock()
        mock_logger.experiment = Mock()
        mock_trainer.logger = mock_logger
        mock_trainer.datamodule = None
        module.trainer = mock_trainer

        # Update confusion matrix
        fake_preds = torch.tensor([0, 1, 2])
        fake_targets = torch.tensor([0, 1, 2])
        module.confusion_matrix.update(fake_preds, fake_targets)

        # Verify confusion matrix has values
        confmat_before = module.confusion_matrix.compute()
        assert confmat_before.sum() > 0

        # Call on_validation_epoch_end
        with patch("matplotlib.pyplot.close"):
            module.on_validation_epoch_end()

        # Update and compute again - should start fresh
        module.confusion_matrix.update(fake_preds, fake_targets)
        confmat_after = module.confusion_matrix.compute()

        # After reset, the confusion matrix should only contain the new batch
        assert confmat_after.sum() == len(fake_preds)

    def test_log_confusion_matrix_plot(self):
        """Test _log_confusion_matrix_plot method."""
        config = DocClassifierConfig(num_classes=3)
        module = DocClassifierModule(config)

        # Mock the trainer with WandB logger
        mock_trainer = Mock()
        mock_logger = Mock()
        mock_experiment = Mock()
        mock_logger.experiment = mock_experiment
        mock_trainer.logger = mock_logger
        mock_trainer.current_epoch = 2  # Mock current_epoch on trainer
        module.trainer = mock_trainer

        # Create a fake confusion matrix
        confmat = torch.tensor([[2, 0, 1], [1, 3, 0], [0, 1, 2]], dtype=torch.float32)
        class_names = ["class_a", "class_b", "class_c"]

        # Call the method
        with patch("matplotlib.pyplot.close") as mock_close:
            module._log_confusion_matrix_plot(confmat, class_names)

            # Verify WandB logging was called
            assert mock_experiment.log.called
            logged_data = mock_experiment.log.call_args[0][0]
            assert Metric.VAL_CONFUSION_MATRIX in logged_data
            assert isinstance(logged_data[Metric.VAL_CONFUSION_MATRIX], wandb.Image)
            assert logged_data["epoch"] == 2

            # Verify figure was closed
            assert mock_close.called

    def test_log_confusion_matrix_table(self):
        """Test _log_confusion_matrix_table method."""
        config = DocClassifierConfig(num_classes=3)
        module = DocClassifierModule(config)

        # Mock the trainer with WandB logger
        mock_trainer = Mock()
        mock_logger = Mock()
        mock_experiment = Mock()
        mock_logger.experiment = mock_experiment
        mock_trainer.logger = mock_logger
        mock_trainer.current_epoch = 2  # Mock current_epoch on trainer
        module.trainer = mock_trainer

        # Create a fake confusion matrix
        confmat = torch.tensor([[2, 0, 1], [1, 3, 0], [0, 1, 2]], dtype=torch.float32)
        class_names = ["class_a", "class_b", "class_c"]

        # Call the method
        module._log_confusion_matrix_table(confmat, class_names)

        # Verify WandB logging was called
        assert mock_experiment.log.called
        logged_data = mock_experiment.log.call_args[0][0]
        assert f"{Metric.VAL_CONFUSION_MATRIX}_table" in logged_data
        assert isinstance(logged_data[f"{Metric.VAL_CONFUSION_MATRIX}_table"], wandb.Table)
        assert logged_data["epoch"] == 2

        # Verify table structure
        table = logged_data[f"{Metric.VAL_CONFUSION_MATRIX}_table"]
        assert table.columns == ["Actual", "class_a", "class_b", "class_c"]
        assert len(table.data) == 3

    def test_log_confusion_matrix_table_without_class_names(self):
        """Test _log_confusion_matrix_table uses default names when class_names is None."""
        config = DocClassifierConfig(num_classes=3)
        module = DocClassifierModule(config)

        # Mock the trainer with WandB logger
        mock_trainer = Mock()
        mock_logger = Mock()
        mock_experiment = Mock()
        mock_logger.experiment = mock_experiment
        mock_trainer.logger = mock_logger
        mock_trainer.current_epoch = 1  # Mock current_epoch on trainer
        module.trainer = mock_trainer

        # Create a fake confusion matrix
        confmat = torch.tensor([[2, 0, 1], [1, 3, 0], [0, 1, 2]], dtype=torch.float32)

        # Call the method with None class_names
        module._log_confusion_matrix_table(confmat, class_names=None)

        # Verify WandB logging was called
        assert mock_experiment.log.called
        logged_data = mock_experiment.log.call_args[0][0]

        # Verify table uses default class names
        table = logged_data[f"{Metric.VAL_CONFUSION_MATRIX}_table"]
        assert table.columns == ["Actual", "Class_0", "Class_1", "Class_2"]
