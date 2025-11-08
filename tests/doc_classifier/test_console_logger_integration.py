"""Unit tests for Console integration with PyTorch Lightning loggers."""

from unittest.mock import MagicMock

import pytest

from traenslenzor.doc_classifier.utils import Console


class TestConsoleLoggerIntegration:
    """Test suite for Console integration with PyTorch Lightning loggers."""

    def test_integrate_with_logger_sets_attributes(self):
        """Test that integrate_with_logger correctly sets logger and global_step."""
        console = Console()
        mock_logger = MagicMock()
        global_step = 42

        result = console.integrate_with_logger(mock_logger, global_step)

        assert console._pl_logger is mock_logger
        assert console._global_step == global_step
        assert result is console  # Check method chaining

    def test_update_global_step(self):
        """Test that update_global_step updates the step counter."""
        console = Console()
        console._global_step = 10

        result = console.update_global_step(100)

        assert console._global_step == 100
        assert result is console  # Check method chaining

    def test_log_without_logger_does_not_fail(self):
        """Test that logging without a logger integration works normally."""
        console = Console.with_prefix("Test")
        console.set_verbose(True)

        # Should not raise any exceptions
        console.log("Test message")
        console.warn("Warning message")
        console.error("Error message")
        console.dbg("Debug message")

    def test_wandb_logger_with_log_text_method(self):
        """Test logging to WandB logger using log_text method."""
        console = Console.with_prefix("Training", "forward")
        mock_logger = MagicMock()
        mock_logger.log_text = MagicMock()

        console.integrate_with_logger(mock_logger, global_step=5)
        console.log("Test info message")

        # Verify log_text was called with correct parameters
        mock_logger.log_text.assert_called_once_with(
            key="Training/forward/info",
            columns=["message"],
            data=[["Test info message"]],
            step=5,
        )

    def test_wandb_experiment_log_method(self):
        """Test logging via WandB experiment object's log method."""
        console = Console.with_prefix("Model")
        mock_logger = MagicMock(spec=["experiment"])  # No log_text method
        mock_experiment = MagicMock()
        mock_experiment.log = MagicMock()
        mock_logger.experiment = mock_experiment

        console.integrate_with_logger(mock_logger, global_step=10)
        console.warn("Warning message")

        # Verify experiment.log was called
        mock_experiment.log.assert_called_once_with(
            {"Model/warning": "Warning message"},
            step=10,
        )

    def test_tensorboard_logger_add_text_method(self):
        """Test logging to TensorBoard logger using add_text method."""
        console = Console.with_prefix("Validation")
        mock_logger = MagicMock(spec=[])  # No log_text method
        mock_experiment = MagicMock(spec=["add_text"])  # Only add_text
        mock_experiment.add_text = MagicMock()
        mock_logger.experiment = mock_experiment

        console.integrate_with_logger(mock_logger, global_step=15)
        console.error("Error occurred")

        # Verify add_text was called
        mock_experiment.add_text.assert_called_once_with(
            "Validation/error",
            "Error occurred",
            15,
        )

    def test_logging_all_levels(self):
        """Test that all log levels (info, warning, error, debug) are logged."""
        console = Console.with_prefix("Test")
        console.set_debug(True)  # Enable debug logging

        mock_logger = MagicMock()
        mock_logger.log_text = MagicMock()

        console.integrate_with_logger(mock_logger, global_step=1)

        console.log("Info message")
        console.warn("Warning message")
        console.error("Error message")
        console.dbg("Debug message")

        # Verify all 4 log levels were called
        assert mock_logger.log_text.call_count == 4

        # Check the keys used
        calls = mock_logger.log_text.call_args_list
        assert calls[0][1]["key"] == "Test/info"
        assert calls[1][1]["key"] == "Test/warning"
        assert calls[2][1]["key"] == "Test/error"
        assert calls[3][1]["key"] == "Test/debug"

    def test_default_prefix_when_none_set(self):
        """Test that 'Console' is used as prefix when none is set."""
        console = Console()
        mock_logger = MagicMock()
        mock_logger.log_text = MagicMock()

        console.integrate_with_logger(mock_logger)
        console.log("Message")

        mock_logger.log_text.assert_called_once()
        assert mock_logger.log_text.call_args[1]["key"] == "Console/info"

    def test_global_step_updates_correctly(self):
        """Test that global_step is used correctly in logging."""
        console = Console.with_prefix("Step")
        mock_logger = MagicMock()
        mock_logger.log_text = MagicMock()

        console.integrate_with_logger(mock_logger, global_step=0)

        console.log("Step 0")
        assert mock_logger.log_text.call_args[1]["step"] == 0

        console.update_global_step(10)
        console.log("Step 10")
        assert mock_logger.log_text.call_args[1]["step"] == 10

        console.update_global_step(100)
        console.log("Step 100")
        assert mock_logger.log_text.call_args[1]["step"] == 100

    def test_exception_handling_does_not_break_training(self):
        """Test that exceptions in logging don't break the console."""
        console = Console.with_prefix("Robust")
        mock_logger = MagicMock()
        mock_logger.log_text = MagicMock(side_effect=Exception("Logger error"))

        console.integrate_with_logger(mock_logger)

        # Should not raise exception
        console.log("This should not crash")
        console.warn("This should not crash either")

    def test_verbose_false_skips_logging(self):
        """Test that verbose=False prevents logging to logger."""
        console = Console.with_prefix("Silent")
        console.set_verbose(False)

        mock_logger = MagicMock()
        mock_logger.log_text = MagicMock()

        console.integrate_with_logger(mock_logger)
        console.log("This should not be logged")

        # log_text should not be called because verbose=False
        mock_logger.log_text.assert_not_called()

    def test_debug_false_skips_debug_logging(self):
        """Test that debug=False prevents debug messages from being logged."""
        console = Console.with_prefix("NoDebug")
        console.set_debug(False)

        mock_logger = MagicMock()
        mock_logger.log_text = MagicMock()

        console.integrate_with_logger(mock_logger)
        console.dbg("This debug message should not be logged")

        # log_text should not be called for debug when is_debug=False
        mock_logger.log_text.assert_not_called()

    def test_hierarchical_prefix_naming(self):
        """Test that multi-part prefixes create hierarchical names."""
        console = Console.with_prefix("Module", "Method", "SubContext")
        mock_logger = MagicMock()
        mock_logger.log_text = MagicMock()

        console.integrate_with_logger(mock_logger)
        console.log("Hierarchical message")

        # The prefix should maintain the hierarchy in the metric name
        called_key = mock_logger.log_text.call_args[1]["key"]
        assert called_key == "Module/Method/SubContext/info"

    def test_builder_pattern_chaining(self):
        """Test that the builder pattern works for fluent API."""
        mock_logger = MagicMock()
        mock_logger.log_text = MagicMock()

        # Chain all setup methods
        console = (
            Console.with_prefix("Chain", "Test")
            .set_verbose(True)
            .set_debug(True)
            .integrate_with_logger(mock_logger, global_step=42)
            .update_global_step(100)
        )

        console.log("Chained message")

        assert console._global_step == 100
        assert console.verbose is True
        assert console.is_debug is True
        mock_logger.log_text.assert_called_once()
        assert mock_logger.log_text.call_args[1]["step"] == 100

    def test_prefix_management_and_timestamp(self):
        console = Console()
        console.set_prefix("Outer", None, "Inner")
        assert console.prefix == "Outer::Inner"
        console.set_timestamp_display(True)
        assert console.show_timestamps is True

        console.set_prefix()
        assert console.prefix is None
        console.unset_prefix()
        assert console.prefix is None

    def test_plog_and_debug_require_flags(self, monkeypatch):
        console = Console.with_prefix("Verbose")
        captured: list[str] = []
        monkeypatch.setattr(Console, "print", lambda self, msg, **_: captured.append(msg))

        console.set_verbose(False)
        console.plog({"state": "hidden"})
        assert not captured

        console.set_debug(True)
        console.dbg("debug message")
        assert captured

    def test_internal_log_to_lightning_no_logger(self):
        console = Console()
        console._log_to_lightning("info", "message")  # Should silently no-op

    def test_update_global_step_returns_self(self):
        console = Console()
        assert console.update_global_step(10) is console


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
