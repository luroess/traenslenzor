"""Unit tests for Console integration with PyTorch Lightning loggers."""

from unittest.mock import MagicMock

import pytest

from traenslenzor.doc_classifier.utils import Console


@pytest.fixture(autouse=True)
def reset_console_shared_state():
    """Ensure each test runs with a clean shared console state."""
    Console._shared_pl_logger = None
    Console._shared_global_step = 0
    yield
    Console._shared_pl_logger = None
    Console._shared_global_step = 0


class TestConsoleLoggerIntegration:
    """Test suite for Console integration with PyTorch Lightning loggers."""

    def test_integrate_with_logger_sets_attributes(self):
        """Test that integrate_with_logger correctly sets shared logger and global_step."""
        mock_logger = MagicMock()
        global_step = 42

        result = Console.integrate_with_logger(mock_logger, global_step)

        # Check shared class variables are set
        assert Console._shared_pl_logger is mock_logger
        assert Console._shared_global_step == global_step
        assert result is Console  # Check method chaining returns class

        # Verify any instance can access the shared state
        console = Console()
        assert console._pl_logger is mock_logger
        assert console._global_step == global_step

    def test_update_global_step(self):
        """Test that update_global_step updates the shared step counter."""
        Console._shared_global_step = 10

        result = Console.update_global_step(100)

        assert Console._shared_global_step == 100
        assert result is Console  # Check method chaining returns class

        # Verify any instance sees the updated step
        console = Console()
        assert console._global_step == 100

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

        Console.integrate_with_logger(mock_logger, global_step=5)
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

        Console.integrate_with_logger(mock_logger, global_step=10)
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

        Console.integrate_with_logger(mock_logger, global_step=15)
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

        Console.integrate_with_logger(mock_logger, global_step=1)

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

        Console.integrate_with_logger(mock_logger)
        console.log("Message")

        mock_logger.log_text.assert_called_once()
        assert mock_logger.log_text.call_args[1]["key"] == "Console/info"

    def test_global_step_updates_correctly(self):
        """Test that global_step is used correctly in logging."""
        console = Console.with_prefix("Step")
        mock_logger = MagicMock()
        mock_logger.log_text = MagicMock()

        Console.integrate_with_logger(mock_logger, global_step=0)

        console.log("Step 0")
        assert mock_logger.log_text.call_args[1]["step"] == 0

        Console.update_global_step(10)
        console.log("Step 10")
        assert mock_logger.log_text.call_args[1]["step"] == 10

        Console.update_global_step(100)
        console.log("Step 100")
        assert mock_logger.log_text.call_args[1]["step"] == 100

    def test_logger_shared_across_instances(self, monkeypatch):
        """Logger integration should propagate to all Console instances."""
        monkeypatch.setattr(Console, "print", lambda self, msg, **_: None)

        Console.with_prefix("First")  # Create but don't need to use
        second = Console.with_prefix("Second")
        mock_logger = MagicMock()
        mock_logger.log_text = MagicMock()

        assert second._pl_logger is None

        Console.integrate_with_logger(mock_logger, global_step=5)

        assert second._pl_logger is mock_logger

        second.log("shared message")
        assert mock_logger.log_text.call_args[1]["key"] == "Second/info"
        assert mock_logger.log_text.call_args[1]["step"] == 5

        Console.update_global_step(11)
        third = Console.with_prefix("Third")
        third.log("future instance uses logger")

        assert mock_logger.log_text.call_args_list[-1][1]["key"] == "Third/info"
        assert mock_logger.log_text.call_args_list[-1][1]["step"] == 11

    def test_exception_handling_does_not_break_training(self):
        """Test that exceptions in logging don't break the console."""
        console = Console.with_prefix("Robust")
        mock_logger = MagicMock()
        mock_logger.log_text = MagicMock(side_effect=Exception("Logger error"))

        Console.integrate_with_logger(mock_logger)

        # Should not raise exception
        console.log("This should not crash")
        console.warn("This should not crash either")

    def test_verbose_false_skips_logging(self):
        """Test that verbose=False prevents logging to logger."""
        console = Console.with_prefix("Silent")
        console.set_verbose(False)

        mock_logger = MagicMock()
        mock_logger.log_text = MagicMock()

        Console.integrate_with_logger(mock_logger)
        console.log("This should not be logged")

        # log_text should not be called because verbose=False
        mock_logger.log_text.assert_not_called()

    def test_debug_false_skips_debug_logging(self):
        """Test that debug=False prevents debug messages from being logged."""
        console = Console.with_prefix("NoDebug")
        console.set_debug(False)

        mock_logger = MagicMock()
        mock_logger.log_text = MagicMock()

        Console.integrate_with_logger(mock_logger)
        console.dbg("This debug message should not be logged")

        # log_text should not be called for debug when is_debug=False
        mock_logger.log_text.assert_not_called()

    def test_hierarchical_prefix_naming(self):
        """Test that multi-part prefixes create hierarchical names."""
        console = Console.with_prefix("Module", "Method", "SubContext")
        mock_logger = MagicMock()
        mock_logger.log_text = MagicMock()

        Console.integrate_with_logger(mock_logger)
        console.log("Hierarchical message")

        # The prefix should maintain the hierarchy in the metric name
        called_key = mock_logger.log_text.call_args[1]["key"]
        assert called_key == "Module/Method/SubContext/info"

    def test_builder_pattern_chaining(self):
        """Test that the builder pattern works for fluent API."""
        mock_logger = MagicMock()
        mock_logger.log_text = MagicMock()

        # Chain all setup methods - note integrate_with_logger is now class method
        Console.integrate_with_logger(mock_logger, global_step=42)
        Console.update_global_step(100)

        console = Console.with_prefix("Chain", "Test").set_verbose(True).set_debug(True)

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
        """Test that update_global_step returns Console class for chaining."""
        result = Console.update_global_step(10)
        assert result is Console  # Returns class, not instance
        assert Console._shared_global_step == 10

    def test_plog_outputs_when_verbose(self, monkeypatch):
        console = Console.with_prefix("VerboseMode")
        console.set_verbose(True)
        captured: list[str] = []
        monkeypatch.setattr(Console, "print", lambda self, msg, **_: captured.append(msg))
        console.plog({"ok": True})
        assert captured

    def test_timestamp_formatting_uses_helper(self, monkeypatch):
        console = Console()
        console.set_timestamp_display(True)
        monkeypatch.setattr(Console, "_get_timestamp", lambda self: "fixed", raising=False)
        formatted = console._format_message("payload")
        assert formatted.startswith("[fixed]")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
