"""Rich-powered console tailored for training and pprinting of instances or other structured data."""

import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

from devtools import pformat
from rich.console import Console as RichConsole
from rich.theme import Theme

if TYPE_CHECKING:
    from lightning.pytorch.loggers.logger import Logger


class Console(RichConsole):
    """Console wrapper that centralises formatting and convenience helpers."""

    # TODO: get prefix automatically from caller (via caller stack?)
    is_debug: bool
    prefix: str | None = None
    _pl_logger: "Logger | None" = None
    _global_step: int = 0

    default_settings = {
        "theme": Theme(
            {
                "config.name": "bold blue",  # Config class names
                "config.field": "green",  # Regular fields
                "config.propagated": "yellow",  # Propagated fields
                "config.value": "white",  # Field values
                "config.type": "dim",  # Type annotations
                "config.doc": "italic dim",  # Documentation
            },
        ),
        "width": 120,
        "force_terminal": True,
        "color_system": "auto",
        "markup": True,
        "highlight": True,
    }

    def __init__(self, **kwargs):
        """Initialise the console with project defaults and user overrides."""
        settings = self.default_settings.copy()
        settings.update(kwargs)
        super().__init__(**settings)
        self.is_debug = False
        self.verbose = True
        self.show_timestamps = False
        self.prefix = None

    @classmethod
    def with_prefix(cls, *parts: str) -> "Console":
        """Create a console instance with a prefixed context.

        Enables builder-style chaining.

        Usage:
        ```python
        console = Console.with_prefix(
            self.__class__.__name__,
            <name_of_the_current_method>
            <further_parts>, # eg. stage, worker_idx...
        )
        ```

        """
        instance = cls()
        instance.set_prefix(*parts)
        return instance

    def set_prefix(self, *parts: str) -> "Console":
        """Set a custom prefix for all log messages.

        Enables builder-style chaining.
        """
        if not parts:
            self.prefix = None
        else:
            self.prefix = "::".join(filter(None, parts))

        return self

    def unset_prefix(self) -> "Console":
        """Unset the prefix for all log messages."""
        self.prefix = None
        return self

    def log(self, message: str) -> None:
        """Emit an informational message when verbosity is enabled."""
        if self.verbose:
            self.print(self._format_message(message))
            if self._pl_logger is not None:
                self._log_to_lightning("info", message)

    def warn(self, message: str) -> None:
        """Emit a warning message and include a short caller stack."""
        if self.verbose:
            self.print(
                f"[bright_yellow]Warning:[/bright_yellow] {self._format_message(message)}\n"
                f"[dim]{self._get_caller_stack()}[/dim]",
            )
            if self._pl_logger is not None:
                self._log_to_lightning("warning", message)

    def error(self, message: str) -> None:
        """Emit an error message and show the relevant caller stack."""
        self.print(
            f"[bright_red]Error:[/bright_red] {self._format_message(message)}\n[dim]{self._get_caller_stack()}[/dim]",
        )
        if self._pl_logger is not None:
            self._log_to_lightning("error", message)

    def plog(self, obj: Any, **kwargs) -> None:
        """Pretty-print an object using the best available formatter."""
        if self.verbose:
            self.print(pformat(obj, **kwargs))

    def dbg(self, message: str) -> None:
        """Emit a debug message when debug mode is enabled."""
        if self.is_debug:
            self.print(
                f"[bold magenta]Debug:[/bold magenta] {self._format_message(message)}",
            )
            if self._pl_logger is not None:
                self._log_to_lightning("debug", message)

    def set_verbose(self, verbose: bool) -> "Console":
        """Toggle verbose logging output."""
        self.verbose = verbose
        return self

    def set_debug(self, is_debug: bool) -> "Console":
        """Enable or disable debug logging while keeping verbose mode sensible."""
        self.is_debug = is_debug
        if is_debug:
            self.verbose = True
        return self

    def set_timestamp_display(self, show_timestamps: bool) -> "Console":
        """Toggle timestamps for subsequent log messages."""
        self.show_timestamps = show_timestamps
        return self

    def _format_message(self, message: str) -> str:
        """Format message with optional timestamp and prefix."""
        if self.prefix:
            # Use rich markup for terminal display
            rich_prefix = self.prefix.replace(
                "::",
                "[/bold cyan][grey]::[/grey][bold cyan]",
            )
            prefix = rf"\[[bold cyan]{rich_prefix}[/bold cyan]]: "
        else:
            prefix = ""
        if self.show_timestamps:
            return f"[{self._get_timestamp()}] {prefix}{message}"
        return f"{prefix}{message}"

    def _get_caller_stack(self) -> str:
        """Get formatted stack trace excluding console internals."""
        stack = traceback.extract_stack()
        # Filter out frames from this file
        current_file = Path(__file__).resolve()
        relevant_frames = [
            frame
            for frame in stack[:-1]  # Exclude current frame
            if Path(frame.filename).resolve() != current_file
        ]
        # Format remaining frames
        return "".join(
            traceback.format_list(relevant_frames[-2:]),
        )  # Show last 2 relevant frames

    def integrate_with_logger(
        self,
        logger: "Logger",
        global_step: int = 0,
    ) -> "Console":
        """Integrate console with PyTorch Lightning logger for WandB/TensorBoard logging.

        Args:
            logger: PyTorch Lightning logger instance (e.g., WandbLogger, TensorBoardLogger).
            global_step: Current training step for metric logging.

        Returns:
            Self for method chaining.

        Example:
            ```python
            # In your LightningModule
            def on_train_start(self):
                self.console.integrate_with_logger(self.logger, self.global_step)

            def training_step(self, batch, batch_idx):
                self.console.update_global_step(self.global_step)
                self.console.log("Processing batch")  # → Terminal + WandB!
            ```
        """
        self._pl_logger = logger
        self._global_step = global_step
        return self

    def update_global_step(self, step: int) -> "Console":
        """Update the global step for subsequent logs.

        Args:
            step: Current global training step.

        Returns:
            Self for method chaining.
        """
        self._global_step = step
        return self

    def _log_to_lightning(self, level: str, message: str) -> None:
        """Log message to PyTorch Lightning logger.

        Args:
            level: Log level (info, warning, error, debug).
            message: Message content without formatting.
        """
        if self._pl_logger is None:
            return

        # Construct metric name with prefix and level
        # Convert :: separators to / for hierarchical metric names
        prefix_clean = self.prefix.replace("::", "/") if self.prefix else "Console"
        metric_name = f"{prefix_clean}/{level}"

        # Log as text to WandB/logger using log_text if available, otherwise skip
        try:
            # WandbLogger has log_text method for logging strings
            if hasattr(self._pl_logger, "log_text"):
                self._pl_logger.log_text(
                    key=metric_name,
                    columns=["message"],
                    data=[[message]],
                    step=self._global_step,
                )
            # For TensorBoard and others, log to experiment directly
            elif hasattr(self._pl_logger, "experiment"):
                exp = self._pl_logger.experiment
                # WandB experiment object
                if hasattr(exp, "log"):
                    exp.log({metric_name: message}, step=self._global_step)
                # TensorBoard experiment
                elif hasattr(exp, "add_text"):
                    exp.add_text(metric_name, message, self._global_step)
        except Exception:
            # Fallback: silent failure to avoid breaking training
            pass
