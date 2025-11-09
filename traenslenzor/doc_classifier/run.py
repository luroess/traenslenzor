from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from traenslenzor.doc_classifier import ExperimentConfig


class CLIExperimentConfig(BaseSettings, ExperimentConfig):
    """CLI-enabled ExperimentConfig with automatic argument parsing.

    Supports loading from TOML files via --config_path argument, with CLI arguments
    overriding config file values.
    """

    config_path: Path | None = None
    """Path to TOML configuration file. If provided, config is loaded from this file first,
    then CLI arguments override specific values."""

    model_config = SettingsConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=True,
        cli_parse_args=True,
        env_prefix="EXPERIMENT_",
    )


if __name__ == "__main__":
    import logging
    import sys

    # Suppress PyTorch Lightning's verbose informational messages
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

    # Extract config_path from CLI args if present
    config_path_arg = None
    for i, arg in enumerate(sys.argv[1:], 1):  # Start from 1 to skip script name
        if arg == "--config_path" and i + 1 < len(sys.argv):
            config_path_arg = Path(sys.argv[i + 1])
            break
        elif arg.startswith("--config_path="):
            config_path_arg = Path(arg.split("=", 1)[1])
            break

    if config_path_arg is not None:
        # Load from TOML file
        if not config_path_arg.exists():
            raise FileNotFoundError(f"Config file not found: {config_path_arg}")

        # Load ExperimentConfig from TOML (creates ONE instance)
        config = ExperimentConfig.from_toml(config_path_arg)
    else:
        # No config file - create from CLI args via pydantic-settings
        config = CLIExperimentConfig()

    # Run the experiment
    config.setup_target_and_run()
