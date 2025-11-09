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
    # Parse CLI arguments first to check for config_path
    cli_config = CLIExperimentConfig()

    # If config_path was provided, load base config from TOML and merge with CLI overrides
    if cli_config.config_path is not None:
        config_path = Path(cli_config.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load base config from TOML
        base_config = ExperimentConfig.from_toml(config_path)

        # Merge CLI overrides by re-parsing with base config as defaults
        # This is done by converting base config to dict and using it as defaults
        base_dict = base_config.model_dump(mode="python")

        # Create new config with CLI overrides
        config = CLIExperimentConfig(**base_dict)
    else:
        # No config file, use CLI arguments only
        config = cli_config

    # Run the experiment
    config.setup_target_and_run()
