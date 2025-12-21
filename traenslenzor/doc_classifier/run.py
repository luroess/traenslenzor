import importlib.util
import inspect
import sys
import warnings
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from traenslenzor.doc_classifier import ExperimentConfig

# Silence Python 3.13 fork-in-thread DeprecationWarning from multiprocessing
warnings.filterwarnings(
    "ignore",
    message=r".*multi-threaded, use of fork\(\) may lead to deadlocks.*",
    category=DeprecationWarning,
)


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


def _load_sweep_experiment(path: Path) -> ExperimentConfig:
    """Load a sweep ExperimentConfig from a Python file."""
    if not path.exists():
        raise FileNotFoundError(f"Sweep config not found: {path}")

    module_name = f"doc_classifier_sweep_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for sweep config: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if hasattr(module, "ResNet50OptunaExperiment"):
        sweep_cls = getattr(module, "ResNet50OptunaExperiment")
        if inspect.isclass(sweep_cls) and issubclass(sweep_cls, ExperimentConfig):
            return sweep_cls()

    for _, obj in module.__dict__.items():
        if (
            inspect.isclass(obj)
            and issubclass(obj, ExperimentConfig)
            and obj is not ExperimentConfig
        ):
            return obj()

    raise ValueError(f"No ExperimentConfig subclass found in sweep config: {path}")


if __name__ == "__main__":
    # Extract config_path / sweep args from CLI
    config_path_arg = None
    sweep_path_arg: Path | None = None
    use_resnet_sweep = False

    for i, arg in enumerate(sys.argv[1:], 1):  # Start from 1 to skip script name
        if arg == "--config_path" and i + 1 < len(sys.argv):
            config_path_arg = Path(sys.argv[i + 1])
        elif arg.startswith("--config_path="):
            config_path_arg = Path(arg.split("=", 1)[1])
        elif arg == "--resnet_sweep":
            use_resnet_sweep = True
        elif arg == "--sweep_path" and i + 1 < len(sys.argv):
            sweep_path_arg = Path(sys.argv[i + 1])
        elif arg.startswith("--sweep_path="):
            sweep_path_arg = Path(arg.split("=", 1)[1])

    if use_resnet_sweep and sweep_path_arg is None:
        sweep_path_arg = Path(".configs/resnet_sweep.py")

    if sweep_path_arg is not None:
        sweep_config = _load_sweep_experiment(sweep_path_arg)
        sweep_config.run_optuna_study()
        raise SystemExit(0)

    if config_path_arg is not None:
        # Load from TOML file
        if not config_path_arg.exists():
            raise FileNotFoundError(f"Config file not found: {config_path_arg}")

        # Load ExperimentConfig from TOML (creates ONE instance)
        config = ExperimentConfig.from_toml(config_path_arg)
    else:
        # No config file - create from CLI args via pydantic-settings
        config = CLIExperimentConfig()

    # Run the experiment (tuning controlled via config.tuner_config)
    config.setup_target_and_run()
