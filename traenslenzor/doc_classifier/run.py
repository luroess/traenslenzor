import importlib.util
import inspect
import sys
import tomllib
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import create_model
from pydantic_settings import BaseSettings, SettingsConfigDict

from traenslenzor.doc_classifier import ExperimentConfig

# Silence Python 3.13 fork-in-thread DeprecationWarning from multiprocessing
warnings.filterwarnings(
    "ignore",
    message=r".*multi-threaded, use of fork\(\) may lead to deadlocks.*",
    category=DeprecationWarning,
)


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge nested dict updates into base."""

    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = value
    return base


@lru_cache(maxsize=1)
def _build_cli_model() -> type[BaseSettings]:
    """Build a BaseSettings-based CLI model that mirrors ExperimentConfig fields.

    We cannot simply inherit from ExperimentConfig because `BaseSettings` and a Pydantic
    model cannot be combined via multiple inheritance (MRO conflict). Instead, we
    dynamically create a settings model that exposes the same CLI surface.
    """

    field_definitions: dict[str, Any] = {
        name: (field.annotation, field)
        for name, field in ExperimentConfig.model_fields.items()
        if not field.exclude
    }

    field_definitions["config_path"] = (Path | None, None)

    return create_model(
        "DocClassifierCLI",
        __base__=BaseSettings,
        __config__=SettingsConfigDict(
            arbitrary_types_allowed=True,
            validate_default=True,
            validate_assignment=True,
            cli_parse_args=True,
            env_prefix="EXPERIMENT_",
        ),
        **field_definitions,
    )


def load_experiment_config(argv: list[str] | None = None) -> ExperimentConfig:
    """Load an ExperimentConfig from TOML (optional) + CLI overrides.

    Args:
        argv: Optional CLI argument list (without program name). If None, parses
            `sys.argv[1:]`.

    Returns:
        ExperimentConfig with CLI overrides applied on top of TOML/defaults.
    """

    cli_model = _build_cli_model()
    cli = cli_model(_cli_parse_args=argv) if argv is not None else cli_model()
    cli_data = cli.model_dump(exclude_unset=True)

    config_path = cli_data.pop("config_path", None)
    base_data: dict[str, Any] = {}
    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        base_data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    merged = _deep_merge(base_data, cli_data)
    return ExperimentConfig.model_validate(merged)


def _load_sweep_experiment_from_python(path: Path) -> ExperimentConfig:
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


def _load_sweep_experiment(path: Path) -> ExperimentConfig:
    """Load a sweep ExperimentConfig from either TOML or Python."""
    if path.suffix == ".toml":
        return ExperimentConfig.from_toml(path)
    if path.suffix == ".py":
        return _load_sweep_experiment_from_python(path)
    raise ValueError(f"Unsupported sweep config format: {path} (expected .toml or .py)")


if __name__ == "__main__":
    # Extract sweep args from CLI (handled before pydantic-settings CLI parsing)
    sweep_path_arg: Path | None = None

    for i, arg in enumerate(sys.argv[1:], 1):  # Start from 1 to skip script name
        if arg == "--resnet_sweep":
            sweep_path_arg = Path("_config/resnet_sweep.toml")
            if not sweep_path_arg.exists():
                sweep_path_arg = Path("_config/resnet_sweep.py")
        elif arg == "--alexnet_sweep":
            sweep_path_arg = Path("_config/alexnet_sweep.toml")
            if not sweep_path_arg.exists():
                sweep_path_arg = Path("_config/alexnet_sweep.py")
        elif arg == "--sweep_path" and i + 1 < len(sys.argv):
            sweep_path_arg = Path(sys.argv[i + 1])
        elif arg.startswith("--sweep_path="):
            sweep_path_arg = Path(arg.split("=", 1)[1])

    if sweep_path_arg is not None:
        sweep_config = _load_sweep_experiment(sweep_path_arg)
        sweep_config.run_optuna_study()
        raise SystemExit(0)

    config = load_experiment_config()

    # Run the experiment (tuning controlled via config.tuner_config)
    config.setup_target_and_run()
