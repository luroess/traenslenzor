from pathlib import Path

from pydantic import Field, ValidationInfo, field_validator

from ..utils import Console, SingletonConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _default_root() -> Path:
    return PROJECT_ROOT


def _default_data_root() -> Path:
    return PROJECT_ROOT / ".data"


class PathConfig(SingletonConfig):
    """Centralise all filesystem locations for the document classifier."""

    root: Path = Field(
        default_factory=_default_root,
    )
    "Project root."
    data_root: Path = Field(default_factory=lambda: Path(".data"))
    hf_cache: Path = Field(default_factory=lambda: Path(".data") / "hf_cache")
    """Directory used for Hugging Face dataset caching."""
    checkpoints: Path = Field(default_factory=lambda: Path(".logs") / "checkpoints")
    """Directory used by Lightning checkpoints."""
    wandb: Path = Field(
        default_factory=lambda: Path(".logs") / "wandb",
    )
    configs_dir: Path = Field(default_factory=lambda: Path(".configs"))
    """Directory containing exported experiment/configuration files (TOML, etc.)."""

    @classmethod
    def _resolve_path(cls, value: str | Path, info: ValidationInfo) -> Path:
        root = info.data.get("root", PROJECT_ROOT)
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        return path.expanduser().resolve()

    @classmethod
    def _ensure_dir(cls, path: Path, field_name: str) -> Path:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            Console.with_prefix(cls.__name__, field_name).log(f"Created directory: {path}")
        return path

    @field_validator("root", mode="before")
    @classmethod
    def _validate_root(cls, value: str | Path) -> Path:
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Configured project root '{path}' does not exist.")
        return path

    @field_validator("checkpoints", "wandb", "data_root", "configs_dir", mode="before")
    @classmethod
    def _resolve_dirs(cls, value: str | Path, info: ValidationInfo) -> Path:
        path = cls._resolve_path(value, info)
        return cls._ensure_dir(path, info.field_name)
