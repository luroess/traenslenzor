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
    optuna_study_uri: str = Field(default=".logs/optuna/{study_name}.db")
    """Uri for Optuna study storage (SQLite). The `{study_name}` placeholder is replaced within the Optuna Config."""
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

    @field_validator("optuna_study_uri", mode="before")
    @classmethod
    def convert_to_uri(cls, v: str, info: ValidationInfo) -> str:
        study_dir = cls._resolve_path(Path(v).parent, info)
        study_dir = cls._ensure_dir(study_dir, info.field_name)
        return f"sqlite:///{(study_dir / Path(v).name).as_posix()}"

    def resolve_checkpoint_path(self, path: str | Path | None) -> Path | None:
        """Resolve a checkpoint path relative to the checkpoints directory.

        Args:
            path: Checkpoint path (absolute, relative, or None).

        Returns:
            Resolved absolute path, or None if input is None/empty.

        Raises:
            FileNotFoundError: If the resolved path does not exist.
        """
        if path in (None, ""):
            return None

        checkpoint_path = Path(path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.checkpoints / checkpoint_path

        checkpoint_path = checkpoint_path.expanduser().resolve()

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path '{checkpoint_path}' does not exist.")

        if not checkpoint_path.suffix == ".ckpt":
            raise FileNotFoundError(f"Checkpoint path '{checkpoint_path}' is not a .ckpt file.")

        return checkpoint_path
