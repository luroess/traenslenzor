"""Configuration + runtime factory for the doc-classifier MCP server.

The goal is to keep this lightweight and friendly for mocked inference while
still allowing a real checkpoint to be plugged in later.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator

from traenslenzor.doc_classifier.configs.path_config import PathConfig
from traenslenzor.doc_classifier.lightning import DocClassifierConfig
from traenslenzor.doc_classifier.mcp_integration.runtime import DocClassifierRuntime
from traenslenzor.doc_classifier.utils import BaseConfig


class DocClassifierMCPConfig(BaseConfig[DocClassifierRuntime]):
    """Factory config that creates a :class:`DocClassifierRuntime` instance."""

    target: type[DocClassifierRuntime] = Field(default=DocClassifierRuntime, exclude=True)

    lit_module_config: DocClassifierConfig = Field(
        default_factory=DocClassifierConfig,
        description="Configuration for the Lightning module.",
    )
    is_mock: bool = Field(
        default=False, description="If True, uses a mock model that returns random predictions. "
    )
    checkpoint_path: Path | None = Field(
        default="resnet50-epoch=5-val_loss=0.52.ckpt",
        description="Optional path to a Lightning checkpoint (.ckpt). Should be relative to .logs/checkpoints/",
    )
    device: str = Field(
        default="cpu",
        description="Torch device string. Only used when a checkpoint is provided.",
    )
    img_size: int = Field(
        default=224,
        ge=32,
        description="Image resize target used by the validation transforms.",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility when using a real model.",
    )
    is_debug: bool = Field(default=False, description="Enable verbose debug logging.")
    verbose: bool = Field(default=True, description="Enable verbose runtime logging.")

    @field_validator("checkpoint_path", mode="before")
    @classmethod
    def validate_checkpoint_path(cls, v: Path | str | None) -> Path | None:
        """Convert string paths to Path objects."""
        return PathConfig().resolve_checkpoint_path(v)


__all__ = ["DocClassifierMCPConfig"]
