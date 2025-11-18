"""Configuration + runtime factory for the doc-classifier MCP server.

The goal is to keep this lightweight and friendly for mocked inference while
still allowing a real checkpoint to be plugged in later.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from traenslenzor.doc_classifier.lightning import DocClassifierConfig
from traenslenzor.doc_classifier.mcp.runtime import DocClassifierRuntime
from traenslenzor.doc_classifier.utils import BaseConfig


class DocClassifierMCPConfig(BaseConfig[DocClassifierRuntime]):
    """Factory config that creates a :class:`DocClassifierRuntime` instance."""

    target: type[DocClassifierRuntime] = Field(default=DocClassifierRuntime, exclude=True)

    lit_module_config: DocClassifierConfig = Field(
        default_factory=DocClassifierConfig,
        description="Configuration for the Lightning module.",
    )
    is_mock: bool = Field(
        default=True, description="If True, uses a mock model that returns random predictions. "
    )
    checkpoint_path: Path | None = Field(
        default=None,
        description="Optional path to a Lightning checkpoint (.ckpt). If omitted, a mock model is used.",
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
    verbose: bool = Field(default=False, description="Enable verbose runtime logging.")

    def setup_target(self) -> DocClassifierRuntime:
        return self.target(self)


__all__ = ["DocClassifierMCPConfig"]
