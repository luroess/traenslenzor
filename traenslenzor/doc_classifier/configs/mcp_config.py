"""Configuration + runtime factory for the doc-classifier MCP server.

The goal is to keep this lightweight and friendly for mocked inference while
still allowing a real checkpoint to be plugged in later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import torch
from pydantic import Field, field_validator

from traenslenzor.doc_classifier.configs.path_config import PathConfig
from traenslenzor.doc_classifier.lightning import DocClassifierConfig
from traenslenzor.doc_classifier.mcp_integration.runtime import DocClassifierRuntime
from traenslenzor.doc_classifier.utils import BaseConfig


class DocClassifierMCPConfig(BaseConfig["DocClassifierRuntime"]):
    """Factory config that creates a :class:`DocClassifierRuntime` instance."""

    @property
    def target(self) -> type["DocClassifierRuntime"]:
        return DocClassifierRuntime

    lit_module_config: DocClassifierConfig = Field(
        default_factory=DocClassifierConfig,
        description="Configuration for the Lightning module.",
    )
    is_mock: bool = Field(
        default=False, description="If True, uses a mock model that returns random predictions. "
    )
    checkpoint_path: Annotated[
        Path | None,
        Field(
            default=Path("alexnet-epoch=15-val_loss=0.03.ckpt"),
            description=(
                "Optional path to a Lightning checkpoint (.ckpt). "
                "If provided, should be relative to .logs/checkpoints/."
            ),
        ),
    ]
    device: Annotated[
        torch.device | Literal["cpu", "cuda", "auto"],
        Field(
            default="auto",
            description="Torch device string. Only used when a checkpoint is provided.",
        ),
    ]
    img_size: int = Field(
        default=224,
        description="Image resize target used by the validation transforms.",
        ge=224,
        le=224,
    )
    normalization_mode: Literal["dataset", "imagenet"] = Field(
        default="imagenet",
        description=(
            "Normalization statistics to use in the MCP inference transform. "
            "Use 'imagenet' for pretrained RGB backbones."
        ),
    )
    convert_to_rgb: bool = Field(
        default=True,
        description="If True, replicate grayscale inputs to 3 channels for RGB backbones.",
    )
    apply_normalization: bool = Field(
        default=True,
        description="If True, apply normalization in the MCP inference transform.",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility when using a real model.",
        ge=37,
        le=69,
    )
    is_debug: bool = Field(default=False, description="Enable verbose debug logging.")
    verbose: bool = Field(default=True, description="Enable verbose runtime logging.")

    @field_validator("checkpoint_path", mode="before")
    @classmethod
    def validate_checkpoint_path(cls, v: Path | None) -> Path | None:
        """Convert string paths to Path objects."""
        return PathConfig().resolve_checkpoint_path(v)

    @field_validator("device", mode="before")
    @classmethod
    def validate_device(cls, v: str | torch.device) -> torch.device:
        """Convert string device to torch.device."""
        if isinstance(v, torch.device):
            return v
        if v == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(v)


__all__ = ["DocClassifierMCPConfig"]
