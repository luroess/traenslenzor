"""Config definitions for the document scanner MCP runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from traenslenzor.doc_classifier.utils import BaseConfig

from .runtime import DocScannerRuntime


class DocScannerMCPConfig(BaseConfig["DocScannerRuntime"]):
    """Top-level configuration for the DocScanner MCP runtime (UVDoc only)."""

    model_config = SettingsConfigDict(
        toml_file=Path(__file__).resolve().parents[2] / "config" / "doc-scanner.toml",
    )

    @property
    def target(self) -> type[DocScannerRuntime]:
        return DocScannerRuntime

    num_filter: int = Field(default=32, ge=32, le=32)
    """Number of filters for UVDocNet."""
    kernel_size: int = Field(default=5, ge=5, le=5)
    """Kernel size for UVDocNet."""
    crop_page: bool = True
    """Whether to crop the unwarped output to the page contour."""
    min_area_ratio: float = 0.15
    """Minimum contour area ratio required for page detection."""
    generate_map_xy: bool = True
    """Whether to generate a dense map_xy."""
    max_map_pixels: int = 5_000_000
    """Max pixel count allowed for map_xy generation."""
    deskew_mode: Literal["uv2d", "uv3d"] = "uv2d"
    """Deskew mode: uv2d (standard) or uv3d (3D-aware reparameterization)."""
    superres_device: str = "CPU"
    """OpenVINO device to run text super-resolution on (e.g. CPU)."""
    superres_allow_download: bool = True
    """Allow downloading the OpenVINO SR model when missing."""
    superres_models_dir: Path = (
        Path(__file__).resolve().parents[2]
        / ".data"
        / "models"
        / "openvino"
        / "text-image-super-resolution-0001"
        / "FP32"
    )
    """Directory containing text-image-super-resolution-0001 XML/BIN files."""
    device: str = "auto"
    """Device to run the model on ("auto", "cpu", "cuda")."""
    verbose: bool = True
    """Enable verbose logging for the runtime and backend."""
    is_debug: bool = False
    """Enable debug logging for the runtime and backend."""
