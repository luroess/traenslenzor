"""Config definitions for the document scanner MCP runtime."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from traenslenzor.doc_classifier.utils import BaseConfig
from traenslenzor.file_server.session_state import DeskewBackend

from .backends import OpenCVDeskewBackend, UVDocDeskewBackend
from .runtime import DocScannerRuntime


class OpenCVDeskewConfig(BaseConfig["OpenCVDeskewBackend"]):
    """Config for the classic OpenCV deskew backend."""

    target: type[OpenCVDeskewBackend] = Field(
        default_factory=lambda: OpenCVDeskewBackend,
        exclude=True,
    )

    fallback_to_original: bool = True
    """Fallback to the full image if corners cannot be found."""
    generate_map_xy: bool = True
    """Whether to generate a dense map_xy when feasible."""
    max_map_pixels: int = 5_000_000
    """Max pixel count allowed for map_xy generation."""
    verbose: bool = True
    """Enable verbose logging for this backend."""
    is_debug: bool = False
    """Enable debug logging for this backend."""


class UVDocDeskewConfig(BaseConfig["UVDocDeskewBackend"]):
    """Config for the UVDoc neural unwarping backend."""

    target: type[UVDocDeskewBackend] = Field(
        default_factory=lambda: UVDocDeskewBackend,
        exclude=True,
    )

    model_path: Path | None = None
    """Optional override path to the UVDoc model weights."""
    num_filter: int = 32
    """Number of filters for UVDocNet."""
    kernel_size: int = 5
    """Kernel size for UVDocNet."""
    input_size: tuple[int, int] | None = None
    """Optional override for UVDoc input size as (width, height)."""
    crop_page: bool = True
    """Whether to crop the unwarped output to the page contour."""
    min_area_ratio: float = 0.15
    """Minimum contour area ratio required for page detection."""
    generate_map_xy: bool = True
    """Whether to generate a dense map_xy when feasible."""
    max_map_pixels: int = 5_000_000
    """Max pixel count allowed for map_xy generation."""
    device: str = "auto"
    """Device to run the model on ("auto", "cpu", "cuda")."""
    verbose: bool = True
    """Enable verbose logging for this backend."""
    is_debug: bool = False
    """Enable debug logging for this backend."""


class DocScannerMCPConfig(BaseConfig["DocScannerRuntime"]):
    """Top-level configuration for the DocScanner MCP runtime."""

    target: type[DocScannerRuntime] = Field(
        default_factory=lambda: DocScannerRuntime,
        exclude=True,
    )

    default_backend: DeskewBackend = DeskewBackend.opencv
    """Default backend to use when none is specified."""
    opencv: OpenCVDeskewConfig = OpenCVDeskewConfig()
    """Config for the OpenCV backend."""
    uvdoc: UVDocDeskewConfig = UVDocDeskewConfig()
    """Config for the UVDoc backend."""
    verbose: bool = True
    """Enable verbose logging for the runtime and backends."""
    is_debug: bool = False
    """Enable debug logging for the runtime and backends."""

    @model_validator(mode="after")
    def _propagate_logging(self) -> "DocScannerMCPConfig":
        for cfg in (self.opencv, self.uvdoc):
            cfg.verbose = self.verbose
            cfg.is_debug = self.is_debug
        return self
