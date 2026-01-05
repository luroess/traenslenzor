"""Deskew backend implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from jaxtyping import Float32, UInt8
from numpy import ndarray as NDArray

from traenslenzor.doc_classifier.utils import Console
from traenslenzor.text_extractor.flatten_image import find_document_corners

from .utils import (
    build_full_image_corners,
    build_map_xy_from_homography,
    compute_map_xy_stride,
    find_page_corners,
    order_points_clockwise,
    sample_map_xy,
    warp_from_corners,
)

if TYPE_CHECKING:
    from .configs import OpenCVDeskewConfig, UVDocDeskewConfig


@dataclass(slots=True)
class DeskewResult:
    """Container for deskew outputs."""

    image_rgb: UInt8[NDArray, "H W 3"]
    """Deskewed RGB image."""
    corners_original: Float32[NDArray, "4 2"] | None
    """Document corners in original image coordinates (UL, UR, LR, LL)."""
    map_xy: Float32[NDArray, "H W 2"] | None
    """Optional map_xy mapping output pixels -> original pixels."""


class OpenCVDeskewBackend:
    """Classic OpenCV deskew backend using contour detection."""

    def __init__(self, config: "OpenCVDeskewConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__, "init")
        self.console.set_verbose(config.verbose).set_debug(config.is_debug)

    def deskew(
        self,
        image_rgb: UInt8[NDArray, "H W 3"],
    ) -> DeskewResult:
        """Deskew via contour detection and perspective warp.

        Args:
            image_rgb (ndarray[uint8]): Input RGB image, shape (H, W, 3).

        Returns:
            DeskewResult containing deskewed RGB image, corners, and optional map_xy.
        """
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        corners = find_document_corners(bgr)

        if corners is None:
            if not self.config.fallback_to_original:
                raise RuntimeError("OpenCV backend could not find document corners.")
            self.console.warn("OpenCV backend failed to find corners; using full image.")
            corners = build_full_image_corners(bgr.shape[0], bgr.shape[1])

        corners = order_points_clockwise(corners)
        warped_bgr, matrix, output_size = warp_from_corners(bgr, corners)
        warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)

        # OpenCV backend does not emit map_xy (avoid misleading grid on simple perspective warp).
        map_xy = None

        return DeskewResult(
            image_rgb=warped_rgb,
            corners_original=corners,
            map_xy=map_xy,
        )


class UVDocDeskewBackend:
    """UVDoc neural unwarping backend (py_reform)."""

    def __init__(self, config: "UVDocDeskewConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__, "init")
        self.console.set_verbose(config.verbose).set_debug(config.is_debug)
        self._model = None
        self._device: str | None = None
        self._input_size: tuple[int, int] | None = None

    def _resolve_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device

        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            from py_reform.models.uvdoc_model import (
                DEFAULT_IMG_SIZE,
                DEFAULT_MODEL_PATH,
                UVDocNet,
            )
        except Exception as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(f"py_reform UVDoc not available: {exc}") from exc

        import torch

        self._device = self._resolve_device()
        self._model = UVDocNet(
            num_filter=self.config.num_filter, kernel_size=self.config.kernel_size
        ).to(self._device)
        self._model.eval()

        model_path = (
            Path(self.config.model_path) if self.config.model_path else Path(DEFAULT_MODEL_PATH)
        )
        if not model_path.exists():
            raise RuntimeError(f"UVDoc model weights not found at {model_path}")

        ckpt = torch.load(model_path, map_location=self._device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        self._model.load_state_dict(state_dict)

        if self.config.input_size is not None:
            self._input_size = (int(self.config.input_size[0]), int(self.config.input_size[1]))
        elif isinstance(DEFAULT_IMG_SIZE, (list, tuple)) and len(DEFAULT_IMG_SIZE) == 2:
            self._input_size = (int(DEFAULT_IMG_SIZE[0]), int(DEFAULT_IMG_SIZE[1]))
        else:
            raise RuntimeError(f"Unexpected UVDoc DEFAULT_IMG_SIZE: {DEFAULT_IMG_SIZE}")

    def deskew(
        self,
        image_rgb: UInt8[NDArray, "H W 3"],
    ) -> DeskewResult:
        """Deskew via UVDoc neural unwarping.

        Args:
            image_rgb (ndarray[uint8]): Input RGB image, shape (H, W, 3).

        Returns:
            DeskewResult containing deskewed RGB image, corners, and optional map_xy.
        """
        import torch
        import torch.nn.functional as F
        from PIL import Image

        self._load_model()
        assert self._model is not None
        assert self._device is not None
        assert self._input_size is not None

        orig_np = image_rgb.astype(np.float32) / 255.0
        h, w = orig_np.shape[:2]

        orig = torch.from_numpy(orig_np).permute(2, 0, 1).unsqueeze(0).to(self._device)
        inp_resized = Image.fromarray(image_rgb).resize(
            tuple(self._input_size), resample=Image.Resampling.BILINEAR
        )
        inp_np = np.array(inp_resized).astype(np.float32) / 255.0
        inp = torch.from_numpy(inp_np).permute(2, 0, 1).unsqueeze(0).to(self._device)

        with torch.no_grad():
            points2d, _points3d = self._model(inp)

        if points2d.dim() == 3:
            points2d = points2d.unsqueeze(0)

        grid_2ch = F.interpolate(points2d, size=(h, w), mode="bilinear", align_corners=True)
        grid = grid_2ch.permute(0, 2, 3, 1).clamp(-1.0, 1.0)

        unwarped = F.grid_sample(orig, grid, align_corners=True)
        unwarped_np = unwarped[0].permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
        unwarped_rgb = (unwarped_np * 255).astype(np.uint8)

        grid_cpu = grid[0].detach().cpu()
        x_norm = grid_cpu[:, :, 0]
        y_norm = grid_cpu[:, :, 1]
        x_in = (x_norm + 1.0) * 0.5 * (w - 1)
        y_in = (y_norm + 1.0) * 0.5 * (h - 1)
        map_xy_full = torch.stack([x_in, y_in], dim=-1).numpy().astype(np.float32)

        corners_unwarped = None
        if self.config.crop_page:
            corners_unwarped, area_ratio = find_page_corners(
                unwarped_rgb, min_area_ratio=self.config.min_area_ratio
            )
            if corners_unwarped is None:
                self.console.warn(
                    f"UVDoc page contour weak (area ratio {area_ratio:.3f}); using full image."
                )
                corners_unwarped = build_full_image_corners(
                    unwarped_rgb.shape[0], unwarped_rgb.shape[1]
                )

        map_xy_stride = compute_map_xy_stride((h, w), self.config.max_map_pixels)

        if corners_unwarped is None:
            output_rgb = unwarped_rgb
            output_size = (unwarped_rgb.shape[0], unwarped_rgb.shape[1])
            map_xy_out = map_xy_full[::map_xy_stride, ::map_xy_stride]
        else:
            unwarped_bgr = cv2.cvtColor(unwarped_rgb, cv2.COLOR_RGB2BGR)
            cropped_bgr, crop_matrix, output_size = warp_from_corners(
                unwarped_bgr, corners_unwarped
            )
            output_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

            map_xy_out = None
            if self.config.generate_map_xy:
                out_h, out_w = output_size
                map_xy_stride = compute_map_xy_stride(output_size, self.config.max_map_pixels)
                xs, ys = np.meshgrid(
                    np.arange(0, out_w, map_xy_stride, dtype=np.float32),
                    np.arange(0, out_h, map_xy_stride, dtype=np.float32),
                )
                grid_out = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)
                inv_crop = np.linalg.inv(crop_matrix)
                coords_unwarped = cv2.perspectiveTransform(grid_out, inv_crop).reshape(
                    xs.shape[0], xs.shape[1], 2
                )
                coords_flat = coords_unwarped.reshape(-1, 2)
                mapped = sample_map_xy(map_xy_full, coords_flat)
                map_xy_out = mapped.reshape(xs.shape[0], xs.shape[1], 2).astype(np.float32)

        corners_original = None
        if corners_unwarped is not None:
            corners_original = sample_map_xy(map_xy_full, corners_unwarped)

        if not self.config.generate_map_xy:
            map_xy_out = None

        return DeskewResult(
            image_rgb=output_rgb,
            corners_original=corners_original,
            map_xy=map_xy_out,
        )
