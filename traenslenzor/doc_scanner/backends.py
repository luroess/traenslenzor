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
    find_page_corners,
    order_points_clockwise,
    sample_map_xy,
    should_generate_map_xy,
    warp_from_corners,
)

if TYPE_CHECKING:
    from .configs import DocScannerDeskewConfig, OpenCVDeskewConfig, UVDocDeskewConfig


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

        map_xy = None
        if self.config.generate_map_xy and should_generate_map_xy(
            output_size, self.config.max_map_pixels
        ):
            map_xy = build_map_xy_from_homography(matrix, output_size)

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

        if corners_unwarped is None:
            output_rgb = unwarped_rgb
            output_size = (unwarped_rgb.shape[0], unwarped_rgb.shape[1])
            map_xy_out = map_xy_full
        else:
            unwarped_bgr = cv2.cvtColor(unwarped_rgb, cv2.COLOR_RGB2BGR)
            cropped_bgr, crop_matrix, output_size = warp_from_corners(
                unwarped_bgr, corners_unwarped
            )
            output_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

            map_xy_out = None
            if self.config.generate_map_xy and should_generate_map_xy(
                output_size, self.config.max_map_pixels
            ):
                out_h, out_w = output_size
                xs, ys = np.meshgrid(
                    np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32)
                )
                grid_out = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)
                inv_crop = np.linalg.inv(crop_matrix)
                coords_unwarped = cv2.perspectiveTransform(grid_out, inv_crop).reshape(
                    out_h, out_w, 2
                )
                coords_flat = coords_unwarped.reshape(-1, 2)
                mapped = sample_map_xy(map_xy_full, coords_flat)
                map_xy_out = mapped.reshape(out_h, out_w, 2).astype(np.float32)

        corners_original = None
        if corners_unwarped is not None:
            corners_original = sample_map_xy(map_xy_full, corners_unwarped)

        if not self.config.generate_map_xy or not should_generate_map_xy(
            (output_rgb.shape[0], output_rgb.shape[1]), self.config.max_map_pixels
        ):
            map_xy_out = None

        return DeskewResult(
            image_rgb=output_rgb,
            corners_original=corners_original,
            map_xy=map_xy_out,
        )


class DocScannerDeskewBackend:
    """DocScanner localization backend using the segmentation module."""

    def __init__(self, config: "DocScannerDeskewConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__, "init")
        self.console.set_verbose(config.verbose).set_debug(config.is_debug)
        self._model = None
        self._device: str | None = None
        self._repo_dir: Path | None = None

    def _resolve_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device

        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self) -> None:
        if self._model is not None:
            return

        repo_dir = Path(self.config.repo_dir)
        if not repo_dir.exists():
            raise RuntimeError(
                f"DocScanner repo not found at {repo_dir}. "
                "Clone it there or update docscanner.repo_dir."
            )

        import sys

        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

        try:
            from seg import U2NETP  # type: ignore
        except Exception as exc:  # pragma: no cover - external dependency
            raise RuntimeError(f"Failed to import DocScanner seg module: {exc}") from exc

        import torch

        self._device = self._resolve_device()
        model = U2NETP(3, 1).to(self._device).eval()

        weights_path = Path(self.config.seg_weights) if self.config.seg_weights else None
        if weights_path is None or not weights_path.exists():
            raise RuntimeError(
                "DocScanner seg weights not found. Expected seg.pth under model_pretrained."
            )

        state = torch.load(weights_path, map_location=self._device)
        model_state = model.state_dict()
        cleaned = {}
        for key, value in state.items():
            cleaned_key = key
            if cleaned_key.startswith("module."):
                cleaned_key = cleaned_key[7:]
            if cleaned_key.startswith("model."):
                cleaned_key = cleaned_key[6:]
            if cleaned_key in model_state and model_state[cleaned_key].shape == value.shape:
                cleaned[cleaned_key] = value
        model_state.update(cleaned)
        model.load_state_dict(model_state)

        self._model = model
        self._repo_dir = repo_dir

    def _detect_corners(
        self,
        image_rgb: UInt8[NDArray, "H W 3"],
    ) -> Float32[NDArray, "4 2"] | None:
        import torch

        assert self._model is not None
        assert self._device is not None

        im = image_rgb.astype(np.float32) / 255.0
        h, w = im.shape[:2]
        im_resized = cv2.resize(im, (288, 288))
        tensor = torch.from_numpy(im_resized.transpose(2, 0, 1)).unsqueeze(0).to(self._device)

        with torch.no_grad():
            mask_pred, *_ = self._model(tensor)

        mask = mask_pred[0, 0].detach().cpu().numpy()
        doc_mask = (mask > self.config.mask_threshold).astype(np.uint8)
        doc_mask = cv2.morphologyEx(doc_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(doc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea).astype(np.float32)
        contour[:, 0, 0] *= w / 288.0
        contour[:, 0, 1] *= h / 288.0

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) != 4:
            rect = cv2.minAreaRect(contour)
            approx = cv2.boxPoints(rect).astype(np.float32).reshape(-1, 1, 2)

        corners = approx.reshape(-1, 2).astype(np.float32)
        return order_points_clockwise(corners)

    def deskew(
        self,
        image_rgb: UInt8[NDArray, "H W 3"],
    ) -> DeskewResult:
        """Deskew via DocScanner segmentation localization.

        Args:
            image_rgb (ndarray[uint8]): Input RGB image, shape (H, W, 3).

        Returns:
            DeskewResult containing deskewed RGB image, corners, and optional map_xy.
        """
        self._load_model()

        corners = self._detect_corners(image_rgb)
        if corners is None:
            if not self.config.fallback_to_original:
                raise RuntimeError("DocScanner backend failed to find document corners.")
            self.console.warn("DocScanner failed to find corners; using full image.")
            corners = build_full_image_corners(image_rgb.shape[0], image_rgb.shape[1])

        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        warped_bgr, matrix, output_size = warp_from_corners(bgr, corners)
        warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)

        map_xy = None
        if self.config.generate_map_xy and should_generate_map_xy(
            output_size, self.config.max_map_pixels
        ):
            map_xy = build_map_xy_from_homography(matrix, output_size)

        return DeskewResult(
            image_rgb=warped_rgb,
            corners_original=corners,
            map_xy=map_xy,
        )
