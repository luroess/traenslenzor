"""Deskew backend implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from jaxtyping import Float32, UInt8
from numpy import ndarray as NDArray
from py_reform.models.uvdoc_model import DEFAULT_IMG_SIZE, DEFAULT_MODEL_PATH, UVDocNet

from traenslenzor.doc_classifier.utils import Console

from .utils import (
    build_full_image_corners,
    compute_map_xy_stride,
    find_page_corners,
    sample_map_xy,
    warp_from_corners,
)

if TYPE_CHECKING:
    from .configs import DocScannerMCPConfig


@dataclass(slots=True)
class DeskewResult:
    """Container for deskew outputs."""

    image_rgb: UInt8[NDArray, "H W 3"]
    """Deskewed RGB image."""
    transformation_matrix: Float32[NDArray, "3 3"] | None
    """Approximate homography mapping original -> deskewed output."""
    corners_original: Float32[NDArray, "4 2"] | None
    """Document corners in original image coordinates (UL, UR, LR, LL)."""
    map_xy: Float32[NDArray, "H W 2"] | None
    """Optional map_xy mapping output pixels -> original pixels."""
    map_xyz: Float32[NDArray, "Gh Gw 3"] | None
    """Optional UVDoc 3D grid (coarse) for surface-aware post-processing."""


class UVDocDeskewBackend:
    """UVDoc neural unwarping backend (py_reform)."""

    def __init__(self, config: "DocScannerMCPConfig") -> None:
        self.config = config
        self.console = (
            Console.with_prefix(self.__class__.__name__)
            .set_verbose(config.verbose)
            .set_debug(config.is_debug)
        )

        self._model: UVDocNet | None = None
        self._device: str | None = None

    def _resolve_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self) -> None:
        if self._model is not None:
            return

        self._device = self._resolve_device()
        self._model = UVDocNet(
            num_filter=self.config.num_filter, kernel_size=self.config.kernel_size
        ).to(self._device)
        self._model.eval()

        model_path = Path(DEFAULT_MODEL_PATH)
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

    def deskew(
        self,
        image_rgb: UInt8[NDArray, "H W 3"],
    ) -> DeskewResult:
        """Deskew (dewarp) a document image via [UVDoc neural unwarping](https://arxiv.org/html/2302.02887v2).

        Instead of explicitly estimating a homography from detected corners, a _dense sampling grid_ is predicted that
        encodes how to resample the warped input image into a flat, deskewed output.

        Conceptually, two artifacts are produced:

        1) An **unwarped RGB image** (optionally cropped to the page rectangle).
        2) A **mapping** that allows projecting the unwarped result (or edits on it) back onto the
           original image coordinate system (used by :mod:`traenslenzor.doc_scanner.backtransform`).

        ## UVDocNet usage and coordinate systems
        ------------------------------------
        - UVDocNet predicts a coarse 2-channel grid (x,y) in the same normalized coordinate system
          used by :func:`torch.nn.functional.grid_sample`: each coordinate is in [-1, 1].
        - The model is run on a resized image of fixed size ``DEFAULT_IMG_SIZE`` because the
          published weights were trained with a fixed input resolution.
        - The predicted grid is upsampled to the original image resolution and applied to the
          **full-resolution** input using ``grid_sample`` to produce the unwarped image.
        - ``align_corners=True`` is used consistently for both the grid upsampling and
          ``grid_sample``. With this convention, -1 and +1 map to the centers of the first/last
          pixels, which is why normalized coordinates are converted to pixels via ``(w - 1)`` and
          ``(h - 1)``.

        High-level procedure
        --------------------
        1) Normalize the input image ``image_rgb`` to float32 in [0, 1] and construct:
           - ``orig``: the full-res tensor (1, 3, H, W) that is resampled from.
           - ``inp``: a resized tensor (1, 3, Hm, Wm) used for UVDocNet inference.
        2) Run UVDocNet on ``inp`` to get ``points2d``: a coarse normalized sampling grid
           (1, 2, Gh, Gw).
        3) Upsample ``points2d`` to full resolution and reshape the sampling grid to the
           grid_sample layout (1, H, W, 2).
        4) Compute ``unwarped = grid_sample(orig, grid)`` to obtain an unwarped image at the same
           resolution as the input.
        5) Convert the normalized grid to a pixel-space mapping:
           ``map_xy_full[y, x] = (x_in, y_in)`` where (x_in, y_in) are float pixel coordinates in
           the **original** input image.
        6) Optionally detect a page quadrilateral on the unwarped image (simple contour-based
           detection) and crop/rectify it. If cropping is enabled, we also derive a corresponding
           ``map_xy_out`` for the cropped output by mapping cropped pixel coordinates back through
           the inverse crop homography and sampling ``map_xy_full``.
        7) Optionally downsample ``map_xy`` according to ``max_map_pixels``. This keeps stored flow
           fields manageable; consumers can upsample as needed.

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

        # --- Prepare original (full-res) source tensor ----------------------------------------
        # image_rgb: (H, W, 3) uint8 in RGB order.
        orig_np = image_rgb.astype(np.float32) / 255.0
        h, w = orig_np.shape[:2]

        # orig: (1, 3, H, W) float32 in [0, 1] on device; this is what we sample from.
        orig = (
            torch.from_numpy(orig_np).permute(2, 0, 1).unsqueeze(0).to(self._device)
        )  # (1, 3, H, W)

        # --- Prepare model input (UVDoc expects a fixed input size) ---------------------------
        # DEFAULT_IMG_SIZE is (Wm, Hm). Resize only for inference; we later upsample the grid.
        inp_resized = Image.fromarray(image_rgb).resize(
            DEFAULT_IMG_SIZE, resample=Image.Resampling.BILINEAR
        )
        inp_np = np.array(inp_resized).astype(np.float32) / 255.0
        # inp: (1, 3, Hm, Wm) float32 in [0, 1] on device.
        inp = torch.from_numpy(inp_np).permute(2, 0, 1).unsqueeze(0).to(self._device)

        # --- Predict a coarse UV sampling grid with UVDocNet ----------------------------------
        # points2d: (B, 2, Gh, Gw) in normalized [-1, 1] image coords for grid_sample (x,y).
        with torch.no_grad():
            points2d, points3d = self._model.forward(inp)

        if points2d.dim() == 3:
            points2d = points2d.unsqueeze(0)
        if points3d.dim() == 3:
            points3d = points3d.unsqueeze(0)

        map_xyz = points3d[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

        # --- Upsample grid to full resolution and unwarp via grid_sample ------------------
        # grid_2ch: (1, 2, H, W) -> grid: (1, H, W, 2) for grid_sample.
        grid_2ch = F.interpolate(points2d, size=(h, w), mode="bilinear", align_corners=True)
        grid = grid_2ch.permute(0, 2, 3, 1).clamp(-1.0, 1.0)

        # unwarped: (1, 3, H, W) float32 in [0, 1]; convert to (H, W, 3) uint8 RGB.
        unwarped = F.grid_sample(orig, grid, align_corners=True)
        unwarped_np = unwarped[0].permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
        unwarped_rgb = (unwarped_np * 255).astype(np.uint8)

        # --- Convert normalized grid to pixel-space map_xy ---------
        # map_xy_full: (H, W, 2) float32, where map_xy_full[y,x] = (x_in, y_in) in original image.
        grid_cpu = grid[0].detach().cpu()
        map_xy_full = (
            torch.stack(
                [
                    (grid_cpu[:, :, 0] + 1.0) * 0.5 * (w - 1),  # x_in
                    (grid_cpu[:, :, 1] + 1.0) * 0.5 * (h - 1),  # y_in
                ],
                dim=-1,
            )
            .numpy()
            .astype(np.float32)
        )

        # --- Optionally detect/crop the page on the unwarped result ---------------------------
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
            corners_unwarped = build_full_image_corners(
                unwarped_rgb.shape[0], unwarped_rgb.shape[1]
            )

        # Choose a stride so map_xy stays under the configured pixel budget.
        map_xy_stride = compute_map_xy_stride((h, w), self.config.max_map_pixels)

        if corners_unwarped is None:
            # No crop: output is the full unwarped image; map_xy is a strided view of map_xy_full.
            output_rgb = unwarped_rgb
            map_xy_out = map_xy_full[::map_xy_stride, ::map_xy_stride]
        else:
            # Crop: warp the unwarped image to the detected page rectangle.
            unwarped_bgr = cv2.cvtColor(unwarped_rgb, cv2.COLOR_RGB2BGR)
            cropped_bgr, crop_matrix, output_size = warp_from_corners(
                unwarped_bgr, corners_unwarped
            )
            output_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

            map_xy_out = None
            if self.config.generate_map_xy:
                # Build a map_xy for the cropped output by inverting the crop homography and
                # sampling map_xy_full in unwarped-space coordinates.
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
            # Map detected page corners from unwarped space back into original-image coordinates.
            corners_original = sample_map_xy(map_xy_full, corners_unwarped)

        transformation_matrix = None
        if corners_original is not None:
            out_h, out_w = output_rgb.shape[:2]
            output_corners = np.array(
                [
                    [0.0, 0.0],
                    [float(out_w - 1), 0.0],
                    [float(out_w - 1), float(out_h - 1)],
                    [0.0, float(out_h - 1)],
                ],
                dtype=np.float32,
            )
            transformation_matrix = cv2.getPerspectiveTransform(
                corners_original.astype(np.float32),
                output_corners,
            ).astype(np.float32)

        if not self.config.generate_map_xy:
            map_xy_out = None

        return DeskewResult(
            image_rgb=output_rgb,
            transformation_matrix=transformation_matrix,
            corners_original=corners_original,
            map_xy=map_xy_out,
            map_xyz=map_xyz,
        )
