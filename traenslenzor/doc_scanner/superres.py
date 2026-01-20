"""OpenVINO-based super-resolution for text images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import httpx
import numpy as np
import openvino as ov

SuperResSource = Literal["raw", "deskewed", "rendered"]

OPENVINO_TEXT_SR_BASE_URL = (
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/"
    "models_bin/1/text-image-super-resolution-0001/FP32"
)
OPENVINO_TEXT_SR_XML = "text-image-super-resolution-0001.xml"
OPENVINO_TEXT_SR_BIN = "text-image-super-resolution-0001.bin"


@dataclass(frozen=True)
class SuperResModelFiles:
    xml: Path
    bin: Path


@dataclass(frozen=True)
class SuperResResult:
    image_rgb: np.ndarray
    scale: int


async def ensure_openvino_text_sr_model(
    *,
    models_dir: Path,
    allow_download: bool,
) -> SuperResModelFiles:
    models_dir.mkdir(parents=True, exist_ok=True)
    xml_path = models_dir / OPENVINO_TEXT_SR_XML
    bin_path = models_dir / OPENVINO_TEXT_SR_BIN
    if xml_path.exists() and bin_path.exists():
        return SuperResModelFiles(xml=xml_path, bin=bin_path)
    if not allow_download:
        raise FileNotFoundError(
            f"Missing OpenVINO SR model files at {models_dir}. "
            "Enable downloads or provide the files."
        )

    async with httpx.AsyncClient() as client:
        xml_resp = await client.get(f"{OPENVINO_TEXT_SR_BASE_URL}/{OPENVINO_TEXT_SR_XML}")
        xml_resp.raise_for_status()
        xml_path.write_bytes(xml_resp.content)

        bin_resp = await client.get(f"{OPENVINO_TEXT_SR_BASE_URL}/{OPENVINO_TEXT_SR_BIN}")
        bin_resp.raise_for_status()
        bin_path.write_bytes(bin_resp.content)

    return SuperResModelFiles(xml=xml_path, bin=bin_path)


def _compile_model(
    *,
    xml_path: Path,
    bin_path: Path,
    device: str,
    height: int,
    width: int,
) -> tuple[ov.CompiledModel, ov.Output]:
    core = ov.Core()
    model = core.read_model(model=str(xml_path), weights=str(bin_path))
    input_layer = model.input(0)
    input_name = input_layer.get_any_name()
    model.reshape({input_name: [1, 1, height, width]})
    compiled = core.compile_model(model, device)
    output = compiled.output(0)
    return compiled, output


def _to_ycrcb(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    return ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]


def _from_ycrcb(y: np.ndarray, cr: np.ndarray, cb: np.ndarray) -> np.ndarray:
    ycrcb = np.stack([y, cr, cb], axis=-1)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def _resize_chroma(channel: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(channel, (width, height), interpolation=cv2.INTER_CUBIC)


def super_resolve_text_image(
    image_rgb: np.ndarray,
    *,
    xml_path: Path,
    bin_path: Path,
    device: str = "CPU",
) -> SuperResResult:
    """Run OpenVINO text-image-super-resolution-0001 on a RGB image.

    The model expects a single-channel input; RGB images are converted to YCrCb, the Y channel is
    super-resolved, and chroma channels are upscaled with bicubic interpolation.
    """

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected image_rgb with shape (H, W, 3).")

    height, width = image_rgb.shape[:2]
    compiled, output = _compile_model(
        xml_path=xml_path,
        bin_path=bin_path,
        device=device,
        height=height,
        width=width,
    )

    y, cr, cb = _to_ycrcb(image_rgb)
    # Open Model Zoo super-resolution demos pass raw 0-255 pixels (no normalization),
    # then scale the output by 255. Keep the same convention here.
    y_in = y.astype(np.float32)[None, None, :, :]

    result = compiled([y_in])[output]
    y_sr = (np.squeeze(result) * 255.0).clip(0, 255).astype(np.uint8)

    out_h, out_w = y_sr.shape[:2]
    cr_up = _resize_chroma(cr, out_w, out_h)
    cb_up = _resize_chroma(cb, out_w, out_h)
    rgb = _from_ycrcb(y_sr, cr_up, cb_up)

    scale = int(round(out_h / max(height, 1)))
    return SuperResResult(image_rgb=rgb, scale=scale)
