import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def diagnose_png(path: Path) -> None:
    """Diagnose PNG file properties for debugging."""
    logger.info("Diagnosing PNG: %s", path)

    with Image.open(path).convert("L") as img:
        logger.info("=== %s ===", path)
        logger.info("Mode: %s, Size: %s, Format: %s", img.mode, img.size, img.format)

        # Check for transparency
        if img.mode == "RGBA":
            alpha = img.split()[3]
            alpha_arr = np.array(alpha)
            logger.info(
                "Alpha channel: min=%s, max=%s, has_transparency=%s",
                alpha_arr.min(),
                alpha_arr.max(),
                alpha_arr.min() < 255,
            )

        # Get pixel value range
        arr = np.array(img)
        logger.info(
            "Array: shape=%s, dtype=%s, range=[%s, %s], mean=%.2f",
            arr.shape,
            arr.dtype,
            arr.min(),
            arr.max(),
            arr.mean(),
        )
