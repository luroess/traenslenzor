"""Runtime orchestrator for the document scanner MCP tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from traenslenzor.doc_classifier.utils import Console
from traenslenzor.file_server.client import FileClient, SessionClient
from traenslenzor.file_server.session_state import BBoxPoint, ExtractedDocument

if TYPE_CHECKING:
    from .backends import DeskewResult
    from .configs import DocScannerMCPConfig


class DocScannerRuntime:
    """Runtime helper that executes the configured UVDoc deskew backend."""

    def __init__(self, config: "DocScannerMCPConfig") -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__, "init")
        self.console.set_verbose(config.verbose).set_debug(config.is_debug)
        self._uvdoc_backend = None

    def _get_backend(self):
        from .backends import UVDocDeskewBackend

        if self._uvdoc_backend is None:
            self._uvdoc_backend = UVDocDeskewBackend(self.config)
        return self._uvdoc_backend

    async def scan_session(
        self,
        session_id: str,
        *,
        crop_document: bool | None = None,
    ) -> ExtractedDocument:
        """Deskew the session's raw document and upload results.

        Args:
            session_id (str): File server session id.
            crop_document (bool | None): Override cropping to the detected page contour.
                When None, defaults to the runtime config.

        Returns:
            ExtractedDocument: Deskewed document metadata including ids and coordinates.
        """
        session = await SessionClient.get(session_id)
        if session.rawDocumentId is None:
            raise RuntimeError("Session has no rawDocumentId; load a document first.")

        image = await FileClient.get_image(session.rawDocumentId)
        if image is None:
            raise RuntimeError(f"Document not found: {session.rawDocumentId}")

        image_rgb = np.array(image.convert("RGB"), dtype=np.uint8)

        deskew_backend = self._get_backend()
        result: DeskewResult = deskew_backend.deskew(
            image_rgb,
            crop_document=crop_document,
        )

        output_image = Image.fromarray(result.image_rgb)
        output_id = await FileClient.put_img(f"{session_id}_deskewed.png", output_image)
        if output_id is None:
            raise RuntimeError("Failed to upload deskewed document image")

        map_xy_id = None
        map_xy_shape = None
        if result.map_xy is not None:
            map_xy_id = await FileClient.put_numpy_array(f"{session_id}_map_xy.npy", result.map_xy)
            if map_xy_id is None:
                raise RuntimeError("Failed to upload map_xy array")
            map_xy_shape = result.map_xy.shape

        transformation_matrix = result.transformation_matrix

        coords = []
        if result.corners_original is not None:
            coords = [
                BBoxPoint(x=float(pt[0]), y=float(pt[1]))
                for pt in np.asarray(result.corners_original, dtype=np.float32)
            ]

        return ExtractedDocument(
            id=output_id,
            documentCoordinates=coords,
            mapXYId=map_xy_id,
            mapXYShape=map_xy_shape,
            transformation_matrix=(
                transformation_matrix.tolist() if transformation_matrix is not None else None
            ),
        )
