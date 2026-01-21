"""FastMCP server for document deskewing and prettified scans."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from PIL import Image
from pydantic import Field

from traenslenzor.doc_classifier.utils import Console
from traenslenzor.doc_scanner.configs import DocScannerMCPConfig
from traenslenzor.doc_scanner.runtime import DocScannerRuntime
from traenslenzor.doc_scanner.superres import (
    ensure_openvino_text_sr_model,
    super_resolve_text_image,
)
from traenslenzor.file_server.client import FileClient, SessionClient
from traenslenzor.file_server.session_state import SessionState, SuperResolvedDocument

ADDRESS = "127.0.0.1"
PORT = 8004
DOC_SCANNER_BASE_PATH = f"http://{ADDRESS}:{PORT}/mcp"
_DOC_SCANNER_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "doc-scanner.toml"

doc_scanner_mcp = FastMCP(
    name="doc-scanner",
    strict_input_validation=True,
)
console = Console.with_prefix("doc-scanner-mcp", "init")

DocScannerMCPConfig.model_rebuild(_types_namespace={"DocScannerRuntime": DocScannerRuntime})

_runtime: DocScannerRuntime | None = None
_runtime_config_signature: dict[str, object] | None = None

_DESKEW_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "extractedDocument": {
            "type": "object",
            "description": "Extracted document metadata from the deskew operation.",
            "properties": {
                "documentCoordinates": {
                    "type": "array",
                    "description": (
                        "Four corner points of the document in original image pixel coordinates "
                        "(order: UL, UR, LR, LL)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "number",
                                "description": "X coordinate in original image pixels.",
                            },
                            "y": {
                                "type": "number",
                                "description": "Y coordinate in original image pixels.",
                            },
                        },
                        "required": ["x", "y"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["documentCoordinates"],
            "additionalProperties": False,
        }
    },
    "required": ["extractedDocument"],
    "additionalProperties": False,
}
_SUPERRES_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "superResolvedDocument": {
            "type": "object",
            "description": "Super-resolved document metadata.",
            "properties": {
                "id": {"type": "string", "description": "File id of the super-resolved image."},
                "sourceId": {"type": "string", "description": "Source image file id."},
                "source": {
                    "type": "string",
                    "description": "Source type (raw, deskewed, rendered).",
                },
                "model": {"type": "string", "description": "Super-resolution model name."},
                "scale": {"type": "integer", "description": "Upscaling factor."},
            },
            "required": ["id", "sourceId", "source", "model", "scale"],
            "additionalProperties": False,
        }
    },
    "required": ["superResolvedDocument"],
    "additionalProperties": False,
}


def _load_doc_scanner_config() -> DocScannerMCPConfig:
    """Load the doc-scanner MCP config, falling back to defaults if needed."""
    config_path = _DOC_SCANNER_CONFIG_PATH
    if not config_path.exists() or config_path.stat().st_size == 0:
        return DocScannerMCPConfig()

    try:
        return DocScannerMCPConfig.from_toml(config_path)
    except Exception as exc:
        console.error(f"Failed to load doc-scanner config at {config_path}: {exc}")
        return DocScannerMCPConfig()


def get_runtime() -> DocScannerRuntime:
    global _runtime
    global _runtime_config_signature

    config = _load_doc_scanner_config()
    config_signature = config.model_dump()
    if _runtime is None or config_signature != _runtime_config_signature:
        _runtime = config.setup_target()
        _runtime_config_signature = config_signature
    return _runtime


@doc_scanner_mcp.tool(
    name="deskew_document",
    title="Deskew document",
    description=(
        "Deskew/prettify the current document and write the extracted document metadata to the session. "
        "The supervisor injects `session_id` automatically; do not ask the user for it. "
        "Returns the extracted document payload."
    ),
    tags={"doc-scanner", "deskew", "session"},
    annotations=ToolAnnotations(
        title="Deskew document",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
    output_schema=_DESKEW_OUTPUT_SCHEMA,
)
async def deskew_document(
    session_id: Annotated[
        str,
        Field(
            description="File server session id.",
            pattern="^[0-9a-fA-F-]{36}$",
        ),
    ],
    context_level: Annotated[
        Literal["minimal", "full"],
        Field(
            default="minimal",
            description=(
                "Controls how much extraction context to include in the response (minimal, full)."
            ),
        ),
    ] = "minimal",
) -> dict[str, object]:
    """Deskew the session document and update the session state.

    Args:
        session_id (str): File server session id injected by the supervisor.
    Returns:
        dict[str, object]: Payload containing the serialized extracted document.
    """
    runtime = get_runtime()
    try:
        extracted = await runtime.scan_session(
            session_id,
            context_level=context_level,
        )
    except Exception as exc:
        console.error(str(exc))
        raise ToolError(str(exc)) from exc

    def update_session(state: SessionState) -> None:
        state.extractedDocument = extracted

    await SessionClient.update(session_id, update_session)

    payload = {
        "documentCoordinates": extracted.documentCoordinates,
    }

    return {"extractedDocument": payload}


@doc_scanner_mcp.tool(
    name="super_resolve_document",
    title="Super-resolve document",
    description=(
        "Upscale a document image for OCR using OpenVINO text-image-super-resolution-0001. "
        "Stores the super-resolved image in the session."
    ),
    tags={"doc-scanner", "superres", "session"},
    annotations=ToolAnnotations(
        title="Super-resolve document",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    output_schema=_SUPERRES_OUTPUT_SCHEMA,
)
async def super_resolve_document(
    session_id: Annotated[
        str,
        Field(
            description="File server session id.",
            pattern="^[0-9a-fA-F-]{36}$",
        ),
    ],
    source: Annotated[
        Literal["raw", "deskewed", "rendered"],
        Field(
            default="deskewed",
            description="Which image to upscale: raw, deskewed, or rendered.",
        ),
    ] = "deskewed",
) -> dict[str, object]:
    """Super-resolve the session document and update the session state."""
    console.log("Loading session for super-resolution.")
    config = _load_doc_scanner_config()
    session = await SessionClient.get(session_id)

    if source == "raw":
        source_id = session.rawDocumentId
    elif source == "rendered":
        source_id = session.renderedDocumentId
    else:
        source_id = session.extractedDocument.id if session.extractedDocument else None

    if not source_id:
        msg = f"Session has no {source} document to super-resolve. Call deskew_document() first."
        console.error(msg)
        raise ToolError(msg)

    console.log(f"Downloading {source} image {source_id}.")
    image = await FileClient.get_image(source_id)
    if image is None:
        msg = f"Source image not found: {source_id}"
        console.error(msg)
        raise ToolError(msg)

    console.log("Ensuring OpenVINO SR model files are available.")
    model_files = await ensure_openvino_text_sr_model(
        models_dir=config.superres_models_dir,
        allow_download=config.superres_allow_download,
    )

    console.log("Running text-image-super-resolution-0001.")
    image_rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    result = super_resolve_text_image(
        image_rgb,
        xml_path=model_files.xml,
        bin_path=model_files.bin,
        device=config.superres_device,
    )

    output_image = Image.fromarray(result.image_rgb)
    output_id = await FileClient.put_img(f"{session_id}_superres.png", output_image)
    if output_id is None:
        msg = "Failed to upload super-resolved image."
        console.error(msg)
        raise ToolError(msg)

    payload = SuperResolvedDocument(
        id=output_id,
        sourceId=source_id,
        source=source,
        model="text-image-super-resolution-0001",
        scale=result.scale,
    )

    def update_session(state: SessionState) -> None:
        state.superResolvedDocument = payload

    console.log("Updating session with super-resolved document metadata.")
    await SessionClient.update(session_id, update_session)
    return {"superResolvedDocument": payload.model_dump()}


async def run() -> None:
    await doc_scanner_mcp.run_async(transport="streamable-http", host=ADDRESS, port=PORT)


__all__ = [
    "doc_scanner_mcp",
    "deskew_document",
    "super_resolve_document",
    "run",
    "DOC_SCANNER_BASE_PATH",
    "ADDRESS",
    "PORT",
]
