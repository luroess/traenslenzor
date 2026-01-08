"""FastMCP server for document deskewing and prettified scans."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from traenslenzor.doc_classifier.utils import Console
from traenslenzor.doc_scanner.configs import DocScannerMCPConfig
from traenslenzor.doc_scanner.runtime import DocScannerRuntime
from traenslenzor.file_server.client import SessionClient
from traenslenzor.file_server.session_state import SessionState

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

_EXTRACTED_DOCUMENT_SCHEMA: dict[str, object] = {
    "type": "object",
    "description": "Deskewed document metadata produced by the doc-scanner.",
    "properties": {
        "id": {
            "type": "string",
            "description": "File id of the deskewed (extracted) document image.",
        },
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
        "mapXYId": {
            "type": ["string", "null"],
            "description": "File id of the map_xy flow field.",
        },
        "mapXYShape": {
            "type": ["array", "null"],
            "items": {"type": "integer"},
            "description": "Shape of map_xy as [H, W, 2].",
        },
    },
    "required": ["id", "documentCoordinates", "mapXYId", "mapXYShape"],
    "additionalProperties": False,
}
_DESKEW_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "extractedDocument": {
            **_EXTRACTED_DOCUMENT_SCHEMA,
            "description": "Extracted document metadata from the deskew operation.",
        }
    },
    "required": ["extractedDocument"],
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
) -> dict[str, object]:
    """Deskew the session document and update the session state.

    Args:
        session_id (str): File server session id injected by the supervisor.
    Returns:
        dict[str, object]: Payload containing the serialized extracted document.
    """
    runtime = get_runtime()
    try:
        extracted = await runtime.scan_session(session_id)
    except Exception as exc:
        console.error(str(exc))
        raise ToolError(str(exc)) from exc

    def update_session(state: SessionState) -> None:
        state.extractedDocument = extracted

    await SessionClient.update(session_id, update_session)
    return {"extractedDocument": extracted.model_dump()}


async def run() -> None:
    await doc_scanner_mcp.run_async(transport="streamable-http", host=ADDRESS, port=PORT)


__all__ = [
    "doc_scanner_mcp",
    "deskew_document",
    "run",
    "DOC_SCANNER_BASE_PATH",
    "ADDRESS",
    "PORT",
]
