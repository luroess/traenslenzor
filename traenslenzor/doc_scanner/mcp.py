"""FastMCP server for document deskewing and prettified scans."""

from __future__ import annotations

from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from traenslenzor.doc_classifier.utils import Console
from traenslenzor.doc_scanner.configs import DocScannerMCPConfig
from traenslenzor.doc_scanner.runtime import DocScannerRuntime
from traenslenzor.file_server.client import SessionClient
from traenslenzor.file_server.session_state import DeskewBackend, SessionState
from traenslenzor.supervisor.config import settings

ADDRESS = "127.0.0.1"
PORT = 8004
DOC_SCANNER_BASE_PATH = f"http://{ADDRESS}:{PORT}/mcp"

doc_scanner_mcp = FastMCP(
    name="doc-scanner",
    strict_input_validation=True,
)
console = Console.with_prefix("doc-scanner-mcp", "init")

DocScannerMCPConfig.model_rebuild(_types_namespace={"DocScannerRuntime": DocScannerRuntime})

_runtime: DocScannerRuntime | None = None


def get_runtime() -> DocScannerRuntime:
    global _runtime
    if _runtime is None:
        config = DocScannerMCPConfig()
        config.uvdoc.device = settings.doc_scanner.uvdoc_device
        _runtime = config.setup_target()
    return _runtime


@doc_scanner_mcp.tool(
    name="deskew_document",
    title="Deskew document",
    description=(
        "Deskew/prettify the current document and write the extracted document metadata to the session. "
        "The supervisor injects `session_id` automatically; do not ask the user for it."
    ),
    tags={"doc-scanner", "deskew", "session"},
    annotations=ToolAnnotations(
        title="Deskew document",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
async def deskew_document(
    session_id: Annotated[
        str,
        Field(
            description="File server session id (injected by the supervisor; do not ask the user).",
            pattern="^[0-9a-fA-F-]{36}$",
        ),
    ],
    backend: Annotated[
        DeskewBackend | None,
        Field(
            description="Optional backend override: opencv or uvdoc.",
            default=None,
        ),
    ] = None,
) -> dict:
    """Deskew the session document and update the session state.

    Args:
        session_id (str): File server session id injected by the supervisor.
        backend (DeskewBackend | None): Optional backend override.

    Returns:
        dict: Serialized ExtractedDocument payload.
    """
    runtime = get_runtime()
    session = await SessionClient.get(session_id)
    preferred_backend = session.deskew_backend if session is not None else None
    try:
        if backend is not None and backend != preferred_backend:
            await SessionClient.update(
                session_id,
                lambda state: setattr(state, "deskew_backend", backend),
            )
        extracted = await runtime.scan_session(session_id)
    except Exception as exc:
        console.error(str(exc))
        raise ToolError(str(exc)) from exc

    def update_session(state: SessionState) -> None:
        state.extractedDocument = extracted
        state.deskew_backend = extracted.backend

    await SessionClient.update(session_id, update_session)
    return extracted.model_dump()


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
