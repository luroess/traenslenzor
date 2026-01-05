"""FastMCP server for the document classifier.

Exposes `classify_document` over MCP and can be run via streamable HTTP,
mirroring the layout detector server.
"""

from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from traenslenzor.doc_classifier.configs.mcp_config import DocClassifierMCPConfig
from traenslenzor.doc_classifier.configs.path_config import PathConfig
from traenslenzor.doc_classifier.mcp_integration.runtime import DocClassifierRuntime
from traenslenzor.doc_classifier.utils import Console
from traenslenzor.file_server.client import SessionClient
from traenslenzor.file_server.session_state import SessionState
from traenslenzor.supervisor.config import settings

ADDRESS = "127.0.0.1"
PORT = 8007
DOC_CLASSIFIER_BASE_PATH = f"http://{ADDRESS}:{PORT}/mcp"

doc_classifier_mcp = FastMCP(
    name="doc-classifier",
    strict_input_validation=True,
)
console = Console.with_prefix("doc-classifier-mcp", "init")

DocClassifierMCPConfig.model_rebuild(
    _types_namespace={"DocClassifierRuntime": DocClassifierRuntime}
)

_runtime: DocClassifierRuntime | None = None
_runtime_checkpoint: str | None = None


def get_runtime() -> DocClassifierRuntime:
    global _runtime
    global _runtime_checkpoint

    desired_checkpoint = settings.doc_classifier.checkpoint_path
    if _runtime is None or desired_checkpoint != _runtime_checkpoint:
        config = DocClassifierMCPConfig(
            checkpoint_path=None,
            device="auto",
        )
        if desired_checkpoint:
            try:
                config.checkpoint_path = PathConfig().resolve_checkpoint_path(desired_checkpoint)
            except FileNotFoundError as exc:
                console.warn(f"Checkpoint '{desired_checkpoint}' not found; using mock. {exc}")
                config.checkpoint_path = None
        _runtime = config.setup_target()
        _runtime_checkpoint = desired_checkpoint
    return _runtime


def reset_runtime() -> None:
    """Force rebuilding the classifier runtime on the next call."""
    global _runtime, _runtime_checkpoint
    _runtime = None
    _runtime_checkpoint = None


_CLASSIFY_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": {"type": "number"},
}


@doc_classifier_mcp.tool(
    name="classify_document",
    title="Classify document",
    description=(
        "Classify the current document and write class probabilities to the session. "
        "The supervisor injects `session_id` automatically; do not ask the user for it. "
        "Returns a mapping of the top-k classes to probabilities."
    ),
    tags={"doc-classifier", "session", "classification"},
    annotations=ToolAnnotations(
        title="Classify document",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    output_schema=_CLASSIFY_OUTPUT_SCHEMA,
)
async def classify_document(
    session_id: Annotated[
        str,
        Field(
            description=(
                "File server session id (injected by the supervisor; do not ask the user)."
            ),
            pattern="^[0-9a-fA-F-]{36}$",
        ),
    ],
) -> dict[str, float]:
    """Classify the session document and update class probabilities.

    Args:
        session_id (str): File server session id injected by the supervisor.

    Returns:
        dict[str, float]: Mapping of class labels to probabilities for the top-k results.
    """
    runtime = get_runtime()
    session = await SessionClient.get(session_id)
    if session.extractedDocument is None:
        # TODO: will the supervisor llm be informed properly about this error?
        msg = "Session has no extractedDocument; run the doc-scanner tool to deskew the document first."
        console.error(msg)
        raise ToolError(msg)

    file_id = session.extractedDocument.id
    if not file_id:
        msg = "Extracted document id missing; run the doc-scanner tool to deskew the document."
        console.error(msg)
        raise ToolError(msg)

    payload = await runtime.classify_file_id(file_id, top_k=3)
    probabilities = runtime.predictions_to_probabilities(payload["predictions"])
    if not probabilities:
        msg = "Classification returned no probabilities."
        console.error(msg)
        raise ToolError(msg)

    def update_session(state: SessionState) -> None:
        state.class_probabilities = probabilities

    await SessionClient.update(session_id, update_session)

    return probabilities


async def run():
    await doc_classifier_mcp.run_async(
        transport="streamable-http", host=ADDRESS, port=PORT, show_banner=False
    )


__all__ = [
    "doc_classifier_mcp",
    "classify_document",
    "run",
    "DOC_CLASSIFIER_BASE_PATH",
    "ADDRESS",
    "PORT",
]
