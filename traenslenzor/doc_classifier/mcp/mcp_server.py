"""FastMCP server for the document classifier.

Exposes `classify_document` over MCP and can be run via streamable HTTP,
mirroring the layout detector server.
"""

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from traenslenzor.doc_classifier.configs.mcp_config import DocClassifierMCPConfig
from traenslenzor.doc_classifier.mcp.runtime import DocClassifierRuntime
from traenslenzor.doc_classifier.mcp.schemas import DocClassifierResponse
from traenslenzor.doc_classifier.utils import Console

ADDRESS = "127.0.0.1"
PORT = 8003
DOC_CLASSIFIER_BASE_PATH = f"http://{ADDRESS}:{PORT}/mcp"

doc_classifier_mcp = FastMCP(
    name="doc-classifier",
    strict_input_validation=True,
)
console = Console.with_prefix("doc-classifier-mcp", "init")

_runtime: DocClassifierRuntime | None = None


def get_runtime() -> DocClassifierRuntime:
    global _runtime
    if _runtime is None:
        _runtime = DocClassifierMCPConfig().setup_target()
    return _runtime


async def classify_document(
    document_id: str,
    top_k: int | str = 3,
) -> DocClassifierResponse:
    """Async helper used by both FastMCP registration and tests."""
    runtime = get_runtime()

    try:
        top_k_int = int(top_k)
    except (TypeError, ValueError) as exc:
        raise ToolError("top_k must be an integer or digit string") from exc

    if not 1 <= top_k_int <= 16:
        raise ToolError("top_k must be between 1 and 16")

    try:
        raw = await runtime.classify_file_id(document_id, top_k=top_k_int)
    except FileNotFoundError as exc:
        raise ToolError(str(exc)) from exc

    probabilities = {
        pred["label"]: float(pred["probability"]) for pred in raw["predictions"][:top_k_int]
    }

    return DocClassifierResponse(
        probabilities=probabilities,
    )


doc_classifier_mcp.tool(
    name="classify_document",
    description=(
        "Classify a document image into one of the supported document classes. "
        "Provide the file id returned by FileClient.put_img."
    ),
)(classify_document)


async def run():
    await doc_classifier_mcp.run_async(transport="streamable-http", host=ADDRESS, port=PORT)


__all__ = [
    "doc_classifier_mcp",
    "classify_document",
    "run",
    "DOC_CLASSIFIER_BASE_PATH",
    "ADDRESS",
    "PORT",
]
