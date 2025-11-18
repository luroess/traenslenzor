"""FastMCP server for the document classifier.

Exposes `classify_document` over MCP and can be run via streamable HTTP,
mirroring the layout detector server.
"""

from pathlib import Path

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from traenslenzor.doc_classifier.configs.mcp_config import DocClassifierMCPConfig
from traenslenzor.doc_classifier.mcp.runtime import DocClassifierRuntime
from traenslenzor.doc_classifier.mcp.schemas import DocClassifierResponse
from traenslenzor.doc_classifier.utils import Console

ADDRESS = "127.0.0.1"
PORT = 8002
DOC_CLASSIFIER_BASE_PATH = f"http://{ADDRESS}:{PORT}/mcp"

doc_classifier_mcp = FastMCP(
    name="doc-classifier",
    strict_input_validation=True,  # enforce Pydantic validation on inputs
)
console = Console.with_prefix("doc-classifier-mcp", "init")

_runtime: DocClassifierRuntime | None = None


def get_runtime() -> DocClassifierRuntime:
    global _runtime
    if _runtime is None:
        _runtime = DocClassifierMCPConfig().setup_target()
    return _runtime


# TODO: direclty use Tool instead of decorator!
@doc_classifier_mcp.tool(
    name="classify_document",
    description=(
        "Classify a document image into one of the supported document classes. "
        "Returns a probability map over the top-k labels."
    ),
    annotations={
        "title": "Classify document",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def classify_document(
    # ctx: Context,
    path: Path,
    top_k: int = 3,
) -> DocClassifierResponse:
    runtime = get_runtime()
    path = runtime.resolve_path(path)

    if not path.exists():
        raise ToolError(f"File not found: {path}")

    # ctx.info(f"Running doc-classifier on {path}")
    raw = runtime.classify_path(path, top_k=top_k)

    probabilities = {
        pred["label"]: float(pred["probability"]) for pred in raw["predictions"][:top_k]
    }

    return DocClassifierResponse(
        top_k=top_k,
        probabilities=probabilities,
        model_version=runtime.model_version,
    )


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
