from langchain_mcp_adapters.client import Connection, MultiServerMCPClient

from traenslenzor.doc_classifier.mcp.mcp_server import DOC_CLASSIFIER_BASE_PATH
from traenslenzor.image_renderer.server import IMAGE_RENDERER_BASE_PATH
from traenslenzor.layout_detector.layout_detector import LAYOUT_DETECTOR_BASE_PATH

MCP_SERVERS: dict[str, Connection] = {
    "layout_detector": {
        "transport": "streamable_http",
        "url": LAYOUT_DETECTOR_BASE_PATH,
    },
    "image_renderer": {
        "transport": "streamable_http",
        "url": IMAGE_RENDERER_BASE_PATH,
    },
    "document_classifier": {
        "transport": "streamable_http",
        "url": DOC_CLASSIFIER_BASE_PATH,
    },
}


async def get_mcp_tools():
    return await MultiServerMCPClient(MCP_SERVERS).get_tools()
