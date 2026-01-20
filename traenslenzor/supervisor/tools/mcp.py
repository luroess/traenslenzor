from langchain_mcp_adapters.client import Connection, MultiServerMCPClient

from traenslenzor.doc_classifier.mcp_integration.mcp_server import DOC_CLASSIFIER_BASE_PATH
from traenslenzor.doc_scanner.mcp import DOC_SCANNER_BASE_PATH
from traenslenzor.font_detector.mcp import FONT_DETECTOR_BASE_PATH
from traenslenzor.image_renderer.mcp import IMAGE_RENDERER_BASE_PATH
from traenslenzor.text_extractor.mcp import TEXT_EXTRACTOR_BASE_PATH
from traenslenzor.translator.mcp import TRANSLATOR_PATH

MCP_SERVERS: dict[str, Connection] = {
    "text_extractor": {
        "transport": "streamable_http",
        "url": TEXT_EXTRACTOR_BASE_PATH,
    },
    "image_renderer": {
        "transport": "streamable_http",
        "url": IMAGE_RENDERER_BASE_PATH,
    },
    "translator": {
        "transport": "streamable_http",
        "url": TRANSLATOR_PATH,
    },
    "font_detector": {
        "transport": "streamable_http",
        "url": FONT_DETECTOR_BASE_PATH,
    },
    "document_classifier": {
        "transport": "streamable_http",
        "url": DOC_CLASSIFIER_BASE_PATH,
    },
    "doc_scanner": {
        "transport": "streamable_http",
        "url": DOC_SCANNER_BASE_PATH,
    },
}


_SERVER_LABELS: dict[str, str] = {
    "text_extractor": "Text Extractor",
    "image_renderer": "Image Renderer",
    "translator": "Translator",
    "font_detector": "Font Detector",
    "document_classifier": "Doc Classifier",
    "doc_scanner": "Doc Scanner",
}


def format_tool_label(tool_name: str) -> str:
    """Format a tool name for display."""
    for sep in ("::", "/", "."):
        if sep in tool_name:
            server, tool = tool_name.split(sep, 1)
            server_label = _SERVER_LABELS.get(server, server.replace("_", " ").title())
            tool_label = tool.replace("_", " ").strip()
            return f"{server_label} · {tool_label}"
    return tool_name.replace("_", " ").strip()


async def get_mcp_tools():
    return await MultiServerMCPClient(MCP_SERVERS).get_tools()
