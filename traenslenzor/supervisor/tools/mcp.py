from langchain_mcp_adapters.client import Connection, MultiServerMCPClient

from traenslenzor.doc_class_detector.mcp import DOC_CLASS_DETECTOR_PATH
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
        "url": DOC_CLASS_DETECTOR_PATH,
    },
}


async def get_mcp_tools():
    return await MultiServerMCPClient(MCP_SERVERS).get_tools()
