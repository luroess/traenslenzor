from langchain_mcp_adapters.client import Connection, MultiServerMCPClient

<<<<<<< HEAD
from traenslenzor.doc_class_detector.mcp import DOC_CLASS_DETECTOR_PATH
from traenslenzor.font_detector.mcp import FONT_DETECTOR_PATH
from traenslenzor.image_renderer.mcp import IMAGE_RENDERER_BASE_PATH
from traenslenzor.text_extractor.mcp import TEXT_EXTRACTOR_BASE_PATH
from traenslenzor.translator.mcp import TRANSLATOR_PATH
=======
from traenslenzor.font_detector.server import FONT_DETECTOR_BASE_PATH
from traenslenzor.image_renderer.server import IMAGE_RENDERER_BASE_PATH
from traenslenzor.layout_detector.layout_detector import LAYOUT_DETECTOR_BASE_PATH
>>>>>>> 4b8d647 (fix: adapted to general mcp server, changed to fastmcp)

MCP_SERVERS: dict[str, Connection] = {
    "text_extractor": {
        "transport": "streamable_http",
        "url": TEXT_EXTRACTOR_BASE_PATH,
    },
    "image_renderer": {
        "transport": "streamable_http",
        "url": IMAGE_RENDERER_BASE_PATH,
    },
<<<<<<< HEAD
    "translator": {
        "transport": "streamable_http",
        "url": TRANSLATOR_PATH,
    },
    "font_detector": {
        "transport": "streamable_http",
        "url": FONT_DETECTOR_PATH,
    },
    "document_classifier": {
        "transport": "streamable_http",
        "url": DOC_CLASS_DETECTOR_PATH,
=======
    "font_detector": {
        "transport": "streamable_http",
        "url": FONT_DETECTOR_BASE_PATH,
>>>>>>> 4b8d647 (fix: adapted to general mcp server, changed to fastmcp)
    },
}


async def get_mcp_tools():
    return await MultiServerMCPClient(MCP_SERVERS).get_tools()
