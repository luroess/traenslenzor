from langchain_mcp_adapters.client import Connection, MultiServerMCPClient

from traenslenzor.image_renderer.server import IMAGE_RENDERER_BASE_PATH
from traenslenzor.text_extractor.text_extractor import TEXT_EXTRACTOR_BASE_PATH

MCP_SERVERS: dict[str, Connection] = {
    "text_extractor": {
        "transport": "streamable_http",
        "url": TEXT_EXTRACTOR_BASE_PATH,
    },
    "image_renderer": {
        "transport": "streamable_http",
        "url": IMAGE_RENDERER_BASE_PATH,
    },
}


async def get_mcp_tools():
    return await MultiServerMCPClient(MCP_SERVERS).get_tools()
