from langchain_mcp_adapters.client import Connection, MultiServerMCPClient

from traenslenzor.layout_detector.layout_detector import LAYOUT_DETECTOR_BASE_PATH

MCP_SERVERS: dict[str, Connection] = {
    "layout_detector": {
        "transport": "streamable_http",
        "url": LAYOUT_DETECTOR_BASE_PATH,
    },
}


async def get_mcp_tools():
    return await MultiServerMCPClient(MCP_SERVERS).get_tools()
