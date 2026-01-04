from traenslenzor.supervisor.tools.document_loader import document_loader
from traenslenzor.supervisor.tools.mcp import get_mcp_tools
from traenslenzor.supervisor.tools.set_target_lang import set_target_language


async def get_tools():
    mcp_tools = await get_mcp_tools()
    return [
        set_target_language,
        document_loader,
        *mcp_tools,
    ]
