from traenslenzor.supervisor.tools.document_loader import document_loader
from traenslenzor.supervisor.tools.mcp import get_mcp_tools
from traenslenzor.supervisor.tools.set_target_lang import set_target_language
from traenslenzor.supervisor.tools.text_replacer import replace_text


async def get_tools():
    mcp_tools = await get_mcp_tools()
    return [
        set_target_language,
        document_loader,
        replace_text,
        *mcp_tools,
    ]
