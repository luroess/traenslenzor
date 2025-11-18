from traenslenzor.doc_classifier.mcp import mcp_server
from traenslenzor.doc_classifier.utils import Console


async def run_tool():
    console = Console.with_prefix("doc-classifier", "mcp-schemas")
    tm = mcp_server.doc_classifier_mcp._tool_manager
    tools = await tm.get_tools()
    for tool in tools.values():
        inp = tool.parameters
        out = tool.output_schema
        console.plog(dict(input=inp, output=out))


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_tool())
