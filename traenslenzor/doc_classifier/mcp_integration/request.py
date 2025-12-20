import asyncio
from pathlib import Path

from PIL import Image

from traenslenzor.doc_classifier.mcp_integration import mcp_server
from traenslenzor.doc_classifier.utils import Console


async def main() -> None:
    console = Console.with_prefix("doc-classifier", "mcp-test")

    # Create a dummy image
    img = Path("/tmp/mock_doc.png")
    Image.new("RGB", (32, 32), color="white").save(img)

    tm = mcp_server.doc_classifier_mcp._tool_manager
    result = await tm.call_tool(
        key="classify_document",
        arguments={"path": str(img), "top_k": 3},
    )

    console.plog(result.structured_content)


if __name__ == "__main__":
    asyncio.run(main())
