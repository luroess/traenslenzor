import asyncio
from pathlib import Path

from PIL import Image

from traenslenzor.doc_classifier.mcp_integration import mcp_server
from traenslenzor.doc_classifier.utils import Console
from traenslenzor.file_server.client import FileClient, SessionClient
from traenslenzor.file_server.session_state import initialize_session


async def main() -> None:
    console = Console.with_prefix("doc-classifier", "mcp-test")

    # Create a dummy image
    img = Path("/tmp/mock_doc.png")
    Image.new("RGB", (32, 32), color="white").save(img)

    file_id = await FileClient.put(str(img))
    if file_id is None:
        raise RuntimeError("Failed to upload mock image to file server.")

    session = initialize_session()
    session.rawDocumentId = file_id
    session_id = await SessionClient.create(session)

    tm = mcp_server.doc_classifier_mcp._tool_manager
    result = await tm.call_tool(
        key="classify_document",
        arguments={"session_id": session_id},
    )

    console.plog(result.structured_content)


if __name__ == "__main__":
    asyncio.run(main())
