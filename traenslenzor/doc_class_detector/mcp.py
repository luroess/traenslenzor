import logging

from fastmcp import FastMCP

from traenslenzor.file_server.client import SessionClient
from traenslenzor.file_server.session_state import SessionState

ADDRESS = "127.0.0.1"
PORT = 8004
DOC_CLASS_DETECTOR_PATH = f"http://{ADDRESS}:{PORT}/mcp"

doc_class_detector = FastMCP("Document Class Detector")

logger = logging.getLogger(__name__)


@doc_class_detector.tool
async def classify_document(session_id: str) -> str:
    """Classify Document.
    Args:
        session_id (str): ID of the current session (e.g., "c12f4b1e-8f47-4a92-b8c1-6e3e9d2f91a4").
    """

    def update_session(session: SessionState):
        session.class_probabilities = {"bullshit": 100.0}

    await SessionClient.update(session_id, update_session)
    return "Document classified"


async def run():
    await doc_class_detector.run_async(
        transport="streamable-http", port=PORT, host=ADDRESS, show_banner=False
    )
