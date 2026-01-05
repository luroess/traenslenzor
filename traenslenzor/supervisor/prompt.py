import logging
from typing import cast

from langchain.agents.middleware import (
    ModelRequest,
    dynamic_prompt,
)

from traenslenzor.file_server.client import SessionClient
from traenslenzor.file_server.session_state import SessionState
from traenslenzor.supervisor.state import SupervisorState

logger = logging.getLogger(__name__)


def has_translated_text(session: SessionState) -> bool:
    """
    Checks if any text element has a translation.
    If so, it is assumed, that the session has been translated.
    """
    if session.text is None:
        return False
    for text in session.text:
        if text.translatedText is not None:
            return True
    return False


def has_text_been_extracted(session: SessionState) -> bool:
    return bool(session.text)


def has_extracted_document(session: SessionState) -> bool:
    return session.extractedDocument is not None


def has_font_been_detected(session: SessionState) -> bool:
    if session.text is None:
        return False
    for text in session.text:
        if text.detectedFont is not None or text.font_size is not None:
            return True
    return False


def has_document_been_classified(session: SessionState) -> bool:
    return session.class_probabilities is not None


def has_result_been_rendered(session: SessionState) -> bool:
    return session.renderedDocumentId is not None


def format_session(session_id: str, session: SessionState) -> str:
    deskew_backend = session.deskew_backend.value if session.deskew_backend else None
    text_count = len(session.text) if session.text else 0
    return f"""
        ✅ the current session_id is '{session_id}'
        {f"✅ the user has selected the language {session.language}" if session.language else "❌ the user has no language selected"}
        {"✅ the user has a document loaded" if session.rawDocumentId else "❌ the user has no document selected"}
        {f"✅ deskew backend: {deskew_backend}" if deskew_backend else "❌ no deskew backend selected"}

        {"✅ extracted document is available" if has_extracted_document(session) else "❌ no extracted document available"}

        {"✅ text was extracted from the document" if has_text_been_extracted(session) else "❌ no text was extracted from the document"}
        {f"✅ text items: {text_count}" if text_count else "❌ no text items recorded"}
        {"✅ the text was translated" if has_translated_text(session) else "❌ the text has not yet been translated"}
        {"✅ the font has been detected" if has_font_been_detected(session) else "❌ the font has not yet been detected"}
        {"✅ the document has been classified" if has_document_been_classified(session) else "❌ the document has not yet been classified"}

        {"✅ the result has been rendered" if has_result_been_rendered(session) else "❌ the result has not yet been rendered"}
    """


@dynamic_prompt
async def context_aware_prompt(request: ModelRequest) -> str:
    state = cast(SupervisorState, request.state)
    session_id = state.get("session_id")
    assert session_id is not None
    session = await SessionClient.get(session_id)

    formatted_session = format_session(session_id, session)
    logger.info("Current Session:")
    logger.info(formatted_session)

    return f"""
    Task:
        You are an image translation assistant.
        Your task is to translate all visible text in an image from the source language into the target language and produce a corresponding translated image.

        When multiple tools are available, determine the execution order based on the required inputs and outputs of each tool, ensuring that all required parameters are available before a tool is invoked.

        Do not describe internal reasoning, planned actions, or tool usage.

        If required information is missing (e.g. target language or document), ask the user a concise clarifying question before proceeding.

        After completing the translation, state the document type the image represents.

    Context:
        {formatted_session}
    """
    # 3. Offer preprocessing options (crop, rotate, enhance). Only continue after the user confirms the image is ready.
