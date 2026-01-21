import asyncio

from traenslenzor.doc_classifier.mcp_integration.mcp_server import run as run_doc_classifier
from traenslenzor.doc_scanner.mcp import run as run_doc_scanner
from traenslenzor.file_server.server import run as run_file_server
from traenslenzor.font_detector.mcp import run as run_font_detector
from traenslenzor.image_renderer.mcp_server import run as run_image_renderer
from traenslenzor.streamlit.run import run as run_streamlit
from traenslenzor.text_extractor.mcp import run as run_text_extractor
from traenslenzor.text_optimizer.mcp import run as run_text_optimizer
from traenslenzor.translator.mcp import run as run_translator


def run():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [
        loop.create_task(run_file_server()),
        loop.create_task(run_image_renderer()),
        loop.create_task(run_text_extractor()),
        loop.create_task(run_translator()),
        # loop.create_task(run_doc_class_detector()),
        loop.create_task(run_doc_classifier()),
        loop.create_task(run_font_detector()),
        loop.create_task(run_doc_scanner()),
        loop.create_task(run_text_optimizer()),
        loop.create_task(run_streamlit()),  # Must be last or weird error :D
    ]
    loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()
