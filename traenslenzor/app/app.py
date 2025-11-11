import asyncio

from traenslenzor.file_server.server import run as run_file_server
from traenslenzor.image_renderer.server import run as run_image_renderer
from traenslenzor.layout_detector.layout_detector import run as run_layout_detector
from traenslenzor.supervisor.supervisor import run as run_supervisor


def run():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [
        loop.create_task(run_file_server()),
        loop.create_task(run_layout_detector()),
        loop.create_task(run_image_renderer()),
        loop.create_task(run_supervisor()),
    ]
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()
