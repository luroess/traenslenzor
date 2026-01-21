import argparse

from traenslenzor.supervisor.config import settings


def main():
    parser = argparse.ArgumentParser(description="Traenslenzor Application")
    parser.add_argument("--server", action="store_true", help="Run in local mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    if args.server:
        settings.llm.ollama_url = "http://wgserver.ddnss.ch:45876"

    if args.debug:
        settings.llm.debug_mode = True

    # Initialize LLM model (ollama)

    from traenslenzor.app import app
    from traenslenzor.logger import setup_logger

    setup_logger()
    app.run()


if __name__ == "__main__":
    main()
