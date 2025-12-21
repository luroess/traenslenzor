import argparse
import os

# disable PADDLEOCR logger scheme
os.environ["PADDLEOCR_DISABLE_AUTO_LOGGING_CONFIG"] = "1"


def main():
    parser = argparse.ArgumentParser(description="Traenslenzor Application")
    parser.add_argument("--local", action="store_true", help="Run in local mode")
    args = parser.parse_args()

    if args.local:
        os.environ["LOCAL_MODE"] = "true"

    # Initialize LLM model (ollama)
    import traenslenzor.supervisor.llm

    from traenslenzor.app import app
    from traenslenzor.logger import setup_logger

    setup_logger()
    app.run()


if __name__ == "__main__":
    main()
