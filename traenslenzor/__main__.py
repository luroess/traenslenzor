import logging
import os

# disable PADDLEOCR logger scheme
os.environ["PADDLEOCR_DISABLE_AUTO_LOGGING_CONFIG"] = "1"

from traenslenzor.app import app


def setup_logger():
    lvl = str(os.getenv("TRAENSLENZOR_LOG_LEVEL", "INFO")).upper()
    level = logging.getLevelNamesMapping().get(lvl, logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)

    for handler in root.handlers:
        handler.setLevel(level)


def main():
    setup_logger()
    app.run()


if __name__ == "__main__":
    main()
