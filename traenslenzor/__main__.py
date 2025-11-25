import logging

from traenslenzor.app import app


def main():
    logging.basicConfig(level=logging.INFO)
    app.run()


if __name__ == "__main__":
    main()
