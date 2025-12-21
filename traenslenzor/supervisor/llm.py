# host model with `OLLAMA_DEBUG=2 ollama serve` to enable debug logging
import logging
import os

import requests
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

SEED = 69
TEMPERATURE = 0

MODEL = "qwen3:4b"

LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() == "true"

if LOCAL_MODE:
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
else:
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://wgserver.ddnss.ch:45876")

try:
    # check ollama server
    requests.get(OLLAMA_URL, timeout=2)
except Exception:
    print("Error: Ollama server not running")
    exit(-1)


def pull_model():
    response = requests.post(f"{OLLAMA_URL}/api/pull", json={"model": MODEL})
    if response.status_code != 200:
        logger.error(f"failed to pull the base model '{MODEL}': \n{response.json()}")
        exit(-1)


def exists_model():
    response = requests.get(f"{OLLAMA_URL}/api/tags")
    if response.status_code != 200:
        logger.error(f"failed to request model information: \n{response.json()}")
        exit(-1)
    present_models = response.json()["models"]
    model_names = [m["name"] for m in present_models]
    return MODEL in model_names


def load_model():
    """Trigger model loading into memory"""
    try:
        requests.post(f"{OLLAMA_URL}/api/chat", json={"model": MODEL})
    except Exception:
        pass


def initialize_model():
    if not exists_model():
        resp = input(f"This application requires {MODEL} LLM. Proceed to download model? (Y/n)\n")
        if resp.strip().lower() not in ["", "y", "yes"]:
            print(f"{MODEL} is required for execution, exiting.")
            exit(0)
        pull_model()
    load_model()


initialize_model()

llm = ChatOllama(
    model=MODEL,
    temperature=TEMPERATURE,
    seed=SEED,
    base_url=OLLAMA_URL,
)
