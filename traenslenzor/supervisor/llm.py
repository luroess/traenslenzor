# host model with `OLLAMA_DEBUG=2 ollama serve` to enable debug logging
import logging
import os
from pathlib import Path

import requests
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

SEED = 69
TEMPERATURE = 0

BASE_MODEL = "llama3.2"
MODEL_NAME = "traenslenzor_2000:0.1"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
with open(Path(__file__).parent / "system.tmpl", "r", encoding="utf-8") as f:
    TEMPLATE = f.read()

try:
    # check ollama server
    requests.get(OLLAMA_URL, timeout=2)
except Exception:
    print("Error: Ollama server not running")
    exit(-1)


def pull_base_model():
    response = requests.post(f"{OLLAMA_URL}/api/pull", json={"model": BASE_MODEL})
    if response.status_code != 200:
        logger.error(f"failed to pull the base model '{BASE_MODEL}': \n{response.json()}")
        exit(-1)


def exists_model():
    response = requests.get(f"{OLLAMA_URL}/api/tags")
    if response.status_code != 200:
        logger.error(f"failed to request model information: \n{response.json()}")
        exit(-1)
    present_models = response.json()["models"]
    model_names = [m["name"] for m in present_models]
    return MODEL_NAME in model_names


def delete():
    logger.info("Deleting model")
    response = requests.delete(f"{OLLAMA_URL}/api/delete", json={"model": MODEL_NAME})
    if response.status_code != 200:
        logger.error(f"failed to delete the model '{MODEL_NAME}': \n{response.json()}")
        exit(-1)


def create():
    logger.debug("Creating model")
    response = requests.post(
        f"{OLLAMA_URL}/api/create",
        json={"model": MODEL_NAME, "from": BASE_MODEL, "template": TEMPLATE},
    )
    if response.status_code != 200:
        logger.error(f"failed to create the model '{MODEL_NAME}': \n{response.json()}")
        exit(-1)


def initialize_model():
    if exists_model():
        delete()
    create()


initialize_model()
llm = ChatOllama(
    model=BASE_MODEL,
    temperature=TEMPERATURE,
    seed=SEED,
    base_url=OLLAMA_URL,
)
