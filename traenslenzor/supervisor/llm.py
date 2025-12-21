# host model with `OLLAMA_DEBUG=2 ollama serve` to enable debug logging
import logging

import requests
from langchain_ollama import ChatOllama

from traenslenzor.supervisor.config import settings

logger = logging.getLogger(__name__)

try:
    # check ollama server
    requests.get(settings.llm.ollama_url, timeout=2)
except Exception:
    print("Error: Ollama server not running")
    exit(-1)


def pull_model():
    response = requests.post(
        f"{settings.llm.ollama_url}/api/pull", json={"model": settings.llm.model}
    )
    if response.status_code != 200:
        logger.error(f"failed to pull the base model '{settings.llm.model}': \n{response.json()}")
        exit(-1)


def exists_model():
    response = requests.get(f"{settings.llm.ollama_url}/api/tags")
    if response.status_code != 200:
        logger.error(f"failed to request model information: \n{response.json()}")
        exit(-1)
    present_models = response.json()["models"]
    model_names = [m["name"] for m in present_models]
    return settings.llm.model in model_names


def initialize_model():
    if exists_model():
        return
    resp = input(
        f"This application requires {settings.llm.model} LLM. Proceed to download model? (Y/n)\n"
    )
    if resp.strip().lower() not in ["", "y", "yes"]:
        print(f"{settings.llm.model} is required for execution, exiting.")
        exit(0)
    pull_model()


initialize_model()

llm = ChatOllama(
    model=settings.llm.model,
    temperature=settings.llm.temperature,
    seed=settings.llm.seed,
    base_url=settings.llm.ollama_url,
)
