# host model with `OLLAMA_DEBUG=2 ollama serve` to enable debug logging
import logging
import os

import requests
from langchain_ollama import ChatOllama

from traenslenzor.supervisor.config import settings

logger = logging.getLogger(__name__)


def configure_ollama_env():
    if settings.llm.flash_attention:
        os.environ.setdefault("OLLAMA_FLASH_ATTENTION", "1")
    if settings.llm.kv_cache_type:
        os.environ.setdefault("OLLAMA_KV_CACHE_TYPE", settings.llm.kv_cache_type)


configure_ollama_env()


def log_llm_config():
    logger.info(
        "LLM config: model=%s, base_url=%s, temperature=%s, seed=%s, num_ctx=%s, "
        "num_predict=%s, num_gpu=%s, num_thread=%s, keep_alive=%s, flash_attention=%s, "
        "kv_cache_type=%s",
        settings.llm.model,
        settings.llm.ollama_url,
        settings.llm.temperature,
        settings.llm.seed,
        settings.llm.num_ctx,
        settings.llm.num_predict,
        settings.llm.num_gpu,
        settings.llm.num_thread,
        settings.llm.keep_alive,
        settings.llm.flash_attention,
        settings.llm.kv_cache_type,
    )


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


def load_model():
    """Trigger model loading into memory"""
    try:
        response = requests.post(
            f"{settings.llm.ollama_url}/api/chat", json={"model": settings.llm.model}
        )
        if response.status_code != 200:
            logger.error(
                "Failed to trigger model loading for '%s'. Status code: %s, response: %s",
                settings.llm.model,
                response.status_code,
                response.text,
            )
    except Exception:
        logger.exception(
            "Unexpected error while triggering model loading for '%s'", settings.llm.model
        )


def initialize_model():
    if not exists_model():
        resp = input(
            f"This application requires {settings.llm.model} LLM. Proceed to download model? (Y/n)\n"
        )
        if resp.strip().lower() not in ["", "y", "yes"]:
            print(f"{settings.llm.model} is required for execution, exiting.")
            exit(0)
        pull_model()
    load_model()


initialize_model()

llm = ChatOllama(
    model=settings.llm.model,
    temperature=settings.llm.temperature,
    seed=settings.llm.seed,
    base_url=settings.llm.ollama_url,
    num_ctx=settings.llm.num_ctx,
    num_predict=settings.llm.num_predict,
    num_gpu=settings.llm.num_gpu,
    num_thread=settings.llm.num_thread,
    keep_alive=settings.llm.keep_alive,
)
