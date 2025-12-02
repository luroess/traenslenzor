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

_llm = None
_initialized = False


def _check_ollama_server():
    """Check if Ollama server is running."""
    try:
        response = requests.get(OLLAMA_URL, timeout=2)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Failed to connect to Ollama server at {OLLAMA_URL}: {e}")
        return False


def pull_base_model():
    """Pull the base model from Ollama. Returns True if successful."""
    logger.info(f"Pulling base model '{BASE_MODEL}'...")
    try:
        response = requests.post(f"{OLLAMA_URL}/api/pull", json={"model": BASE_MODEL}, timeout=300)
        if response.status_code != 200:
            logger.error(f"Failed to pull the base model '{BASE_MODEL}': {response.text}")
            return False
        logger.info(f"Successfully pulled base model '{BASE_MODEL}'")
        return True
    except Exception as e:
        logger.error(f"Exception while pulling base model: {e}")
        return False


def exists_model():
    """Check if the custom model exists in Ollama."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            logger.error(f"Failed to request model information: {response.text}")
            return False
        present_models = response.json().get("models", [])
        model_names = [m["name"] for m in present_models]
        return MODEL_NAME in model_names
    except Exception as e:
        logger.error(f"Exception while checking if model exists: {e}")
        return False


def delete():
    """Delete the custom model from Ollama."""
    logger.info(f"Deleting model '{MODEL_NAME}'...")
    try:
        response = requests.delete(f"{OLLAMA_URL}/api/delete", json={"model": MODEL_NAME})
        if response.status_code != 200:
            logger.error(f"Failed to delete the model '{MODEL_NAME}': {response.text}")
            return False
        logger.info(f"Successfully deleted model '{MODEL_NAME}'")
        return True
    except Exception as e:
        logger.error(f"Exception while deleting model: {e}")
        return False


def create():
    """Create the custom model in Ollama. Returns True if successful."""
    logger.info(f"Creating model '{MODEL_NAME}'...")
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/create",
            json={"model": MODEL_NAME, "from": BASE_MODEL, "template": TEMPLATE},
            timeout=60,
        )
        if response.status_code != 200:
            logger.error(f"Failed to create the model '{MODEL_NAME}': {response.text}")
            return False
        logger.info(f"Successfully created model '{MODEL_NAME}'")
        return True
    except Exception as e:
        logger.error(f"Exception while creating model: {e}")
        return False


def initialize_model():
    """Initialize the model, pulling base model if needed. Returns True if successful."""
    global _initialized

    if _initialized:
        return True

    # Check if Ollama server is running
    if not _check_ollama_server():
        logger.error(
            f"Ollama server not running at {OLLAMA_URL}. "
            "Please start it with 'docker compose up -d' or 'ollama serve'"
        )
        return False

    # Check if base model exists, if not try to pull it
    base_model_exists = exists_model() or BASE_MODEL in [
        m["name"] for m in requests.get(f"{OLLAMA_URL}/api/tags").json().get("models", [])
    ]

    if not base_model_exists:
        logger.info(f"Base model '{BASE_MODEL}' not found, attempting to pull...")
        if not pull_base_model():
            logger.error(
                f"Failed to pull base model '{BASE_MODEL}'. "
                "This may be due to network connectivity issues. "
                "Please ensure you have internet access and try again, or manually pull the model with:\n"
                f"  ollama pull {BASE_MODEL}"
            )
            return False

    # Delete existing custom model if it exists
    if exists_model():
        logger.info(f"Custom model '{MODEL_NAME}' already exists, deleting...")
        if not delete():
            logger.warning("Failed to delete existing model, continuing anyway...")

    # Create the custom model
    if not create():
        logger.error(f"Failed to create custom model '{MODEL_NAME}'")
        return False

    _initialized = True
    logger.info("Model initialization complete")
    return True


def get_llm():
    """Get the LLM instance, initializing if necessary."""
    global _llm

    if _llm is None:
        if not initialize_model():
            raise RuntimeError(
                f"Failed to initialize Ollama model. Please ensure:\n"
                f"1. Ollama server is running (docker compose up -d or ollama serve)\n"
                f"2. You have internet access to download the '{BASE_MODEL}' model\n"
                f"3. Or manually pull the model with: ollama pull {BASE_MODEL}"
            )

        _llm = ChatOllama(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            seed=SEED,
            base_url=OLLAMA_URL,
        )

    return _llm


# Export get_llm as the primary interface
__all__ = ["get_llm", "initialize_model"]
