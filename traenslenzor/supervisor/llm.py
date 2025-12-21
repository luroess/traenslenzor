"""Ollama LLM initialization for the supervisor."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import requests
from langchain_ollama import ChatOllama
from pydantic import field_validator, model_validator

from traenslenzor.doc_classifier.utils import BaseConfig

logger = logging.getLogger(__name__)

DEFAULT_SEED = 69
DEFAULT_TEMPERATURE = 0.0
DEFAULT_FUNCTIONGEMMA_MODEL = "functiongemma:latest"
DEFAULT_QWEN3_MODEL = "qwen3:4b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_QWEN3_URL = "http://wgserver.ddnss.ch:45876"

_llm: ChatOllama | None = None
_configured_settings: SupervisorLLMConfig | None = None


class LLMBackend(Enum):
    """Supported LLM model selections for the supervisor."""

    FUNCTIONGEMMA = "functiongemma"
    QWEN3 = "qwen3"

    def default_model_name(self) -> str:
        """Return the default Ollama model tag for this backend."""
        match self:
            case LLMBackend.FUNCTIONGEMMA:
                return DEFAULT_FUNCTIONGEMMA_MODEL
            case LLMBackend.QWEN3:
                return DEFAULT_QWEN3_MODEL

    def default_base_url(self) -> str:
        """Return the default Ollama base URL for this backend."""
        match self:
            case LLMBackend.FUNCTIONGEMMA:
                return DEFAULT_OLLAMA_URL
            case LLMBackend.QWEN3:
                return DEFAULT_QWEN3_URL


# TODO: pulling and ensurance of model availability is AI slop! Shouldn't it be straightforward using ollama directly instead of HTTP API!?
class SupervisorLLMConfig(BaseConfig[ChatOllama]):
    """Factory config for the supervisor's `ChatOllama` runtime client."""

    model: LLMBackend = LLMBackend.FUNCTIONGEMMA
    """LLM backend family selection."""

    model_name: str = DEFAULT_FUNCTIONGEMMA_MODEL
    """Ollama model tag/name used for the request."""

    base_url: str = DEFAULT_OLLAMA_URL
    """Ollama API base URL used for the request."""

    temperature: float = DEFAULT_TEMPERATURE
    """Sampling temperature passed to `ChatOllama`."""

    seed: int = DEFAULT_SEED
    """Random seed forwarded to `ChatOllama`."""

    auto_pull: bool = False
    """If True, auto-pull missing models via the Ollama HTTP API."""

    @classmethod
    def get_settings(cls) -> SupervisorLLMConfig:
        """Return the currently configured LLM settings."""
        if _configured_settings is not None:
            return _configured_settings
        return cls()

    @model_validator(mode="before")
    @classmethod
    def _apply_backend_defaults(cls, data: Any) -> Any:
        """Fill backend-dependent defaults when fields are omitted."""
        if not isinstance(data, dict):
            return data

        raw_model = data.get("model", LLMBackend.FUNCTIONGEMMA)
        try:
            model = LLMBackend(raw_model) if isinstance(raw_model, str) else raw_model
        except ValueError:
            return data

        if "model_name" not in data or data.get("model_name") in (None, ""):
            data["model_name"] = model.default_model_name()
        if "base_url" not in data or data.get("base_url") in (None, ""):
            data["base_url"] = model.default_base_url()

        return data

    @field_validator("model_name", mode="before")
    @classmethod
    def _normalize_model_name(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("model_name must be a string.")
        normalized = value.strip()
        if not normalized:
            raise ValueError("model_name must not be empty.")
        return normalized

    @field_validator("base_url", mode="before")
    @classmethod
    def _normalize_base_url(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("base_url must be a string.")
        normalized = value.rstrip("/")
        if not normalized:
            raise ValueError("base_url must not be empty.")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://.")
        return normalized

    def apply(self) -> "SupervisorLLMConfig":
        """Apply this config as the process-wide supervisor LLM settings.

        Returns:
            SupervisorLLMConfig: The applied config (self).
        """
        global _configured_settings
        _configured_settings = self
        reset_llm_cache()
        return self

    def setup_target(self) -> ChatOllama:  # type: ignore[override]
        """Instantiate a configured `ChatOllama` client.

        Returns:
            ChatOllama: Configured chat model client.
        """
        self.ensure_model_available()
        return ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            seed=self.seed,
            base_url=self.base_url,
        )

    def ensure_model_available(self) -> None:
        """Ensure the configured model exists on the target Ollama server."""
        installed = self.list_installed_models()
        if self.is_model_installed(installed_models=installed):
            return

        if self.auto_pull:
            logger.info(
                "Ollama model '%s' not found on %s; pulling it now...",
                self.model_name,
                self.base_url,
            )
            self.pull_model()
            return

        raise RuntimeError(
            f"Required Ollama model '{self.model_name}' is not installed on {self.base_url}. "
            f"Run `OLLAMA_HOST={self.base_url} ollama pull {self.model_name}` or enable auto-pull."
        )

    def list_installed_models(self) -> set[str]:
        """Return the set of model tags installed on the configured Ollama server."""
        url = f"{self.base_url}/api/tags"
        try:
            response = requests.get(url, timeout=5)
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to list Ollama models at {url}.") from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to list Ollama models at {url}: {response.status_code} {response.text}"
            )

        payload = response.json()
        models = payload.get("models")
        if not isinstance(models, list):
            raise RuntimeError(f"Malformed response from {url}: {payload!r}")

        names: set[str] = set()
        for entry in models:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if isinstance(name, str):
                names.add(name)
        return names

    def is_model_installed(self, *, installed_models: set[str]) -> bool:
        """Return True when the configured model appears in `installed_models`."""
        if self.model_name in installed_models:
            return True
        if ":" in self.model_name:
            return False
        return any(name.startswith(f"{self.model_name}:") for name in installed_models)

    def pull_model(self) -> None:
        """Pull the configured model from the Ollama registry."""
        url = f"{self.base_url}/api/pull"
        try:
            response = requests.post(
                url,
                json={"model": self.model_name, "stream": False},
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to pull Ollama model '{self.model_name}' via {url}."
            ) from exc

        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Unexpected response from Ollama while pulling '{self.model_name}': "
                f"{response.status_code} {response.text}"
            ) from exc

        if response.status_code != 200:
            raise RuntimeError(f"Failed to pull Ollama model '{self.model_name}': {payload}")
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(
                f"Failed to pull Ollama model '{self.model_name}': {payload['error']}"
            )


def get_llm_settings() -> SupervisorLLMConfig:
    """Return the currently configured LLM settings."""
    if _configured_settings is not None:
        return _configured_settings
    return SupervisorLLMConfig()


def reset_llm_cache() -> None:
    """Clear the cached LLM instance."""
    global _llm
    _llm = None


def get_llm() -> ChatOllama:
    """Get s configured `ChatOllama` instance.

    The instance is cached at module level so repeated calls return the same object.

    Returns:
        ChatOllama: Configured chat model client.
    """

    global _llm
    if _llm is not None:
        return _llm

    settings = get_llm_settings()
    _llm = settings.setup_target()
    return _llm


def main() -> None:
    """CLI entrypoint: validate LLM connectivity and model availability."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_llm_settings()
    model_name = settings.model_name
    get_llm()
    print(f"OK: Ollama model '{model_name}' available at {settings.base_url}")


if __name__ == "__main__":
    main()
