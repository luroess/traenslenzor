from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class LLMConfig(BaseModel):
    seed: int = 69
    temperature: float = 0
    ollama_url: str = "http://localhost:11434"
    model: str = "qwen3:4b"
    debug_mode: bool = False
    num_ctx: int | None = None
    num_predict: int | None = None
    num_gpu: int | None = None
    num_thread: int | None = None
    keep_alive: int | str | None = None
    flash_attention: bool = False
    kv_cache_type: str | None = None


class SupervisorConfig(BaseSettings):
    llm: LLMConfig = LLMConfig()

    model_config = SettingsConfigDict(
        toml_file=Path(__file__).parent.parent.parent / "config/supervisor.toml"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls),)


settings = SupervisorConfig()
