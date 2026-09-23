"""Runtime configuration shared by both the flexible and rigid architectures.

Everything is read by `pydantic-settings` from the real environment or a `.env` file at the project
root (see `example.env` for the template) — anchored to the project, not the current directory, so
the web UI finds its API key no matter where it's launched from. Values are validated up front: a
typo in `CONSULTANT_BOT_SEARCH_STRATEGY` or `CONSULTANT_BOT_LOG_LEVEL` fails at startup with a clear
message instead of an error deep inside graph construction or logging setup.

Read settings through `get_settings()` at build/call time rather than at import time, so nothing
is frozen into default arguments before the environment is final.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SearchStrategyName = Literal["filter", "tfidf", "embedding"]

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # SecretStr keeps the key out of `repr(settings)` and so out of any log or traceback.
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.3, validation_alias="CONSULTANT_BOT_LLM_TEMPERATURE")
    # Without a timeout, one stalled request hangs the user's turn indefinitely.
    llm_timeout_seconds: float = Field(
        default=60.0, gt=0, validation_alias="CONSULTANT_BOT_LLM_TIMEOUT_SECONDS"
    )
    llm_max_retries: int = Field(default=2, ge=0, validation_alias="CONSULTANT_BOT_LLM_MAX_RETRIES")

    # "filter" (Phase 1), "tfidf" (Phase 2) or "embedding" (Phase 3).
    search_strategy: SearchStrategyName = Field(
        default="filter", validation_alias="CONSULTANT_BOT_SEARCH_STRATEGY"
    )
    search_top_k: int = Field(default=5, ge=1, validation_alias="CONSULTANT_BOT_SEARCH_TOP_K")
    products_path: Path = Field(
        default=PROJECT_ROOT / "products.json", validation_alias="CONSULTANT_BOT_PRODUCTS_PATH"
    )
    log_level: LogLevel = Field(default="INFO", validation_alias="CONSULTANT_BOT_LOG_LEVEL")

    @field_validator("log_level", mode="before")
    @classmethod
    def _uppercase_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value


@lru_cache
def get_settings() -> Settings:
    """The process-wide settings, loaded once on first use."""
    return Settings()
