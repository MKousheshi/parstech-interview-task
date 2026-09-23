"""Runtime configuration shared by both the flexible and rigid architectures.

OpenAI connection details (API key, base URL, model name) are loaded from a `.env` file (see
`example.env` at the repo root for the template) or the real environment, via `pydantic-settings`.
Everything else keeps the plain `os.environ`-with-defaults style used before.
"""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """OpenAI connection settings, read from `.env` (or the environment) once at import time."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"


settings = Settings()

OPENAI_API_KEY = settings.openai_api_key
OPENAI_BASE_URL = settings.openai_base_url
LLM_MODEL = settings.openai_model
LLM_TEMPERATURE = float(os.environ.get("CONSULTANT_BOT_LLM_TEMPERATURE", "0.3"))

# One of "filter" (Phase 1), "tfidf" (Phase 2), "embedding" (Phase 3).
SEARCH_STRATEGY = os.environ.get("CONSULTANT_BOT_SEARCH_STRATEGY", "filter")
SEARCH_TOP_K = int(os.environ.get("CONSULTANT_BOT_SEARCH_TOP_K", "5"))
