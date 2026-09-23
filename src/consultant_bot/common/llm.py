"""Shared factory for the default OpenAI chat model, used by every LLM-touching node.

Reads connection details from `consultant_bot.common.config.get_settings()` (itself loaded from
`.env`/the environment). `api_key`/`base_url` are only passed through when set, so that omitting
them from `.env` falls back to `langchain-openai`'s own `OPENAI_API_KEY`/`OPENAI_BASE_URL` env
lookup.
"""

from typing import Any

from langchain_openai import ChatOpenAI

from consultant_bot.common.config import get_settings


def build_chat_model() -> ChatOpenAI:
    settings = get_settings()
    kwargs: dict[str, Any] = {
        "model": settings.openai_model,
        "temperature": settings.llm_temperature,
    }
    if settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**kwargs)
