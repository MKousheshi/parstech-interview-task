"""Shared factory for the default OpenAI chat model, used by every LLM-touching node.

Reads connection details from `consultant_bot.common.config` (itself loaded from `.env`/the
environment). `api_key`/`base_url` are only passed through when set, so that omitting them from
`.env` falls back to `langchain-openai`'s own `OPENAI_API_KEY`/`OPENAI_BASE_URL` env lookup.
"""

from typing import Any

from langchain_openai import ChatOpenAI

from consultant_bot.common import config


def build_chat_model() -> ChatOpenAI:
    kwargs: dict[str, Any] = {"model": config.LLM_MODEL, "temperature": config.LLM_TEMPERATURE}
    if config.OPENAI_API_KEY:
        kwargs["api_key"] = config.OPENAI_API_KEY
    if config.OPENAI_BASE_URL:
        kwargs["base_url"] = config.OPENAI_BASE_URL
    return ChatOpenAI(**kwargs)
