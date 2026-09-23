"""Shared factory for the default OpenAI chat model, used by every LLM-touching node.

`OPENAI_API_KEY` is picked up from the environment by `langchain-openai` itself.
"""

from langchain_openai import ChatOpenAI

from consultant_bot.common import config


def build_chat_model() -> ChatOpenAI:
    return ChatOpenAI(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
