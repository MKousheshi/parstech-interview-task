"""Conversation state for the flexible (agentic) architecture."""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from consultant_bot.common.entities import Entities
from consultant_bot.common.search.base import ProductHit


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    entities: Entities
    consultation_requested: bool
    consultation_offered: bool
    consultation_done: bool
    last_shown_products: list[ProductHit] | None
