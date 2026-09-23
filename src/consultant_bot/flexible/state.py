"""Conversation state for the flexible (agentic) architecture."""

from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from consultant_bot.common.entities import Entities
from consultant_bot.common.search.base import ProductHit


class State(TypedDict):
    # Only `messages` is present from the first turn on; every other field appears once a node
    # first writes it, which is why nodes read them with `state.get(...)`.
    messages: Annotated[list[BaseMessage], add_messages]
    entities: NotRequired[Entities]
    consultation_requested: NotRequired[bool]
    consultation_offered: NotRequired[bool]
    consultation_done: NotRequired[bool]
    last_shown_products: NotRequired[list[ProductHit] | None]
    # The latest free-knowledge analysis text, written by `analysis` and read by `suggestion`.
    analysis: NotRequired[str | None]
