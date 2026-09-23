"""Conversation state for the rigid (flow-based) architecture."""

from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from consultant_bot.common.entities import Entities
from consultant_bot.common.search.base import ProductHit

# The 4 entity field names, as a type: `awaiting_field` can only ever name one of them.
EntityField = Literal["business_type", "customer_type", "location", "sales_channel"]

Intent = Literal["search", "consultation", "unclear"]


class State(TypedDict):
    # Only `messages` is present from the first turn on; every other field appears once a node
    # first writes it, which is why nodes read them with `state.get(...)`.
    messages: Annotated[list[BaseMessage], add_messages]
    entities: NotRequired[Entities]
    # The entity the next user message will be stored into verbatim, skipping intent routing.
    awaiting_field: NotRequired[EntityField | None]
    # `route_intent`'s label for this turn, read by its conditional edge.
    intent: NotRequired[Intent | None]
    consultation_done: NotRequired[bool]
    last_search_results: NotRequired[list[ProductHit] | None]
