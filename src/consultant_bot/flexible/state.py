"""Conversation state for the flexible (agentic) architecture."""

from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps

from consultant_bot.common.entities import Entities
from consultant_bot.common.search.base import ProductHit


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    entities: Entities
    consultation_requested: bool
    consultation_offered: bool
    consultation_done: bool
    last_shown_products: list[ProductHit] | None
    # Required by LangGraph's prebuilt create_react_agent (used inside the assistant node) when
    # given a custom state_schema; it tracks the ReAct loop's remaining recursion budget.
    remaining_steps: NotRequired[RemainingSteps]
