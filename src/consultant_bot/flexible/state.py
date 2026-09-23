"""Conversation state for the flexible (agentic) architecture."""

from typing import Annotated, Any, NotRequired, Protocol, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from consultant_bot.common.entities import Entities


class State(TypedDict):
    # Only `messages` is present from the first turn on; every other field appears once a node
    # first writes it, which is why nodes read them with `state.get(...)`.
    messages: Annotated[list[BaseMessage], add_messages]
    entities: NotRequired[Entities]
    consultation_requested: NotRequired[bool]
    consultation_offered: NotRequired[bool]
    consultation_done: NotRequired[bool]
    # The latest free-knowledge analysis text, written by `analysis` and read by `suggestion`.
    analysis: NotRequired[str | None]


class Node(Protocol):
    """What the node factories return. A Protocol rather than `Callable[[State], ...]` because
    LangGraph's `add_node` matches nodes against a protocol whose parameter is named `state`, which
    a bare `Callable` (positional-only) doesn't satisfy for mypy.
    """

    def __call__(self, state: State) -> dict[str, Any]: ...
