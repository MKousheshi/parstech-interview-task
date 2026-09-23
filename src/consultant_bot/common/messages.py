"""Reading text out of conversation messages: front-end-ready replies and the latest user input.

Used by the web UI (`webui.py`) to decide what to show: a single turn can append more than one AI
message — the assistant's own reply, then the `analysis` and `suggestion` pair when the
consultation fires — and all of them are part of the answer. The rigid variant's nodes read the
user's raw message with `latest_user_text`.
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def message_text(message: BaseMessage) -> str:
    """The message's text content, whether the provider returned a string or content blocks."""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    # Some providers return content as a list of blocks; keep only the textual ones.
    return "".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()


def as_reply(response: Any) -> BaseMessage:
    """A model's response as a message that can go into `messages`.

    Nodes take a `LanguageModelLike`, which may return a plain `str`. `add_messages` would turn a
    bare string into a `HumanMessage`, recording the model's reply as if the user had said it.
    """
    if isinstance(response, BaseMessage):
        return response
    return AIMessage(content=str(response))


def reply_texts(messages: Sequence[BaseMessage]) -> list[str]:
    """The user-visible AI replies among `messages`, in order.

    Skips the ReAct loop's intermediate messages: a `ToolMessage` is internal bookkeeping, and the
    `AIMessage` that only carries a tool call has no text of its own, so rendering either would
    show the user a blank turn.
    """
    return [
        text
        for message in messages
        if isinstance(message, AIMessage) and (text := message_text(message))
    ]


def latest_user_text(messages: Sequence[BaseMessage]) -> str:
    """The text of the most recent user message, or "" if there is none."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message_text(message)
    return ""
