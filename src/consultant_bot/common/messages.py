"""Turning a graph invocation's appended messages into front-end-ready reply text.

Used by the web UI (`webui.py`) to decide what to show: a single turn can append more than one AI
message — the assistant's own reply, then the `analysis` and `suggestion` pair when the
consultation fires — and all of them are part of the answer.
"""

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage


def _text_of(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    # Some providers return content as a list of blocks; keep only the textual ones.
    return "".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()


def reply_texts(messages: Sequence[BaseMessage]) -> list[str]:
    """The user-visible AI replies among `messages`, in order.

    Skips the ReAct loop's intermediate messages: a `ToolMessage` is internal bookkeeping, and the
    `AIMessage` that only carries a tool call has no text of its own, so rendering either would
    show the user a blank turn.
    """
    return [
        text
        for message in messages
        if isinstance(message, AIMessage) and (text := _text_of(message))
    ]
