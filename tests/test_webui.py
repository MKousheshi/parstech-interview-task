"""Covers the web UI's per-turn reply collection without booting Gradio or an LLM.

Runs against a real compiled graph + `MemorySaver` rather than a fake app, since the thing worth
checking is exactly the checkpointer interaction: reading the prior message count back out of the
thread's own state, so the slice of "what this turn appended" is right even with several browser
sessions sharing one process.
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from consultant_bot.webui import turn_replies


class _State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _build_app() -> Any:
    """A graph whose single turn appends the same shape a consultation-firing turn does."""

    def reply(_state: _State) -> dict[str, Any]:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search_products", "args": {}, "id": "c1"}],
                ),
                ToolMessage(content="- محصول", tool_call_id="c1"),
                AIMessage(content="پاسخ دستیار"),
                AIMessage(content="تحلیل کسب‌وکار"),
                AIMessage(content="پیشنهاد محصول"),
            ]
        }

    graph = StateGraph(_State)
    graph.add_node("reply", reply)  # type: ignore[call-overload]
    graph.add_edge(START, "reply")
    graph.add_edge("reply", END)
    return graph.compile(checkpointer=MemorySaver())


def test_turn_returns_every_reply_the_turn_appended() -> None:
    app = _build_app()

    replies = turn_replies(app, "thread-1", "سلام")

    assert replies == ["پاسخ دستیار", "تحلیل کسب‌وکار", "پیشنهاد محصول"]


def test_turn_returns_only_the_current_turn_not_the_whole_history() -> None:
    app = _build_app()

    turn_replies(app, "thread-1", "سلام")
    replies = turn_replies(app, "thread-1", "سؤال دوم")

    assert replies == ["پاسخ دستیار", "تحلیل کسب‌وکار", "پیشنهاد محصول"]


def test_threads_are_isolated_from_each_other() -> None:
    app = _build_app()

    turn_replies(app, "thread-1", "سلام")
    replies = turn_replies(app, "thread-2", "سلام")

    assert replies == ["پاسخ دستیار", "تحلیل کسب‌وکار", "پیشنهاد محصول"]
