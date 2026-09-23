"""The checkpointer must round-trip the package's own state types without LangGraph's warning."""

import logging
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from consultant_bot.common.checkpoint import build_checkpointer
from consultant_bot.common.entities import Entities
from consultant_bot.common.search.base import ProductHit
from consultant_bot.common.search.filter_search import FilterSearch
from consultant_bot.common.search.products import load_products
from consultant_bot.flexible.nodes.assistant import build_assistant_node
from consultant_bot.flexible.state import State

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "products_fixture.json"


class _ToolCallingFakeModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ToolCallingFakeModel":
        return self


def test_state_types_round_trip_through_the_checkpointer_without_warnings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = _ToolCallingFakeModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "search_products", "args": {"query": "تلگرام"}, "id": "a"}
                    ],
                ),
                AIMessage(content="پیدا شد"),
            ]
        )
    )
    graph = StateGraph(State)
    graph.add_node(  # type: ignore[call-overload]
        "assistant", build_assistant_node(llm, FilterSearch(load_products(FIXTURE_PATH)), top_k=5)
    )
    graph.add_edge(START, "assistant")
    graph.add_edge("assistant", END)
    app = graph.compile(checkpointer=build_checkpointer())
    config: RunnableConfig = {"configurable": {"thread_id": "t"}}

    with caplog.at_level(logging.WARNING):
        app.invoke(
            {"messages": [HumanMessage(content="تلگرام")], "entities": Entities(location="تهران")},
            config,
        )
        values = app.get_state(config).values

    assert "unregistered type" not in caplog.text
    assert values["entities"] == Entities(location="تهران")
    assert isinstance(values["last_shown_products"][0], ProductHit)
    assert values["last_shown_products"][0].product.id == 7569
