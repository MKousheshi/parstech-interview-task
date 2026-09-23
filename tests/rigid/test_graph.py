from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from consultant_bot.architectures import ARCHITECTURES
from consultant_bot.common.analysis import analysis_messages
from consultant_bot.common.checkpoint import build_checkpointer
from consultant_bot.common.entities import Entities
from consultant_bot.common.messages import reply_texts
from consultant_bot.common.search.filter_search import FilterSearch
from consultant_bot.common.search.products import load_products
from consultant_bot.rigid.graph import (
    _route_after_capture,
    _route_start,
    assemble_graph,
    build_graph,
)
from consultant_bot.rigid.nodes.ask_entity import QUESTIONS
from consultant_bot.rigid.nodes.canned import FALLBACK_MESSAGE, IDLE_MESSAGE
from consultant_bot.rigid.nodes.product_search import RESULTS_HEADER
from consultant_bot.rigid.nodes.route_intent import IntentLabel
from consultant_bot.rigid.state import Intent
from tests.support import COMPLETE_ENTITIES, FIXTURE_PATH, ScriptedRunnable


def test_build_graph_wires_all_expected_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    # A dummy key is enough: ChatOpenAI checks one is present but never calls the API here.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")

    app = build_graph()

    assert {
        "route_intent",
        "capture_entity",
        "ask_entity",
        "product_search",
        "fallback",
        "idle_reply",
        "analysis",
        "suggestion",
    } <= set(app.get_graph().nodes)


def test_rigid_is_registered_as_an_architecture() -> None:
    assert ARCHITECTURES["rigid"] is build_graph


def test_route_start_captures_while_a_field_is_pending() -> None:
    assert _route_start({"messages": [], "awaiting_field": "location"}) == "capture_entity"
    assert _route_start({"messages": [], "awaiting_field": None}) == "route_intent"
    assert _route_start({"messages": []}) == "route_intent"


def test_route_after_capture_runs_analysis_only_once_complete() -> None:
    assert _route_after_capture({"messages": [], "entities": COMPLETE_ENTITIES}) == "analysis"
    partial = Entities(business_type="کافه")
    assert _route_after_capture({"messages": [], "entities": partial}) == "ask_entity"


class _Conversation:
    """Runs turns through a compiled rigid graph on fakes and returns each turn's replies."""

    def __init__(self, *intents: Intent) -> None:
        self.classifier = ScriptedRunnable(*(IntentLabel(intent=i) for i in intents))
        self.llm = ScriptedRunnable(AIMessage(content="تحلیل"), AIMessage(content="پیشنهاد"))
        self.app = assemble_graph(
            self.classifier,
            self.llm,
            FilterSearch(load_products(FIXTURE_PATH)),
            top_k=5,
            checkpointer=build_checkpointer(),
        )
        self.config: RunnableConfig = {"configurable": {"thread_id": "t"}}

    def say(self, text: str) -> list[str]:
        before = len(self.state().get("messages", []))
        result = self.app.invoke({"messages": [HumanMessage(content=text)]}, self.config)
        return reply_texts(result["messages"][before:])

    def state(self) -> dict[str, Any]:
        return self.app.get_state(self.config).values


def test_full_consultation_asks_in_order_then_runs_analysis_and_suggestion() -> None:
    chat = _Conversation("consultation", "consultation")

    assert chat.say("می‌خوام مشاوره بگیرم") == [QUESTIONS["business_type"]]
    assert chat.say("کافه") == [QUESTIONS["customer_type"]]
    assert chat.say("B2C") == [QUESTIONS["location"]]
    assert chat.say("تهران") == [QUESTIONS["sales_channel"]]
    assert chat.say("اینستاگرام") == ["تحلیل", "پیشنهاد"]

    state = chat.state()
    assert state["entities"] == COMPLETE_ENTITIES
    assert state["consultation_done"] is True
    # Answers skip classification: only the opening message was classified.
    assert len(chat.classifier.inputs) == 1
    assert chat.llm.inputs[0] == analysis_messages(COMPLETE_ENTITIES)

    assert chat.say("یک مشاوره دیگه می‌خوام") == [IDLE_MESSAGE]


def test_a_pending_field_swallows_an_off_topic_reply() -> None:
    chat = _Conversation("consultation")

    chat.say("مشاوره")
    assert chat.say("محصولات تلگرام رو نشونم بده") == [QUESTIONS["customer_type"]]

    assert chat.state()["entities"].business_type == "محصولات تلگرام رو نشونم بده"


def test_search_and_unclear_turns_end_after_one_reply() -> None:
    chat = _Conversation("search", "unclear")

    [search_reply] = chat.say("تلگرام")
    assert search_reply.startswith(RESULTS_HEADER.format(query="تلگرام"))
    assert chat.say("سلام") == [FALLBACK_MESSAGE]
    assert not chat.llm.was_called
