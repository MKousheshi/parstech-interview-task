from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig

from consultant_bot.architectures import ARCHITECTURES
from consultant_bot.common.analysis import analysis_messages
from consultant_bot.common.checkpoint import build_checkpointer
from consultant_bot.common.entities import Entities
from consultant_bot.common.messages import reply_texts
from consultant_bot.common.search.filter_search import FilterSearch
from consultant_bot.common.search.products import load_products
from consultant_bot.scripted.graph import _route_after_capture, assemble_graph, build_graph
from consultant_bot.scripted.nodes.ask_entity import PENDING_REMINDER, QUESTIONS
from consultant_bot.scripted.nodes.canned import (
    FALLBACK_MESSAGE,
    IDLE_MESSAGE,
    PENDING_FALLBACK_MESSAGE,
)
from consultant_bot.scripted.nodes.product_search import RESULTS_HEADER
from consultant_bot.scripted.nodes.route_intent import IntentLabel
from consultant_bot.scripted.state import Intent
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


def test_scripted_is_registered_as_an_architecture() -> None:
    assert ARCHITECTURES["scripted"] is build_graph


def test_route_after_capture_runs_analysis_only_once_complete() -> None:
    assert _route_after_capture({"messages": [], "entities": COMPLETE_ENTITIES}) == "analysis"
    partial = Entities(business_type="کافه")
    assert _route_after_capture({"messages": [], "entities": partial}) == "ask_entity"


class _Conversation:
    """Runs turns through a compiled scripted graph on fakes and returns each turn's replies."""

    def __init__(self, *labels: Intent | IntentLabel) -> None:
        self.classifier = ScriptedRunnable(
            *(IntentLabel(intent=label) if isinstance(label, str) else label for label in labels)
        )
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
    chat = _Conversation("consultation", "answer", "answer", "answer", "answer", "consultation")

    assert chat.say("می‌خوام مشاوره بگیرم") == [QUESTIONS["business_type"]]
    assert chat.say("کافه") == [QUESTIONS["customer_type"]]
    assert chat.say("B2C") == [QUESTIONS["location"]]
    assert chat.say("تهران") == [QUESTIONS["sales_channel"]]
    assert chat.say("اینستاگرام") == ["تحلیل", "پیشنهاد"]

    state = chat.state()
    assert state["entities"] == COMPLETE_ENTITIES
    assert state["consultation_done"] is True
    # Every turn is classified, answers included.
    assert len(chat.classifier.inputs) == 5
    assert chat.llm.inputs[0] == analysis_messages(COMPLETE_ENTITIES)

    assert chat.say("یک مشاوره دیگه می‌خوام") == [IDLE_MESSAGE]


def test_a_consultation_request_stating_all_4_entities_is_answered_at_once() -> None:
    label = IntentLabel(intent="consultation", **COMPLETE_ENTITIES.model_dump())
    chat = _Conversation(label)

    assert chat.say("یه کافه تو تهران دارم، B2C، از اینستاگرام می‌فروشم. مشاوره بده") == [
        "تحلیل",
        "پیشنهاد",
    ]
    assert chat.state()["entities"] == COMPLETE_ENTITIES
    assert chat.state()["consultation_done"] is True


def test_only_the_entities_not_yet_stated_are_asked_for() -> None:
    opening = IntentLabel(intent="consultation", business_type="کافه", location="تهران")
    chat = _Conversation(opening, "answer", "answer")

    assert chat.say("یه کافه تو تهران دارم، مشاوره می‌خوام") == [QUESTIONS["customer_type"]]
    assert chat.say("B2C") == [QUESTIONS["sales_channel"]]
    assert chat.say("اینستاگرام") == ["تحلیل", "پیشنهاد"]


def test_entities_mentioned_in_a_search_are_not_recorded() -> None:
    chat = _Conversation(IntentLabel(intent="search", business_type="کافه"), "consultation")

    chat.say("طراحی سایت برای کافه")
    assert chat.say("مشاوره می‌خوام") == [QUESTIONS["business_type"]]


def test_a_search_mid_consultation_keeps_the_question_pending() -> None:
    chat = _Conversation("consultation", "search", "answer")

    chat.say("مشاوره")
    [search_reply] = chat.say("تلگرام")

    assert search_reply.startswith(RESULTS_HEADER.format(query="تلگرام"))
    assert search_reply.endswith(PENDING_REMINDER.format(question=QUESTIONS["business_type"]))
    assert chat.state()["awaiting_field"] == "business_type"
    assert not chat.state()["entities"].known()

    assert chat.say("کافه") == [QUESTIONS["customer_type"]]
    assert chat.state()["entities"].business_type == "کافه"


def test_an_unclear_reply_mid_consultation_repeats_the_question() -> None:
    chat = _Conversation("consultation", "unclear", "consultation")

    chat.say("مشاوره")
    expected = PENDING_FALLBACK_MESSAGE.format(question=QUESTIONS["business_type"])
    assert chat.say("سلام") == [expected]
    # Asking for a consultation again while one is underway repeats the open question.
    assert chat.say("مشاوره می‌خوام") == [QUESTIONS["business_type"]]


def test_search_and_unclear_turns_end_after_one_reply() -> None:
    chat = _Conversation("search", "unclear")

    [search_reply] = chat.say("تلگرام")
    assert search_reply.startswith(RESULTS_HEADER.format(query="تلگرام"))
    assert chat.say("سلام") == [FALLBACK_MESSAGE]
    assert not chat.llm.was_called


class _FailsOnce(Runnable[Any, Any]):
    """An LLM whose first call times out and whose later calls succeed."""

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("LLM request timed out")
        return AIMessage(content="پاسخ")


def test_a_consultation_that_failed_after_the_last_answer_can_be_retried() -> None:
    chat = _Conversation("consultation", "answer", "answer", "answer", "answer", "consultation")
    chat.app = assemble_graph(
        chat.classifier,
        _FailsOnce(),
        FilterSearch(load_products(FIXTURE_PATH)),
        top_k=5,
        checkpointer=build_checkpointer(),
    )
    for answer in ["مشاوره", "کافه", "B2C", "تهران"]:
        chat.say(answer)
    with pytest.raises(TimeoutError):
        chat.say("اینستاگرام")

    assert chat.say("مشاوره می‌خوام") == ["پاسخ", "پاسخ"]
    assert chat.state()["consultation_done"] is True
