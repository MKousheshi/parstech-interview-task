import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from consultant_bot.common.entities import Entities
from consultant_bot.scripted.nodes.ask_entity import QUESTIONS
from consultant_bot.scripted.nodes.route_intent import (
    SYSTEM_PROMPT,
    IntentLabel,
    build_route_intent_node,
    route_after_intent,
    system_prompt,
)
from consultant_bot.scripted.state import Intent, State
from tests.support import ScriptedRunnable


def test_classifies_only_the_latest_user_message() -> None:
    classifier = ScriptedRunnable(IntentLabel(intent="search"))
    node = build_route_intent_node(classifier)

    result = node(
        {
            "messages": [
                HumanMessage(content="پیام قبلی"),
                AIMessage(content="پاسخ قبلی"),
                HumanMessage(content="یک محصول تلگرامی می‌خوام"),
            ]
        }
    )

    assert result == {"intent": "search", "stated_entities": Entities()}
    [received] = classifier.inputs
    assert isinstance(received[0], SystemMessage)
    assert [m.content for m in received[1:]] == ["یک محصول تلگرامی می‌خوام"]


def test_answer_is_only_offered_while_a_question_is_pending() -> None:
    idle = system_prompt({"messages": []})
    assert "answer" not in idle
    assert "business_type" in idle  # entities are extracted either way

    pending = system_prompt({"messages": [], "awaiting_field": "location"})
    assert pending.startswith(SYSTEM_PROMPT)
    assert "answer" in pending
    assert QUESTIONS["location"] in pending


def test_the_pending_question_reaches_the_classifier() -> None:
    classifier = ScriptedRunnable(IntentLabel(intent="answer"))
    node = build_route_intent_node(classifier)

    result = node({"messages": [HumanMessage(content="تهران")], "awaiting_field": "location"})

    assert result["intent"] == "answer"
    [received] = classifier.inputs
    assert QUESTIONS["location"] in received[0].content


def test_the_stated_entities_are_reported_without_the_label() -> None:
    label = IntentLabel(intent="consultation", business_type="کافه", location="null")
    node = build_route_intent_node(ScriptedRunnable(label))

    result = node({"messages": [HumanMessage(content="کافه دارم، مشاوره می‌خوام")]})

    # A placeholder like "null" counts as not stated.
    assert result["stated_entities"] == Entities(business_type="کافه")
    assert type(result["stated_entities"]) is not IntentLabel


def test_an_answer_is_captured_only_while_a_field_is_pending() -> None:
    pending: State = {"messages": [], "intent": "answer", "awaiting_field": "location"}
    assert route_after_intent(pending) == "capture_entity"
    assert route_after_intent({"messages": [], "intent": "answer"}) == "fallback"


@pytest.mark.parametrize(
    ("intent", "consultation_done", "expected"),
    [
        ("search", False, "product_search"),
        ("unclear", False, "fallback"),
        ("consultation", False, "capture_entity"),
        ("consultation", True, "idle_reply"),
        ("search", True, "product_search"),
    ],
)
def test_route_after_intent(intent: Intent, consultation_done: bool, expected: str) -> None:
    state: State = {"messages": [], "intent": intent, "consultation_done": consultation_done}

    assert route_after_intent(state) == expected


def test_route_after_intent_falls_back_without_a_label() -> None:
    assert route_after_intent({"messages": []}) == "fallback"
