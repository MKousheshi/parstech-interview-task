import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from consultant_bot.scripted.nodes.route_intent import (
    IntentLabel,
    build_route_intent_node,
    route_after_intent,
)
from consultant_bot.scripted.state import Intent, State
from tests.support import COMPLETE_ENTITIES, ScriptedRunnable


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

    assert result == {"intent": "search"}
    [received] = classifier.inputs
    assert isinstance(received[0], SystemMessage)
    assert [m.content for m in received[1:]] == ["یک محصول تلگرامی می‌خوام"]


@pytest.mark.parametrize(
    ("intent", "consultation_done", "expected"),
    [
        ("search", False, "product_search"),
        ("unclear", False, "fallback"),
        ("consultation", False, "ask_entity"),
        ("consultation", True, "idle_reply"),
        ("search", True, "product_search"),
    ],
)
def test_route_after_intent(intent: Intent, consultation_done: bool, expected: str) -> None:
    state: State = {"messages": [], "intent": intent, "consultation_done": consultation_done}

    assert route_after_intent(state) == expected


def test_consultation_with_all_entities_but_not_done_retries_the_analysis() -> None:
    state: State = {"messages": [], "intent": "consultation", "entities": COMPLETE_ENTITIES}

    assert route_after_intent(state) == "analysis"


def test_route_after_intent_falls_back_without_a_label() -> None:
    assert route_after_intent({"messages": []}) == "fallback"
