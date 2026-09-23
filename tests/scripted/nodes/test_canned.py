from langchain_core.messages import HumanMessage

from consultant_bot.scripted.nodes.ask_entity import QUESTIONS
from consultant_bot.scripted.nodes.canned import (
    FALLBACK_MESSAGE,
    IDLE_MESSAGE,
    PENDING_FALLBACK_MESSAGE,
    fallback,
    idle_reply,
)


def test_fallback_replies_with_the_canned_message_whatever_was_said() -> None:
    for text in ["سلام", "هوا چطوره؟"]:
        result = fallback({"messages": [HumanMessage(content=text)]})
        assert [m.content for m in result["messages"]] == [FALLBACK_MESSAGE]


def test_fallback_repeats_the_pending_question_mid_consultation() -> None:
    result = fallback({"messages": [HumanMessage(content="سلام")], "awaiting_field": "location"})

    expected = PENDING_FALLBACK_MESSAGE.format(question=QUESTIONS["location"])
    assert [m.content for m in result["messages"]] == [expected]


def test_idle_reply_replies_with_the_canned_message() -> None:
    result = idle_reply({"messages": [HumanMessage(content="مشاوره می‌خوام")]})

    assert [m.content for m in result["messages"]] == [IDLE_MESSAGE]
