from langchain_core.messages import HumanMessage

from consultant_bot.rigid.nodes.canned import FALLBACK_MESSAGE, IDLE_MESSAGE, fallback, idle_reply


def test_fallback_replies_with_the_canned_message_whatever_was_said() -> None:
    for text in ["سلام", "هوا چطوره؟"]:
        result = fallback({"messages": [HumanMessage(content=text)]})
        assert [m.content for m in result["messages"]] == [FALLBACK_MESSAGE]


def test_idle_reply_replies_with_the_canned_message() -> None:
    result = idle_reply({"messages": [HumanMessage(content="مشاوره می‌خوام")]})

    assert [m.content for m in result["messages"]] == [IDLE_MESSAGE]
