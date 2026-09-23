from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from consultant_bot.common.messages import latest_user_text, reply_texts


def test_returns_every_ai_reply_in_order() -> None:
    messages = [
        HumanMessage(content="سلام"),
        AIMessage(content="پاسخ دستیار"),
        AIMessage(content="تحلیل کسب‌وکار"),
        AIMessage(content="پیشنهاد محصول"),
    ]

    assert reply_texts(messages) == ["پاسخ دستیار", "تحلیل کسب‌وکار", "پیشنهاد محصول"]


def test_skips_tool_call_stubs_and_tool_messages() -> None:
    messages = [
        HumanMessage(content="یک محصول اینستاگرامی میخوام"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_products", "args": {"query": "اینستاگرام"}, "id": "c1"}],
        ),
        ToolMessage(content="- مدیریت پیج اینستاگرام", tool_call_id="c1"),
        AIMessage(content="این گزینه مناسب است"),
    ]

    assert reply_texts(messages) == ["این گزینه مناسب است"]


def test_skips_whitespace_only_replies() -> None:
    assert reply_texts([AIMessage(content="   \n ")]) == []


def test_reads_text_out_of_block_style_content() -> None:
    messages = [AIMessage(content=[{"type": "text", "text": "پاسخ"}, {"type": "other"}])]

    assert reply_texts(messages) == ["پاسخ"]


def test_latest_user_text_is_the_last_human_message_trimmed() -> None:
    messages = [
        HumanMessage(content="اول"),
        AIMessage(content="پاسخ"),
        HumanMessage(content="  دوم  "),
        AIMessage(content="پاسخ دوم"),
    ]

    assert latest_user_text(messages) == "دوم"


def test_latest_user_text_is_empty_without_a_user_message() -> None:
    assert latest_user_text([AIMessage(content="سلام")]) == ""
