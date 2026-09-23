from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from consultant_bot.flexible.nodes.analysis import build_analysis_node

COMPLETE_ENTITIES = {
    "business_type": "کافه",
    "customer_type": "B2C",
    "location": "تهران",
    "sales_channel": "اینستاگرام",
}


class RecordingFakeLLM:
    """Records exactly what messages it was invoked with, to verify context isolation."""

    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.received_messages: list | None = None  # type: ignore[type-arg]

    def invoke(self, messages: list) -> AIMessage:  # type: ignore[type-arg]
        self.received_messages = messages
        return self.response


def test_analysis_call_contains_only_a_system_and_entities_summary_message() -> None:
    fake_llm = RecordingFakeLLM(AIMessage(content="تحلیل کسب‌وکار"))
    node = build_analysis_node(fake_llm)  # type: ignore[arg-type]
    prior_messages = [
        HumanMessage(content="یک محصول برای مدیریت پیج اینستاگرام میخوام"),
        AIMessage(content="حتماً، این چند گزینه مناسب است: مدیریت پیج اینستاگرام (اقتصادی)"),
    ]

    node(
        {
            "messages": prior_messages,
            "entities": COMPLETE_ENTITIES,
            "last_shown_products": ["should never reach the analysis prompt"],
        }
    )

    assert fake_llm.received_messages is not None
    assert len(fake_llm.received_messages) == 2
    assert isinstance(fake_llm.received_messages[0], SystemMessage)
    assert isinstance(fake_llm.received_messages[1], HumanMessage)
    for message in fake_llm.received_messages:
        assert "اینستاگرام (اقتصادی)" not in message.content
    assert fake_llm.received_messages[1] not in prior_messages


def test_analysis_summary_includes_all_four_entities() -> None:
    fake_llm = RecordingFakeLLM(AIMessage(content="تحلیل"))
    node = build_analysis_node(fake_llm)  # type: ignore[arg-type]

    node({"messages": [], "entities": COMPLETE_ENTITIES, "last_shown_products": None})

    summary = fake_llm.received_messages[1].content  # type: ignore[index]
    for value in COMPLETE_ENTITIES.values():
        assert value in summary


def test_analysis_appends_the_llm_response_as_a_message() -> None:
    response = AIMessage(content="تحلیل نهایی")
    fake_llm = RecordingFakeLLM(response)
    node = build_analysis_node(fake_llm)  # type: ignore[arg-type]

    result = node({"messages": [], "entities": COMPLETE_ENTITIES, "last_shown_products": None})

    assert result["messages"] == [response]
