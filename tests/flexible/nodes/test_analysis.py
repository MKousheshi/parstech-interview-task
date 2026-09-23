from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from consultant_bot.common.search.base import ProductHit
from consultant_bot.common.search.products import Product
from consultant_bot.flexible.nodes.analysis import build_analysis_node
from tests.support import COMPLETE_ENTITIES, ScriptedRunnable

SHOWN_PRODUCT = Product(
    id=1,
    name="مدیریت پیج اینستاگرام (اقتصادی)",
    description="",
    short_description="",
    categories=[],
    price="1",
    permalink="https://example.com/1",
)


def test_analysis_call_contains_only_a_system_and_entities_summary_message() -> None:
    fake_llm = ScriptedRunnable(AIMessage(content="تحلیل کسب‌وکار"))
    node = build_analysis_node(fake_llm)
    prior_messages = [
        HumanMessage(content="یک محصول برای مدیریت پیج اینستاگرام میخوام"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_products", "args": {"query": "اینستاگرام"}, "id": "a"}],
        ),
        ToolMessage(
            content="- مدیریت پیج اینستاگرام (اقتصادی) (1 تومان): https://example.com/1",
            tool_call_id="a",
            name="search_products",
            artifact=[ProductHit(product=SHOWN_PRODUCT, score=1.0)],
        ),
        AIMessage(content="حتماً، این چند گزینه مناسب است: مدیریت پیج اینستاگرام (اقتصادی)"),
    ]

    node(
        {
            "messages": prior_messages,
            "entities": COMPLETE_ENTITIES,
        }
    )

    [received] = fake_llm.inputs
    assert len(received) == 2
    assert isinstance(received[0], SystemMessage)
    assert isinstance(received[1], HumanMessage)
    for message in received:
        assert "اینستاگرام (اقتصادی)" not in message.content
    assert received[1] not in prior_messages


def test_analysis_summary_includes_all_four_entities() -> None:
    fake_llm = ScriptedRunnable(AIMessage(content="تحلیل"))
    node = build_analysis_node(fake_llm)

    node({"messages": [], "entities": COMPLETE_ENTITIES})

    summary = fake_llm.inputs[0][1].content
    for value in COMPLETE_ENTITIES.known().values():
        assert value in summary


def test_analysis_appends_the_llm_response_and_stores_its_text() -> None:
    response = AIMessage(content="تحلیل نهایی")
    fake_llm = ScriptedRunnable(response)
    node = build_analysis_node(fake_llm)

    result = node({"messages": [], "entities": COMPLETE_ENTITIES})

    assert result["messages"] == [response]
    assert result["analysis"] == "تحلیل نهایی"
