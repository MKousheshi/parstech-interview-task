from langchain_core.messages import AIMessage, HumanMessage

from consultant_bot.common.analysis import analysis_messages
from consultant_bot.common.search.base import ProductHit
from consultant_bot.common.search.products import Product
from consultant_bot.rigid.nodes.analysis import build_analysis_node
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


def test_analysis_is_invoked_with_only_the_prompt_and_entity_summary() -> None:
    fake_llm = ScriptedRunnable(AIMessage(content="تحلیل کسب‌وکار"))
    node = build_analysis_node(fake_llm)

    node(
        {
            "messages": [
                HumanMessage(content="مدیریت پیج اینستاگرام"),
                AIMessage(content="نتایج: مدیریت پیج اینستاگرام (اقتصادی)"),
            ],
            "entities": COMPLETE_ENTITIES,
            "last_search_results": [ProductHit(product=SHOWN_PRODUCT, score=1.0)],
        }
    )

    assert fake_llm.inputs == [analysis_messages(COMPLETE_ENTITIES)]


def test_analysis_appends_the_llm_response() -> None:
    response = AIMessage(content="تحلیل نهایی")
    node = build_analysis_node(ScriptedRunnable(response))

    result = node({"messages": [], "entities": COMPLETE_ENTITIES})

    assert result == {"messages": [response]}
