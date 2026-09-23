from langchain_core.messages import HumanMessage

from consultant_bot.common.search.filter_search import FilterSearch
from consultant_bot.common.search.products import load_products
from consultant_bot.scripted.nodes.ask_entity import PENDING_REMINDER, QUESTIONS
from consultant_bot.scripted.nodes.product_search import (
    NO_RESULTS_MESSAGE,
    RESULTS_HEADER,
    build_product_search_node,
)
from consultant_bot.scripted.state import Node
from tests.support import FIXTURE_PATH


def _node(top_k: int = 5) -> Node:
    return build_product_search_node(FilterSearch(load_products(FIXTURE_PATH)), top_k=top_k)


def test_formats_the_hits_for_the_raw_message_under_a_fixed_header() -> None:
    result = _node()({"messages": [HumanMessage(content="تلگرام")]})

    [message] = result["messages"]
    assert message.content.startswith(RESULTS_HEADER.format(query="تلگرام"))
    assert "خدمات ارسال پیام انبوه تلگرام" in message.content
    assert result["last_search_results"][0].product.id == 7569


def test_reports_nothing_found_for_an_unmatched_query() -> None:
    result = _node()({"messages": [HumanMessage(content="بیمه")]})

    assert result["messages"][0].content == NO_RESULTS_MESSAGE.format(query="بیمه")
    assert result["last_search_results"] == []


def test_returns_at_most_top_k_hits() -> None:
    result = _node(top_k=1)({"messages": [HumanMessage(content="خدمات سایت تلگرام")]})

    assert len(result["last_search_results"]) == 1


def test_a_search_mid_consultation_ends_with_the_pending_question() -> None:
    result = _node()({"messages": [HumanMessage(content="بیمه")], "awaiting_field": "location"})

    reminder = PENDING_REMINDER.format(question=QUESTIONS["location"])
    assert result["messages"][0].content == (
        f"{NO_RESULTS_MESSAGE.format(query='بیمه')}\n\n{reminder}"
    )
    assert "awaiting_field" not in result
