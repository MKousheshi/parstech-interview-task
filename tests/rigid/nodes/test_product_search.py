from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage

from consultant_bot.common.search.filter_search import FilterSearch
from consultant_bot.common.search.products import load_products
from consultant_bot.rigid.nodes.product_search import (
    NO_RESULTS_MESSAGE,
    RESULTS_HEADER,
    build_product_search_node,
)
from consultant_bot.rigid.state import State
from tests.support import FIXTURE_PATH


def _node(top_k: int = 5) -> Callable[[State], dict[str, Any]]:
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
