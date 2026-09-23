from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from consultant_bot.common.search.base import ProductHit
from consultant_bot.common.search.products import Product
from consultant_bot.flexible.nodes.suggestion import SearchQuery, build_suggestion_node
from consultant_bot.flexible.state import State
from tests.support import COMPLETE_ENTITIES, ScriptedRunnable


def _product(product_id: int) -> Product:
    return Product(
        id=product_id,
        name=f"محصول {product_id}",
        description="",
        short_description="",
        categories=[],
        price="1000000",
        permalink=f"https://example.com/{product_id}",
    )


class FakeSearchStrategy:
    relevance_threshold = 0.3

    def __init__(self, hits: list[ProductHit]) -> None:
        self.hits = hits
        self.received_query: str | None = None

    def search(self, query: str, category: str | None = None, top_k: int = 5) -> list[ProductHit]:
        self.received_query = query
        return self.hits


def _state() -> State:
    return {
        "messages": [AIMessage(content="تحلیل کسب‌وکار شما این است...")],
        "entities": COMPLETE_ENTITIES,
        "analysis": "تحلیل کسب‌وکار شما این است...",
    }


def _recorded_hits(result: dict[str, Any]) -> list[ProductHit]:
    [record] = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    return list(record.artifact)


def test_hits_above_threshold_are_formatted_and_shown() -> None:
    hits = [ProductHit(product=_product(1), score=0.5)]
    strategy = FakeSearchStrategy(hits)
    formatter = ScriptedRunnable(AIMessage(content="پیشنهاد نهایی"))
    node = build_suggestion_node(
        ScriptedRunnable(SearchQuery(query="کافه")),
        strategy,
        formatter,
        top_k=5,
        relevance_threshold=0.1,
    )

    result = node(_state())

    assert formatter.was_called is True
    assert result["messages"][-1] == AIMessage(content="پیشنهاد نهایی")
    assert _recorded_hits(result) == hits
    assert result["consultation_done"] is True


def test_retrieval_is_recorded_as_a_valid_tool_call_pair_before_the_reply() -> None:
    """Price and link survive in the history even if the formatted prose drops them."""
    hits = [ProductHit(product=_product(1), score=0.5)]
    node = build_suggestion_node(
        ScriptedRunnable(SearchQuery(query="کافه", category="اینستاگرام")),
        FakeSearchStrategy(hits),
        ScriptedRunnable(AIMessage(content="پیشنهاد نهایی")),
        top_k=5,
        relevance_threshold=0.1,
    )

    call, record, reply = node(_state())["messages"]

    assert isinstance(call, AIMessage)
    [tool_call] = call.tool_calls
    assert tool_call["name"] == "search_products"
    assert tool_call["args"] == {"query": "کافه", "category": "اینستاگرام"}
    assert isinstance(record, ToolMessage)
    assert record.tool_call_id == tool_call["id"]
    assert "1000000" in record.content
    assert "https://example.com/1" in record.content
    assert reply.content == "پیشنهاد نهایی"


def test_hits_below_threshold_trigger_honest_fallback_without_calling_formatter() -> None:
    hits = [ProductHit(product=_product(1), score=0.05)]
    strategy = FakeSearchStrategy(hits)
    formatter = ScriptedRunnable(AIMessage(content="نباید این صدا زده شود"))
    node = build_suggestion_node(
        ScriptedRunnable(SearchQuery(query="کافه")),
        strategy,
        formatter,
        top_k=5,
        relevance_threshold=0.1,
    )

    result = node(_state())

    assert formatter.was_called is False
    assert "متأسفانه" in result["messages"][-1].content
    assert _recorded_hits(result) == []
    assert result["consultation_done"] is True


def test_no_hits_at_all_still_sets_consultation_done_and_records_an_empty_search() -> None:
    strategy = FakeSearchStrategy([])
    formatter = ScriptedRunnable(AIMessage(content="نباید این صدا زده شود"))
    node = build_suggestion_node(
        ScriptedRunnable(SearchQuery(query="کافه")),
        strategy,
        formatter,
        top_k=5,
        relevance_threshold=0.1,
    )

    result = node(_state())

    assert formatter.was_called is False
    assert _recorded_hits(result) == []
    assert result["messages"][1].content == "هیچ محصول مرتبطی یافت نشد."
    assert result["consultation_done"] is True


def test_search_is_called_with_the_formulated_query_and_category() -> None:
    strategy = FakeSearchStrategy([])
    node = build_suggestion_node(
        ScriptedRunnable(SearchQuery(query="مدیریت پیج اینستاگرام", category="اینستاگرام")),
        strategy,
        ScriptedRunnable(AIMessage(content="")),
        top_k=5,
        relevance_threshold=0.1,
    )

    node(_state())

    assert strategy.received_query == "مدیریت پیج اینستاگرام"


def test_query_formulation_reads_the_analysis_field_not_the_last_message() -> None:
    formulator = ScriptedRunnable(SearchQuery(query="کافه"))
    node = build_suggestion_node(
        formulator,
        FakeSearchStrategy([]),
        ScriptedRunnable(AIMessage(content="")),
        top_k=5,
        relevance_threshold=0.1,
    )
    state = _state()
    state["messages"].append(AIMessage(content="پیام نامرتبط بعدی"))

    node(state)

    prompt = formulator.inputs[0][1].content
    assert "تحلیل کسب‌وکار شما این است..." in prompt
    assert "پیام نامرتبط بعدی" not in prompt


def test_threshold_defaults_to_the_strategys_own_relevance_threshold() -> None:
    # 0.2 clears a 0.1 floor but not FakeSearchStrategy's own 0.3 one.
    strategy = FakeSearchStrategy([ProductHit(product=_product(1), score=0.2)])
    formatter = ScriptedRunnable(AIMessage(content=""))
    node = build_suggestion_node(
        ScriptedRunnable(SearchQuery(query="کافه")), strategy, formatter, top_k=5
    )

    result = node(_state())

    assert formatter.was_called is False
    assert _recorded_hits(result) == []
