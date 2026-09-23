from langchain_core.messages import AIMessage

from consultant_bot.common.search.base import ProductHit
from consultant_bot.common.search.products import Product
from consultant_bot.flexible.nodes.suggestion import SearchQuery, build_suggestion_node

COMPLETE_ENTITIES = {
    "business_type": "کافه",
    "customer_type": "B2C",
    "location": "تهران",
    "sales_channel": "اینستاگرام",
}


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


class FakeQueryFormulator:
    def __init__(self, response: SearchQuery) -> None:
        self.response = response

    def invoke(self, _messages: list) -> SearchQuery:  # type: ignore[type-arg]
        return self.response


class FakeSearchStrategy:
    def __init__(self, hits: list[ProductHit]) -> None:
        self.hits = hits
        self.received_query: str | None = None

    def search(self, query: str, category: str | None = None, top_k: int = 5) -> list[ProductHit]:
        self.received_query = query
        return self.hits


class RecordingFakeLLM:
    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.was_called = False

    def invoke(self, _messages: list) -> AIMessage:  # type: ignore[type-arg]
        self.was_called = True
        return self.response


def _state():  # type: ignore[no-untyped-def]
    return {
        "messages": [AIMessage(content="تحلیل کسب‌وکار شما این است...")],
        "entities": COMPLETE_ENTITIES,
    }


def test_hits_above_threshold_are_formatted_and_shown() -> None:
    hits = [ProductHit(product=_product(1), score=0.5)]
    strategy = FakeSearchStrategy(hits)
    formatter = RecordingFakeLLM(AIMessage(content="پیشنهاد نهایی"))
    node = build_suggestion_node(
        FakeQueryFormulator(SearchQuery(query="کافه")),
        strategy,
        formatter,
        relevance_threshold=0.1,
    )

    result = node(_state())

    assert formatter.was_called is True
    assert result["messages"] == [AIMessage(content="پیشنهاد نهایی")]
    assert result["last_shown_products"] == hits
    assert result["consultation_done"] is True


def test_hits_below_threshold_trigger_honest_fallback_without_calling_formatter() -> None:
    hits = [ProductHit(product=_product(1), score=0.05)]
    strategy = FakeSearchStrategy(hits)
    formatter = RecordingFakeLLM(AIMessage(content="نباید این صدا زده شود"))
    node = build_suggestion_node(
        FakeQueryFormulator(SearchQuery(query="کافه")),
        strategy,
        formatter,
        relevance_threshold=0.1,
    )

    result = node(_state())

    assert formatter.was_called is False
    assert "متأسفانه" in result["messages"][0].content
    assert result["last_shown_products"] == []
    assert result["consultation_done"] is True


def test_no_hits_at_all_still_sets_consultation_done_and_empty_last_shown() -> None:
    strategy = FakeSearchStrategy([])
    formatter = RecordingFakeLLM(AIMessage(content="نباید این صدا زده شود"))
    node = build_suggestion_node(
        FakeQueryFormulator(SearchQuery(query="کافه")), strategy, formatter, relevance_threshold=0.1
    )

    result = node(_state())

    assert formatter.was_called is False
    assert result["last_shown_products"] == []
    assert result["consultation_done"] is True


def test_search_is_called_with_the_formulated_query_and_category() -> None:
    strategy = FakeSearchStrategy([])
    node = build_suggestion_node(
        FakeQueryFormulator(SearchQuery(query="مدیریت پیج اینستاگرام", category="اینستاگرام")),
        strategy,
        RecordingFakeLLM(AIMessage(content="")),
        relevance_threshold=0.1,
    )

    node(_state())

    assert strategy.received_query == "مدیریت پیج اینستاگرام"
