from langchain_core.messages import AIMessage

from consultant_bot.common.entities import Entities
from consultant_bot.common.search.base import ProductHit, format_hits
from consultant_bot.common.search.products import Product
from consultant_bot.rigid.nodes.suggestion import (
    NO_RESULTS_MESSAGE,
    build_query,
    build_suggestion_node,
)
from consultant_bot.rigid.state import State
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
        self.received: list[tuple[str, str | None, int]] = []

    def search(self, query: str, category: str | None = None, top_k: int = 5) -> list[ProductHit]:
        self.received.append((query, category, top_k))
        return self.hits


def _state() -> State:
    return {"messages": [AIMessage(content="تحلیل کسب‌وکار")], "entities": COMPLETE_ENTITIES}


def test_query_joins_business_type_and_sales_channel() -> None:
    assert build_query(COMPLETE_ENTITIES) == "کافه اینستاگرام"
    assert build_query(Entities(business_type="کافه")) == "کافه"


def test_runs_one_search_with_the_template_query_and_top_k() -> None:
    strategy = FakeSearchStrategy([ProductHit(product=_product(1), score=0.9)])
    node = build_suggestion_node(strategy, ScriptedRunnable(AIMessage(content="پیشنهاد")), top_k=3)

    node(_state())

    assert strategy.received == [("کافه اینستاگرام", None, 3)]


def test_formatter_gets_every_hit_even_below_the_strategy_threshold() -> None:
    # No relevance floor in the rigid variant: a 0.01 hit is written up like any other.
    hits = [
        ProductHit(product=_product(1), score=0.9),
        ProductHit(product=_product(2), score=0.01),
    ]
    formatter = ScriptedRunnable(AIMessage(content="پیشنهاد"))
    node = build_suggestion_node(FakeSearchStrategy(hits), formatter, top_k=5)

    result = node(_state())

    [received] = formatter.inputs
    prompt = received[1].content
    assert format_hits(hits) in prompt
    assert COMPLETE_ENTITIES.summary() in prompt
    assert result["messages"] == [AIMessage(content="پیشنهاد")]
    assert result["last_search_results"] == hits
    assert result["consultation_done"] is True


def test_no_hits_at_all_skips_the_formatter_for_a_fixed_message() -> None:
    formatter = ScriptedRunnable(AIMessage(content="پیشنهاد"))
    node = build_suggestion_node(FakeSearchStrategy([]), formatter, top_k=5)

    result = node(_state())

    assert not formatter.was_called
    assert [m.content for m in result["messages"]] == [NO_RESULTS_MESSAGE]
    assert result["last_search_results"] == []
    assert result["consultation_done"] is True
