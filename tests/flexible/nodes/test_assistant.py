from pathlib import Path
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from consultant_bot.common.search.base import ProductHit
from consultant_bot.common.search.filter_search import FilterSearch
from consultant_bot.common.search.products import Product, load_products
from consultant_bot.flexible.nodes.assistant import (
    _build_system_prompt,
    build_assistant_node,
    is_complete_but_unrequested_and_unoffered,
    searched_hits,
)

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "products_fixture.json"

PRODUCT = Product(
    id=1,
    name="مدیریت پیج اینستاگرام",
    description="",
    short_description="",
    categories=["اینستاگرام"],
    price="2500000",
    permalink="https://example.com/instagram",
)

COMPLETE_ENTITIES = {
    "business_type": "کافه",
    "customer_type": "B2C",
    "location": "تهران",
    "sales_channel": "اینستاگرام",
}


def _state(**overrides):  # type: ignore[no-untyped-def]
    base = {
        "entities": COMPLETE_ENTITIES,
        "consultation_requested": False,
        "consultation_offered": False,
        "consultation_done": False,
    }
    base.update(overrides)
    return base


def test_true_when_complete_and_unrequested_and_unoffered() -> None:
    assert is_complete_but_unrequested_and_unoffered(_state()) is True


def test_false_when_entities_incomplete() -> None:
    entities = dict(COMPLETE_ENTITIES)
    del entities["location"]
    assert is_complete_but_unrequested_and_unoffered(_state(entities=entities)) is False


def test_false_when_already_requested() -> None:
    assert is_complete_but_unrequested_and_unoffered(_state(consultation_requested=True)) is False


def test_false_when_already_offered() -> None:
    assert is_complete_but_unrequested_and_unoffered(_state(consultation_offered=True)) is False


def test_false_when_already_done() -> None:
    assert is_complete_but_unrequested_and_unoffered(_state(consultation_done=True)) is False


def test_prompt_carries_price_and_link_for_the_last_shown_products() -> None:
    """Follow-ups like "how much is it?" are meant to be answerable without re-running search."""
    state = _state(last_shown_products=[ProductHit(product=PRODUCT, score=1.0)])

    prompt = _build_system_prompt(state)

    assert "مدیریت پیج اینستاگرام" in prompt
    assert "2500000" in prompt
    assert "https://example.com/instagram" in prompt


def test_prompt_says_nothing_shown_yet_when_state_is_empty() -> None:
    assert "(هیچ)" in _build_system_prompt(_state(last_shown_products=None))


class _ToolCallingFakeModel(GenericFakeChatModel):
    """Replays scripted AI messages; `bind_tools` is a no-op since the tool calls are scripted."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ToolCallingFakeModel":
        return self


def _search_call(call_id: str, query: str) -> dict[str, Any]:
    return {"name": "search_products", "args": {"query": query}, "id": call_id}


def _run_assistant(scripted: list[AIMessage], **state_overrides: Any) -> dict[str, Any]:
    llm = _ToolCallingFakeModel(messages=iter(scripted))
    node = build_assistant_node(llm, FilterSearch(load_products(FIXTURE_PATH)), top_k=5)
    state = _state(entities={}, messages=[HumanMessage(content="سلام")], last_shown_products=None)
    state.update(state_overrides)
    return node(state)  # type: ignore[no-any-return]


def test_parallel_search_calls_in_one_step_merge_into_last_shown_products() -> None:
    """Compound requests are decomposed into parallel tool calls; they must not collide."""
    result = _run_assistant(
        [
            AIMessage(
                content="",
                tool_calls=[_search_call("a", "تلگرام"), _search_call("b", "اینستاگرام")],
            ),
            AIMessage(content="این‌ها را پیدا کردم"),
        ]
    )

    shown_ids = [hit.product.id for hit in result["last_shown_products"]]
    assert 7569 in shown_ids
    assert 9177 in shown_ids
    assert len(shown_ids) == len(set(shown_ids))


def test_turn_without_search_keeps_previous_last_shown_products() -> None:
    previous = [ProductHit(product=PRODUCT, score=1.0)]

    result = _run_assistant(
        [AIMessage(content="قیمتش ۲.۵ میلیون است")], last_shown_products=previous
    )

    assert result["last_shown_products"] == previous


def test_searched_hits_is_none_without_a_search_and_empty_for_an_empty_search() -> None:
    assert searched_hits([AIMessage(content="سلام")]) is None
    empty_search = ToolMessage(content="", tool_call_id="a", name="search_products", artifact=[])
    assert searched_hits([empty_search]) == []
