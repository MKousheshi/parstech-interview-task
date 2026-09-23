from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from consultant_bot.common.entities import Entities
from consultant_bot.common.search.base import ProductHit
from consultant_bot.common.search.filter_search import FilterSearch
from consultant_bot.common.search.products import Product, load_products
from consultant_bot.flexible.nodes.assistant import (
    _build_system_prompt,
    build_assistant_node,
    is_complete_but_unrequested_and_unoffered,
    searched_hits,
)
from consultant_bot.flexible.state import State
from tests.support import COMPLETE_ENTITIES, FIXTURE_PATH, ToolCallingFakeModel

PRODUCT = Product(
    id=1,
    name="مدیریت پیج اینستاگرام",
    description="",
    short_description="",
    categories=["اینستاگرام"],
    price="2500000",
    permalink="https://example.com/instagram",
)


def _state(**overrides: Any) -> State:
    base: dict[str, Any] = {
        "messages": [],
        "entities": COMPLETE_ENTITIES,
        "consultation_requested": False,
        "consultation_offered": False,
        "consultation_done": False,
    }
    base.update(overrides)
    return cast(State, base)


def test_true_when_complete_and_unrequested_and_unoffered() -> None:
    assert is_complete_but_unrequested_and_unoffered(_state()) is True


def test_false_when_entities_incomplete() -> None:
    entities = COMPLETE_ENTITIES.model_copy(update={"location": None})
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


def _search_call(call_id: str, query: str) -> dict[str, Any]:
    return {"name": "search_products", "args": {"query": query}, "id": call_id}


def _run_assistant(scripted: list[AIMessage], **state_overrides: Any) -> dict[str, Any]:
    llm = ToolCallingFakeModel(messages=iter(scripted))
    node = build_assistant_node(llm, FilterSearch(load_products(FIXTURE_PATH)), top_k=5)
    state = _state(
        **{
            "entities": Entities(),
            "messages": [HumanMessage(content="سلام")],
            "last_shown_products": None,
            **state_overrides,
        }
    )
    return node(state)


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


def test_turn_without_search_leaves_last_shown_products_untouched() -> None:
    previous = [ProductHit(product=PRODUCT, score=1.0)]

    result = _run_assistant(
        [AIMessage(content="قیمتش ۲.۵ میلیون است")], last_shown_products=previous
    )

    assert "last_shown_products" not in result


def test_searched_hits_is_none_without_a_search_and_empty_for_an_empty_search() -> None:
    assert searched_hits([AIMessage(content="سلام")]) is None
    empty_search = ToolMessage(content="", tool_call_id="a", name="search_products", artifact=[])
    assert searched_hits([empty_search]) == []


def test_every_model_call_gets_a_system_prompt_built_from_current_state() -> None:
    ToolCallingFakeModel.received = []

    _run_assistant([AIMessage(content="سلام!")], entities=Entities(business_type="کافه"))

    [call] = ToolCallingFakeModel.received
    assert isinstance(call[0], SystemMessage)
    assert "کافه" in call[0].content
    assert call[1].content == "سلام"
