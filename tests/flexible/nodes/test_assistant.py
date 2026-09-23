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


def test_prompt_carries_no_product_list_of_its_own() -> None:
    """Earlier results are read from the history, not re-injected into the system prompt."""
    history = [
        AIMessage(content="", tool_calls=[_search_call("a", "اینستاگرام")]),
        ToolMessage(
            content="- مدیریت پیج اینستاگرام (2500000 تومان): https://example.com/instagram",
            tool_call_id="a",
            name="search_products",
        ),
    ]

    prompt = _build_system_prompt(_state(messages=history))

    assert "https://example.com/instagram" not in prompt


def _search_call(call_id: str, query: str) -> dict[str, Any]:
    return {"name": "search_products", "args": {"query": query}, "id": call_id}


def _run_assistant(scripted: list[AIMessage], **state_overrides: Any) -> dict[str, Any]:
    llm = ToolCallingFakeModel(messages=iter(scripted))
    node = build_assistant_node(llm, FilterSearch(load_products(FIXTURE_PATH)), top_k=5)
    state = _state(
        **{
            "entities": Entities(),
            "messages": [HumanMessage(content="سلام")],
            **state_overrides,
        }
    )
    return node(state)


def test_parallel_search_results_stay_in_the_history_with_their_hits() -> None:
    """Compound requests are decomposed into parallel tool calls; each result is kept."""
    result = _run_assistant(
        [
            AIMessage(
                content="",
                tool_calls=[_search_call("a", "تلگرام"), _search_call("b", "اینستاگرام")],
            ),
            AIMessage(content="این‌ها را پیدا کردم"),
        ]
    )

    searches = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [search.tool_call_id for search in searches] == ["a", "b"]
    shown_ids = {hit.product.id for search in searches for hit in search.artifact}
    assert {7569, 9177} <= shown_ids
    assert set(result) == {"messages"}


def test_earlier_search_results_reach_the_model_on_a_later_turn() -> None:
    """A follow-up about an older result is answerable: the old ToolMessage is still in context."""
    ToolCallingFakeModel.received = []
    earlier = [
        HumanMessage(content="پکیج اینستاگرام دارید؟"),
        AIMessage(content="", tool_calls=[_search_call("a", "اینستاگرام")]),
        ToolMessage(
            content="- مدیریت پیج اینستاگرام (2500000 تومان): https://example.com/instagram",
            tool_call_id="a",
            name="search_products",
            artifact=[ProductHit(product=PRODUCT, score=1.0)],
        ),
        AIMessage(content="بله، این پکیج موجود است."),
        HumanMessage(content="قیمتش چند بود؟"),
    ]

    _run_assistant([AIMessage(content="۲.۵ میلیون تومان")], messages=earlier)

    [call] = ToolCallingFakeModel.received
    assert any("2500000" in str(message.content) for message in call)


def test_every_model_call_gets_a_system_prompt_built_from_current_state() -> None:
    ToolCallingFakeModel.received = []

    _run_assistant([AIMessage(content="سلام!")], entities=Entities(business_type="کافه"))

    [call] = ToolCallingFakeModel.received
    assert isinstance(call[0], SystemMessage)
    assert "«کافه»" in call[0].content
    assert call[1].content == "سلام"
