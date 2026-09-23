from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from consultant_bot.common.entities import Entities
from consultant_bot.flexible.nodes.extract_entities import (
    RECENT_MESSAGES_WINDOW,
    ExtractedEntities,
    build_extract_entities_node,
    recent_context,
)
from consultant_bot.flexible.state import State
from tests.support import ScriptedRunnable


def _state(**overrides: Any) -> State:
    base: dict[str, Any] = {
        "messages": [HumanMessage(content="hi")],
        "entities": Entities(),
        "consultation_requested": False,
        "consultation_offered": False,
        "consultation_done": False,
        "last_shown_products": None,
    }
    base.update(overrides)
    return cast(State, base)


def _react_turn(index: int) -> list[BaseMessage]:
    """One turn's worth of history as the assistant's ReAct loop leaves it behind."""
    return [
        HumanMessage(content=f"سؤال {index}"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_products", "args": {"query": "x"}, "id": f"c{index}"}],
        ),
        ToolMessage(content="- محصول", tool_call_id=f"c{index}"),
        AIMessage(content=f"پاسخ {index}"),
    ]


def test_recent_context_drops_tool_traffic() -> None:
    context = recent_context(_react_turn(1))

    assert [m.content for m in context] == ["سؤال 1", "پاسخ 1"]


def test_recent_context_never_starts_on_an_orphaned_tool_message() -> None:
    history = [message for index in range(5) for message in _react_turn(index)]

    context = recent_context(history)

    assert len(context) == RECENT_MESSAGES_WINDOW
    assert not any(isinstance(m, ToolMessage) for m in context)
    assert not any(isinstance(m, AIMessage) and m.tool_calls for m in context)


def test_fills_entities_from_first_mention() -> None:
    node = build_extract_entities_node(ScriptedRunnable(ExtractedEntities(business_type="کافه")))

    update = node(_state())

    assert update["entities"] == Entities(business_type="کافه")
    assert "consultation_requested" not in update


def test_merges_across_calls_leaving_unmentioned_fields_untouched() -> None:
    extractor = ScriptedRunnable(
        ExtractedEntities(business_type="کافه"), ExtractedEntities(location="تهران")
    )
    node = build_extract_entities_node(extractor)

    first_update = node(_state())
    second_state = _state(entities=first_update["entities"])
    second_update = node(second_state)

    assert second_update["entities"] == Entities(business_type="کافه", location="تهران")


def test_overwrites_on_new_mention() -> None:
    extractor = ScriptedRunnable(ExtractedEntities(customer_type="B2B"))
    node = build_extract_entities_node(extractor)

    update = node(_state(entities=Entities(customer_type="B2C")))

    assert update["entities"].customer_type == "B2B"


def test_vague_answer_leaves_field_unset() -> None:
    extractor = ScriptedRunnable(ExtractedEntities())
    node = build_extract_entities_node(extractor)

    update = node(_state(entities=Entities(business_type="کافه")))

    assert update["entities"] == Entities(business_type="کافه")


def test_consultation_requested_is_set_when_extractor_detects_it() -> None:
    extractor = ScriptedRunnable(ExtractedEntities(wants_consultation=True))
    node = build_extract_entities_node(extractor)

    update = node(_state())

    assert update["consultation_requested"] is True


def test_consultation_requested_stays_true_once_set() -> None:
    extractor = ScriptedRunnable(ExtractedEntities(wants_consultation=False))
    node = build_extract_entities_node(extractor)

    update = node(_state(consultation_requested=True))

    assert "consultation_requested" not in update


def test_consultation_done_clears_when_entity_changes_after_completion() -> None:
    extractor = ScriptedRunnable(ExtractedEntities(customer_type="B2B"))
    node = build_extract_entities_node(extractor)

    update = node(
        _state(
            entities=Entities(customer_type="B2C"),
            consultation_requested=True,
            consultation_done=True,
        )
    )

    assert update["consultation_done"] is False


def test_consultation_done_stays_true_when_nothing_changes() -> None:
    extractor = ScriptedRunnable(ExtractedEntities())
    node = build_extract_entities_node(extractor)

    update = node(
        _state(
            entities=Entities(customer_type="B2C"),
            consultation_requested=True,
            consultation_done=True,
        )
    )

    assert "consultation_done" not in update
