import pytest
from langchain_core.messages import AIMessage

from consultant_bot.common.entities import ENTITY_FIELDS, Entities
from consultant_bot.scripted.nodes.ask_entity import QUESTIONS, ask_entity, next_missing_field
from tests.support import COMPLETE_ENTITIES


def test_questions_follow_the_canonical_field_order() -> None:
    assert tuple(QUESTIONS) == ENTITY_FIELDS


def test_asks_for_business_type_first_on_a_fresh_conversation() -> None:
    result = ask_entity({"messages": []})

    [message] = result["messages"]
    assert isinstance(message, AIMessage)
    assert message.content == QUESTIONS["business_type"]
    assert result["awaiting_field"] == "business_type"


def test_skips_fields_that_are_already_set() -> None:
    entities = Entities(business_type="کافه", customer_type="B2C")

    result = ask_entity({"messages": [], "entities": entities})

    assert result["awaiting_field"] == "location"
    assert result["messages"][0].content == QUESTIONS["location"]


def test_walks_the_fields_in_order_one_at_a_time() -> None:
    entities = Entities()
    asked = []
    while (field := next_missing_field(entities)) is not None:
        asked.append(field)
        entities = entities.model_copy(update={field: "مقدار"})

    assert tuple(asked) == ENTITY_FIELDS


def test_refuses_to_run_once_all_entities_are_known() -> None:
    with pytest.raises(ValueError):
        ask_entity({"messages": [], "entities": COMPLETE_ENTITIES})
