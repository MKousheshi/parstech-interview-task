from langchain_core.messages import AIMessage, HumanMessage

from consultant_bot.common.entities import Entities
from consultant_bot.scripted.nodes.ask_entity import next_missing_field
from consultant_bot.scripted.nodes.capture_entity import capture_entity
from tests.support import COMPLETE_ENTITIES


def test_stores_the_stated_value_into_the_pending_field_and_clears_it() -> None:
    result = capture_entity(
        {
            "messages": [AIMessage(content="سؤال"), HumanMessage(content="یه کافه دارم")],
            "intent": "answer",
            "awaiting_field": "business_type",
            "stated_entities": Entities(business_type="کافه"),
        }
    )

    assert result["entities"] == Entities(business_type="کافه")
    assert result["awaiting_field"] is None


def test_a_consultation_request_stating_all_4_fills_them_all() -> None:
    result = capture_entity(
        {"messages": [], "intent": "consultation", "stated_entities": COMPLETE_ENTITIES}
    )

    assert result["entities"] == COMPLETE_ENTITIES


def test_fields_already_set_are_not_overwritten() -> None:
    result = capture_entity(
        {
            "messages": [HumanMessage(content="ما تو شیراز هستیم")],
            "intent": "answer",
            "entities": Entities(business_type="کافه", location="تهران"),
            "awaiting_field": "customer_type",
            "stated_entities": Entities(location="شیراز", customer_type="B2C"),
        }
    )

    assert result["entities"] == Entities(
        business_type="کافه", customer_type="B2C", location="تهران"
    )


def test_an_answer_stating_no_entity_is_stored_verbatim_as_the_pending_field() -> None:
    result = capture_entity(
        {
            "messages": [HumanMessage(content="  مردم عادی  ")],
            "intent": "answer",
            "awaiting_field": "customer_type",
            "stated_entities": Entities(),
        }
    )

    assert result["entities"] == Entities(customer_type="مردم عادی")


def test_an_answer_stating_another_field_leaves_the_pending_one_to_be_asked_again() -> None:
    result = capture_entity(
        {
            "messages": [HumanMessage(content="از اینستاگرام می‌فروشیم")],
            "intent": "answer",
            "awaiting_field": "business_type",
            "stated_entities": Entities(sales_channel="اینستاگرام"),
        }
    )

    assert result["entities"] == Entities(sales_channel="اینستاگرام")
    assert next_missing_field(result["entities"]) == "business_type"


def test_a_consultation_request_is_never_stored_verbatim() -> None:
    result = capture_entity(
        {
            "messages": [HumanMessage(content="مشاوره می‌خوام")],
            "intent": "consultation",
            "awaiting_field": "location",
        }
    )

    assert result["entities"] == Entities()


def test_an_empty_reply_leaves_the_field_unset_so_it_is_asked_again() -> None:
    result = capture_entity(
        {
            "messages": [HumanMessage(content="   ")],
            "intent": "answer",
            "entities": Entities(business_type="کافه", customer_type="B2C"),
            "awaiting_field": "location",
        }
    )

    assert next_missing_field(result["entities"]) == "location"
