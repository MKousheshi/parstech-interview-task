import pytest
from langchain_core.messages import AIMessage, HumanMessage

from consultant_bot.common.entities import Entities
from consultant_bot.scripted.nodes.ask_entity import next_missing_field
from consultant_bot.scripted.nodes.capture_entity import capture_entity


def test_stores_the_trimmed_reply_into_the_pending_field_and_clears_it() -> None:
    result = capture_entity(
        {
            "messages": [AIMessage(content="سؤال"), HumanMessage(content="  کافه  ")],
            "awaiting_field": "business_type",
        }
    )

    assert result["entities"] == Entities(business_type="کافه")
    assert result["awaiting_field"] is None


def test_leaves_other_fields_untouched() -> None:
    result = capture_entity(
        {
            "messages": [HumanMessage(content="تهران")],
            "entities": Entities(business_type="کافه", customer_type="B2C"),
            "awaiting_field": "location",
        }
    )

    assert result["entities"] == Entities(
        business_type="کافه", customer_type="B2C", location="تهران"
    )


def test_an_off_topic_reply_is_stored_verbatim() -> None:
    # The scripted design's documented limitation: a pending field swallows whatever comes next.
    reply = "راستش می‌خوام محصولات رو جست‌وجو کنم"

    result = capture_entity(
        {"messages": [HumanMessage(content=reply)], "awaiting_field": "location"}
    )

    assert result["entities"].location == reply


def test_an_empty_reply_leaves_the_field_unset_so_it_is_asked_again() -> None:
    result = capture_entity(
        {"messages": [HumanMessage(content="   ")], "awaiting_field": "business_type"}
    )

    assert next_missing_field(result["entities"]) == "business_type"


def test_refuses_to_run_without_a_pending_field() -> None:
    with pytest.raises(ValueError):
        capture_entity({"messages": [HumanMessage(content="کافه")]})
