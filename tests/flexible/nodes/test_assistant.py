from consultant_bot.common.search.base import ProductHit
from consultant_bot.common.search.products import Product
from consultant_bot.flexible.nodes.assistant import (
    _build_system_prompt,
    is_complete_but_unrequested_and_unoffered,
)

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
