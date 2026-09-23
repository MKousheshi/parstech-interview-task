from consultant_bot.flexible.nodes.assistant import is_complete_but_unrequested_and_unoffered

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
