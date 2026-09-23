from consultant_bot.flexible.nodes.completion_check import completion_check

COMPLETE_ENTITIES = {
    "business_type": "کافه",
    "customer_type": "B2C",
    "location": "تهران",
    "sales_channel": "اینستاگرام",
}

INCOMPLETE_ENTITIES = {"business_type": "کافه"}


def _state(**overrides):  # type: ignore[no-untyped-def]
    base = {
        "entities": COMPLETE_ENTITIES,
        "consultation_requested": True,
        "consultation_done": False,
    }
    base.update(overrides)
    return base


def test_incomplete_entities_never_fires() -> None:
    state = _state(entities=INCOMPLETE_ENTITIES, consultation_requested=True)
    assert completion_check(state) is False


def test_complete_but_unrequested_does_not_fire() -> None:
    state = _state(consultation_requested=False)
    assert completion_check(state) is False


def test_requested_but_incomplete_does_not_fire() -> None:
    state = _state(entities=INCOMPLETE_ENTITIES, consultation_requested=True)
    assert completion_check(state) is False


def test_complete_and_requested_fires() -> None:
    state = _state(consultation_requested=True, consultation_done=False)
    assert completion_check(state) is True


def test_already_done_does_not_refire() -> None:
    state = _state(consultation_requested=True, consultation_done=True)
    assert completion_check(state) is False
