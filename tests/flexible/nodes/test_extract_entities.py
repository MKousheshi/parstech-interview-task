from langchain_core.messages import HumanMessage

from consultant_bot.flexible.nodes.extract_entities import (
    ExtractedEntities,
    build_extract_entities_node,
)


class FakeExtractor:
    """Stands in for the structured-output LLM, returning one canned response per call."""

    def __init__(self, responses: list[ExtractedEntities]) -> None:
        self._responses = list(responses)

    def invoke(self, _messages: list) -> ExtractedEntities:  # type: ignore[type-arg]
        return self._responses.pop(0)


def _state(**overrides):  # type: ignore[no-untyped-def]
    base = {
        "messages": [HumanMessage(content="hi")],
        "entities": {},
        "consultation_requested": False,
        "consultation_offered": False,
        "consultation_done": False,
        "last_shown_products": None,
    }
    base.update(overrides)
    return base


def test_fills_entities_from_first_mention() -> None:
    node = build_extract_entities_node(FakeExtractor([ExtractedEntities(business_type="کافه")]))

    update = node(_state())

    assert update["entities"] == {"business_type": "کافه"}
    assert "consultation_requested" not in update


def test_merges_across_calls_leaving_unmentioned_fields_untouched() -> None:
    extractor = FakeExtractor(
        [
            ExtractedEntities(business_type="کافه"),
            ExtractedEntities(location="تهران"),
        ]
    )
    node = build_extract_entities_node(extractor)

    first_update = node(_state())
    second_state = _state(entities=first_update["entities"])
    second_update = node(second_state)

    assert second_update["entities"] == {"business_type": "کافه", "location": "تهران"}


def test_overwrites_on_new_mention() -> None:
    extractor = FakeExtractor([ExtractedEntities(customer_type="B2B")])
    node = build_extract_entities_node(extractor)

    update = node(_state(entities={"customer_type": "B2C"}))

    assert update["entities"]["customer_type"] == "B2B"


def test_vague_answer_leaves_field_unset() -> None:
    extractor = FakeExtractor([ExtractedEntities()])
    node = build_extract_entities_node(extractor)

    update = node(_state(entities={"business_type": "کافه"}))

    assert update["entities"] == {"business_type": "کافه"}


def test_consultation_requested_is_set_when_extractor_detects_it() -> None:
    extractor = FakeExtractor([ExtractedEntities(wants_consultation=True)])
    node = build_extract_entities_node(extractor)

    update = node(_state())

    assert update["consultation_requested"] is True


def test_consultation_requested_stays_true_once_set() -> None:
    extractor = FakeExtractor([ExtractedEntities(wants_consultation=False)])
    node = build_extract_entities_node(extractor)

    update = node(_state(consultation_requested=True))

    assert "consultation_requested" not in update


def test_consultation_done_clears_when_entity_changes_after_completion() -> None:
    extractor = FakeExtractor([ExtractedEntities(customer_type="B2B")])
    node = build_extract_entities_node(extractor)

    update = node(
        _state(
            entities={"customer_type": "B2C"},
            consultation_requested=True,
            consultation_done=True,
        )
    )

    assert update["consultation_done"] is False


def test_consultation_done_stays_true_when_nothing_changes() -> None:
    extractor = FakeExtractor([ExtractedEntities()])
    node = build_extract_entities_node(extractor)

    update = node(
        _state(
            entities={"customer_type": "B2C"},
            consultation_requested=True,
            consultation_done=True,
        )
    )

    assert "consultation_done" not in update
