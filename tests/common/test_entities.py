from consultant_bot.common.entities import ENTITY_FIELDS, ENTITY_LABELS, Entities

COMPLETE = Entities(
    business_type="کافه", customer_type="B2C", location="تهران", sales_channel="اینستاگرام"
)


def test_is_complete_only_when_all_four_fields_are_set() -> None:
    assert COMPLETE.is_complete() is True
    assert Entities().is_complete() is False
    assert COMPLETE.model_copy(update={"location": None}).is_complete() is False
    assert COMPLETE.model_copy(update={"location": ""}).is_complete() is False


def test_known_and_missing_split_the_fields_by_label() -> None:
    entities = Entities(business_type="کافه")

    assert entities.known() == {ENTITY_LABELS["business_type"]: "کافه"}
    assert entities.missing_labels() == [
        ENTITY_LABELS[field] for field in ENTITY_FIELDS if field != "business_type"
    ]


def test_summary_has_one_labeled_line_per_known_field() -> None:
    lines = COMPLETE.summary().splitlines()

    assert len(lines) == 4
    assert lines[0] == f"- {ENTITY_LABELS['business_type']}: کافه"
