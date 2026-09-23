from typing import get_args

from consultant_bot.common.entities import ENTITY_FIELDS
from consultant_bot.rigid.state import EntityField


def test_entity_field_literal_matches_the_shared_field_order() -> None:
    assert get_args(EntityField) == ENTITY_FIELDS
