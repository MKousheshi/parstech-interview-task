"""capture_entity node: stores the user's reply verbatim as the value of the pending field.

Zero LLM calls and no validation: whatever was typed (trimmed) becomes the value, even if it was
clearly meant as something else — the scripted design's documented limitation. An empty reply leaves
the field unset, so `ask_entity` asks for it again.
"""

from typing import Any

from consultant_bot.common.entities import Entities
from consultant_bot.common.messages import latest_user_text
from consultant_bot.scripted.state import State


def capture_entity(state: State) -> dict[str, Any]:
    field = state.get("awaiting_field")
    if field is None:
        # The graph only routes here while a field is pending.
        raise ValueError("capture_entity reached with no pending field")
    entities = state.get("entities") or Entities()
    value = latest_user_text(state["messages"])
    return {
        "entities": entities.model_copy(update={field: value}),
        "awaiting_field": None,
    }
