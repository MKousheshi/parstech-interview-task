"""capture_entity node: records the entities of an `answer` or `consultation` turn.

Zero LLM calls. The entities `route_intent` found stated in the message fill the fields that are
still unset; a field that's already set is never overwritten (there's no correction mechanism). So a
consultation request that states all 4 goes straight to the analysis.

An `answer` that states no entity at all is stored verbatim as the pending field, so an answer the
classifier couldn't pin to a field isn't lost. An answer stating some *other* field leaves the
pending one unset, and `ask_entity` asks for it again; so does an empty reply.

The graph continues to `analysis` once all 4 are known, else to `ask_entity`. That also covers a
consultation whose 4th answer was captured but whose analysis or suggestion call then failed (a
timeout, a rate limit): the next consultation request finds all 4 set and retries it.
"""

from typing import Any

from consultant_bot.common.entities import ENTITY_FIELDS, Entities
from consultant_bot.common.messages import latest_user_text
from consultant_bot.scripted.state import State


def capture_entity(state: State) -> dict[str, Any]:
    entities = state.get("entities") or Entities()
    stated = state.get("stated_entities") or Entities()
    updates = {
        field: value
        for field in ENTITY_FIELDS
        if (value := stated.value(field)) and not entities.value(field)
    }
    field = state.get("awaiting_field")
    if state.get("intent") == "answer" and field is not None and not stated.known():
        updates[field] = latest_user_text(state["messages"])
    return {"entities": entities.model_copy(update=updates), "awaiting_field": None}
