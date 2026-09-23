"""completion_check: the one deterministic fork in the graph, no LLM call.

Fires analysis+suggestion only once all 4 entities are present *and* the user has explicitly
asked for (or agreed to) a consultation — entity completeness alone is deliberately not enough,
see `docs/DECISIONS.md`'s 2026-09-22 "consultation firing requires an explicit request" entry.
"""

from consultant_bot.common.entities import ENTITY_FIELDS, Entities
from consultant_bot.flexible.state import State


def completion_check(state: State) -> bool:
    entities: Entities = state.get("entities", {})
    all_present = all(entities.get(field) for field in ENTITY_FIELDS)
    return (
        all_present
        and state.get("consultation_requested", False)
        and not state.get("consultation_done", False)
    )
