"""completion_check: the one deterministic fork in the graph, no LLM call.

Fires analysis+suggestion only once all 4 entities are present *and* the user has explicitly
asked for (or agreed to) a consultation — entity completeness alone is deliberately not enough,
see `docs/DECISIONS.md`'s 2026-09-22 "consultation firing requires an explicit request" entry.
"""

from consultant_bot.agentic.state import State
from consultant_bot.common.entities import Entities


def completion_check(state: State) -> bool:
    entities = state.get("entities") or Entities()
    return (
        entities.is_complete()
        and state.get("consultation_requested", False)
        and not state.get("consultation_done", False)
    )
