"""The in-memory checkpointer every architecture's graph is compiled with.

Graph state carries a few of this package's own types (`Entities`, and `ProductHit`/`Product` in
`last_shown_products`). LangGraph's checkpoint serializer only warns when deserializing types it
wasn't told about today, and will refuse them in a future release, so they're registered
explicitly here rather than relying on that permissive default.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from consultant_bot.common.entities import Entities
from consultant_bot.common.search.base import ProductHit
from consultant_bot.common.search.products import Product

STATE_TYPES: tuple[type, ...] = (Entities, ProductHit, Product)


def build_checkpointer() -> MemorySaver:
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[(t.__module__, t.__qualname__) for t in STATE_TYPES]
    )
    return MemorySaver(serde=serde)
