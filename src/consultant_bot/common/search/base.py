"""Search strategy interface shared by all product search implementations."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from consultant_bot.common.search.products import Product


@dataclass
class ProductHit:
    product: Product
    score: float


class SearchStrategy(Protocol):
    def search(
        self, query: str, category: str | None = None, top_k: int = 5
    ) -> list[ProductHit]: ...


def format_hits(hits: Sequence[ProductHit]) -> str:
    """One line per hit — name, price and link.

    The single shape used everywhere hits reach an LLM: the `search_products` tool's result, the
    `suggestion` node's formatting prompt, and the `last_shown_products` block in the assistant's
    system prompt. Keeping them identical is what lets a follow-up question ("how much was the
    second one?") be answered from state without re-running search.
    """
    return "\n".join(
        f"- {hit.product.name} ({hit.product.price} تومان): {hit.product.permalink}" for hit in hits
    )
