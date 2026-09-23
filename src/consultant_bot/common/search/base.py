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


def search_with_category_fallback(
    strategy: SearchStrategy, query: str, category: str | None, top_k: int
) -> list[ProductHit]:
    """Searches within `category`, retrying across the whole catalog if that finds nothing.

    Categories reaching a search are LLM guesses, not picks from the real category list, so a
    guess that matches no category would otherwise turn a perfectly answerable query into "no
    products found".
    """
    hits = strategy.search(query, category=category, top_k=top_k)
    if category and not hits:
        hits = strategy.search(query, category=None, top_k=top_k)
    return hits


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
