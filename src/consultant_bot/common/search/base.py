"""Search strategy interface and the helpers shared by all product search implementations."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from consultant_bot.common.search.products import Product


@dataclass
class ProductHit:
    product: Product
    score: float


class SearchStrategy(Protocol):
    @property
    def relevance_threshold(self) -> float:
        """Scores at or below this are noise for this strategy's scoring scale.

        Scales differ per strategy (a keyword-match fraction vs. a cosine similarity), so the
        floor a caller applies has to come from the strategy it's actually holding.
        """
        ...

    def search(
        self, query: str, category: str | None = None, top_k: int = 5
    ) -> list[ProductHit]: ...


def product_text(product: Product) -> str:
    """The text every strategy searches over: name, descriptions and category names."""
    return " ".join(
        [product.name, product.description, product.short_description, *product.categories]
    )


def category_indices(products: Sequence[Product], category: str | None) -> list[int]:
    """Indices of the products in `category` (case-insensitive substring), or all if `None`."""
    if not category:
        return list(range(len(products)))
    category_lower = category.lower()
    return [
        i
        for i, product in enumerate(products)
        if any(category_lower in c.lower() for c in product.categories)
    ]


def search_relevant(
    strategy: SearchStrategy,
    query: str,
    category: str | None,
    top_k: int,
    *,
    min_score: float | None = None,
) -> list[ProductHit]:
    """The hits that clear the relevance floor, searching within `category` first.

    The floor is `min_score`, or the strategy's own `relevance_threshold` when not given. Without
    it, the similarity strategies return their `top_k` best matches however unrelated they are,
    and an LLM shown five products for an off-catalog question will usually recommend one.

    Categories reaching a search are LLM guesses, not picks from the real category list, so when
    nothing relevant is found within `category` the search is retried across the whole catalog.
    Otherwise a guess that matches no category would turn a perfectly answerable query into "no
    products found".
    """
    floor = strategy.relevance_threshold if min_score is None else min_score

    def relevant(searched_category: str | None) -> list[ProductHit]:
        hits = strategy.search(query, category=searched_category, top_k=top_k)
        return [hit for hit in hits if hit.score > floor]

    hits = relevant(category)
    if category and not hits:
        hits = relevant(None)
    return hits


def format_hits(hits: Sequence[ProductHit]) -> str:
    """One line per hit — name, price and link.

    The single shape used everywhere hits reach an LLM: the `search_products` tool's result, the
    `suggestion` node's formatting prompt and the retrieval record it leaves in the history.
    Because every result that reaches the history has name, price and link, a follow-up question
    ("how much was the second one?") can be answered from it without re-running search.
    """
    return "\n".join(
        f"- {hit.product.name} ({hit.product.price} تومان): {hit.product.permalink}" for hit in hits
    )
