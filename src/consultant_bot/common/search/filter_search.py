"""Phase 1 search strategy: naive substring/keyword matching plus an explicit category filter."""

from dataclasses import dataclass

from consultant_bot.common.search.base import ProductHit
from consultant_bot.common.search.products import Product


def _haystack(product: Product) -> str:
    return " ".join(
        [product.name, product.description, product.short_description, *product.categories]
    ).lower()


@dataclass
class FilterSearch:
    """Bare-minimum keyword search: no ranking sophistication, just substring matching."""

    products: list[Product]

    def search(self, query: str, category: str | None = None, top_k: int = 5) -> list[ProductHit]:
        candidates = self.products
        if category:
            category_lower = category.lower()
            candidates = [
                p for p in candidates if any(category_lower in c.lower() for c in p.categories)
            ]

        tokens = [t for t in query.strip().lower().split() if t]
        if not tokens:
            return [ProductHit(product=p, score=1.0) for p in candidates][:top_k]

        hits = []
        for product in candidates:
            haystack = _haystack(product)
            matched = sum(1 for token in tokens if token in haystack)
            if matched == 0:
                continue
            hits.append(ProductHit(product=product, score=matched / len(tokens)))

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]
