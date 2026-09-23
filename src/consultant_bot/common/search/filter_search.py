"""Phase 1 search strategy: naive substring/keyword matching plus an explicit category filter.

Query tokens are normalized and stripped of Persian stopwords first (see `text.py`); without that,
words like "و" or "در" substring-match almost every product and every query scores as relevant.
"""

from dataclasses import dataclass

from consultant_bot.common.search.base import ProductHit
from consultant_bot.common.search.products import Product
from consultant_bot.common.search.text import keyword_tokens, normalize


def _haystack(product: Product) -> str:
    fields = [product.name, product.description, product.short_description, *product.categories]
    return normalize(" ".join(fields))


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

        if not query.strip():
            return [ProductHit(product=p, score=1.0) for p in candidates][:top_k]

        tokens = keyword_tokens(query)
        if not tokens:
            return []

        hits = []
        for product in candidates:
            haystack = _haystack(product)
            matched = sum(1 for token in tokens if token in haystack)
            if matched == 0:
                continue
            hits.append(ProductHit(product=product, score=matched / len(tokens)))

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]
