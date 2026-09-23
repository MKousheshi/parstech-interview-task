"""Phase 1 search strategy: naive substring/keyword matching plus an explicit category filter.

Query tokens are normalized and stripped of Persian stopwords first (see `text.py`); without that,
words like "و" or "در" substring-match almost every product and every query scores as relevant.
"""

from dataclasses import dataclass, field
from typing import ClassVar

from consultant_bot.common.search.base import ProductHit, category_indices, product_text
from consultant_bot.common.search.products import Product
from consultant_bot.common.search.text import keyword_tokens, normalize


@dataclass
class FilterSearch:
    """Bare-minimum keyword search: no ranking sophistication, just substring matching."""

    # Any hit this returns already matched at least one token (score > 0 by construction), so
    # 0.0 means "any real match".
    relevance_threshold: ClassVar[float] = 0.0

    products: list[Product]
    _haystacks: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._haystacks = [normalize(product_text(product)) for product in self.products]

    def search(self, query: str, category: str | None = None, top_k: int = 5) -> list[ProductHit]:
        indices = category_indices(self.products, category)

        if not query.strip():
            return [ProductHit(product=self.products[i], score=1.0) for i in indices][:top_k]

        tokens = keyword_tokens(query)
        if not tokens:
            return []

        hits = []
        for i in indices:
            matched = sum(1 for token in tokens if token in self._haystacks[i])
            if matched == 0:
                continue
            hits.append(ProductHit(product=self.products[i], score=matched / len(tokens)))

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]
