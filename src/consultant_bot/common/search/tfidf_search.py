"""Phase 2 search strategy: TF-IDF vectorization + cosine similarity ranking."""

from dataclasses import dataclass, field
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from consultant_bot.common.search.base import ProductHit
from consultant_bot.common.search.products import Product


def _document(product: Product) -> str:
    return " ".join(
        [product.name, product.description, product.short_description, *product.categories]
    )


@dataclass
class TfidfSearch:
    """Ranks products by cosine similarity between the query and each product's TF-IDF vector.

    Category names are folded into the corpus text rather than handled separately, so a
    free-text query mentioning a category naturally contributes to relevance.
    """

    products: list[Product]
    _vectorizer: TfidfVectorizer = field(init=False, repr=False)
    _matrix: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._vectorizer = TfidfVectorizer()
        documents = [_document(product) for product in self.products]
        self._matrix = self._vectorizer.fit_transform(documents)

    def search(self, query: str, category: str | None = None, top_k: int = 5) -> list[ProductHit]:
        indices = list(range(len(self.products)))
        if category:
            category_lower = category.lower()
            indices = [
                i
                for i in indices
                if any(category_lower in c.lower() for c in self.products[i].categories)
            ]
        if not indices:
            return []

        if not query.strip():
            return [ProductHit(product=self.products[i], score=1.0) for i in indices][:top_k]

        query_vector = self._vectorizer.transform([query])
        candidate_matrix = self._matrix[indices]
        similarities = cosine_similarity(query_vector, candidate_matrix)[0]

        hits = [
            ProductHit(product=self.products[i], score=float(score))
            for i, score in zip(indices, similarities, strict=True)
            if score > 0
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]
