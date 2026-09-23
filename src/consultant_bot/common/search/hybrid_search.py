"""Hybrid search strategy: a weighted blend of a lexical and a semantic strategy's scores.

In the app it blends TF-IDF (exact-term precision) with embeddings (synonyms and paraphrases),
but it takes any two strategies, so tests can run it on fakes without loading the model.

The blend is a weighted sum of the two *raw* scores, not rank fusion (RRF) or per-query min-max
normalization. Both of those always give the best candidate a top score, even for an off-catalog
query, so `relevance_threshold` could no longer tell a real match from the least-bad one. Raw
cosine scores keep their absolute meaning, so the blended floor is the same blend of the two
strategies' own floors.
"""

import sys
from dataclasses import dataclass

from consultant_bot.common.search.base import ProductHit, SearchStrategy

# Embedding scores already spread wider than TF-IDF's (roughly 0.15-0.75 vs. 0-0.3 on this
# catalog), so an even split still lets semantics lead while exact terms break near-ties.
DEFAULT_LEXICAL_WEIGHT = 0.5


@dataclass
class HybridSearch:
    lexical: SearchStrategy
    semantic: SearchStrategy
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT

    def __post_init__(self) -> None:
        if not 0 <= self.lexical_weight <= 1:
            raise ValueError(f"lexical_weight must be between 0 and 1, got {self.lexical_weight}")

    @property
    def relevance_threshold(self) -> float:
        return self._blend(self.lexical.relevance_threshold, self.semantic.relevance_threshold)

    def _blend(self, lexical_score: float, semantic_score: float) -> float:
        return self.lexical_weight * lexical_score + (1 - self.lexical_weight) * semantic_score

    def search(self, query: str, category: str | None = None, top_k: int = 5) -> list[ProductHit]:
        # Every candidate from both sides, so each product is scored by both before the cut. A
        # product one side leaves out (TF-IDF drops zero-similarity ones) scores 0 there.
        lexical = {
            hit.product.id: hit
            for hit in self.lexical.search(query, category=category, top_k=sys.maxsize)
        }
        semantic = {
            hit.product.id: hit
            for hit in self.semantic.search(query, category=category, top_k=sys.maxsize)
        }

        hits = [
            ProductHit(
                product=hit.product,
                score=self._blend(
                    lexical[product_id].score if product_id in lexical else 0.0,
                    semantic[product_id].score if product_id in semantic else 0.0,
                ),
            )
            for product_id, hit in (lexical | semantic).items()
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]
