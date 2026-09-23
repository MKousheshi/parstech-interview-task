"""Phase 3 search strategy: multilingual sentence-embedding semantic search.

Model choice: `paraphrase-multilingual-MiniLM-L12-v2` (local `sentence-transformers`, no API
calls). It's a small (~118MB), well-established multilingual model covering Persian among its
50+ languages, chosen over a larger multilingual model for fast local inference on a laptop-scale
demo where semantic recall matters more than state-of-the-art multilingual STS benchmarks.
"""

from dataclasses import dataclass, field
from typing import Any

from sentence_transformers import SentenceTransformer

from consultant_bot.common.search.base import ProductHit
from consultant_bot.common.search.products import Product

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def is_model_cached(model_name: str = EMBEDDING_MODEL_NAME) -> bool:
    """Checks whether `model_name` is already cached locally, without touching the network."""
    try:
        SentenceTransformer(model_name, local_files_only=True, device="cpu")
    except Exception:
        return False
    return True


def _document(product: Product) -> str:
    return " ".join(
        [product.name, product.description, product.short_description, *product.categories]
    )


@dataclass
class EmbeddingSearch:
    """Ranks products by cosine similarity between query and product sentence embeddings."""

    products: list[Product]
    model_name: str = EMBEDDING_MODEL_NAME
    _model: SentenceTransformer = field(init=False, repr=False)
    _embeddings: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Pinned to CPU: this is a small model over a tiny catalog, and letting
        # sentence-transformers auto-pick CUDA breaks on GPUs the installed torch build doesn't
        # have kernels for (e.g. older compute-capability cards).
        self._model = SentenceTransformer(self.model_name, device="cpu")
        documents = [_document(product) for product in self.products]
        self._embeddings = self._model.encode(
            documents, normalize_embeddings=True, convert_to_numpy=True
        )

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

        query_embedding = self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        candidate_embeddings = self._embeddings[indices]
        similarities = candidate_embeddings @ query_embedding

        hits = [
            ProductHit(product=self.products[i], score=float(score))
            for i, score in zip(indices, similarities, strict=True)
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]
