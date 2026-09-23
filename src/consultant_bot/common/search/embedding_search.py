"""Phase 3 search strategy: multilingual sentence-embedding semantic search.

Model choice: `paraphrase-multilingual-MiniLM-L12-v2` (local `sentence-transformers`, no API
calls). It's a small (~118MB), well-established multilingual model covering Persian among its
50+ languages, chosen over a larger multilingual model for fast local inference on a laptop-scale
demo where semantic recall matters more than state-of-the-art multilingual STS benchmarks.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from sentence_transformers import SentenceTransformer

from consultant_bot.common.search.base import ProductHit, category_indices, product_text
from consultant_bot.common.search.products import Product

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def is_model_cached(model_name: str = EMBEDDING_MODEL_NAME) -> bool:
    """Checks whether `model_name` is in the local Hugging Face cache, without loading it or
    touching the network. Bare names resolve under `sentence-transformers/`, as they do for
    `SentenceTransformer` itself.
    """
    repo_id = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    try:
        snapshot_download(repo_id, local_files_only=True)
    except LocalEntryNotFoundError:
        return False
    return True


@dataclass
class EmbeddingSearch:
    """Ranks products by cosine similarity between query and product sentence embeddings."""

    # Cosine similarity between sentence embeddings, which runs higher than TF-IDF's for
    # unrelated text; below this it's noise.
    relevance_threshold: ClassVar[float] = 0.2

    products: list[Product]
    model_name: str = EMBEDDING_MODEL_NAME
    _model: SentenceTransformer = field(init=False, repr=False)
    _embeddings: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Pinned to CPU: this is a small model over a tiny catalog, and letting
        # sentence-transformers auto-pick CUDA breaks on GPUs the installed torch build doesn't
        # have kernels for (e.g. older compute-capability cards).
        self._model = SentenceTransformer(self.model_name, device="cpu")
        documents = [product_text(product) for product in self.products]
        self._embeddings = self._model.encode(
            documents, normalize_embeddings=True, convert_to_numpy=True
        )

    def search(self, query: str, category: str | None = None, top_k: int = 5) -> list[ProductHit]:
        indices = category_indices(self.products, category)
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
