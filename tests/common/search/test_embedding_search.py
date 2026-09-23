"""Phase 3 tests. Skipped unless the embedding model is already cached locally: downloading a
~118MB model on every test run isn't practical for CI, and this environment may not have network
access to the model hub at all — see docs/TODO_FLEXIBLE.md's Phase 3 notes.
"""

import pytest

from consultant_bot.common.search.embedding_search import EmbeddingSearch, is_model_cached
from consultant_bot.common.search.products import Product, load_products
from tests.support import FIXTURE_PATH

pytestmark = pytest.mark.skipif(
    not is_model_cached(),
    reason="embedding model not cached locally; skipping instead of downloading it in tests",
)


@pytest.fixture(scope="module")
def products() -> list[Product]:
    return load_products(FIXTURE_PATH)


@pytest.fixture(scope="module")
def search(products: list[Product]) -> EmbeddingSearch:
    return EmbeddingSearch(products)


def test_relevant_query_ranks_matching_product_first(search: EmbeddingSearch) -> None:
    hits = search.search("طراحی سایت فروشگاهی")
    assert hits[0].product.id == 7341
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_category_filter_narrows_candidates(search: EmbeddingSearch) -> None:
    hits = search.search("مدیریت", category="اینستاگرام")
    assert all(hit.product.id == 9177 for hit in hits)


def test_empty_query_with_category_returns_filtered_candidates(search: EmbeddingSearch) -> None:
    hits = search.search("", category="اینستاگرام")
    assert [hit.product.id for hit in hits] == [9177]


def test_top_k_limits_result_count(search: EmbeddingSearch) -> None:
    hits = search.search("بازاریابی دیجیتال", top_k=2)
    assert len(hits) == 2
