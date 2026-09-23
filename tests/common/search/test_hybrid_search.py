import pytest

from consultant_bot.common.search.base import ProductHit, search_relevant
from consultant_bot.common.search.embedding_search import EmbeddingSearch, is_model_cached
from consultant_bot.common.search.hybrid_search import HybridSearch
from consultant_bot.common.search.products import Product, load_products
from consultant_bot.common.search.tfidf_search import TfidfSearch
from tests.support import FIXTURE_PATH


def _product(product_id: int) -> Product:
    return Product(
        id=product_id,
        name=f"محصول {product_id}",
        description="",
        short_description="",
        categories=[],
        price="1000000",
        permalink=f"https://example.com/{product_id}",
    )


class FakeStrategy:
    def __init__(self, scores: dict[int, float], relevance_threshold: float = 0.0) -> None:
        self.scores = scores
        self.relevance_threshold = relevance_threshold
        self.calls: list[tuple[str, str | None, int]] = []

    def search(self, query: str, category: str | None = None, top_k: int = 5) -> list[ProductHit]:
        self.calls.append((query, category, top_k))
        hits = [ProductHit(product=_product(i), score=s) for i, s in self.scores.items()]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]


def _scores(hits: list[ProductHit]) -> dict[int, float]:
    return {hit.product.id: round(hit.score, 6) for hit in hits}


def test_scores_are_the_weighted_blend_of_both_sides() -> None:
    hybrid = HybridSearch(
        lexical=FakeStrategy({1: 0.4, 2: 0.2}),
        semantic=FakeStrategy({1: 0.6, 2: 0.8}),
        lexical_weight=0.25,
    )

    assert _scores(hybrid.search("q")) == {1: 0.55, 2: 0.65}


def test_a_product_one_side_leaves_out_scores_zero_there() -> None:
    hybrid = HybridSearch(lexical=FakeStrategy({1: 0.4}), semantic=FakeStrategy({2: 0.6}))

    assert _scores(hybrid.search("q")) == {1: 0.2, 2: 0.3}


def test_hits_are_sorted_by_blended_score_and_cut_to_top_k() -> None:
    hybrid = HybridSearch(
        lexical=FakeStrategy({1: 0.9, 2: 0.0, 3: 0.1}),
        semantic=FakeStrategy({1: 0.1, 2: 0.2, 3: 1.0}),
    )

    assert [hit.product.id for hit in hybrid.search("q", top_k=2)] == [3, 1]


def test_both_sides_are_searched_in_full_with_the_same_query_and_category() -> None:
    lexical, semantic = FakeStrategy({}), FakeStrategy({})
    HybridSearch(lexical=lexical, semantic=semantic).search("q", category="سئو", top_k=2)

    for side in (lexical, semantic):
        [(query, category, top_k)] = side.calls
        assert (query, category) == ("q", "سئو")
        assert top_k > 2  # every candidate, so the cut happens after blending


def test_relevance_threshold_is_the_same_blend_of_both_floors() -> None:
    hybrid = HybridSearch(
        lexical=FakeStrategy({}, relevance_threshold=0.1),
        semantic=FakeStrategy({}, relevance_threshold=0.2),
        lexical_weight=0.5,
    )

    assert hybrid.relevance_threshold == pytest.approx(0.15)


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_a_weight_outside_zero_to_one_is_rejected(weight: float) -> None:
    with pytest.raises(ValueError, match="lexical_weight"):
        HybridSearch(lexical=FakeStrategy({}), semantic=FakeStrategy({}), lexical_weight=weight)


needs_model = pytest.mark.skipif(
    not is_model_cached(),
    reason="embedding model not cached locally; skipping instead of downloading it in tests",
)


@pytest.fixture(scope="module")
def products() -> list[Product]:
    return load_products(FIXTURE_PATH)


@pytest.fixture(scope="module")
def embedding(products: list[Product]) -> EmbeddingSearch:
    return EmbeddingSearch(products)


@pytest.fixture(scope="module")
def hybrid(products: list[Product], embedding: EmbeddingSearch) -> HybridSearch:
    return HybridSearch(lexical=TfidfSearch(products), semantic=embedding)


@needs_model
def test_relevant_query_ranks_matching_product_first(hybrid: HybridSearch) -> None:
    hits = search_relevant(hybrid, "طراحی سایت فروشگاهی", None, top_k=5)
    assert hits[0].product.id == 7341


@needs_model
def test_off_catalog_query_finds_nothing_relevant(hybrid: HybridSearch) -> None:
    assert search_relevant(hybrid, "دستور پخت قورمه سبزی", None, top_k=5) == []
