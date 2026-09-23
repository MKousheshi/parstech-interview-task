import pytest

from consultant_bot.common.search.products import Product, load_products
from consultant_bot.common.search.tfidf_search import TfidfSearch
from tests.support import FIXTURE_PATH


@pytest.fixture
def products() -> list[Product]:
    return load_products(FIXTURE_PATH)


def test_no_match_returns_empty_list(products: list[Product]) -> None:
    hits = TfidfSearch(products).search("هواپیما خودرو")
    assert hits == []


def test_relevant_query_ranks_matching_product_first(products: list[Product]) -> None:
    hits = TfidfSearch(products).search("طراحی سایت")
    assert hits[0].product.id == 7341
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_category_filter_narrows_candidates(products: list[Product]) -> None:
    hits = TfidfSearch(products).search("مدیریت", category="اینستاگرام")
    assert all(hit.product.id == 9177 for hit in hits)


def test_empty_query_with_category_returns_filtered_candidates(products: list[Product]) -> None:
    hits = TfidfSearch(products).search("", category="اینستاگرام")
    assert [hit.product.id for hit in hits] == [9177]


def test_top_k_limits_result_count(products: list[Product]) -> None:
    hits = TfidfSearch(products).search("طراحی سایت مشاوره اینستاگرام", top_k=2)
    assert len(hits) == 2
