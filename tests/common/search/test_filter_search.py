from pathlib import Path

import pytest

from consultant_bot.common.search.filter_search import FilterSearch
from consultant_bot.common.search.products import Product, load_products

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "products_fixture.json"


@pytest.fixture
def products() -> list[Product]:
    return load_products(FIXTURE_PATH)


def test_keyword_match_returns_matching_product(products: list[Product]) -> None:
    hits = FilterSearch(products).search("بیمه")
    assert hits == []


def test_keyword_match_finds_expected_product(products: list[Product]) -> None:
    hits = FilterSearch(products).search("تلگرام")
    assert [hit.product.id for hit in hits] == [7569]


def test_category_filter_narrows_to_matching_category(products: list[Product]) -> None:
    hits = FilterSearch(products).search("", category="اینستاگرام")
    assert [hit.product.id for hit in hits] == [9177]


def test_empty_query_returns_all_candidates_up_to_top_k(products: list[Product]) -> None:
    hits = FilterSearch(products).search("", top_k=2)
    assert len(hits) == 2


def test_ranking_prefers_products_matching_more_query_tokens(products: list[Product]) -> None:
    hits = FilterSearch(products).search("گوگل اینستاگرام")
    assert hits[0].product.id == 7341
    assert hits[0].score == 1.0
    assert {hit.product.id for hit in hits[1:]} == {9177, 7574}
    assert all(hit.score == pytest.approx(0.5) for hit in hits[1:])


def test_top_k_limits_result_count(products: list[Product]) -> None:
    hits = FilterSearch(products).search("مشاوره", top_k=1)
    assert len(hits) == 1
