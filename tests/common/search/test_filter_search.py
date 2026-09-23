import pytest

from consultant_bot.common.search.filter_search import FilterSearch
from consultant_bot.common.search.products import Product, load_products
from tests.support import FIXTURE_PATH


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


def test_stopwords_alone_do_not_match_anything(products: list[Product]) -> None:
    assert FilterSearch(products).search("و در برای از") == []


def test_stopwords_do_not_dilute_the_score_of_a_real_match(products: list[Product]) -> None:
    hits = FilterSearch(products).search("برای تلگرام و")
    assert [(hit.product.id, hit.score) for hit in hits] == [(7569, 1.0)]


def test_arabic_letter_variants_match_their_persian_forms(products: list[Product]) -> None:
    # "اینستاگرام" typed with Arabic yeh (ي) still finds the Persian-spelled product.
    hits = FilterSearch(products).search("اينستاگرام")
    assert 9177 in [hit.product.id for hit in hits]


def test_zwnj_and_space_spellings_match_the_same_products(products: list[Product]) -> None:
    search = FilterSearch(products)
    assert search.search("تلگرام می\u200cخواهم") == search.search("تلگرام میخواهم")
