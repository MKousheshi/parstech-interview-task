from pathlib import Path

from consultant_bot.common.search.products import load_products, strip_html

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "products_fixture.json"


def test_load_products_returns_expected_count() -> None:
    products = load_products(FIXTURE_PATH)
    assert len(products) == 5


def test_html_stripped_from_description_and_short_description() -> None:
    products = load_products(FIXTURE_PATH)
    instagram_product = next(p for p in products if p.id == 9177)
    assert "<" not in instagram_product.description
    assert "<" not in instagram_product.short_description
    assert "مدیریت و افزایش تعامل" in instagram_product.description


def test_categories_are_flattened_to_names() -> None:
    products = load_products(FIXTURE_PATH)
    google_ads_product = next(p for p in products if p.id == 7574)
    assert google_ads_product.categories == ["تبلیغات گوگل (Google Ads)", "مشاوره دیجیتال مارکتینگ"]


def test_irrelevant_woocommerce_fields_are_dropped() -> None:
    products = load_products(FIXTURE_PATH)
    product = products[0]
    assert not hasattr(product, "sku")
    assert not hasattr(product, "_links")
    assert not hasattr(product, "attributes")


def test_strip_html_collapses_whitespace() -> None:
    assert strip_html("<p>hello   <b>world</b></p>\n<p>!</p>") == "hello world !"
