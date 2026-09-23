from consultant_bot.common.search.base import ProductHit, format_hits, search_with_category_fallback
from consultant_bot.common.search.products import Product

PRODUCT = Product(
    id=1,
    name="مدیریت پیج اینستاگرام",
    description="",
    short_description="",
    categories=["اینستاگرام"],
    price="2500000",
    permalink="https://example.com/instagram",
)


def test_format_hits_includes_name_price_and_link() -> None:
    line = format_hits([ProductHit(product=PRODUCT, score=1.0)])

    assert "مدیریت پیج اینستاگرام" in line
    assert "2500000" in line
    assert "https://example.com/instagram" in line


def test_format_hits_is_empty_for_no_hits() -> None:
    assert format_hits([]) == ""


def test_format_hits_writes_one_line_per_hit() -> None:
    hits = [ProductHit(product=PRODUCT, score=1.0), ProductHit(product=PRODUCT, score=0.5)]

    assert len(format_hits(hits).splitlines()) == 2


class _CategoryAwareStrategy:
    """Finds `PRODUCT` for any query, but only when the category is absent or "اینستاگرام"."""

    def __init__(self) -> None:
        self.categories_searched: list[str | None] = []

    def search(self, query: str, category: str | None = None, top_k: int = 5) -> list[ProductHit]:
        self.categories_searched.append(category)
        if category in (None, "اینستاگرام"):
            return [ProductHit(product=PRODUCT, score=1.0)]
        return []


def test_category_fallback_keeps_category_results_when_there_are_any() -> None:
    strategy = _CategoryAwareStrategy()

    hits = search_with_category_fallback(strategy, "پیج", "اینستاگرام", top_k=5)

    assert [hit.product.id for hit in hits] == [1]
    assert strategy.categories_searched == ["اینستاگرام"]


def test_category_fallback_retries_without_a_category_that_matches_nothing() -> None:
    strategy = _CategoryAwareStrategy()

    hits = search_with_category_fallback(strategy, "پیج", "شبکه‌های اجتماعی", top_k=5)

    assert [hit.product.id for hit in hits] == [1]
    assert strategy.categories_searched == ["شبکه‌های اجتماعی", None]
