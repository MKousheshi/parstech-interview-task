from consultant_bot.common.search.base import ProductHit, format_hits
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
