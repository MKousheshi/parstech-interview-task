"""Loads and normalizes the WooCommerce product export into plain `Product` records."""

import json
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from consultant_bot.common.config import get_settings


@dataclass(frozen=True)
class Product:
    id: int
    name: str
    description: str
    short_description: str
    categories: list[str]
    price: str
    permalink: str


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML-to-text extractor: keeps element text, drops all markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        self._chunks.append(data)

    def get_text(self) -> str:
        return "".join(self._chunks)


def strip_html(html: str) -> str:
    """Strips HTML tags, collapsing whitespace left behind by removed markup."""
    parser = _HTMLTextExtractor()
    parser.feed(html)
    parser.close()
    return " ".join(parser.get_text().split())


def _normalize_product(raw: dict) -> Product:
    return Product(
        id=raw["id"],
        name=raw["name"],
        description=strip_html(raw.get("description", "")),
        short_description=strip_html(raw.get("short_description", "")),
        categories=[category["name"] for category in raw.get("categories", [])],
        price=raw.get("price", ""),
        permalink=raw.get("permalink", ""),
    )


def load_products(path: Path | str | None = None) -> list[Product]:
    """Loads a WooCommerce product export and normalizes each record into a `Product`.

    Defaults to the configured catalog (`CONSULTANT_BOT_PRODUCTS_PATH`, else the project's
    `products.json`).
    """
    with Path(path or get_settings().products_path).open(encoding="utf-8") as f:
        raw_products = json.load(f)
    return [_normalize_product(raw) for raw in raw_products]
