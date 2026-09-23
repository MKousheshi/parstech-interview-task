"""Search strategy interface shared by all product search implementations."""

from dataclasses import dataclass
from typing import Protocol

from consultant_bot.common.search.products import Product


@dataclass
class ProductHit:
    product: Product
    score: float


class SearchStrategy(Protocol):
    def search(
        self, query: str, category: str | None = None, top_k: int = 5
    ) -> list[ProductHit]: ...
