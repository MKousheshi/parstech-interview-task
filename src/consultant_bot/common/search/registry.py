"""Builds a search strategy by name, shared by both architectures.

`EmbeddingSearch` is imported only when asked for: importing it pulls in `sentence-transformers`
and `torch`, which would otherwise slow every startup even with the default filter strategy.
"""

from consultant_bot.common.config import SearchStrategyName, get_settings
from consultant_bot.common.search.base import SearchStrategy
from consultant_bot.common.search.filter_search import FilterSearch
from consultant_bot.common.search.products import Product, load_products
from consultant_bot.common.search.tfidf_search import TfidfSearch


def build_strategy(name: SearchStrategyName, products: list[Product]) -> SearchStrategy:
    if name == "filter":
        return FilterSearch(products)
    if name == "tfidf":
        return TfidfSearch(products)
    from consultant_bot.common.search.embedding_search import EmbeddingSearch

    return EmbeddingSearch(products)


def build_active_strategy() -> SearchStrategy:
    """Builds the strategy named by `CONSULTANT_BOT_SEARCH_STRATEGY` over the configured catalog."""
    return build_strategy(get_settings().search_strategy, load_products())
