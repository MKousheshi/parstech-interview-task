"""LangChain tool wrapping the active SearchStrategy.

Each call returns the formatted hits as the `ToolMessage` content (what the LLM reads) and the raw
`ProductHit` list as its artifact. The tool deliberately does *not* write `last_shown_products`
itself: the assistant is told to issue one call per need, and OpenAI runs those as parallel tool
calls in a single step, where several writes to the same state key are rejected outright. The
`assistant` node instead folds every artifact from the turn into one `last_shown_products` update.
"""

from collections.abc import Callable

from langchain_core.tools import BaseTool, tool

from consultant_bot.common import config
from consultant_bot.common.search.base import (
    ProductHit,
    SearchStrategy,
    format_hits,
    search_with_category_fallback,
)
from consultant_bot.common.search.embedding_search import EmbeddingSearch
from consultant_bot.common.search.filter_search import FilterSearch
from consultant_bot.common.search.products import Product, load_products
from consultant_bot.common.search.tfidf_search import TfidfSearch

_STRATEGY_BUILDERS: dict[str, Callable[[list[Product]], SearchStrategy]] = {
    "filter": FilterSearch,
    "tfidf": TfidfSearch,
    "embedding": EmbeddingSearch,
}


def build_active_strategy() -> SearchStrategy:
    """Builds the `SearchStrategy` selected by `config.SEARCH_STRATEGY`, over the full catalog."""
    products = load_products()
    builder = _STRATEGY_BUILDERS[config.SEARCH_STRATEGY]
    return builder(products)


SEARCH_PRODUCTS_TOOL_NAME = "search_products"

NO_HITS_MESSAGE = "هیچ محصول مرتبطی یافت نشد."


def build_search_products_tool(
    strategy: SearchStrategy, top_k: int = config.SEARCH_TOP_K
) -> BaseTool:
    """Builds a `search_products` tool bound to a specific `SearchStrategy` instance."""

    @tool(SEARCH_PRODUCTS_TOOL_NAME, response_format="content_and_artifact")
    def search_products(query: str, category: str | None = None) -> tuple[str, list[ProductHit]]:
        """Search the store's product catalog for products matching a query.

        Use one focused query per distinct need. For a compound request covering several
        different needs, call this tool once per need instead of combining them into one query.

        Args:
            query: A short, focused search query describing what the customer needs.
            category: An optional category name to narrow the search; ignored if no product
                falls under it.
        """
        hits = search_with_category_fallback(strategy, query, category, top_k)
        return format_hits(hits) or NO_HITS_MESSAGE, hits

    return search_products
