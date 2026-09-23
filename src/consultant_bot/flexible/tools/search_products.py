"""LangChain tool wrapping the active SearchStrategy.

On every call, it also updates `state.last_shown_products` so follow-up turns ("what's the price
of it?") can be answered directly from state instead of re-running search on an under-specified
query.
"""

from collections.abc import Callable
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.types import Command

from consultant_bot.common import config
from consultant_bot.common.search.base import ProductHit, SearchStrategy
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


def _format_hits(hits: list[ProductHit]) -> str:
    if not hits:
        return "هیچ محصول مرتبطی یافت نشد."
    lines = [
        f"- {hit.product.name} ({hit.product.price} تومان): {hit.product.permalink}" for hit in hits
    ]
    return "\n".join(lines)


def build_search_products_tool(
    strategy: SearchStrategy, top_k: int = config.SEARCH_TOP_K
) -> BaseTool:
    """Builds a `search_products` tool bound to a specific `SearchStrategy` instance."""

    @tool
    def search_products(
        query: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        category: str | None = None,
    ) -> Command:
        """Search the store's product catalog for products matching a query.

        Use one focused query per distinct need. For a compound request covering several
        different needs, call this tool once per need instead of combining them into one query.

        Args:
            query: A short, focused search query describing what the customer needs.
            category: An optional exact category name to narrow the search.
        """
        hits = strategy.search(query, category=category, top_k=top_k)
        return Command(
            update={
                "last_shown_products": hits,
                "messages": [ToolMessage(content=_format_hits(hits), tool_call_id=tool_call_id)],
            }
        )

    return search_products
