"""LangChain tool wrapping the active SearchStrategy.

Each call returns the formatted hits as the `ToolMessage` content (what the LLM reads) and the raw
`ProductHit` list as its artifact. Only hits above the strategy's relevance threshold are returned,
the same floor the `suggestion` node applies, so an off-catalog question gets "nothing found"
rather than the closest unrelated products. The tool writes no state of its own: its `ToolMessage`
stays in the conversation history, which is what later follow-up questions ("how much was the
second one?") are answered from.
"""

import logging

from langchain_core.tools import BaseTool, tool

from consultant_bot.common.search.base import (
    ProductHit,
    SearchStrategy,
    format_hits,
    search_relevant,
)

logger = logging.getLogger(__name__)

SEARCH_PRODUCTS_TOOL_NAME = "search_products"

NO_HITS_MESSAGE = "هیچ محصول مرتبطی یافت نشد."


def build_search_products_tool(strategy: SearchStrategy, *, top_k: int) -> BaseTool:
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
        hits = search_relevant(strategy, query, category, top_k)
        # The query is derived from what the user typed, so it stays out of INFO logs.
        logger.info("search_products: %d hit(s)", len(hits))
        logger.debug("search_products(%r, category=%r)", query, category)
        return format_hits(hits) or NO_HITS_MESSAGE, hits

    return search_products
