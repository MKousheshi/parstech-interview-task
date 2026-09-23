"""product_search node: the user's raw message as the search query, template-formatted results.

Zero LLM calls: no query rewriting, no decomposition of compound requests, no category guess.
`last_search_results` is written for display only — nothing downstream reads it back to answer a
follow-up question.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage

from consultant_bot.common.messages import latest_user_text
from consultant_bot.common.search.base import SearchStrategy, format_hits
from consultant_bot.rigid.state import State

RESULTS_HEADER = "نتایج جست‌وجو برای «{query}»:"

NO_RESULTS_MESSAGE = "محصولی مطابق با «{query}» در فروشگاه پیدا نشد."


def build_product_search_node(
    strategy: SearchStrategy, *, top_k: int
) -> Callable[[State], dict[str, Any]]:
    def product_search(state: State) -> dict[str, Any]:
        query = latest_user_text(state["messages"])
        hits = strategy.search(query, top_k=top_k)
        if hits:
            text = f"{RESULTS_HEADER.format(query=query)}\n{format_hits(hits)}"
        else:
            text = NO_RESULTS_MESSAGE.format(query=query)
        return {"messages": [AIMessage(content=text)], "last_search_results": hits}

    return product_search
