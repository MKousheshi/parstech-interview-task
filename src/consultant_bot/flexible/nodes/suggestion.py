"""suggestion node: query formulation, retrieval with a relevance floor, and grounded formatting.

Two internal steps, both deterministic in sequence: (1) a small LLM call turns the 4 entities plus
the analysis text (`state["analysis"]`, written by the `analysis` node) into a short, focused
search query and an optional category guess; (2) `search_products` is called directly against the
active `SearchStrategy` — not through the assistant's tool loop — and a strategy-appropriate
relevance threshold is applied. Only hits that clear it are shown to the formatting LLM, together
with the entities and the analysis so the write-up addresses this business, and it's told not to
invent products. If no hit clears the threshold, the formatting call is skipped for a fixed
"nothing found" reply.

The retrieval is also recorded in `messages` as a `search_products` call/result pair, placed just
before the formatted reply and holding exactly the hits that cleared the threshold. The formatting
LLM's prose may leave out a price or link, so without this pair a later follow-up about a suggested
product would have nothing to answer from. With it, the suggestion's products are in the history in
the same shape as the assistant's own searches.

`consultation_done` is always set at the end, whether or not any hit cleared the threshold — the
consultation itself is still considered complete either way.
"""

import logging
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel, LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from consultant_bot.common.entities import Entities
from consultant_bot.common.messages import as_reply
from consultant_bot.common.search.base import (
    ProductHit,
    SearchStrategy,
    format_hits,
    search_with_category_fallback,
)
from consultant_bot.flexible.state import Node, State
from consultant_bot.flexible.tools.search_products import NO_HITS_MESSAGE, SEARCH_PRODUCTS_TOOL_NAME

logger = logging.getLogger(__name__)

NO_RESULTS_MESSAGE = (
    "متأسفانه در حال حاضر محصول یا بسته مرتبطی در فروشگاه برای این نیاز پیدا نکردم."
)

QUERY_FORMULATION_SYSTEM_PROMPT = """\
بر اساس اطلاعات کسب‌وکار کاربر و تحلیل ارائه‌شده، یک عبارت جست‌وجوی کوتاه و متمرکز برای جست‌وجو \
در کاتالوگ محصولات یک فروشگاه دیجیتال مارکتینگ بساز. در صورت مشخص بودن یک دسته‌بندی مرتبط، آن را \
هم به‌عنوان حدس بگذار.\
"""

FORMATTING_SYSTEM_PROMPT = """\
بر اساس اطلاعات کسب‌وکار کاربر، تحلیل ارائه‌شده و فهرست محصولاتی که از کاتالوگ واقعی فروشگاه \
بازیابی شده‌اند، یک پیشنهاد محصول/بسته مناسب برای این کسب‌وکار بنویس. فقط از محصولات فهرست‌شده \
استفاده کن و هرگز محصول یا قیمتی که در فهرست نیست را اختراع نکن.\
"""


class SearchQuery(BaseModel):
    query: str = Field(description="یک عبارت جست‌وجوی کوتاه و متمرکز")
    category: str | None = Field(default=None, description="نام دسته‌بندی حدسی، در صورت مشخص بودن")


def retrieval_record(search_query: SearchQuery, hits: list[ProductHit]) -> list[BaseMessage]:
    """The suggestion's retrieval as a `search_products` call/result pair for the history.

    Shaped exactly like what the assistant's own tool loop leaves behind (same tool name, same
    `format_hits` content, hits as the artifact), so the pair is valid chat-API input on later
    turns and the assistant reads it like any other search.
    """
    call_id = f"call_{uuid4().hex}"
    args: dict[str, Any] = {"query": search_query.query}
    if search_query.category:
        args["category"] = search_query.category
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": SEARCH_PRODUCTS_TOOL_NAME, "args": args, "id": call_id}],
        ),
        ToolMessage(
            content=format_hits(hits) or NO_HITS_MESSAGE,
            tool_call_id=call_id,
            name=SEARCH_PRODUCTS_TOOL_NAME,
            artifact=hits,
        ),
    ]


def build_query_formulator(llm: BaseChatModel) -> Runnable[Any, SearchQuery]:
    return llm.with_structured_output(SearchQuery)  # type: ignore[return-value]


def build_suggestion_node(
    query_formulator: Runnable[Any, SearchQuery],
    strategy: SearchStrategy,
    formatting_llm: LanguageModelLike,
    *,
    top_k: int,
    relevance_threshold: float | None = None,
) -> Node:
    threshold = strategy.relevance_threshold if relevance_threshold is None else relevance_threshold

    def suggestion(state: State) -> dict[str, Any]:
        entities = state.get("entities") or Entities()
        analysis_text = state.get("analysis") or ""

        search_query = query_formulator.invoke(
            [
                SystemMessage(content=QUERY_FORMULATION_SYSTEM_PROMPT),
                HumanMessage(content=f"{entities.summary()}\n\nتحلیل کسب‌وکار:\n{analysis_text}"),
            ]
        )

        raw_hits = search_with_category_fallback(
            strategy, search_query.query, search_query.category, top_k
        )
        hits = [hit for hit in raw_hits if hit.score > threshold]
        logger.info(
            "suggestion query %r (category %r): %d of %d hit(s) cleared threshold %.2f",
            search_query.query,
            search_query.category,
            len(hits),
            len(raw_hits),
            threshold,
        )

        if hits:
            message = as_reply(
                formatting_llm.invoke(
                    [
                        SystemMessage(content=FORMATTING_SYSTEM_PROMPT),
                        HumanMessage(
                            content=f"اطلاعات کسب‌وکار:\n{entities.summary()}\n\n"
                            f"تحلیل کسب‌وکار:\n{analysis_text}\n\n"
                            f"محصولات:\n{format_hits(hits)}"
                        ),
                    ]
                )
            )
        else:
            message = AIMessage(content=NO_RESULTS_MESSAGE)

        return {
            "messages": [*retrieval_record(search_query, hits), message],
            "consultation_done": True,
        }

    return suggestion
