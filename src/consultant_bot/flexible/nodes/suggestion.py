"""suggestion node: query formulation, retrieval with a relevance floor, and grounded formatting.

Two internal steps, both deterministic in sequence: (1) a small LLM call turns the 4 entities plus
the analysis text (`state["analysis"]`, written by the `analysis` node) into a short, focused
search query and an optional category guess; (2) `search_products` is called directly against the
active `SearchStrategy` — not through the assistant's tool loop — a strategy-appropriate relevance
threshold is applied, and only hits that clear it are ever shown to the formatting LLM, which is
explicitly told not to invent products and to say so honestly if none did.

`consultation_done` and `last_shown_products` are always set at the end, whether or not any hit
cleared the threshold — the consultation itself is still considered complete either way.
"""

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel, LanguageModelLike
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from consultant_bot.common.entities import Entities
from consultant_bot.common.search.base import (
    SearchStrategy,
    format_hits,
    search_with_category_fallback,
)
from consultant_bot.flexible.state import State

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
بر اساس فهرست محصولات زیر که از کاتالوگ واقعی فروشگاه بازیابی شده‌اند، یک پیشنهاد محصول/بسته \
مناسب برای کاربر بنویس. فقط از محصولات فهرست‌شده استفاده کن و هرگز محصول یا قیمتی که در فهرست \
نیست را اختراع نکن.\
"""


class SearchQuery(BaseModel):
    query: str = Field(description="یک عبارت جست‌وجوی کوتاه و متمرکز")
    category: str | None = Field(default=None, description="نام دسته‌بندی حدسی، در صورت مشخص بودن")


def build_query_formulator(llm: BaseChatModel) -> Runnable[Any, SearchQuery]:
    return llm.with_structured_output(SearchQuery)  # type: ignore[return-value]


def build_suggestion_node(
    query_formulator: Runnable[Any, SearchQuery],
    strategy: SearchStrategy,
    formatting_llm: LanguageModelLike,
    *,
    top_k: int,
    relevance_threshold: float | None = None,
) -> Callable[[State], dict[str, Any]]:
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
            message = formatting_llm.invoke(
                [
                    SystemMessage(content=FORMATTING_SYSTEM_PROMPT),
                    HumanMessage(content=format_hits(hits)),
                ]
            )
        else:
            message = AIMessage(content=NO_RESULTS_MESSAGE)

        return {
            "messages": [message],
            "last_shown_products": hits,
            "consultation_done": True,
        }

    return suggestion
