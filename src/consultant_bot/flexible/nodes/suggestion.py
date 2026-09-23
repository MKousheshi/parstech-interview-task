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

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from consultant_bot.common import config
from consultant_bot.common.entities import ENTITY_FIELDS, Entities
from consultant_bot.common.llm import build_chat_model
from consultant_bot.common.search.base import (
    SearchStrategy,
    format_hits,
    search_with_category_fallback,
)
from consultant_bot.flexible.state import State

ENTITY_LABELS: dict[str, str] = {
    "business_type": "نوع کسب‌وکار",
    "customer_type": "نوع مشتریان (B2B یا B2C)",
    "location": "موقعیت جغرافیایی",
    "sales_channel": "کانال فروش مجازی (وب‌سایت یا پیج)",
}

# Strategy-appropriate relevance floors: FilterSearch already discards non-matches internally (any
# hit it returns has score > 0 by construction), so 0.0 is effectively "any real match"; TF-IDF and
# embedding scores are cosine similarities, where low-single-digit-percent scores are noise.
RELEVANCE_THRESHOLDS: dict[str, float] = {"filter": 0.0, "tfidf": 0.1, "embedding": 0.2}

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


def _entities_summary(entities: Entities) -> str:
    return "\n".join(
        f"- {ENTITY_LABELS[field]}: {entities[field]}"  # type: ignore[literal-required]
        for field in ENTITY_FIELDS
    )


def build_query_formulator() -> Runnable[Any, SearchQuery]:
    return build_chat_model().with_structured_output(SearchQuery)  # type: ignore[return-value]


def build_suggestion_node(
    query_formulator: Runnable[Any, SearchQuery],
    strategy: SearchStrategy,
    formatting_llm: LanguageModelLike,
    top_k: int = config.SEARCH_TOP_K,
    relevance_threshold: float | None = None,
) -> Callable[[State], dict[str, Any]]:
    threshold = (
        RELEVANCE_THRESHOLDS[config.SEARCH_STRATEGY]
        if relevance_threshold is None
        else relevance_threshold
    )

    def suggestion(state: State) -> dict[str, Any]:
        entities: Entities = state.get("entities", {})
        analysis_text = state.get("analysis") or ""

        search_query = query_formulator.invoke(
            [
                SystemMessage(content=QUERY_FORMULATION_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"{_entities_summary(entities)}\n\nتحلیل کسب‌وکار:\n{analysis_text}"
                ),
            ]
        )

        raw_hits = search_with_category_fallback(
            strategy, search_query.query, search_query.category, top_k
        )
        hits = [hit for hit in raw_hits if hit.score > threshold]

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
