"""suggestion node: deterministic search query, no relevance floor, grounded LLM formatting.

The query is a fixed template over the entities (`build_query`), not an LLM call — cheaper and
predictable, but weaker than the flexible variant's LLM-formulated query when the entities don't
use the catalog's vocabulary. Whatever the strategy's top-`k` returns is written up, with no
relevance threshold (the flexible variant's fix, deliberately left out here); only a completely
empty result skips the formatting call for a fixed message.

`consultation_done` and `last_search_results` are set either way.
"""

import logging
from typing import Any

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from consultant_bot.common.entities import Entities
from consultant_bot.common.messages import as_reply
from consultant_bot.common.search.base import SearchStrategy, format_hits
from consultant_bot.rigid.state import Node, State

logger = logging.getLogger(__name__)

NO_RESULTS_MESSAGE = "متأسفانه محصول یا بسته‌ای در فروشگاه برای این کسب‌وکار پیدا نکردم."

FORMATTING_SYSTEM_PROMPT = """\
بر اساس اطلاعات کسب‌وکار کاربر و فهرست محصولاتی که از کاتالوگ واقعی فروشگاه بازیابی شده‌اند، یک \
پیشنهاد محصول/بسته مناسب برای این کسب‌وکار بنویس. فقط از محصولات فهرست‌شده استفاده کن و هرگز \
محصول یا قیمتی که در فهرست نیست را اختراع نکن.\
"""


def build_query(entities: Entities) -> str:
    """The search query: business type and sales channel, joined — no LLM involved."""
    return " ".join(value for value in (entities.business_type, entities.sales_channel) if value)


def build_suggestion_node(
    strategy: SearchStrategy, formatting_llm: LanguageModelLike, *, top_k: int
) -> Node:
    def suggestion(state: State) -> dict[str, Any]:
        entities = state.get("entities") or Entities()
        query = build_query(entities)
        hits = strategy.search(query, top_k=top_k)
        logger.info("suggestion query %r: %d hit(s)", query, len(hits))

        if hits:
            message = as_reply(
                formatting_llm.invoke(
                    [
                        SystemMessage(content=FORMATTING_SYSTEM_PROMPT),
                        HumanMessage(
                            content=f"اطلاعات کسب‌وکار:\n{entities.summary()}\n\n"
                            f"محصولات:\n{format_hits(hits)}"
                        ),
                    ]
                )
            )
        else:
            message = AIMessage(content=NO_RESULTS_MESSAGE)

        return {"messages": [message], "last_search_results": hits, "consultation_done": True}

    return suggestion
