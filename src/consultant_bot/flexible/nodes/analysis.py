"""analysis node: free-knowledge business analysis, isolated from product data and the search tool.

A fresh LLM call built from a synthetic summary of the 4 entities only — deliberately *not* a
continuation of `state["messages"]` and *not* the same tool-bound model instance as `assistant`,
so neither prior search results in the conversation history nor the search tool itself can leak
into this call. This isolation is structural (the node only ever receives a plain, non-tool-bound
`llm` and builds its own 2-message list), not just a prompting convention, since it's a documented
hard requirement — see `docs/DECISIONS.md`.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import HumanMessage, SystemMessage

from consultant_bot.common.entities import ENTITY_FIELDS, Entities
from consultant_bot.flexible.state import State

ENTITY_LABELS: dict[str, str] = {
    "business_type": "نوع کسب‌وکار",
    "customer_type": "نوع مشتریان (B2B یا B2C)",
    "location": "موقعیت جغرافیایی",
    "sales_channel": "کانال فروش مجازی (وب‌سایت یا پیج)",
}

SYSTEM_PROMPT = """\
تو یک مشاور کسب‌وکار هستی. صرفاً بر اساس دانش عمومی خودت (بدون دسترسی به هیچ کاتالوگ محصول یا \
داده اختصاصی فروشگاهی)، برای کسب‌وکار زیر یک تحلیل اولیه و توصیه‌های عملی برای رشد آن ارائه بده. \
به هیچ محصول یا خدمت خاصی از هیچ فروشگاهی اشاره نکن — معرفی محصول در مرحله‌ای جداگانه انجام \
می‌شود.\
"""


def _entities_summary(entities: Entities) -> str:
    return "\n".join(
        f"- {ENTITY_LABELS[field]}: {entities[field]}"  # type: ignore[literal-required]
        for field in ENTITY_FIELDS
    )


def build_analysis_node(llm: LanguageModelLike) -> Callable[[State], dict[str, Any]]:
    def analysis(state: State) -> dict[str, Any]:
        entities: Entities = state.get("entities", {})
        summary = _entities_summary(entities)
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=summary)])
        return {"messages": [response]}

    return analysis
