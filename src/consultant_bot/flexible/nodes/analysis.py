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

from consultant_bot.common.entities import Entities
from consultant_bot.common.messages import message_text
from consultant_bot.flexible.state import State

SYSTEM_PROMPT = """\
تو یک مشاور کسب‌وکار هستی. صرفاً بر اساس دانش عمومی خودت (بدون دسترسی به هیچ کاتالوگ محصول یا \
داده اختصاصی فروشگاهی)، برای کسب‌وکار زیر یک تحلیل اولیه و توصیه‌های عملی برای رشد آن ارائه بده. \
به هیچ محصول یا خدمت خاصی از هیچ فروشگاهی اشاره نکن — معرفی محصول در مرحله‌ای جداگانه انجام \
می‌شود.\
"""


def build_analysis_node(llm: LanguageModelLike) -> Callable[[State], dict[str, Any]]:
    def analysis(state: State) -> dict[str, Any]:
        entities = state.get("entities") or Entities()
        response = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=entities.summary())]
        )
        text = response if isinstance(response, str) else message_text(response)
        return {"messages": [response], "analysis": text}

    return analysis
