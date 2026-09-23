"""route_intent node: a single structured-output classification of the latest user message.

Runs only when no entity field is pending. Picks exactly one of `search` / `consultation` /
`unclear` and writes it to `intent`; `route_after_intent` — a pure function on the conditional
edge — turns that label into the next node.
"""

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from consultant_bot.common.messages import latest_user_text
from consultant_bot.rigid.state import Intent, State

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
تو دسته‌بند پیام‌های کاربر در چت‌بات یک فروشگاه محصولات و خدمات دیجیتال مارکتینگ هستی. پیام \
کاربر را دقیقاً در یکی از این سه دسته قرار بده:

- search: کاربر دنبال یک محصول یا خدمت از فروشگاه است، یا درباره محصولات سؤال می‌کند.
- consultation: کاربر می‌خواهد برای کسب‌وکارش مشاوره بگیرد.
- unclear: هیچ‌کدام از دو مورد بالا نیست، یا منظور کاربر مشخص نیست (مثلاً سلام و احوال‌پرسی).\
"""


class IntentLabel(BaseModel):
    intent: Intent = Field(description="دسته پیام کاربر: search، consultation یا unclear")


def build_default_classifier(llm: BaseChatModel) -> Runnable[Any, IntentLabel]:
    return llm.with_structured_output(IntentLabel)  # type: ignore[return-value]


def build_route_intent_node(
    classifier: Runnable[Any, IntentLabel],
) -> Callable[[State], dict[str, Any]]:
    def route_intent(state: State) -> dict[str, Any]:
        text = latest_user_text(state["messages"])
        label = classifier.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=text)]
        )
        logger.info("classified intent: %s", label.intent)
        return {"intent": label.intent}

    return route_intent


def route_after_intent(state: State) -> str:
    """The node that handles this turn's classified intent."""
    match state.get("intent"):
        case "search":
            return "product_search"
        case "consultation":
            return "idle_reply" if state.get("consultation_done") else "ask_entity"
        case _:
            return "fallback"
