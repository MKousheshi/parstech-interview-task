"""route_intent node: a single structured-output classification of the latest user message.

Every turn starts here. Picks exactly one of `answer` / `search` / `consultation` / `unclear` and
writes it to `intent`; `route_after_intent` — a pure function on the conditional edge — turns that
label into the next node.

`answer` is only offered while an entity question is pending: the prompt then shows the classifier
that question, since a bare "تهران" can't be told apart from a search without it. So a search or an
off-topic message mid-consultation is handled as such instead of being stored as the entity, and the
pending question stays open for a later turn.

The same call also reports which of the 4 entities the message states (`stated_entities`), so "I
run a cafe in Tehran, B2C, selling on Instagram — advise me" needs no follow-up questions.
`capture_entity` merges them, but only on an `answer` or `consultation` turn: an entity mentioned in
a search ("site design for a cafe") isn't taken as the user's own business.
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import Field

from consultant_bot.common.entities import ENTITY_FIELDS, Entities, ReportedEntities
from consultant_bot.common.messages import latest_user_text
from consultant_bot.scripted.nodes.ask_entity import QUESTIONS
from consultant_bot.scripted.state import Intent, Node, State

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
تو دسته‌بند پیام‌های کاربر در چت‌بات یک فروشگاه محصولات و خدمات دیجیتال مارکتینگ هستی. پیام \
کاربر را دقیقاً در یکی از این دسته‌ها قرار بده:

- search: کاربر دنبال یک محصول یا خدمت از فروشگاه است، یا درباره محصولات سؤال می‌کند.
- consultation: کاربر می‌خواهد برای کسب‌وکارش مشاوره بگیرد.
- unclear: هیچ‌کدام از موارد دیگر نیست، یا منظور کاربر مشخص نیست (مثلاً سلام و احوال‌پرسی).\
"""

# Appended only while an entity question is pending, so `answer` is never offered otherwise.
ANSWER_PROMPT = """

آخرین سؤالی که از کاربر پرسیده شده این است: «{question}»

- answer: پیام کاربر پاسخ همین سؤال است، حتی اگر فقط یک کلمه باشد (مثلاً نام یک شهر یا یک \
کانال فروش). اگر پیام پاسخ این سؤال است، answer را انتخاب کن، نه search.\
"""

ENTITIES_PROMPT = """

همچنین هر کدام از ۴ ویژگی زیر را که کاربر در همین پیام صراحتاً درباره کسب‌وکار خودش گفته، کوتاه \
استخراج کن و بقیه را خالی (null) بگذار. چیزی را حدس نزن:

- business_type: نوع کسب‌وکار (مثلاً کافه)
- customer_type: نوع مشتریان، B2B یا B2C
- location: موقعیت جغرافیایی (مثلاً تهران)
- sales_channel: کانال فروش مجازی (مثلاً وب‌سایت یا پیج اینستاگرام)\
"""


class IntentLabel(ReportedEntities):
    intent: Intent = Field(
        description="دسته پیام کاربر: answer (فقط وقتی سؤالی منتظر پاسخ است)، search، "
        "consultation یا unclear"
    )


def build_default_classifier(llm: BaseChatModel) -> Runnable[Any, IntentLabel]:
    return llm.with_structured_output(IntentLabel)  # type: ignore[return-value]


def system_prompt(state: State) -> str:
    """The classifier's instructions, with the `answer` label added while a question is pending."""
    field = state.get("awaiting_field")
    answer = "" if field is None else ANSWER_PROMPT.format(question=QUESTIONS[field])
    return SYSTEM_PROMPT + answer + ENTITIES_PROMPT


def build_route_intent_node(
    classifier: Runnable[Any, IntentLabel],
) -> Node:
    def route_intent(state: State) -> dict[str, Any]:
        text = latest_user_text(state["messages"])
        label = classifier.invoke(
            [SystemMessage(content=system_prompt(state)), HumanMessage(content=text)]
        )
        stated = Entities(**label.model_dump(include=set(ENTITY_FIELDS)))
        # Field names only at INFO: the values are the user's own words.
        named = [field for field in ENTITY_FIELDS if stated.value(field)]
        logger.info("classified intent: %s, stated: %s", label.intent, named)
        return {"intent": label.intent, "stated_entities": stated}

    return route_intent


def route_after_intent(state: State) -> str:
    """The node that handles this turn's classified intent."""
    match state.get("intent"):
        # With nothing pending there's no question to answer, so a stray `answer` falls back.
        case "answer" if state.get("awaiting_field"):
            return "capture_entity"
        case "search":
            return "product_search"
        case "consultation":
            return "idle_reply" if state.get("consultation_done") else "capture_entity"
        case _:
            return "fallback"
