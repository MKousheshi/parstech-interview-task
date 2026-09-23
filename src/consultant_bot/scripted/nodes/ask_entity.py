"""ask_entity node: the fixed question for the next unset entity, in the spec's literal order.

Zero LLM calls. Sets `awaiting_field`, so `route_intent` can recognize the user's next message
as the answer for `capture_entity` to record. Fields the user already stated are skipped, and
asking again while a question is still open (a consultation request mid-consultation) repeats it.
"""

from typing import Any

from langchain_core.messages import AIMessage

from consultant_bot.common.entities import Entities
from consultant_bot.scripted.state import EntityField, State

# One fixed question per field, in `ENTITY_FIELDS` order (which this dict's order also defines).
QUESTIONS: dict[EntityField, str] = {
    "business_type": (
        "کسب‌وکار شما در چه زمینه‌ای فعالیت می‌کند؟ (مثلاً کافه، فروشگاه پوشاک یا آموزشگاه)"
    ),
    "customer_type": ("مشتریان شما بیشتر کسب‌وکارها هستند (B2B) یا مصرف‌کنندگان نهایی (B2C)؟"),
    "location": "کسب‌وکار شما در کدام شهر یا منطقه فعالیت می‌کند؟",
    "sales_channel": (
        "فروش مجازی شما بیشتر از چه کانالی انجام می‌شود؟ (مثلاً وب‌سایت یا پیج اینستاگرام)"
    ),
}


PENDING_REMINDER = "در ضمن، هنوز منتظر پاسخ این سؤال هستم: {question}"


def with_pending_reminder(text: str, state: State) -> str:
    """`text`, followed by the still-unanswered entity question if there is one, so a detour
    (a search mid-consultation) leads back to where the consultation left off."""
    field = state.get("awaiting_field")
    if field is None:
        return text
    return f"{text}\n\n{PENDING_REMINDER.format(question=QUESTIONS[field])}"


def next_missing_field(entities: Entities) -> EntityField | None:
    """The first unset field in the canonical order, or `None` once all 4 are known."""
    for field in QUESTIONS:
        if not entities.value(field):
            return field
    return None


def ask_entity(state: State) -> dict[str, Any]:
    field = next_missing_field(state.get("entities") or Entities())
    if field is None:
        # The graph only routes here while a field is still missing.
        raise ValueError("ask_entity reached with all 4 entities already known")
    return {"messages": [AIMessage(content=QUESTIONS[field])], "awaiting_field": field}
