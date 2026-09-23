"""extract_entities node: unconditional structured-output LLM call, merged into state each turn.

Runs before the assistant replies so it always sees up-to-date entity state, including anything
just corrected. See `docs/ARCHITECTURE_FLEXIBLE.md`'s `extract_entities` section for the merge
rules this implements.
"""

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from consultant_bot.common.entities import ENTITY_FIELDS, Entities
from consultant_bot.common.llm import build_chat_model
from consultant_bot.flexible.state import State

# How many recent messages to give the extractor, for pronoun/reference resolution and for
# detecting an affirmative reply to a consultation offer made a turn or two ago.
RECENT_MESSAGES_WINDOW = 6

SYSTEM_PROMPT = """\
از پیام‌های اخیر مکالمه، ۴ ویژگی زیر را در صورتی که کاربر صراحتاً و با اطمینان بیان کرده باشد \
استخراج کن. اگر پاسخ کاربر مبهم یا نامشخص بود (مثلاً «نمی‌دانم» یا «مهم نیست»)، آن فیلد را خالی \
(null) بگذار به‌جای ثبت یک مقدار نامطمئن:

- business_type: نوع کسب‌وکار
- customer_type: نوع مشتریان، B2B یا B2C
- location: موقعیت جغرافیایی
- sales_channel: کانال فروش مجازی (وب‌سایت یا پیج)

همچنین wants_consultation را true کن اگر کاربر صراحتاً درخواست مشاوره کسب‌وکار کرده، یا در پاسخ \
به پیشنهاد قبلی دستیار برای انجام مشاوره، پاسخ مثبت داده باشد.\
"""


class ExtractedEntities(BaseModel):
    business_type: str | None = Field(default=None)
    customer_type: str | None = Field(default=None)
    location: str | None = Field(default=None)
    sales_channel: str | None = Field(default=None)
    wants_consultation: bool = Field(default=False)


def build_default_extractor() -> Runnable[Any, ExtractedEntities]:
    return build_chat_model().with_structured_output(ExtractedEntities)  # type: ignore[return-value]


def recent_context(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """The last `RECENT_MESSAGES_WINDOW` conversational messages — tool traffic excluded.

    The assistant's ReAct loop interleaves `AIMessage(tool_calls=...)`/`ToolMessage` pairs into the
    history, and a fixed-size window over the raw list can begin on a `ToolMessage` whose
    originating tool call fell outside it — which OpenAI rejects outright ("a message with role
    'tool' must be a response to a preceding message with tool_calls"). Tool traffic carries no
    entity information anyway, so it's dropped *before* the window is taken, which also means the
    window always spends its budget on actual user/assistant turns.
    """
    conversational: list[BaseMessage] = [
        message
        for message in messages
        if isinstance(message, HumanMessage)
        or (isinstance(message, AIMessage) and not message.tool_calls)
    ]
    return conversational[-RECENT_MESSAGES_WINDOW:]


def build_extract_entities_node(
    extractor: Runnable[Any, ExtractedEntities],
) -> Callable[[State], dict[str, Any]]:
    def extract_entities(state: State) -> dict[str, Any]:
        recent_messages = recent_context(state["messages"])
        extracted = extractor.invoke([SystemMessage(content=SYSTEM_PROMPT), *recent_messages])

        entities: Entities = dict(state.get("entities", {}))  # type: ignore[assignment]
        changed = False
        for entity_field in ENTITY_FIELDS:
            value = getattr(extracted, entity_field)
            if value and entities.get(entity_field) != value:
                entities[entity_field] = value  # type: ignore[literal-required]
                changed = True

        update: dict[str, Any] = {"entities": entities}
        if extracted.wants_consultation:
            update["consultation_requested"] = True
        if changed and state.get("consultation_done"):
            update["consultation_done"] = False
        return update

    return extract_entities
