"""extract_entities node: unconditional structured-output LLM call, merged into state each turn.

Runs before the assistant replies so it always sees up-to-date entity state, including anything
just corrected. See `docs/ARCHITECTURE_FLEXIBLE.md`'s `extract_entities` section for the merge
rules this implements.

Given older messages, the extractor returns answers the user gave turns ago, often reworded
("کافی‌شاپ" for a stored "کافه"; seen with gpt-4o-mini even when told not to). Any change clears
`consultation_done`, and `consultation_requested` stays set, so that rewording would re-run the
whole analysis and suggestion on a turn that had nothing to do with them. Three guards stop it:
the extractor only sees the latest user message and the assistant reply before it, the prompt
shows the stored values and asks only for what that message states or corrects, and values that
differ only in spacing or Arabic/Persian letter variants don't count as a change.
"""

import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import Field

from consultant_bot.common.entities import ENTITY_FIELDS, Entities
from consultant_bot.common.search.text import normalize
from consultant_bot.flexible.state import Node, State

logger = logging.getLogger(__name__)

# The latest user message plus the assistant reply just before it: enough to resolve a short answer
# to the assistant's question ("شیراز") or a "yes" to its consultation offer. Older messages are
# deliberately left out, since the extractor would re-read answers stored turns ago.
RECENT_MESSAGES_WINDOW = 2

SYSTEM_PROMPT = """\
از آخرین پیام کاربر، ۴ ویژگی زیر را در صورتی که کاربر صراحتاً و با اطمینان بیان یا اصلاح کرده \
باشد استخراج کن. پیام‌های قبلی فقط برای فهم ارجاع‌ها هستند (مثلاً پاسخ کوتاه «تهران» به سؤال دستیار \
درباره شهر)؛ مقداری را که فقط در پیام‌های قبلی گفته شده دوباره برنگردان. اگر پاسخ کاربر مبهم یا \
نامشخص بود (مثلاً «نمی‌دانم» یا «مهم نیست»)، آن فیلد را خالی (null) بگذار به‌جای ثبت یک مقدار \
نامطمئن:

- business_type: نوع کسب‌وکار
- customer_type: نوع مشتریان، B2B یا B2C
- location: موقعیت جغرافیایی
- sales_channel: کانال فروش مجازی (وب‌سایت یا پیج)

مقادیر ثبت‌شده تاکنون (فقط اگر آخرین پیام کاربر یکی از آن‌ها را تغییر داده، مقدار جدید را برگردان):
{known_entities}

همچنین wants_consultation را true کن اگر کاربر صراحتاً درخواست مشاوره کسب‌وکار کرده، یا در پاسخ \
به پیشنهاد قبلی دستیار برای انجام مشاوره، پاسخ مثبت داده باشد.\
"""


class ExtractedEntities(Entities):
    wants_consultation: bool = Field(default=False)


def build_default_extractor(llm: BaseChatModel) -> Runnable[Any, ExtractedEntities]:
    return llm.with_structured_output(ExtractedEntities)  # type: ignore[return-value]


def recent_context(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """The last `RECENT_MESSAGES_WINDOW` conversational messages, tool traffic excluded.

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


def _comparable(value: str | None) -> str:
    """`value` with spacing and letter variants unified, for deciding whether it really changed."""
    return " ".join(normalize(value or "").split())


def build_extract_entities_node(
    extractor: Runnable[Any, ExtractedEntities],
) -> Node:
    def extract_entities(state: State) -> dict[str, Any]:
        entities = state.get("entities") or Entities()
        prompt = SYSTEM_PROMPT.format(known_entities=entities.summary() or "(هنوز هیچ‌کدام)")
        recent_messages = recent_context(state["messages"])
        extracted = extractor.invoke([SystemMessage(content=prompt), *recent_messages])

        # A field the extractor left empty means "not mentioned (clearly) this turn", never
        # "clear it": only non-empty values that differ from what's stored are merged in.
        changes = {
            field: value
            for field in ENTITY_FIELDS
            if (value := extracted.value(field))
            and _comparable(value) != _comparable(entities.value(field))
        }
        changed = bool(changes)
        if changed:
            logger.info("entities updated: %s", changes)

        update: dict[str, Any] = {"entities": entities.model_copy(update=changes)}
        if extracted.wants_consultation:
            update["consultation_requested"] = True
        if changed and state.get("consultation_done"):
            update["consultation_done"] = False
        return update

    return extract_entities
