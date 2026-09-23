"""assistant node: the flexible variant's conversational core.

A tool-calling LLM node (LangChain's `create_agent` ReAct-style loop) with `search_products` as its
only tool; its system prompt is rebuilt from current state before every model call via a
`dynamic_prompt` middleware. See `docs/ARCHITECTURE_FLEXIBLE.md`'s `assistant` section for the full
behavioral spec this implements: answering whatever the user actually said (chat, product query,
follow-up, off-topic), decomposing compound search requests into multiple tool calls, never
pre-empting the dedicated analysis/suggestion pair, and proactively offering a consultation once
entities are complete but unrequested and unoffered.
"""

from collections.abc import Callable, Sequence
from typing import Any, NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage

from consultant_bot.common.entities import Entities
from consultant_bot.common.search.base import ProductHit, SearchStrategy, format_hits
from consultant_bot.flexible.state import State
from consultant_bot.flexible.tools.search_products import (
    SEARCH_PRODUCTS_TOOL_NAME,
    build_search_products_tool,
)

SYSTEM_PROMPT_TEMPLATE = """\
تو دستیار فروشگاه محصولات دیجیتال مارکتینگ هستی و دو کار انجام می‌دهی: جست‌وجوی محصول، و کمک به \
جمع‌آوری اطلاعات برای یک مشاوره کسب‌وکار. طبیعی و مفید صحبت کن؛ به هر چیزی که کاربر واقعاً گفته \
(چه گپ عادی، چه سؤال درباره محصول، چه سؤال پیگیری درباره نتایج قبلی، چه موضوعی نامرتبط) پاسخ بده.

اطلاعات مشاوره جمع‌آوری‌شده تاکنون:
{known_entities}

اطلاعات باقی‌مانده (در صورت مناسب بودن می‌توانی به‌آرامی و بدون فشار بپرسی):
{missing_entities}

وضعیت مشاوره: درخواست‌شده={consultation_requested}، قبلاً پیشنهادشده={consultation_offered}، \
قبلاً انجام‌شده={consultation_done}

آخرین محصولات نمایش‌داده‌شده به کاربر — برای پاسخ به سؤالات پیگیری (مثل قیمت یا لینک) مستقیماً از \
همین فهرست استفاده کن و جست‌وجو را دوباره اجرا نکن:
{last_shown_products}

دستورالعمل‌های مهم:
1. اگر کاربر در یک پیام چند نیاز متفاوت مطرح کرد (مثلاً چند دسته محصول جدا)، به‌جای یک فراخوانی \
   واحد ابزار search_products با کل جمله، برای هر نیاز جداگانه یک‌بار ابزار را فراخوانی کن.
2. اگر هر ۴ ویژگی بالا کامل است، خودت تحلیل کسب‌وکار یا پیشنهاد محصول جدید ارائه نده — این کار \
   به‌طور خودکار توسط گره‌های اختصاصی انجام می‌شود؛ فقط به‌طور طبیعی پاسخ بده و اجازه بده ادامه \
   فرایند در همین نوبت انجام شود.
{proactive_offer_instruction}\
"""

PROACTIVE_OFFER_INSTRUCTION = (
    "3. هر ۴ ویژگی اکنون کامل است و کاربر هنوز مشاوره‌ای درخواست نکرده و پیشنهادی هم داده نشده — "
    "ضمن پاسخ به آنچه کاربر گفته، همین حالا به‌طور طبیعی پیشنهاد بده که یک تحلیل کسب‌وکار و "
    "معرفی محصولات مناسب برایش انجام شود."
)


def is_complete_but_unrequested_and_unoffered(state: State) -> bool:
    """The proactive-offer condition: computed in plain code, never left to the LLM to notice."""
    entities = state.get("entities") or Entities()
    return (
        entities.is_complete()
        and not state.get("consultation_requested", False)
        and not state.get("consultation_done", False)
        and not state.get("consultation_offered", False)
    )


def _build_system_prompt(state: State) -> str:
    entities = state.get("entities") or Entities()
    known = (
        "، ".join(f"{label}: {value}" for label, value in entities.known().items())
        or "(هنوز هیچ‌کدام)"
    )
    missing = "، ".join(entities.missing_labels()) or "(هیچ)"
    # Name, price and link — not just names: the follow-up questions this block exists to serve
    # ("how much is it?", "send me the link") can't be answered from a bare list of names.
    products_text = format_hits(state.get("last_shown_products") or []) or "(هیچ)"

    return SYSTEM_PROMPT_TEMPLATE.format(
        known_entities=known,
        missing_entities=missing,
        consultation_requested=state.get("consultation_requested", False),
        consultation_offered=state.get("consultation_offered", False),
        consultation_done=state.get("consultation_done", False),
        last_shown_products=products_text,
        proactive_offer_instruction=(
            PROACTIVE_OFFER_INSTRUCTION if is_complete_but_unrequested_and_unoffered(state) else ""
        ),
    )


def searched_hits(messages: Sequence[BaseMessage]) -> list[ProductHit] | None:
    """Every product the `search_products` calls among `messages` returned, deduplicated in order.

    `None` when no search ran, so the caller can leave the previous turn's `last_shown_products`
    in place for follow-up questions; a search that found nothing yields `[]`, which does replace
    it.
    """
    searches = [
        message
        for message in messages
        if isinstance(message, ToolMessage) and message.name == SEARCH_PRODUCTS_TOOL_NAME
    ]
    if not searches:
        return None
    hits: list[ProductHit] = []
    seen_ids: set[int] = set()
    for search in searches:
        for hit in search.artifact or []:
            if hit.product.id not in seen_ids:
                seen_ids.add(hit.product.id)
                hits.append(hit)
    return hits


class _AgentLoopState(AgentState[Any]):
    """The agent loop's own state: its messages, plus the parent fields the system prompt reads."""

    entities: NotRequired[Entities]
    consultation_requested: NotRequired[bool]
    consultation_offered: NotRequired[bool]
    consultation_done: NotRequired[bool]
    last_shown_products: NotRequired[list[ProductHit] | None]


_AGENT_LOOP_KEYS = frozenset(_AgentLoopState.__annotations__)


@dynamic_prompt
def _system_prompt(request: ModelRequest) -> str:
    return _build_system_prompt(request.state)  # type: ignore[arg-type]


def build_assistant_node(
    llm: BaseChatModel, search_strategy: SearchStrategy, *, top_k: int
) -> Callable[[State], dict[str, Any]]:
    search_tool = build_search_products_tool(search_strategy, top_k=top_k)
    agent = create_agent(
        model=llm,
        tools=[search_tool],
        middleware=[_system_prompt],
        state_schema=_AgentLoopState,
    )

    def assistant(state: State) -> dict[str, Any]:
        should_offer = is_complete_but_unrequested_and_unoffered(state)
        agent_input = {key: value for key, value in state.items() if key in _AGENT_LOOP_KEYS}
        result = agent.invoke(agent_input)  # type: ignore[call-overload]
        new_messages = result["messages"][len(state["messages"]) :]

        update: dict[str, Any] = {"messages": new_messages}
        hits = searched_hits(new_messages)
        if hits is not None:
            update["last_shown_products"] = hits
        if should_offer:
            update["consultation_offered"] = True
        return update

    return assistant
