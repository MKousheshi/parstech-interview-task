"""assistant node: the flexible variant's conversational core.

A tool-calling LLM node (LangGraph's prebuilt ReAct-style loop) with `search_products` as its only
tool. See `docs/ARCHITECTURE_FLEXIBLE.md`'s `assistant` section for the full behavioral spec this
implements: answering whatever the user actually said (chat, product query, follow-up, off-topic),
decomposing compound search requests into multiple tool calls, never pre-empting the dedicated
analysis/suggestion pair, and proactively offering a consultation once entities are complete but
unrequested and unoffered.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage, SystemMessage

# Deprecated in favor of langchain.agents.create_agent, but that replacement only takes a static
# system_prompt (str/SystemMessage) rather than a per-invocation callable — this node needs the
# dynamic prompt to reflect current entity/consultation state each turn, so sticking with this
# (still fully functional, just flagged for removal in LangGraph v2.0) rather than losing that.
from langgraph.prebuilt import create_react_agent

from consultant_bot.common import config
from consultant_bot.common.entities import Entities
from consultant_bot.common.search.base import SearchStrategy, format_hits
from consultant_bot.flexible.state import State
from consultant_bot.flexible.tools.search_products import build_search_products_tool

ENTITY_LABELS: dict[str, str] = {
    "business_type": "نوع کسب‌وکار",
    "customer_type": "نوع مشتریان (B2B یا B2C)",
    "location": "موقعیت جغرافیایی",
    "sales_channel": "کانال فروش مجازی (وب‌سایت یا پیج)",
}

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
    entities: Entities = state.get("entities", {})
    all_present = all(entities.get(field) for field in ENTITY_LABELS)
    return (
        all_present
        and not state.get("consultation_requested", False)
        and not state.get("consultation_done", False)
        and not state.get("consultation_offered", False)
    )


def _build_system_prompt(state: State) -> str:
    entities: Entities = state.get("entities", {})
    known = (
        "، ".join(
            f"{label}: {entities[field]}"  # type: ignore[literal-required]
            for field, label in ENTITY_LABELS.items()
            if entities.get(field)
        )
        or "(هنوز هیچ‌کدام)"
    )
    missing = (
        "، ".join(label for field, label in ENTITY_LABELS.items() if not entities.get(field))
        or "(هیچ)"
    )
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


def _prompt(state: State) -> list[BaseMessage]:
    return [SystemMessage(content=_build_system_prompt(state)), *state["messages"]]


def build_assistant_node(
    llm: LanguageModelLike, search_strategy: SearchStrategy, top_k: int = config.SEARCH_TOP_K
) -> Callable[[State], dict[str, Any]]:
    search_tool = build_search_products_tool(search_strategy, top_k=top_k)
    react_agent = create_react_agent(
        model=llm,
        tools=[search_tool],
        prompt=_prompt,
        state_schema=State,
    )

    def assistant(state: State) -> dict[str, Any]:
        should_offer = is_complete_but_unrequested_and_unoffered(state)
        result: dict[str, Any] = react_agent.invoke(state)
        if should_offer:
            result["consultation_offered"] = True
        return result

    return assistant
