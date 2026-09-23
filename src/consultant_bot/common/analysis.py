"""The free-knowledge business analysis prompt, shared by both architectures' `analysis` nodes.

The prompt is built from a synthetic summary of the 4 entities only — never from the conversation
history — so neither prior search results nor any other product data can reach the call. That
isolation is a documented hard requirement (see `docs/DECISIONS.md`), so both variants build the
call's input here rather than each keeping their own copy.
"""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from consultant_bot.common.entities import Entities

ANALYSIS_SYSTEM_PROMPT = """\
تو یک مشاور کسب‌وکار هستی. صرفاً بر اساس دانش عمومی خودت (بدون دسترسی به هیچ کاتالوگ محصول یا \
داده اختصاصی فروشگاهی)، برای کسب‌وکار زیر یک تحلیل اولیه و توصیه‌های عملی برای رشد آن ارائه بده. \
به هیچ محصول یا خدمت خاصی از هیچ فروشگاهی اشاره نکن — معرفی محصول در مرحله‌ای جداگانه انجام \
می‌شود.\
"""


def analysis_messages(entities: Entities) -> list[BaseMessage]:
    """The complete input to an analysis call: the system prompt plus the entity summary."""
    return [SystemMessage(content=ANALYSIS_SYSTEM_PROMPT), HumanMessage(content=entities.summary())]
