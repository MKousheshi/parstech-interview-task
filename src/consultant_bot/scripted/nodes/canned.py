"""The two fixed-reply nodes: `fallback` for unclassifiable input, `idle_reply` for asking to
start a consultation that has already been completed. Zero LLM calls, and the reply never depends
on what the user actually said.
"""

from typing import Any

from langchain_core.messages import AIMessage

from consultant_bot.scripted.state import State

FALLBACK_MESSAGE = (
    "متوجه منظورتان نشدم. می‌خواهید در میان محصولات فروشگاه جست‌وجو کنید، یا یک مشاوره کوتاه "
    "برای کسب‌وکارتان شروع کنیم؟"
)

IDLE_MESSAGE = (
    "مشاوره کسب‌وکار شما قبلاً انجام شده است. اگر بخواهید، می‌توانم در جست‌وجوی محصولات "
    "فروشگاه کمکتان کنم."
)


def fallback(state: State) -> dict[str, Any]:
    return {"messages": [AIMessage(content=FALLBACK_MESSAGE)]}


def idle_reply(state: State) -> dict[str, Any]:
    return {"messages": [AIMessage(content=IDLE_MESSAGE)]}
