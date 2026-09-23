"""The two fixed-reply nodes: `fallback` for unclassifiable input, `idle_reply` for asking to
start a consultation that has already been completed. Zero LLM calls, and the reply never depends
on what the user actually said — only on whether an entity question is still pending, in which case
`fallback` asks it again instead of offering the two modes.
"""

from typing import Any

from langchain_core.messages import AIMessage

from consultant_bot.scripted.nodes.ask_entity import QUESTIONS
from consultant_bot.scripted.state import State

FALLBACK_MESSAGE = (
    "متوجه منظورتان نشدم. می‌خواهید در میان محصولات فروشگاه جست‌وجو کنید، یا یک مشاوره کوتاه "
    "برای کسب‌وکارتان شروع کنیم؟"
)

PENDING_FALLBACK_MESSAGE = "متوجه منظورتان نشدم. {question}"

IDLE_MESSAGE = (
    "مشاوره کسب‌وکار شما قبلاً انجام شده است. اگر بخواهید، می‌توانم در جست‌وجوی محصولات "
    "فروشگاه کمکتان کنم."
)


def fallback(state: State) -> dict[str, Any]:
    field = state.get("awaiting_field")
    if field is None:
        return {"messages": [AIMessage(content=FALLBACK_MESSAGE)]}
    return {
        "messages": [AIMessage(content=PENDING_FALLBACK_MESSAGE.format(question=QUESTIONS[field]))]
    }


def idle_reply(state: State) -> dict[str, Any]:
    return {"messages": [AIMessage(content=IDLE_MESSAGE)]}
