"""Gradio web UI, the chatbot's front end.

Builds the selected graph from `ARCHITECTURES` with an in-memory checkpointer: one compiled graph
shared by the process, one `thread_id` per browser session (assigned on page load), so concurrent
visitors get independent conversations without needing a real database.

The `Chatbot`/`Textbox` components are configured `rtl=True` since the assistant's responses are in
Persian. `WELCOME_MESSAGE` is seeded directly into the `Chatbot`'s initial value rather than run
through the graph, so it displays instantly without an LLM call and doesn't count as a turn against
the graph's own conversation state.

The page is a plain `Chatbot` + `Textbox` rather than `gr.ChatInterface`. `ChatInterface` adds undo
and retry buttons that only edit what the browser shows: the graph's checkpointed conversation
would keep the undone message, and a retry would send the same message into it a second time. The
one history control kept is the chatbot's clear button, and clearing starts a new `thread_id`, so
the bot forgets exactly what the user just watched disappear.

`MemorySaver` keeps every checkpoint of every thread until the process exits, so a thread is
deleted from it once nothing can reach it any more: when the chat is cleared, and when Gradio
drops the browser session.
"""

import argparse
import logging
import uuid

import gradio as gr
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from consultant_bot.architectures import ARCHITECTURES
from consultant_bot.common.checkpoint import build_checkpointer
from consultant_bot.common.config import get_settings
from consultant_bot.common.messages import reply_texts

WELCOME_MESSAGE = (
    "سلام! من دستیار فروشگاه محصولات دیجیتال مارکتینگ هستم. می‌تونم توی پیدا کردن محصول "
    "مناسب کمکت کنم، و اگر بخوای، برای کسب‌وکارت یک مشاوره کوتاه هم انجام بدم. چطور می‌تونم "
    "کمکت کنم؟"
)

NO_REPLY_MESSAGE = "متأسفانه پاسخی تولید نشد. لطفاً دوباره تلاش کنید."

ERROR_MESSAGE = "متأسفانه در پردازش پیام خطایی رخ داد. لطفاً چند لحظه بعد دوباره تلاش کنید."

logger = logging.getLogger(__name__)


def turn_replies(app: CompiledStateGraph, thread_id: str, message: str) -> list[str]:
    """Runs one turn and returns every reply it appended, in order.

    A turn can produce more than one: when the consultation fires, the assistant's own reply is
    followed by the analysis and then the suggestion. Gradio renders a returned list as separate
    message bubbles, so all of them reach the user — showing only the last would silently drop the
    business analysis, which is a required output in its own right.

    The prior message count comes from the checkpointer rather than being tracked here, so this
    stays correct across concurrent browser sessions sharing one process.
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    previous_count = len(app.get_state(config).values.get("messages", []))
    result = app.invoke({"messages": [HumanMessage(content=message)]}, config=config)
    return reply_texts(result["messages"][previous_count:])


def user_facing_replies(app: CompiledStateGraph, thread_id: str, message: str) -> list[str]:
    """`turn_replies`, but never empty and never raising: what the chat window should show."""
    try:
        replies = turn_replies(app, thread_id, message)
    except Exception:
        # An LLM/API failure (timeout, rate limit, bad key) shouldn't surface as Gradio's generic
        # error toast; log the details and tell the user in their own language.
        logger.exception("turn failed for thread %s", thread_id)
        return [ERROR_MESSAGE]
    # A turn that somehow appended nothing would otherwise render as no reply at all, which reads
    # as a hung UI; say so instead.
    return replies or [NO_REPLY_MESSAGE]


ChatHistory = list[dict[str, str]]


def new_thread_id() -> str:
    return str(uuid.uuid4())


def welcome_history() -> ChatHistory:
    return [{"role": "assistant", "content": WELCOME_MESSAGE}]


def add_user_message(message: str, history: ChatHistory) -> tuple[ChatHistory, str]:
    """Shows the user's message right away and empties the textbox; blank input is ignored."""
    if not message.strip():
        return history, ""
    return [*history, {"role": "user", "content": message}], ""


def add_replies(app: CompiledStateGraph, history: ChatHistory, thread_id: str) -> ChatHistory:
    """Runs the turn for the user message `add_user_message` just showed and appends its replies."""
    if not history or history[-1]["role"] != "user":
        return history
    replies = user_facing_replies(app, thread_id, history[-1]["content"])
    return [*history, *({"role": "assistant", "content": reply} for reply in replies)]


def start_over(checkpointer: BaseCheckpointSaver, old_thread_id: str) -> tuple[ChatHistory, str]:
    """What clearing the chat does: forget the old conversation and start a new one."""
    checkpointer.delete_thread(old_thread_id)
    return welcome_history(), new_thread_id()


def build_demo(arch: str = "flexible") -> gr.Blocks:
    checkpointer = build_checkpointer()
    app = ARCHITECTURES[arch](checkpointer)

    with gr.Blocks(title="دستیار مشاوره کسب‌وکار") as demo:
        gr.Markdown("# دستیار مشاوره کسب‌وکار", rtl=True)
        # A callable value is called on every page load, so each browser session gets its own.
        # Gradio calls `delete_callback` when the session goes away (the tab is closed), which
        # frees that conversation's checkpoints instead of keeping them until the process exits.
        thread_id = gr.State(new_thread_id, delete_callback=checkpointer.delete_thread)
        chatbot = gr.Chatbot(rtl=True, label="دستیار مشاوره کسب‌وکار", value=welcome_history)
        textbox = gr.Textbox(
            rtl=True, text_align="right", placeholder="پیام خود را بنویسید...", show_label=False
        )

        textbox.submit(add_user_message, [textbox, chatbot], [chatbot, textbox], queue=False).then(
            lambda history, tid: add_replies(app, history, tid), [chatbot, thread_id], chatbot
        )
        chatbot.clear(
            lambda tid: start_over(checkpointer, tid),
            inputs=thread_id,
            outputs=[chatbot, thread_id],
        )

    return demo  # type: ignore[no-any-return]


def main() -> None:
    parser = argparse.ArgumentParser(prog="consultant-bot-web")
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), default="flexible")
    args = parser.parse_args()
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    build_demo(args.arch).launch()


if __name__ == "__main__":
    main()
