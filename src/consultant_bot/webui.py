"""Gradio web UI, the chatbot's front end.

Builds the selected graph from `ARCHITECTURES` with an in-memory checkpointer: one compiled graph
shared by the process, one `thread_id` per browser session (assigned on page load), so concurrent
visitors get independent conversations without needing a real database.

The `Chatbot`/`Textbox` components are configured `rtl=True` since the assistant's responses are in
Persian. `WELCOME_MESSAGE` is seeded directly into the `Chatbot`'s initial value rather than run
through the graph, so it displays instantly without an LLM call and doesn't count as a turn against
the graph's own conversation state.
"""

import argparse
import logging
import uuid

import gradio as gr
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
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


def build_demo(arch: str = "flexible") -> gr.Blocks:
    app = ARCHITECTURES[arch](build_checkpointer())

    def respond(message: str, history: list[dict[str, str]], thread_id: str) -> list[str]:
        return user_facing_replies(app, thread_id, message)

    with gr.Blocks(title="دستیار مشاوره کسب‌وکار") as demo:
        thread_id = gr.State()
        demo.load(lambda: str(uuid.uuid4()), outputs=thread_id)
        gr.ChatInterface(
            fn=respond,
            additional_inputs=[thread_id],
            chatbot=gr.Chatbot(
                rtl=True,
                label="دستیار مشاوره کسب‌وکار",
                value=[{"role": "assistant", "content": WELCOME_MESSAGE}],
            ),
            textbox=gr.Textbox(rtl=True, text_align="right", placeholder="پیام خود را بنویسید..."),
            title="دستیار مشاوره کسب‌وکار",
        )

    return demo  # type: ignore[no-any-return]


def main() -> None:
    parser = argparse.ArgumentParser(prog="consultant-bot-web")
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), default="flexible")
    args = parser.parse_args()
    logging.basicConfig(
        level=get_settings().log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    build_demo(args.arch).launch()


if __name__ == "__main__":
    main()
