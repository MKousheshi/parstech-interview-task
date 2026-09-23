"""Gradio web UI, an alternative front end to the CLI REPL in `cli.py`.

Reuses the same `ARCHITECTURES` graph builders and in-memory checkpointer approach as the CLI: one
compiled graph shared by the process, one `thread_id` per browser session (assigned on page load),
so concurrent visitors get independent conversations without needing a real database.

The `Chatbot`/`Textbox` components are configured `rtl=True` since the assistant's responses are in
Persian.
"""

import argparse
import uuid

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from consultant_bot.cli import ARCHITECTURES


def build_demo(arch: str = "flexible") -> gr.Blocks:
    app = ARCHITECTURES[arch](MemorySaver())

    def respond(message: str, history: list[dict[str, str]], thread_id: str) -> str:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        result = app.invoke({"messages": [HumanMessage(content=message)]}, config=config)
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        return str(ai_messages[-1].content) if ai_messages else ""

    with gr.Blocks(title="دستیار مشاوره کسب‌وکار") as demo:
        thread_id = gr.State()
        demo.load(lambda: str(uuid.uuid4()), outputs=thread_id)
        gr.ChatInterface(
            fn=respond,
            additional_inputs=[thread_id],
            chatbot=gr.Chatbot(rtl=True, label="دستیار مشاوره کسب‌وکار"),
            textbox=gr.Textbox(rtl=True, text_align="right", placeholder="پیام خود را بنویسید..."),
            title="دستیار مشاوره کسب‌وکار",
        )

    return demo  # type: ignore[no-any-return]


def main() -> None:
    parser = argparse.ArgumentParser(prog="consultant-bot-web")
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), default="flexible")
    args = parser.parse_args()
    build_demo(args.arch).launch()


if __name__ == "__main__":
    main()
