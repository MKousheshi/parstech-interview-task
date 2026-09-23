"""REPL entry point shared by both architectures.

Reads a line from stdin, invokes the selected architecture's graph, prints the newly appended AI
message(s), and repeats until EOF or the user types "exit". Conversation state persists only for
the lifetime of the process (one `thread_id`, one in-memory checkpointer).
"""

import argparse
import sys
import uuid
from collections.abc import Callable

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from consultant_bot.common.messages import reply_texts
from consultant_bot.flexible.graph import build_graph as build_flexible_graph

GraphBuilder = Callable[[BaseCheckpointSaver | None], CompiledStateGraph]

ARCHITECTURES: dict[str, GraphBuilder] = {
    "flexible": build_flexible_graph,
}


def run(arch: str) -> None:
    app = ARCHITECTURES[arch](MemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}
    previous_count = 0

    for line in sys.stdin:
        text = line.strip()
        if not text or text.lower() == "exit":
            break

        result = app.invoke({"messages": [HumanMessage(content=text)]}, config=config)
        new_messages = result["messages"][previous_count:]
        previous_count = len(result["messages"])
        for reply in reply_texts(new_messages):
            print(reply)


def main() -> None:
    parser = argparse.ArgumentParser(prog="consultant-bot")
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), default="flexible")
    args = parser.parse_args()
    run(args.arch)


if __name__ == "__main__":
    main()
