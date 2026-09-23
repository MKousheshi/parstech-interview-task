"""Registry of the chatbot architectures a front end can build, keyed by `--arch` name."""

from collections.abc import Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from consultant_bot.agentic.graph import build_graph as build_agentic_graph
from consultant_bot.scripted.graph import build_graph as build_scripted_graph

GraphBuilder = Callable[[BaseCheckpointSaver | None], CompiledStateGraph]

ARCHITECTURES: dict[str, GraphBuilder] = {
    "agentic": build_agentic_graph,
    "scripted": build_scripted_graph,
}
