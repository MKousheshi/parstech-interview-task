"""Registry of the chatbot architectures a front end can build, keyed by `--arch` name."""

from collections.abc import Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from consultant_bot.flexible.graph import build_graph as build_flexible_graph
from consultant_bot.rigid.graph import build_graph as build_rigid_graph

GraphBuilder = Callable[[BaseCheckpointSaver | None], CompiledStateGraph]

ARCHITECTURES: dict[str, GraphBuilder] = {
    "flexible": build_flexible_graph,
    "rigid": build_rigid_graph,
}
