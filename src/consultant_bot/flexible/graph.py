"""Builds the flexible (agentic) architecture's StateGraph: the real node topology.

`extract_entities -> assistant` (a ReAct-style tool-calling loop, `search_products` as its only
tool) `-> completion_check gate -> (analysis -> suggestion | END)`. See
`docs/ARCHITECTURE_FLEXIBLE.md` for the full per-node behavioral spec.
"""

import logging

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from consultant_bot.common.config import get_settings
from consultant_bot.common.llm import build_chat_model
from consultant_bot.common.search.registry import build_active_strategy
from consultant_bot.flexible.nodes.analysis import build_analysis_node
from consultant_bot.flexible.nodes.assistant import build_assistant_node
from consultant_bot.flexible.nodes.completion_check import completion_check
from consultant_bot.flexible.nodes.extract_entities import (
    build_default_extractor,
    build_extract_entities_node,
)
from consultant_bot.flexible.nodes.suggestion import build_query_formulator, build_suggestion_node
from consultant_bot.flexible.state import State

logger = logging.getLogger(__name__)


def _route_after_assistant(state: State) -> str:
    if completion_check(state):
        logger.info("consultation requested with all 4 entities known: running analysis")
        return "analysis"
    return END


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    top_k = get_settings().search_top_k
    strategy = build_active_strategy()
    # One client shared by every node. Tool binding and structured output each wrap it in a new
    # runnable rather than mutating it, so `analysis` still gets a plain, tool-free model.
    llm = build_chat_model()

    graph = StateGraph(State)
    # mypy can't match a plain `Callable[[State], dict[str, Any]]` against add_node's generic
    # `StateNode[NodeInputT, ...]` overloads, even though it's a perfectly valid node at runtime.
    graph.add_node(  # type: ignore[call-overload]
        "extract_entities", build_extract_entities_node(build_default_extractor(llm))
    )
    graph.add_node(  # type: ignore[call-overload]
        "assistant", build_assistant_node(llm, strategy, top_k=top_k)
    )
    graph.add_node("analysis", build_analysis_node(llm))  # type: ignore[call-overload]
    graph.add_node(  # type: ignore[call-overload]
        "suggestion",
        build_suggestion_node(build_query_formulator(llm), strategy, llm, top_k=top_k),
    )

    graph.add_edge(START, "extract_entities")
    graph.add_edge("extract_entities", "assistant")
    graph.add_conditional_edges("assistant", _route_after_assistant)
    graph.add_edge("analysis", "suggestion")
    graph.add_edge("suggestion", END)

    return graph.compile(checkpointer=checkpointer)
