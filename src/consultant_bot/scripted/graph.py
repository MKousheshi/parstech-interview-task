"""Builds the scripted (flow-based) architecture's StateGraph.

`START -> route_intent`, which routes every turn: `search -> product_search`; `unclear ->
fallback`; `consultation` once done `-> idle_reply`; otherwise `answer` and `consultation ->
capture_entity -> (all 4 entities known? analysis -> suggestion : ask_entity)`. Every other node
ends the turn. See `docs/ARCHITECTURE_SCRIPTED.md` for the full per-node behavioral spec.
"""

import logging
from typing import Any

from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from consultant_bot.common.config import get_settings
from consultant_bot.common.entities import Entities
from consultant_bot.common.llm import build_chat_model
from consultant_bot.common.search.base import SearchStrategy
from consultant_bot.common.search.registry import build_active_strategy
from consultant_bot.scripted.nodes.analysis import build_analysis_node
from consultant_bot.scripted.nodes.ask_entity import ask_entity
from consultant_bot.scripted.nodes.canned import fallback, idle_reply
from consultant_bot.scripted.nodes.capture_entity import capture_entity
from consultant_bot.scripted.nodes.product_search import build_product_search_node
from consultant_bot.scripted.nodes.route_intent import (
    IntentLabel,
    build_default_classifier,
    build_route_intent_node,
    route_after_intent,
)
from consultant_bot.scripted.nodes.suggestion import build_suggestion_node
from consultant_bot.scripted.state import State

logger = logging.getLogger(__name__)


def _route_after_capture(state: State) -> str:
    if (state.get("entities") or Entities()).is_complete():
        logger.info("all 4 entities known: running analysis")
        return "analysis"
    return "ask_entity"


def assemble_graph(
    classifier: Runnable[Any, IntentLabel],
    llm: LanguageModelLike,
    strategy: SearchStrategy,
    *,
    top_k: int,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """The graph over injected LLM pieces and strategy, so tests can run it on fakes."""
    graph = StateGraph(State)
    graph.add_node("route_intent", build_route_intent_node(classifier))
    graph.add_node("capture_entity", capture_entity)
    graph.add_node("ask_entity", ask_entity)
    graph.add_node("product_search", build_product_search_node(strategy, top_k=top_k))
    graph.add_node("fallback", fallback)
    graph.add_node("idle_reply", idle_reply)
    graph.add_node("analysis", build_analysis_node(llm))
    graph.add_node("suggestion", build_suggestion_node(strategy, llm, top_k=top_k))

    graph.add_edge(START, "route_intent")
    graph.add_conditional_edges("capture_entity", _route_after_capture, ["analysis", "ask_entity"])
    graph.add_conditional_edges(
        "route_intent",
        route_after_intent,
        ["capture_entity", "product_search", "fallback", "idle_reply"],
    )
    graph.add_edge("analysis", "suggestion")
    for node in ["ask_entity", "product_search", "fallback", "idle_reply", "suggestion"]:
        graph.add_edge(node, END)

    return graph.compile(checkpointer=checkpointer)


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    # One client shared by every LLM-touching node; structured output wraps it in a new runnable
    # rather than mutating it, so `analysis` and `suggestion` still get the plain model.
    llm = build_chat_model()
    return assemble_graph(
        build_default_classifier(llm),
        llm,
        build_active_strategy(),
        top_k=get_settings().search_top_k,
        checkpointer=checkpointer,
    )
