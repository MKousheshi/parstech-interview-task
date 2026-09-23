"""Builds the flexible (agentic) architecture's StateGraph.

Currently a trivial echo stub so `cli.py` has something real to invoke while the actual nodes
(`extract_entities`, `assistant`, `completion_check`, `analysis`, `suggestion`) are built out one
by one in later steps.
"""

from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from consultant_bot.flexible.state import State


def _echo(state: State) -> dict:
    last_message = state["messages"][-1]
    return {"messages": [AIMessage(content=f"echo: {last_message.content}")]}


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    graph = StateGraph(State)
    graph.add_node("echo", _echo)
    graph.add_edge(START, "echo")
    graph.add_edge("echo", END)
    return graph.compile(checkpointer=checkpointer)
