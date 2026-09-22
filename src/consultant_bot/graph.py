from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    greeting: str


def say_hello(state: State) -> State:
    return {"greeting": "Hello, world!"}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("say_hello", say_hello)
    graph.add_edge(START, "say_hello")
    graph.add_edge("say_hello", END)
    return graph.compile()
