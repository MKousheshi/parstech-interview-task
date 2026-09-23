from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from consultant_bot.flexible.graph import build_graph


def test_stub_graph_echoes_input() -> None:
    app = build_graph()
    result = app.invoke({"messages": [HumanMessage(content="hello")]})
    last_message = result["messages"][-1]
    assert isinstance(last_message, AIMessage)
    assert last_message.content == "echo: hello"


def test_stub_graph_persists_messages_across_turns_with_checkpointer() -> None:
    app = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-thread"}}

    app.invoke({"messages": [HumanMessage(content="first")]}, config=config)
    result = app.invoke({"messages": [HumanMessage(content="second")]}, config=config)

    contents = [message.content for message in result["messages"]]
    assert contents == ["first", "echo: first", "second", "echo: second"]
