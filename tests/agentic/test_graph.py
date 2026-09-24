import pytest
from langgraph.graph import END

from consultant_bot.agentic import graph
from consultant_bot.agentic.graph import _route_after_assistant, build_graph
from consultant_bot.agentic.state import State
from consultant_bot.common.entities import Entities
from consultant_bot.common.search.filter_search import FilterSearch
from consultant_bot.common.search.products import load_products

COMPLETE_ENTITIES = Entities(
    business_type="کافه", customer_type="B2C", location="تهران", sales_channel="اینستاگرام"
)


def test_build_graph_wires_all_expected_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Building the graph constructs real ChatOpenAI instances (which validate that *a* key is
    # present, but never call the API), so a dummy key is enough — no network access happens here.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    # The default hybrid strategy would load the embedding model; wiring doesn't depend on it.
    monkeypatch.setattr(graph, "build_active_strategy", lambda: FilterSearch(load_products()))

    app = build_graph()

    node_names = set(app.get_graph().nodes.keys())
    assert {"extract_entities", "assistant", "analysis", "suggestion"} <= node_names


def test_route_after_assistant_goes_to_analysis_when_complete_and_requested() -> None:
    state: State = {
        "messages": [],
        "entities": COMPLETE_ENTITIES,
        "consultation_requested": True,
        "consultation_done": False,
    }
    assert _route_after_assistant(state) == "analysis"


def test_route_after_assistant_ends_when_not_yet_requested() -> None:
    state: State = {
        "messages": [],
        "entities": COMPLETE_ENTITIES,
        "consultation_requested": False,
        "consultation_done": False,
    }
    assert _route_after_assistant(state) == END
