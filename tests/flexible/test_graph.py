from langgraph.graph import END

from consultant_bot.common.entities import Entities
from consultant_bot.flexible.graph import _route_after_assistant, build_graph

COMPLETE_ENTITIES = Entities(
    business_type="کافه", customer_type="B2C", location="تهران", sales_channel="اینستاگرام"
)


def test_build_graph_wires_all_expected_nodes(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Building the graph constructs real ChatOpenAI instances (which validate that *a* key is
    # present, but never call the API), so a dummy key is enough — no network access happens here.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")

    app = build_graph()

    node_names = set(app.get_graph().nodes.keys())
    assert {"extract_entities", "assistant", "analysis", "suggestion"} <= node_names


def test_route_after_assistant_goes_to_analysis_when_complete_and_requested() -> None:
    state = {
        "entities": COMPLETE_ENTITIES,
        "consultation_requested": True,
        "consultation_done": False,
    }
    assert _route_after_assistant(state) == "analysis"


def test_route_after_assistant_ends_when_not_yet_requested() -> None:
    state = {
        "entities": COMPLETE_ENTITIES,
        "consultation_requested": False,
        "consultation_done": False,
    }
    assert _route_after_assistant(state) == END
