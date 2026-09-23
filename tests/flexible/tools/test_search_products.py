from pathlib import Path

from langchain_core.messages import ToolMessage

from consultant_bot.common.search.filter_search import FilterSearch
from consultant_bot.common.search.products import load_products
from consultant_bot.flexible.tools.search_products import build_search_products_tool

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "products_fixture.json"


def _invoke_tool(tool, query: str, category: str | None = None):  # type: ignore[no-untyped-def]
    args = {"query": query}
    if category is not None:
        args["category"] = category
    return tool.invoke({"type": "tool_call", "name": tool.name, "args": args, "id": "call-1"})


def test_search_products_returns_expected_hits_and_updates_state() -> None:
    products = load_products(FIXTURE_PATH)
    tool = build_search_products_tool(FilterSearch(products), top_k=5)

    result = _invoke_tool(tool, "تلگرام")

    assert result.update["last_shown_products"][0].product.id == 7569
    tool_message = result.update["messages"][0]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.tool_call_id == "call-1"
    assert "تلگرام" in tool_message.content


def test_search_products_no_match_reports_nothing_found() -> None:
    products = load_products(FIXTURE_PATH)
    tool = build_search_products_tool(FilterSearch(products), top_k=5)

    result = _invoke_tool(tool, "بیمه")

    assert result.update["last_shown_products"] == []
    assert "یافت نشد" in result.update["messages"][0].content


def test_search_products_respects_top_k() -> None:
    products = load_products(FIXTURE_PATH)
    tool = build_search_products_tool(FilterSearch(products), top_k=1)

    result = _invoke_tool(tool, "مشاوره")

    assert len(result.update["last_shown_products"]) == 1


def test_search_products_applies_category_filter() -> None:
    products = load_products(FIXTURE_PATH)
    tool = build_search_products_tool(FilterSearch(products), top_k=5)

    result = _invoke_tool(tool, "", category="اینستاگرام")

    assert [hit.product.id for hit in result.update["last_shown_products"]] == [9177]
