from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from consultant_bot.agentic.tools.search_products import build_search_products_tool
from consultant_bot.common.search.filter_search import FilterSearch
from consultant_bot.common.search.products import load_products
from tests.support import FIXTURE_PATH


def _invoke_tool(tool: BaseTool, query: str, category: str | None = None) -> ToolMessage:
    args = {"query": query}
    if category is not None:
        args["category"] = category
    message = tool.invoke({"type": "tool_call", "name": tool.name, "args": args, "id": "call-1"})
    assert isinstance(message, ToolMessage)
    return message


def test_search_products_returns_formatted_content_and_hits_as_artifact() -> None:
    products = load_products(FIXTURE_PATH)
    tool = build_search_products_tool(FilterSearch(products), top_k=5)

    result = _invoke_tool(tool, "تلگرام")

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-1"
    assert "تلگرام" in result.content
    assert result.artifact[0].product.id == 7569


def test_search_products_no_match_reports_nothing_found() -> None:
    products = load_products(FIXTURE_PATH)
    tool = build_search_products_tool(FilterSearch(products), top_k=5)

    result = _invoke_tool(tool, "بیمه")

    assert result.artifact == []
    assert "یافت نشد" in result.content


def test_search_products_respects_top_k() -> None:
    products = load_products(FIXTURE_PATH)
    tool = build_search_products_tool(FilterSearch(products), top_k=1)

    result = _invoke_tool(tool, "مشاوره")

    assert len(result.artifact) == 1


def test_search_products_applies_category_filter() -> None:
    products = load_products(FIXTURE_PATH)
    tool = build_search_products_tool(FilterSearch(products), top_k=5)

    result = _invoke_tool(tool, "", category="اینستاگرام")

    assert [hit.product.id for hit in result.artifact] == [9177]


def test_search_products_ignores_a_guessed_category_that_matches_nothing() -> None:
    products = load_products(FIXTURE_PATH)
    tool = build_search_products_tool(FilterSearch(products), top_k=5)

    result = _invoke_tool(tool, "تلگرام", category="دسته‌ای که وجود ندارد")

    assert [hit.product.id for hit in result.artifact] == [7569]
