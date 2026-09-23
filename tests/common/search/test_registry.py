import subprocess
import sys

import pytest

from consultant_bot.common.search.embedding_search import EmbeddingSearch, is_model_cached
from consultant_bot.common.search.filter_search import FilterSearch
from consultant_bot.common.search.hybrid_search import HybridSearch
from consultant_bot.common.search.products import load_products
from consultant_bot.common.search.registry import build_strategy
from consultant_bot.common.search.tfidf_search import TfidfSearch
from tests.support import FIXTURE_PATH


@pytest.mark.parametrize(("name", "expected"), [("filter", FilterSearch), ("tfidf", TfidfSearch)])
def test_build_strategy_returns_the_named_strategy(name: str, expected: type) -> None:
    strategy = build_strategy(name, load_products(FIXTURE_PATH))  # type: ignore[arg-type]
    assert isinstance(strategy, expected)


def test_building_the_graph_module_does_not_import_sentence_transformers() -> None:
    # A fresh interpreter: in-process, another test module may already have imported it.
    code = (
        "import sys, consultant_bot.flexible.graph; "
        "sys.exit('sentence_transformers' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False)  # noqa: S603
    assert result.returncode == 0


@pytest.mark.skipif(not is_model_cached(), reason="embedding model not cached locally")
def test_hybrid_blends_tfidf_with_embeddings() -> None:
    strategy = build_strategy("hybrid", load_products(FIXTURE_PATH))
    assert isinstance(strategy, HybridSearch)
    assert isinstance(strategy.lexical, TfidfSearch)
    assert isinstance(strategy.semantic, EmbeddingSearch)
