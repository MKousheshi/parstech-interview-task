"""Side-by-side comparison of the three search strategies against realistic Persian queries.

Run manually to sanity-check relevance across phases:
`uv run python scripts/compare_search.py`
"""

from consultant_bot.common.config import SearchStrategyName
from consultant_bot.common.search.base import ProductHit, SearchStrategy
from consultant_bot.common.search.embedding_search import is_model_cached
from consultant_bot.common.search.products import load_products
from consultant_bot.common.search.registry import build_strategy

QUERIES = [
    "مدیریت پیج اینستاگرام",
    "طراحی سایت فروشگاهی",
    "تبلیغات در گوگل",
    "ارسال پیامک تبلیغاتی",
    # Compound/multi-facet query, issued as a single call on purpose: shows how each phase's
    # single-query-vector ranking degrades on it compared to decomposing it into separate calls.
    "یک سایت فروشگاهی می‌خوام و همچنین تبلیغات گوگل و مدیریت اینستاگرام",
]

TOP_K = 5


def _print_hits(hits: list[ProductHit]) -> None:
    if not hits:
        print("    (no hits)")
        return
    for hit in hits:
        print(f"    {hit.score:.3f}  [{hit.product.id}] {hit.product.name}")


def main() -> None:
    products = load_products()
    names: list[SearchStrategyName] = ["filter", "tfidf"]
    if is_model_cached():
        names.append("embedding")
    else:
        print(
            "(skipping embedding strategy: model not cached locally — run once with network "
            "access to download and cache it)\n"
        )
    strategies: dict[str, SearchStrategy] = {name: build_strategy(name, products) for name in names}

    for query in QUERIES:
        print(f"=== query: {query!r} ===")
        for name, strategy in strategies.items():
            print(f"  -- {name} --")
            _print_hits(strategy.search(query, top_k=TOP_K))
        print()


if __name__ == "__main__":
    main()
