"""Side-by-side comparison of the three search strategies against realistic Persian queries.

Run manually to sanity-check relevance across phases:
`uv run python -m consultant_bot.common.search.eval`
"""

from consultant_bot.common.search.base import ProductHit, SearchStrategy
from consultant_bot.common.search.filter_search import FilterSearch
from consultant_bot.common.search.products import load_products
from consultant_bot.common.search.tfidf_search import TfidfSearch

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
    strategies: dict[str, SearchStrategy] = {
        "filter": FilterSearch(products),
        "tfidf": TfidfSearch(products),
    }
    from consultant_bot.common.search.embedding_search import EmbeddingSearch, is_model_cached

    if is_model_cached():
        strategies["embedding"] = EmbeddingSearch(products)
    else:
        print(
            "(skipping embedding strategy: model not cached locally — run once with network "
            "access to download and cache it)\n"
        )

    for query in QUERIES:
        print(f"=== query: {query!r} ===")
        for name, strategy in strategies.items():
            print(f"  -- {name} --")
            _print_hits(strategy.search(query, top_k=TOP_K))
        print()


if __name__ == "__main__":
    main()
