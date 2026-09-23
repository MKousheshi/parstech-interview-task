from consultant_bot.common.search.text import keyword_tokens, normalize


def test_normalize_unifies_arabic_letters_and_zwnj() -> None:
    assert normalize("كيف‌ها") == "کیف ها"


def test_keyword_tokens_drops_stopwords_and_single_characters() -> None:
    assert keyword_tokens("یک سایت فروشگاهی می‌خوام و ۱") == ["سایت", "فروشگاهی"]
