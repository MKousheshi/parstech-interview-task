"""Persian text normalization and keyword tokenization shared by the search strategies."""

# Arabic code points that look identical to their Persian counterparts but compare unequal, plus
# the zero-width non-joiner (ZWNJ), which joins or separates the same word inconsistently
# ("می‌خواهم" vs "میخواهم" vs "می خواهم") — treating it as a space makes both sides agree.
_CHAR_MAP = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "‌": " "})

# Function words and filler common in Persian queries. Substring-matched, any of these hits nearly
# every product description, which turns keyword scores into noise.
STOPWORDS: frozenset[str] = frozenset(
    {
        "و", "در", "به", "از", "که", "را", "رو", "با", "برای", "این", "آن", "اون", "یا", "هم",
        "تا", "اگر", "می", "نمی", "ها", "های", "ای", "یک", "یه", "است", "هست", "هستم", "بود",
        "کن", "کنم", "کنید", "کنیم", "بر", "روی", "ما", "من", "تو", "شما", "او", "چه", "چی",
        "هر", "همه", "نیز", "پس", "اما", "ولی", "باید", "خوام", "خواهم", "میخوام", "میخواهم",
        "دارم", "داریم", "همچنین", "بدم", "بده", "یعنی", "مثل", "چند", "چطور", "چگونه",
    }
)  # fmt: skip


def normalize(text: str) -> str:
    """Lowercases and unifies Persian/Arabic letter variants and ZWNJ, for matching only."""
    return text.translate(_CHAR_MAP).lower()


def keyword_tokens(query: str) -> list[str]:
    """The query's meaningful tokens: normalized, stopwords and single characters dropped."""
    return [
        token for token in normalize(query).split() if len(token) > 1 and token not in STOPWORDS
    ]
