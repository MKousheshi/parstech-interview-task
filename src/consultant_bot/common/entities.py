"""Shared entity schema collected during the business consultation flow.

The 4 fields, their Persian labels and the "are all 4 known" check live here once, so every node
that summarizes or checks entities agrees on them.
"""

from pydantic import BaseModel, ConfigDict, field_validator

# A value longer than this is cut short where it's quoted into a system prompt. Real answers
# ("کافه", "تهران") are a few words; a long one is either a pasted paragraph or an attempt to
# smuggle instructions into the system role.
MAX_QUOTED_VALUE_CHARS = 80


def quote_user_value(value: str) -> str:
    """`value` wrapped in «», shortened and with guillemets removed, for use inside a system prompt.

    Entity values are the user's own words. Quoting them marks them as data, and removing any «»
    they contain stops a value from closing the quote early and continuing as prompt text.
    """
    text = " ".join(value.replace("«", "").replace("»", "").split())
    if len(text) > MAX_QUOTED_VALUE_CHARS:
        text = text[:MAX_QUOTED_VALUE_CHARS].rstrip() + "…"
    return f"«{text}»"


ENTITY_LABELS: dict[str, str] = {
    "business_type": "نوع کسب‌وکار",
    "customer_type": "نوع مشتریان (B2B یا B2C)",
    "location": "موقعیت جغرافیایی",
    "sales_channel": "کانال فروش مجازی (وب‌سایت یا پیج)",
}


class Entities(BaseModel):
    """The 4 consultation entities; `None` means not yet known."""

    model_config = ConfigDict(frozen=True)

    business_type: str | None = None
    customer_type: str | None = None
    location: str | None = None
    sales_channel: str | None = None

    def value(self, field: str) -> str | None:
        """The value of one of the `ENTITY_FIELDS`, by name."""
        value: str | None = getattr(self, field)
        return value

    def is_complete(self) -> bool:
        return all(self.value(field) for field in ENTITY_FIELDS)

    def known(self) -> dict[str, str]:
        """Label -> value for every field that's set, in `ENTITY_FIELDS` order."""
        return {
            ENTITY_LABELS[field]: value for field in ENTITY_FIELDS if (value := self.value(field))
        }

    def missing_labels(self) -> list[str]:
        return [ENTITY_LABELS[field] for field in ENTITY_FIELDS if not self.value(field)]

    def quoted_summary(self) -> str:
        """`summary()` with each value run through `quote_user_value`, for system prompts."""
        return "\n".join(
            f"- {label}: {quote_user_value(value)}" for label, value in self.known().items()
        )

    def summary(self) -> str:
        """One "- label: value" line per field, for LLM prompts built from the entities alone."""
        return "\n".join(f"- {label}: {value}" for label, value in self.known().items())


ENTITY_FIELDS: tuple[str, ...] = tuple(ENTITY_LABELS)


# What an LLM sometimes writes as text when it means "no value". Read literally, "null" would be
# stored as the user's business type.
PLACEHOLDER_VALUES = frozenset({"null", "none", "n/a", "unknown", "نامشخص", "ندارد"})


class ReportedEntities(Entities):
    """`Entities` as an LLM's structured output reports them: a placeholder string means unset."""

    @field_validator(*ENTITY_FIELDS, mode="before")
    @classmethod
    def _placeholder_means_unset(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in PLACEHOLDER_VALUES:
            return None
        return value
