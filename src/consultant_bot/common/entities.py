"""Shared entity schema collected during the business consultation flow.

The 4 fields, their Persian labels and the "are all 4 known" check live here once, so every node
that summarizes or checks entities agrees on them.
"""

from pydantic import BaseModel, ConfigDict

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

    def summary(self) -> str:
        """One "- label: value" line per field, for LLM prompts built from the entities alone."""
        return "\n".join(f"- {label}: {value}" for label, value in self.known().items())


ENTITY_FIELDS: tuple[str, ...] = tuple(ENTITY_LABELS)
