"""Shared entity schema collected during the business consultation flow."""

from typing import TypedDict


class Entities(TypedDict, total=False):
    business_type: str
    customer_type: str
    location: str
    sales_channel: str


# The 4 field names of `Entities`, kept as a single source of truth for code that needs to check
# "are all 4 present" or iterate over them (an `Entities` dict may omit unset fields entirely,
# since it's `total=False`, so that check can't be a bare `all(entities.values())`).
ENTITY_FIELDS: tuple[str, ...] = ("business_type", "customer_type", "location", "sales_channel")
