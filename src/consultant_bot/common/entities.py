"""Shared entity schema collected during the business consultation flow."""

from typing import TypedDict


class Entities(TypedDict, total=False):
    business_type: str
    customer_type: str
    location: str
    sales_channel: str
