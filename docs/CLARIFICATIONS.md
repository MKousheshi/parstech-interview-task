# Clarifications

Questions and observations about the original task description (`TASK_SPEC.md`), separated out so the original spec stays untouched.

## Products data

- `products.json` has been added to the repo (29 items). It is a raw WooCommerce REST API product export, not a purpose-built catalog: each item carries many WooCommerce-specific fields (`sku`, `stock_status`, `attributes`, `variations`, `upsell_ids`, `_links`, etc.) alongside the useful ones (`name`, `description`, `short_description`, `price`/`regular_price`/`sale_price`, `categories`, `images`).
- `description` and `short_description` contain raw HTML (tables, headings) rather than plain text — needs cleanup/extraction before being fed to the search or the LLM.
- All product content is in Persian (e.g. "مدیریت پیج اینستاگرام (اقتصادی)" — Instagram page management), consistent with the digital-marketing-store premise.
- Not yet decided which fields are actually relevant for search/recommendation vs. noise to strip — see `DECISIONS.md`.

## Still open

Nothing currently open. Resolved items have moved to `DECISIONS.md`.
