"""Runtime configuration shared by both the flexible and rigid architectures.

`OPENAI_API_KEY` is deliberately not read here: `langchain-openai` picks it up from the
environment directly wherever a chat model is constructed.
"""

import os

LLM_MODEL = os.environ.get("CONSULTANT_BOT_LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.environ.get("CONSULTANT_BOT_LLM_TEMPERATURE", "0.3"))

# One of "filter" (Phase 1), "tfidf" (Phase 2), "embedding" (Phase 3).
SEARCH_STRATEGY = os.environ.get("CONSULTANT_BOT_SEARCH_STRATEGY", "filter")
SEARCH_TOP_K = int(os.environ.get("CONSULTANT_BOT_SEARCH_TOP_K", "5"))
