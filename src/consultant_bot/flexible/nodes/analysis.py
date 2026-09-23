"""analysis node: free-knowledge business analysis, isolated from product data and the search tool.

A fresh LLM call built from a synthetic summary of the 4 entities only — deliberately *not* a
continuation of `state["messages"]` and *not* the same tool-bound model instance as `assistant`,
so neither prior search results in the conversation history nor the search tool itself can leak
into this call. This isolation is structural (the node only ever receives a plain, non-tool-bound
`llm` and builds its input with `common.analysis.analysis_messages`), not just a prompting
convention, since it's a documented hard requirement — see `docs/DECISIONS.md`.
"""

from typing import Any

from langchain_core.language_models import LanguageModelLike

from consultant_bot.common.analysis import analysis_messages
from consultant_bot.common.entities import Entities
from consultant_bot.common.messages import message_text
from consultant_bot.flexible.state import Node, State


def build_analysis_node(llm: LanguageModelLike) -> Node:
    def analysis(state: State) -> dict[str, Any]:
        entities = state.get("entities") or Entities()
        response = llm.invoke(analysis_messages(entities))
        text = response if isinstance(response, str) else message_text(response)
        return {"messages": [response], "analysis": text}

    return analysis
