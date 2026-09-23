"""analysis node: free-knowledge business analysis from the 4 collected entities only.

Same hard requirement as the flexible variant: one plain LLM call whose input is built by
`common.analysis.analysis_messages` from the entities alone — no conversation history, no product
data. Its reply is shown to the user; `suggestion` doesn't read it back.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import LanguageModelLike

from consultant_bot.common.analysis import analysis_messages
from consultant_bot.common.entities import Entities
from consultant_bot.rigid.state import State


def build_analysis_node(llm: LanguageModelLike) -> Callable[[State], dict[str, Any]]:
    def analysis(state: State) -> dict[str, Any]:
        response = llm.invoke(analysis_messages(state.get("entities") or Entities()))
        return {"messages": [response]}

    return analysis
