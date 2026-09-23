"""Fixtures and typed test doubles shared across the test suite."""

from pathlib import Path
from typing import Any, ClassVar, TypeVar

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableConfig

from consultant_bot.common.entities import Entities

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "products_fixture.json"

COMPLETE_ENTITIES = Entities(
    business_type="کافه", customer_type="B2C", location="تهران", sales_channel="اینستاگرام"
)

T = TypeVar("T")


class ScriptedRunnable(Runnable[Any, T]):
    """Stands in for an LLM or structured-output chain: replays scripted responses in order
    (repeating the last one once they run out) and records every input it was invoked with.
    """

    def __init__(self, *responses: T) -> None:
        self._responses = list(responses)
        self.inputs: list[Any] = []

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> T:
        self.inputs.append(input)
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]

    @property
    def was_called(self) -> bool:
        return bool(self.inputs)


class ToolCallingFakeModel(GenericFakeChatModel):
    """A chat model for agent loops: replays scripted AI messages (tool calls included — so
    `bind_tools` is a no-op) and records the messages of every model call in `received`.
    """

    received: ClassVar[list[list[BaseMessage]]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ToolCallingFakeModel":
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        self.received.append(list(messages))
        return super()._generate(messages, *args, **kwargs)
