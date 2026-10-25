from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Sequence, Union

Message = Dict[str, str]


class Backend(ABC):
    """Anything that can continue a chat. Backends receive the full message list
    (system prompt included) and return the assistant's raw text."""

    name = "backend"

    @abstractmethod
    def generate(self, messages: Sequence[Message], max_new_tokens: int = 384) -> str:
        ...

    def generate_batch(self, batch: Sequence[Sequence[Message]], max_new_tokens: int = 384) -> List[str]:
        return [self.generate(m, max_new_tokens) for m in batch]

    def warm_up(self) -> None:
        """Optional: start waking the model server (used by the web app on page load)."""


class ScriptedBackend(Backend):
    """Returns canned responses. Used by the tests and for offline demos of the
    agent loop. Pass a list (consumed in order) or a function of the messages."""

    name = "scripted"

    def __init__(self, responses: Union[Sequence[str], Callable[[Sequence[Message]], str]]):
        self._fn = responses if callable(responses) else None
        self._queue = list(responses) if not callable(responses) else []
        self.calls: List[List[Message]] = []

    def generate(self, messages: Sequence[Message], max_new_tokens: int = 384) -> str:
        self.calls.append(list(messages))
        if self._fn is not None:
            return self._fn(messages)
        if not self._queue:
            raise RuntimeError("ScriptedBackend ran out of responses")
        return self._queue.pop(0)
