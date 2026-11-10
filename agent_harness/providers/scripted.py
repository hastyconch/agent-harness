"""A deterministic provider for tests and offline demos."""

from __future__ import annotations

import copy
from typing import Callable, List, Union

from ..conversation import ModelRequest, ModelResponse
from .base import Provider, ProviderError

Step = Union[ModelResponse, Exception, Callable[[ModelRequest], ModelResponse]]


class ScriptedProvider(Provider):
    """Replays a fixed list of responses. A step may be a response, an
    exception to raise, or a function of the request. Every request is kept in
    ``requests`` (deep-copied) so tests can inspect exactly what was sent."""

    name = "scripted"

    def __init__(self, steps: List[Step], context_window: int = 200_000):
        self.steps = list(steps)
        self.context_window = context_window
        self.requests: List[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(copy.deepcopy(request))
        if not self.steps:
            raise ProviderError("ScriptedProvider ran out of steps")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step):
            return step(request)
        return step
