"""The one interface the agent loop needs from an LLM backend."""

from __future__ import annotations

import abc

from ..conversation import ModelRequest, ModelResponse


class ProviderError(Exception):
    """Raised by adapters. ``retryable`` tells the loop whether backing off and
    trying again could help (rate limits, overload, network)."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class Provider(abc.ABC):
    name: str = "provider"
    context_window: int = 200_000

    @abc.abstractmethod
    def complete(self, request: ModelRequest) -> ModelResponse:
        """Send one request and return the full (non-partial) response."""
