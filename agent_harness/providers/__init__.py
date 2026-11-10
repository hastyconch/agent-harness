from .base import Provider, ProviderError
from .scripted import ScriptedProvider

__all__ = ["Provider", "ProviderError", "ScriptedProvider", "AnthropicProvider"]


def __getattr__(name):
    # Import lazily so `anthropic` stays an optional dependency.
    if name == "AnthropicProvider":
        from .anthropic import AnthropicProvider

        return AnthropicProvider
    raise AttributeError(name)
