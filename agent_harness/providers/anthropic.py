"""Claude adapter (Anthropic Python SDK).

Install with ``pip install anthropic``. The SDK is imported lazily so the rest
of the harness has no third-party dependencies.

Notes on the request shape:
- Thinking is adaptive and effort is set explicitly (Claude Opus 5.5 defaults
  to ``medium``; agentic work usually wants ``high``).
- Thinking blocks are kept as ``ProviderBlock(reasoning=True)`` and replayed
  unchanged; the context manager strips them whenever it rewrites history,
  because they are only valid with the exact prefix that produced them.
- Server-side refusal fallback is on by default (``fallbacks="default"``).
- Requests stream (large ``max_tokens``) and tools use eager input streaming,
  which means tool inputs can arrive truncated; the registry validates every
  input against its schema before running anything.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..conversation import (
    Block,
    Message,
    ModelRequest,
    ModelResponse,
    ProviderBlock,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    Usage,
)
from .base import Provider, ProviderError

_STOP_REASONS = {
    "end_turn": StopReason.END_TURN,
    "stop_sequence": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "refusal": StopReason.REFUSAL,
    "pause_turn": StopReason.PAUSED,
}

_REASONING_TYPES = {"thinking", "redacted_thinking"}

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        max_tokens: int = 64_000,
        effort: str = "high",
        context_window: int = 1_000_000,
        fallbacks: Optional[str] = "default",
        client: Any = None,
    ):
        if client is None:
            import anthropic  # lazy: only needed for this adapter

            client = anthropic.Anthropic()  # credentials resolved from env / `ant auth login`
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.context_window = context_window
        self.fallbacks = fallbacks

    # ------------------------------------------------------------ outbound

    def _block_out(self, block: Block) -> Optional[Dict[str, Any]]:
        if isinstance(block, TextBlock):
            return {"type": "text", "text": block.text}
        if isinstance(block, ToolCall):
            return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.arguments}
        if isinstance(block, ToolResultBlock):
            return {
                "type": "tool_result",
                "tool_use_id": block.tool_call_id,
                "content": block.content,
                "is_error": block.is_error,
            }
        if isinstance(block, ProviderBlock) and block.provider == self.name:
            return block.data
        return None  # another provider's opaque block

    def _messages_out(self, messages: List[Message]) -> List[Dict[str, Any]]:
        out = []
        for m in messages:
            content = [b for b in (self._block_out(x) for x in m.content) if b is not None]
            if content:
                out.append({"role": m.role, "content": content})
        return out

    # ------------------------------------------------------------- inbound

    @staticmethod
    def _dump(block: Any) -> Dict[str, Any]:
        if hasattr(block, "to_dict"):
            return block.to_dict()
        return block.model_dump(exclude_none=True)

    def _block_in(self, block: Any) -> Block:
        if block.type == "text":
            return TextBlock(block.text)
        if block.type == "tool_use":
            args = block.input
            if isinstance(args, dict):
                return ToolCall(block.id, block.name, args)
            return ToolCall(block.id, block.name, {}, parse_error="arguments were not a JSON object: %r" % (args,))
        return ProviderBlock(self.name, self._dump(block), reasoning=block.type in _REASONING_TYPES)

    # ---------------------------------------------------------------- call

    def complete(self, request: ModelRequest) -> ModelResponse:
        import anthropic

        params: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens or self.max_tokens,
            "system": request.system,
            "messages": self._messages_out(request.messages),
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
        }
        if request.tools:
            params["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                    "eager_input_streaming": True,
                }
                for t in request.tools
            ]
        betas = []
        if self.fallbacks:
            betas.append(FALLBACK_BETA)
            params["fallbacks"] = self.fallbacks
        if betas:
            params["betas"] = betas

        try:
            with self.client.beta.messages.stream(**params) as stream:
                msg = stream.get_final_message()
        except anthropic.RateLimitError as e:
            raise ProviderError("rate limited: %s" % e, retryable=True) from e
        except anthropic.InternalServerError as e:
            raise ProviderError("server error: %s" % e, retryable=True) from e
        except anthropic.APIConnectionError as e:
            raise ProviderError("connection error: %s" % e, retryable=True) from e
        except anthropic.APIStatusError as e:
            raise ProviderError("API error %s: %s" % (e.status_code, e), retryable=False) from e

        details = getattr(msg, "stop_details", None)
        u = msg.usage
        # input_tokens excludes cached prompt tokens; the context size is the sum.
        prompt_tokens = u.input_tokens + (getattr(u, "cache_creation_input_tokens", 0) or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0)
        return ModelResponse(
            content=[self._block_in(b) for b in msg.content],
            stop_reason=_STOP_REASONS.get(msg.stop_reason, StopReason.OTHER),
            usage=Usage(prompt_tokens, u.output_tokens),
            stop_details=self._dump(details) if details is not None else None,
        )
